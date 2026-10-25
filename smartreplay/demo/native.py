"""Native session host client: durable archives, fenced IPC and lossless snapshots."""
from __future__ import annotations

from collections import OrderedDict
import ctypes
from ctypes import wintypes
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import threading
import time
import uuid
import wave

from .capture import Match, MatchStore, NO_WINDOW, ffmpeg_binary
from .native_process import attach_native_process


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + ".tmp")
    pending.write_text(json.dumps(value, allow_nan=False, indent=2), encoding="utf-8")
    pending.replace(path)


def _inside(path: Path, root: Path) -> Path:
    path = Path(path).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("A native media path is outside its session directory.")
    return path


def _alive(pid):
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        return False
    if os.name == "nt":
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            return False
        try:
            code = wintypes.DWORD()
            return bool(kernel.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


@dataclass(frozen=True)
class NativeSnapshot:
    match: Match
    origin_s: float
    session_id: str

    @property
    def path(self):
        return self.match.path

    @property
    def duration(self):
        return self.match.duration


class NativeRecorder:
    def __init__(self, root: Path, store: MatchStore, binary: Path | None = None, cache_mb=128):
        self.root, self.store = Path(root).resolve(), store
        self.session_root = self.root / "sessions"
        self.control_dir = self.root / "native"
        repo = Path(__file__).resolve().parents[2]
        self.checkout = repo / "experiments/gamehq-passive"
        self.binary = Path(binary or os.environ.get("SMARTREPLAY_NATIVE_EXE") or
                           self.checkout / "out-passive/GameHQPassiveReplay.exe").resolve()
        if not isinstance(cache_mb, int) or isinstance(cache_mb, bool) or not 8 <= cache_mb <= 1024:
            raise ValueError("The replay cache must be between 8 and 1024 MiB.")
        self.cache_mb = cache_mb
        self.process = None
        self.settings_file = None
        self._log = None
        self._rpc_lock, self._snapshot_lock = threading.Lock(), threading.Lock()
        self._snapshots = OrderedDict()
        self.session_root.mkdir(parents=True, exist_ok=True)
        (self.control_dir / "commands").mkdir(parents=True, exist_ok=True)
        (self.control_dir / "responses").mkdir(parents=True, exist_ok=True)

    def command(self):
        command = [str(self.binary), "--full-sessions", "--session-root", str(self.session_root),
                "--control-dir", str(self.control_dir), "--media-root", str(self.store.root),
                "--replay-cache-mb", str(self.cache_mb), "--log", str(self.control_dir / "native.jsonl")]
        if self.settings_file is not None:
            command += ["--settings", str(Path(self.settings_file).resolve())]
        return command

    def status(self):
        unavailable = {"available": False, "recording": False, "session_id": None, "active_session": None,
                       "cache": {"bytes": 0, "budget_bytes": self.cache_mb * 1024 * 1024}, "error": ""}
        try:
            data = json.loads((self.control_dir / "status.json").read_text(encoding="utf-8-sig"))
            if not isinstance(data, dict) or data.get("version") != 1 or not isinstance(data.get("instance_id"), str):
                raise ValueError("Unsupported native recorder status.")
            updated = datetime.fromisoformat(data["updated_at"].replace("Z", "+00:00"))
            age = (datetime.now(timezone.utc) - updated).total_seconds()
            if age < -5 or age > 8 or not _alive(data.get("pid")):
                return {**unavailable, "error": "Native recorder status is stale. Start the native recorder again."}
            return {**data, "available": True}
        except FileNotFoundError:
            return unavailable
        except (ValueError, TypeError, KeyError, OSError):
            return {**unavailable, "error": "Native recorder status is unreadable. Start it again."}

    def start(self):
        if self.status()["available"]:
            return self.request("start")
        if self.process and self.process.poll() is None:
            return {"ok": True, "starting": True}
        if not self.binary.is_file():
            raise RuntimeError("Build the native host first: experiments/gamehq-passive/spike/passive-replay/build.ps1")
        env = os.environ.copy()
        kit = self.checkout / "tools/Qt/6.8.3/mingw_64"
        compiler = self.checkout / "tools/Qt/Tools/mingw1310_64/bin"
        env["PATH"] = f"{kit / 'bin'}{os.pathsep}{compiler}{os.pathsep}{env.get('PATH', '')}"
        env["QML_IMPORT_PATH"] = str(kit / "qml")
        env["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(kit / "plugins/platforms")
        env["QT_FORCE_STDERR_LOGGING"] = "1"
        self._log = (self.control_dir / "host-stderr.log").open("ab")
        try:
            self.process = subprocess.Popen(self.command(), cwd=self.binary.parent, env=env,
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=self._log, creationflags=NO_WINDOW)
        except Exception:
            self._log.close()
            raise
        return {"ok": True, "starting": True}

    def recover(self, owner_lease):
        """Reconnect an orphan only after the prior Python service lost its lease."""
        if (getattr(owner_lease, "file", None) is None
                or Path(owner_lease.path).resolve() != self.root / "service.lock"):
            raise RuntimeError("Recorder recovery requires exclusive service ownership.")
        path = self.control_dir / "status.json"
        if not path.exists():
            return False
        try:
            prior = json.loads(path.read_text(encoding="utf-8-sig"))
            pid, instance = prior["pid"], prior["instance_id"]
            if prior.get("version") != 1 or not isinstance(instance, str) or not instance:
                raise ValueError("Invalid identity")
            process = attach_native_process(pid, self.binary, prior["updated_at"])
        except (KeyError, ValueError, TypeError) as exc:
            raise RuntimeError("Cannot verify the old recorder's status.") from exc
        if process is None:
            return False
        try:
            try:
                lock_pid = int((self.control_dir / "native.lock").read_text().splitlines()[0])
            except (OSError, ValueError, IndexError):
                lock_pid = None
            if lock_pid != pid:
                raise RuntimeError("Cannot verify that the old recorder owns this recording directory.")
            state = self.status()
            if state.get("available") and state.get("instance_id") == instance:
                try:
                    self.request("start", expected_instance=instance)
                    # Old in-flight answers died with their Python service.
                    voice = state.get("voice") or {}
                    if voice.get("request_id") and voice.get("state") in ("transcribing", "searching"):
                        self.voice_feedback({"id": voice["request_id"], "instance_id": instance},
                            "error", "Deja reconnected. Hold your voice shortcut to ask again.")
                    self.process = process
                    return True
                except RuntimeError:
                    pass
            # An unresponsive orphan cannot publish or accept new footage safely.
            # Try finalizing first; terminate only the verified, retained handle.
            try:
                self.request("stop", timeout=3, expected_instance=instance)
            except RuntimeError:
                pass
            process.terminate()
            process.wait(timeout=5)
            return False
        finally:
            if self.process is not process:
                process.close_handle()

    def request(self, kind, timeout=10, expected_instance=None, **fields):
        with self._rpc_lock:
            state = self.status()
            if not state["available"]:
                raise RuntimeError(state["error"] or "Start the native recorder first.")
            if expected_instance is not None and state.get("instance_id") != expected_instance:
                raise RuntimeError("The native recorder instance changed. Ask again in the current session.")
            instance, request_id = state["instance_id"], uuid.uuid4().hex
            command = {"id": request_id, "type": kind, "instance_id": instance, **fields}
            atomic_json(self.control_dir / "commands" / f"{request_id}.json", command)
            response_path = self.control_dir / "responses" / f"{request_id}.json"
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                if response_path.exists():
                    response = json.loads(response_path.read_text(encoding="utf-8-sig"))
                    response_path.unlink(missing_ok=True)
                    if not isinstance(response, dict) or response.get("id") != request_id or response.get("instance_id") != instance:
                        raise RuntimeError("The native reply belongs to a different recorder instance.")
                    if not response.get("ok"):
                        raise RuntimeError(str(response.get("error") or "The native request failed."))
                    return response
                current = self.status()
                if not current["available"] or current.get("instance_id") != instance:
                    exit_code = self.process.poll() if self.process else None
                    detail = current.get("error") or "The native recorder instance changed."
                    if exit_code is not None:
                        detail += f" Host exit code: {exit_code}."
                    raise RuntimeError("Native recorder unavailable during request: " + detail)
                time.sleep(.05)
            raise RuntimeError("The native recorder did not respond in time. The recording may still be running.")

    def flush(self, session_id, timeout=10):
        return self.request("flush", timeout=timeout, session_id=session_id)

    def stop(self):
        return self.request("stop", timeout=20)

    def play(self, match: Match, start_s: float, end_s: float, *, instance_id=None, voice_id=None, session_id=None):
        _inside(match.path, self.store.root)
        if not all(isinstance(t, (int, float)) and not isinstance(t, bool) and math.isfinite(t)
                   for t in (start_s, end_s)) or not 0 <= start_s < end_s <= match.duration + .05:
            raise ValueError("Invalid native replay bounds.")
        voice = {"voice_id": voice_id, "session_id": session_id} if voice_id is not None else {}
        return self.request("play", expected_instance=instance_id, path=str(match.path),
                            start_s=start_s, end_s=min(end_s, match.duration), **voice)

    def voice_feedback(self, request, state, message, transcript=""):
        """Progress never waits on capture RPC and always targets the originating host."""
        instance = request.get("instance_id")
        current = self.status()
        if not instance or not current.get("available") or current.get("instance_id") != instance:
            return False
        if state not in ("transcribing", "searching", "replaying", "no_match", "error", "idle"):
            raise ValueError("Invalid voice feedback state.")
        command_id = uuid.uuid4().hex
        atomic_json(self.control_dir / "commands" / f"{command_id}.json", {
            "id": command_id, "type": "voice_status", "instance_id": instance, "no_reply": True,
            "voice_id": request["id"], "state": state, "message": str(message)[:700],
            "transcript": str(transcript)[:800]})
        return True

    def claim_voice_request(self):
        """Atomically consume one valid current-generation manifest, including bad entries."""
        requests = self.control_dir / "voice" / "requests"
        voice_root = self.control_dir / "voice"
        state = self.status()
        self._clean_voice_orphans(state)
        for manifest in sorted(requests.glob("*.json")):
            # On Windows two simultaneous renames can resolve the same source
            # handle before either move completes. Exclusive creation makes the
            # claim belong to exactly one client before moving the manifest.
            claim_lock = manifest.with_suffix(".claim")
            try:
                with claim_lock.open("xb"):
                    pass
            except FileExistsError:
                continue
            claimed = manifest.with_name(manifest.name + ".claimed-" + uuid.uuid4().hex)
            try:
                manifest.rename(claimed)
            except FileNotFoundError:
                claim_lock.unlink(missing_ok=True)
                continue  # A second service client won the atomic claim.
            except OSError:
                claim_lock.unlink(missing_ok=True)
                raise
            request = None
            expected = voice_root / f"{manifest.stem}.wav" if re.fullmatch(r"[0-9a-f]{32}", manifest.stem) else None
            try:
                if claimed.stat().st_size > 16384:
                    raise ValueError("Voice request is too large.")
                request = json.loads(claimed.read_text(encoding="utf-8-sig"))
                if (not isinstance(request, dict) or type(request.get("version")) is not int
                        or request["version"] != 1 or expected is None or request.get("id") != manifest.stem
                        or not isinstance(request.get("session_id"), str)
                        or not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", request["session_id"])
                        or not isinstance(request.get("instance_id"), str)
                        or not state.get("available") or request["instance_id"] != state.get("instance_id")):
                    raise ValueError("Voice request belongs to an invalid or expired recorder session.")
                created = datetime.fromisoformat(request["created_at"].replace("Z", "+00:00"))
                if created.tzinfo is None or created.utcoffset().total_seconds() != 0:
                    raise ValueError("Voice request time must use UTC.")
                age = (datetime.now(timezone.utc) - created).total_seconds()
                if not -5 <= age <= 120:
                    raise ValueError("Voice request has expired. Hold push to talk and ask again.")
                supplied = request.get("path")
                if (not isinstance(supplied, str) or not Path(supplied).is_absolute()
                        or Path(supplied).resolve() != expected.absolute()
                        or expected.resolve() != expected.absolute()):
                    raise ValueError("Voice audio must stay in its request directory.")
                if not expected.is_file() or not 44 < expected.stat().st_size <= 3 * 1024 * 1024:
                    raise ValueError("Voice audio is missing or exceeds its size limit.")
                with wave.open(str(expected), "rb") as audio:
                    frames, rate = audio.getnframes(), audio.getframerate()
                    if (audio.getcomptype() != "NONE" or audio.getsampwidth() != 2
                            or audio.getnchannels() not in (1, 2) or not 8000 <= rate <= 48000
                            or frames <= 0 or frames > rate * 15
                            or len(audio.readframes(frames)) != frames * audio.getnchannels() * 2):
                        raise ValueError("Voice audio must be complete PCM audio of at most 15 seconds.")
                return {**request, "path": expected}
            except (ValueError, TypeError, KeyError, AttributeError, OSError, EOFError, wave.Error) as exc:
                if expected:
                    feedback = {"id": manifest.stem, "instance_id": state.get("instance_id")}
                    if isinstance(request, dict):
                        feedback["instance_id"] = request.get("instance_id")
                    try:
                        self.voice_feedback(feedback, "error", str(exc) or "Voice audio could not be read.")
                    except (OSError, RuntimeError, ValueError):
                        pass
                if expected:
                    expected.unlink(missing_ok=True)
            finally:
                try:
                    claimed.unlink(missing_ok=True)
                finally:
                    claim_lock.unlink(missing_ok=True)
        return None

    def _clean_voice_orphans(self, state):
        """A crashed service/host cannot retain expired microphone files forever."""
        directory = self.control_dir / "voice"
        requests = directory / "requests"
        cutoff = time.time() - 120
        voice = state.get("voice") or {}
        active = voice.get("request_id") if voice.get("state") in ("listening", "transcribing", "searching") else None
        for artifact in requests.glob("*"):
            match = re.fullmatch(r"([0-9a-f]{32})(?:\.claim|\.json\.claimed-.+)", artifact.name)
            try:
                if match and match[1] != active and artifact.stat().st_mtime < cutoff:
                    artifact.unlink(missing_ok=True)
            except OSError:
                continue
        for audio in directory.glob("*.wav"):
            try:
                if (re.fullmatch(r"[0-9a-f]{32}", audio.stem) and audio.stem != active
                        and audio.stat().st_mtime < cutoff
                        and not (requests / f"{audio.stem}.json").exists()
                        and not (requests / f"{audio.stem}.claim").exists()):
                    audio.unlink(missing_ok=True)
            except OSError:
                continue

    def _validate(self, manifest):
        if not isinstance(manifest, dict) or not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", str(manifest.get("id", ""))):
            raise ValueError("Invalid native session identity.")
        duration = manifest.get("duration_s")
        if isinstance(duration, bool) or not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration < 0:
            raise ValueError("Invalid native session duration.")
        segments = manifest.get("segments")
        if not isinstance(segments, list):
            raise ValueError("Invalid native session segments.")
        prior = 0.0
        directory = self.session_root / manifest["id"]
        for segment in segments:
            if not isinstance(segment, dict) or not isinstance(segment.get("path"), str) or not Path(segment["path"]).is_absolute():
                raise ValueError("Invalid native segment path.")
            _inside(Path(segment["path"]), directory)
            a, b = segment.get("start_s"), segment.get("end_s")
            if not all(isinstance(t, (int, float)) and not isinstance(t, bool) and math.isfinite(t) for t in (a, b)):
                raise ValueError("Invalid native segment timestamps.")
            if abs(a - prior) > .05 or b <= a or b > duration + .1:
                raise ValueError("Native session timestamps are inconsistent.")
            prior = b
        if abs(prior - duration) > .05:
            raise ValueError("Native session duration does not match its finalized timeline.")
        return manifest

    def session(self, session_id):
        if not isinstance(session_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", session_id):
            raise ValueError("Invalid native session identity.")
        try:
            data = json.loads((self.session_root / session_id / "session.json").read_text(encoding="utf-8-sig"))
        except FileNotFoundError as exc:
            raise ValueError("That recording session is not available.") from exc
        if not isinstance(data, dict) or data.get("id") != session_id:
            raise ValueError("Native session identity does not match its directory.")
        return self._validate(data)

    def sessions(self):
        items = []
        for path in self.session_root.glob("*/session.json"):
            try:
                items.append(self.session(path.parent.name))
            except (ValueError, KeyError, OSError):
                continue
        return sorted(items, key=lambda item: item.get("created_at", ""), reverse=True)

    def snapshot(self, session, start_s, end_s):
        session = self._validate(session)
        if not all(isinstance(t, (int, float)) and not isinstance(t, bool) and math.isfinite(t) for t in (start_s, end_s)):
            raise ValueError("Invalid session replay range.")
        if not 0 <= start_s < end_s <= session["duration_s"] + .05:
            raise ValueError("Replay range is outside recorded session footage.")
        selected = [segment for segment in session["segments"] if segment["end_s"] > start_s and segment["start_s"] < end_s]
        if not selected:
            raise ValueError("No finalized footage is available in this range yet.")
        for segment in selected:
            if not Path(segment["path"]).is_file():
                raise ValueError("A recorded session segment is missing; this interval cannot be replayed.")
        for left, right in zip(selected, selected[1:]):
            if abs(left["end_s"] - right["start_s"]) > .1:
                raise ValueError("Recorded session footage is missing from this interval.")
        key = hashlib.sha256(json.dumps([(s["path"], Path(s["path"]).stat().st_size,
                Path(s["path"]).stat().st_mtime_ns) for s in selected]).encode()).hexdigest()
        with self._snapshot_lock:
            if key in self._snapshots and self._snapshots[key].path.exists():
                self._snapshots.move_to_end(key)
                return self._snapshots[key]
            output = self.store.allocate()
            listing = output.with_name("segments.txt")
            rows = []
            for segment in selected:
                escaped = str(Path(segment["path"]).resolve()).replace("\\", "/").replace("'", "'\\''")
                rows.append(f"file '{escaped}'\nduration {segment['end_s'] - segment['start_s']:.9f}")
            listing.write_text("\n".join(rows) + "\n", encoding="utf-8")
            command = [ffmpeg_binary(), "-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0",
                       "-i", str(listing), "-map", "0:v:0", "-map", "0:a?", "-c", "copy", "-movflags", "+faststart", str(output)]
            try:
                result = subprocess.run(command, capture_output=True, timeout=120, creationflags=NO_WINDOW)
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError("Preparing the replay took too long. Recording is still independent of this request.") from exc
            if result.returncode:
                output.with_name("remux-error.log").write_bytes(result.stderr)
                raise RuntimeError("The native footage could not be assembled into a replay. Recording has not been stopped.")
            match = self.store.finish(output, "native")
            snapshot = NativeSnapshot(match, float(selected[0]["start_s"]), session["id"])
            metadata_path = output.with_suffix(".json")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata.update({"origin_s": snapshot.origin_s, "session_id": snapshot.session_id})
            atomic_json(metadata_path, metadata)
            self._snapshots[key] = snapshot
            while len(self._snapshots) > 128:
                self._snapshots.popitem(last=False)
            return snapshot

    def close(self):
        if self.process and self.process.poll() is None:
            try:
                self.stop()
            except RuntimeError:
                pass
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        if self.process and hasattr(self.process, "close_handle"):
            self.process.close_handle()
        if self._log:
            self._log.close()
