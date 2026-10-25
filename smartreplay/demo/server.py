"""Loopback-only demo service; long work runs outside HTTP request threads."""
from __future__ import annotations

from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import re
import threading
import time

from .. import player_settings as ps
from ..providers import PROVIDERS
from .capture import MatchStore, game_window
from .native import NativeRecorder
from .search import create_model, find_moments

STATIC = Path(__file__).with_name("static")
_SESSION_MONITOR_INTERVAL_S = 1.0
_VOICE_POLL_INTERVAL_S = .25


class DemoSession:
    def __init__(self, root: Path, model_factory=None, recorder_factory=None, window_finder=game_window,
                 native_factory=NativeRecorder, session_search_factory=None, voice_transcriber=None,
                 settings_file: Path | None = None, native_recovery=None):
        self.root = Path(root).resolve()
        self.store = MatchStore(self.root / "matches")
        # With a settings file, model.provider (edited in deja) picks the backend and
        # is followed live; SMARTREPLAY_DEMO_BACKEND still wins at startup.
        self.settings_file = Path(settings_file) if settings_file else None
        self.backend = os.environ.get("SMARTREPLAY_DEMO_BACKEND") or self._settings_provider() or "codex"
        self._model_transport = None
        self._model_generation = 0
        self.model_factory = model_factory or self._make_model
        self.window_finder = window_finder
        self.native = native_factory(self.root, self.store)
        self.native.settings_file = self.settings_file
        if native_recovery is not None:
            native_recovery(self.native)
        self.session_search_factory = session_search_factory
        self.queries = None
        self.selected_session_id = None
        self.import_mode = False
        self.capture_control_busy = False
        self.capture_error = ""
        self.capture_worker = None
        self.lock = threading.RLock()
        self.busy = False
        self.phase = "idle"
        self.match = None
        self.result = None
        self.error = ""
        self.recorder = None
        self.worker = None
        self.started_at = None
        self.record_seconds = 25
        self.model_status = {"state": "unchecked", "message": "Check the model connection before presenting."}
        self._lifecycle_lock = threading.Lock()
        self._closed = False
        self._backend_changing = False
        self._monitor_requested = False
        self._monitor_worker = None
        self._monitor_stop = None
        self._monitor_wake = None
        self._monitor_session_id = None
        self._monitor_error = ""
        self.voice_transcriber = voice_transcriber
        self.voice = {"state": "idle", "message": "Hold your push-to-talk shortcut to ask about gameplay.",
                      "request_id": "", "transcript": ""}
        self.voice_model = {"state": "waiting", "message": "Speech recognition prepares when the recorder starts."}
        self._voice_stop = threading.Event()
        self._active_voice_request = None
        self._voice_worker = threading.Thread(target=self._watch_voice, name="native-voice-monitor", daemon=True)
        self._voice_worker.start()
        self._settings_worker = None
        if self.settings_file:
            self._settings_worker = threading.Thread(target=self._watch_settings, name="model-provider-setting",
                                                     daemon=True)
            self._settings_worker.start()
        native = self._native_status()
        if native.get("available") and (native.get("enabled") or native.get("recording")):
            self._ensure_monitor()

    def _make_model(self):
        # Share the connection, not mutable timeout/API settings used by indexing.
        with self.lock:
            if self._closed or self._backend_changing:
                raise RuntimeError("This model connection is closed or changing.")
            model = create_model(self.backend, self.root / "model", transport=self._model_transport)
            if self.backend == "codex":
                self._model_transport = getattr(model, "transport", None)
                model._owns_transport = False
            return model

    def _settings_provider(self):
        if not self.settings_file:
            return None
        return ps.load(self.settings_file)[0]["model"]["provider"]

    def _settings_stamp(self):
        try:
            return self.settings_file.stat().st_mtime_ns
        except OSError:
            return None

    def _watch_settings(self):
        """Follow model.provider: switch backend when deja saves a new one, then check it.

        Only saves after startup count, so SMARTREPLAY_DEMO_BACKEND keeps its startup choice.
        """
        seen = self._settings_stamp()
        while not self._voice_stop.wait(1.0):  # set by close()
            stamp = self._settings_stamp()
            if stamp == seen:
                continue
            provider = self._settings_provider()
            with self.lock:
                if self._closed:
                    return
                waiting = self.busy or self._backend_changing
            if provider == self.backend:
                seen = stamp
                continue
            if waiting:
                continue  # look again next second, once the current operation is done
            try:
                self.set_backend(provider, persist=False)
                self.check_model()
                seen = stamp
            except (RuntimeError, ValueError):
                pass

    def _save_provider(self, backend):
        settings, _ = ps.load(self.settings_file)
        if settings["model"]["provider"] == backend:
            return
        settings["model"]["provider"] = backend
        try:
            ps.save(settings, self.settings_file)
        except (ps.SettingsError, OSError):
            pass  # the switch itself stands; deja shows the file's problems

    def _close_model_transport(self):
        with self.lock:
            transport, self._model_transport = self._model_transport, None
        if transport is not None:
            transport.close()

    def _native_status(self):
        fallback = {"available": False, "enabled": False, "recording": False,
                    "session_id": None, "active_session": None, "cache": {}, "error": ""}
        try:
            state = self.native.status()
            if (not isinstance(state, dict)
                    or not all(isinstance(state.get(name), bool) for name in ("available", "recording"))
                    or ("enabled" in state and not isinstance(state["enabled"], bool))):
                raise ValueError("Native recorder status is unreadable.")
            return {**fallback, **state}
        except Exception as exc:
            return {**fallback, "error": str(exc)[:700] or "Native recorder status is unreadable."}

    @staticmethod
    def _native_session_id(state):
        session_id = state.get("session_id")
        if not session_id and isinstance(state.get("active_session"), dict):
            session_id = state["active_session"].get("id")
        return session_id if isinstance(session_id, str) and session_id else None

    def _ensure_monitor(self, *, wake=False):
        with self.lock:
            if self._closed:
                return
            self._monitor_requested = True
            if self._backend_changing:
                return
            if self._monitor_worker and self._monitor_worker.is_alive():
                if wake:
                    self._monitor_wake.set()
                return
            stop, wake_signal = threading.Event(), threading.Event()
            self._monitor_stop, self._monitor_wake = stop, wake_signal
            self._monitor_worker = threading.Thread(target=self._monitor_sessions, args=(stop, wake_signal),
                                                     name="native-session-monitor", daemon=True)
            self._monitor_worker.start()

    def _stop_monitor(self):
        with self.lock:
            worker = self._monitor_worker
            if self._monitor_stop:
                self._monitor_stop.set()
                self._monitor_wake.set()
            self._monitor_worker = None
        if worker and worker is not threading.current_thread():
            worker.join(timeout=2)

    def _monitor_sessions(self, stop, wake):
        while not stop.is_set():
            try:
                state = self._native_status()
                current = self._native_session_id(state) if state.get("available") else None
                with self.lock:
                    previous = self._monitor_session_id
                    if current:
                        self._monitor_session_id = current
                # Refresh a session's final partial window even after status clears its identity.
                for session_id in dict.fromkeys(s for s in (previous, current) if s):
                    manifest = self.native.session(session_id)
                    with self.lock:
                        if stop.is_set() or self._closed or self._backend_changing:
                            return
                        self._queries().refresh(manifest)
                with self.lock:
                    if not stop.is_set():
                        self._monitor_error = str(state.get("error") or "")[:700]
            except Exception as exc:
                with self.lock:
                    if not stop.is_set():
                        self._monitor_error = str(exc)[:700] or "Background session indexing failed."
            wake.wait(_SESSION_MONITOR_INTERVAL_S)
            wake.clear()

    def _watch_voice(self):
        warmed = False
        while not self._voice_stop.is_set():
            try:
                if not callable(getattr(self.native, "claim_voice_request", None)):
                    return
                state = self._native_status()
                with self.lock:
                    queries = self.queries
                    priority = self.busy or (state.get("voice") or {}).get("state") in (
                        "listening", "transcribing", "searching")
                reserve = getattr(queries, "set_interactive_pending", None)
                if callable(reserve):
                    reserve(priority)
                # Wait for a real host before loading the local speech model. The watcher
                # itself starts even when capture is absent, without requiring a page poll.
                if state.get("available") and state.get("instance_id"):
                    if self.voice_transcriber is None:
                        from .voice import VoiceTranscriber
                        self.voice_transcriber = VoiceTranscriber()
                    if not warmed:
                        warmed = True
                        with self.lock:
                            self.voice_model = {"state": "warming", "message": "Preparing local speech recognition…"}
                        warmup = getattr(self.voice_transcriber, "warmup", None)
                        try:
                            if callable(warmup):
                                warmup()
                            with self.lock:
                                self.voice_model = {"state": "ready", "message": "Local speech recognition is ready."}
                        except Exception as exc:
                            with self.lock:
                                self.voice_model = {"state": "error", "message": str(exc)[:700]}
                self._poll_voice_request()
            except Exception as exc:
                with self.lock:
                    if not self._closed:
                        self.voice = {**self.voice, "state": "error", "message": str(exc)[:700] or "Voice requests could not be read."}
            self._voice_stop.wait(_VOICE_POLL_INTERVAL_S)

    def _poll_voice_request(self):
        claim = getattr(self.native, "claim_voice_request", None)
        if not callable(claim):
            return
        with self.lock:
            if self._closed or self.busy or self._backend_changing:
                return
            request = claim()
            if request is None:
                return
            self._begin("transcribing")
            self._active_voice_request = request
            self.result = None
            self._run(lambda: self._answer_voice(request))

    def _voice_update(self, request, state, message, transcript=""):
        with self.lock:
            self.voice = {"state": state, "message": message, "transcript": transcript,
                          "request_id": request["id"]}
            if self._closed:
                return
        # Feedback is best effort. A stopped or replaced host must not turn a valid
        # search into a new command addressed to the replacement recorder.
        try:
            self.native.voice_feedback(request, state, message, transcript)
        except (OSError, RuntimeError, ValueError):
            pass

    def _assert_voice_current(self, request):
        with self.lock:
            if self._closed:
                raise RuntimeError("This replay session is closed. The voice request was canceled.")
        state = self._native_status()
        if not state.get("available") or state.get("instance_id") != request["instance_id"]:
            raise RuntimeError("The recorder changed while processing speech. Hold push to talk and ask again.")
        voice = state.get("voice")
        if isinstance(voice, dict) and "request_id" in voice and (
                voice.get("request_id") != request["id"] or voice.get("state") not in ("transcribing", "searching")):
            raise RuntimeError("The voice request was canceled or expired. Hold push to talk and ask again.")

    @staticmethod
    def _confirmed_interval(item, duration):
        a, b = item.get("start_s"), item.get("end_s")
        return (item.get("verdict") == "yes"
                and all(isinstance(t, (int, float)) and not isinstance(t, bool) and math.isfinite(t) for t in (a, b))
                and 0 <= a < b <= duration)

    def _answer_voice(self, request):
        transcript = ""
        try:
            self._assert_voice_current(request)
            self._voice_update(request, "transcribing", "Transcribing your question locally…")
            try:
                transcript = self.voice_transcriber.transcribe(request["path"])
            finally:
                request["path"].unlink(missing_ok=True)
            if not isinstance(transcript, str) or not transcript.strip() or len(transcript) > 800:
                raise ValueError("No clear question was heard. Hold push to talk and try again.")
            transcript = transcript.strip()
            with self.lock:
                self.voice_model = {"state": "ready", "message": "Local speech recognition is ready."}
            self._assert_voice_current(request)
            with self.lock:
                self.phase = "searching"
                self.selected_session_id, self.import_mode = request["session_id"], False
            self._voice_update(request, "searching", "Finding and checking your moment. Recording continues…", transcript)
            snapshot, result = self._search_session(transcript, request["session_id"], "session", None)
            self._assert_voice_current(request)
            confirmed = next((item for item in result["matches"]
                              if snapshot and self._confirmed_interval(item, snapshot.duration)), None)
            result["native_auto_played"] = False
            if confirmed:
                # Search exports only a short evidence window. Once its start is
                # confirmed, freeze a new replay through the current capture edge,
                # including gameplay recorded while the model was searching.
                start = snapshot.origin_s + confirmed["start_s"]
                self._voice_update(request, "searching", "Preparing replay through the present…", transcript)
                self.native.flush(request["session_id"])
                self._assert_voice_current(request)
                manifest = self.native.session(request["session_id"])
                playback = self.native.snapshot(manifest, start, manifest["duration_s"])
                self._assert_voice_current(request)
                playback_start = start - playback.origin_s
                playback_end = min(playback.duration, manifest["duration_s"] - playback.origin_s)
                self.native.play(playback.match, playback_start, playback_end,
                                 instance_id=request["instance_id"], voice_id=request["id"], session_id=request["session_id"])
                result["native_auto_played"] = True
                result["native_playback"] = {"match_id": playback.match.id,
                    "session_start_s": start, "session_end_s": playback.origin_s + playback_end}
            if self._publish_session_result(snapshot, result):
                self._voice_update(request, "replaying" if confirmed else "no_match",
                    "Replaying the confirmed moment." if confirmed else "No matching moment was confirmed. Try another description.", transcript)
        except Exception as exc:
            message = str(exc)[:700] or "The voice question failed. Hold push to talk and retry."
            with self.lock:
                self.error = message
            self._voice_update(request, "error", message, transcript)
        finally:
            try:
                request["path"].unlink(missing_ok=True)
            finally:
                with self.lock:
                    if self._active_voice_request is request:
                        self._active_voice_request = None

    def _voice_snapshot(self, native):
        voice = dict(self.voice)
        host = native.get("voice")
        if isinstance(host, dict) and (host.get("state") == "listening"
                or (host.get("request_id") and host.get("request_id") != voice.get("request_id"))
                or ("request_id" in host and host.get("state") in ("error", "replaying", "no_match")
                    and host.get("request_id") == voice.get("request_id"))
                or ("request_id" in host and not host.get("request_id") and host.get("state") == "idle")
                or not voice.get("request_id")):
            transcript = voice.get("transcript", "")
            voice.update(host)
            if not voice.get("transcript") and host.get("request_id") in ("", self.voice.get("request_id")):
                voice["transcript"] = transcript
        return {**voice, "model": dict(self.voice_model)}

    def snapshot(self):
        window = self.window_finder()
        native = self._native_status()
        if native.get("available") and (native.get("enabled") or native.get("recording")):
            self._ensure_monitor()
        archive_error = ""
        try:
            sessions = self.native.sessions()
        except Exception as exc:
            sessions = []
            archive_error = str(exc)[:700] or "Recording session history is unreadable."
        with self.lock:
            selected = self.selected_session_id
            if not selected and not self.import_mode:
                selected = native.get("session_id") or (sessions[0]["id"] if sessions else None)
            manifest = next((s for s in sessions if s["id"] == selected), None)
        indexing = {"indexed_until_s": 0, "duration_s": manifest["duration_s"] if manifest else 0,
                    "indexing": False, "pending": 0, "error": ""}
        if manifest:
            try:
                with self.lock:
                    queries = self._queries()
                    queries.refresh(manifest)
                    indexing = queries.status(manifest["id"])
            except Exception as exc:
                indexing["error"] = str(exc)[:700] or "Session indexing failed."
        with self.lock:
            indexing["error"] = "; ".join(dict.fromkeys(str(message) for message in (
                indexing.get("error"), self._monitor_error, archive_error, native.get("error")) if message))
            return {"busy": self.busy, "phase": self.phase, "match": self.match.public() if self.match else None,
                    "result": self.result, "error": self.error or self.capture_error,
                    "model": dict(self.model_status), "backend": self.backend,
                    "voice": self._voice_snapshot(native),
                    "native": {k: native.get(k) for k in ("available", "enabled", "recording", "session_id", "cache", "error")},
                    "capture_control_busy": self.capture_control_busy,
                    "sessions": [{k: s.get(k) for k in ("id", "game", "created_at", "state", "duration_s")} for s in sessions],
                    "selected_session_id": selected, "index": indexing,
                    "game": {"running": bool(window), "title": window["title"] if window else "",
                             "minimized": window.get("minimized", False) if window else False},
                    "recorded_s": round(time.monotonic() - self.started_at, 1) if self.started_at and self.phase == "recording" else 0,
                    "record_seconds": self.record_seconds}

    def _queries(self):
        with self.lock:
            if self._closed:
                raise RuntimeError("This replay session is closed.")
            if self._backend_changing:
                raise RuntimeError("The model connection is changing. Session indexing will resume shortly.")
            if self.queries is None:
                if self.session_search_factory is None:
                    from .session_search import SessionSearch
                    self.session_search_factory = SessionSearch
                generation = self._model_generation
                def model_factory():
                    with self.lock:
                        if self._closed:
                            raise RuntimeError("This model connection is closed.")
                        if self._backend_changing or generation != self._model_generation:
                            raise RuntimeError("The model connection changed; old indexing work was discarded.")
                        return self.model_factory()
                self.queries = self.session_search_factory(self.root / "indexes", model_factory, self.native.snapshot)
            return self.queries

    def _begin(self, phase):
        if self._closed:
            raise RuntimeError("This replay session is closed.")
        if self._backend_changing:
            raise RuntimeError("Wait for the model connection change to finish.")
        if self.busy:
            raise RuntimeError("Wait for the current operation to finish.")
        self.busy, self.phase, self.error = True, phase, ""

    def _run(self, action):
        def work():
            try:
                action()
            except Exception as exc:
                with self.lock:
                    self.error = str(exc)[:700] or "The operation failed. Please retry."
            finally:
                with self.lock:
                    self.busy = False
                    self.phase = "ready" if self.match else "idle"
                    self.recorder = None
        self.worker = threading.Thread(target=work, daemon=True)
        self.worker.start()

    def import_video(self, source: Path):
        with self.lock:
            self._begin("finalizing")
        try:
            match = self.store.import_video(source)
            if match.duration > 35:
                raise ValueError("Use a short recording of at most 35 seconds.")
            with self.lock:
                self.match, self.result, self.error = match, None, ""
                self.import_mode = True
                self.selected_session_id = None
            return match
        finally:
            with self.lock:
                self.busy = False
                self.phase = "ready" if self.match else "idle"

    def record(self, seconds=25):
        # Capture control has its own job state. A question never owns the recorder.
        with self.lock:
            self.import_mode = False
        self._capture_control(self.native.start, monitor=True)

    def stop(self):
        def stop_recording():
            # Remember the identity before the host clears it in its stopped status.
            session_id = self._native_session_id(self._native_status())
            if session_id:
                with self.lock:
                    self._monitor_session_id = session_id
            self.native.stop()
        self._capture_control(stop_recording, monitor=True)

    def _capture_control(self, action, *, monitor=False):
        def work():
            try:
                with self.lock:
                    if self._closed:
                        return
                action()
                if monitor:
                    self._ensure_monitor(wake=True)
            except Exception as exc:
                with self.lock:
                    self.capture_error = str(exc)[:700]
            finally:
                with self.lock:
                    self.capture_control_busy = False
        with self.lock:
            if self._closed:
                raise RuntimeError("This replay session is closed.")
            if self.capture_control_busy:
                raise RuntimeError("Wait for the recording control to finish.")
            self.capture_control_busy, self.capture_error = True, ""
            self.capture_worker = threading.Thread(target=work, daemon=True)
            self.capture_worker.start()

    def ask(self, question, match_id=None, session_id=None, scope="session", start_s=None):
        if not isinstance(question, str) or not question.strip() or len(question) > 800:
            raise ValueError("Enter a question of 1–800 characters.")
        if session_id:
            return self._ask_session(question, session_id, scope, start_s)
        with self.lock:
            if not self.match or self.match.id != match_id:
                raise ValueError("Select the current match before asking a question.")
            self._begin("searching")
            match = self.match
            self.result = None

            def search():
                model = self.model_factory()
                result = find_moments(match.path, question, model)
                result["match_id"] = match.id
                result["answered_at"] = datetime.now(timezone.utc).isoformat()
                # Preserve the evidence with the exact recording used for the answer.
                answer = match.path.with_name(f"answer-{time.time_ns()}.json")
                answer.write_text(json.dumps(result, indent=2), encoding="utf-8")
                with self.lock:
                    self.result = result
            self._run(search)

    def _ask_session(self, question, session_id, scope, start_s):
        if scope not in ("session", "recent", "window"):
            raise ValueError("Choose session, recent footage or a timestamp window.")
        self.native.session(session_id)
        with self.lock:
            self._begin("searching")
            self.selected_session_id, self.import_mode, self.result = session_id, False, None
            def search():
                snapshot, result = self._search_session(question, session_id, scope, start_s)
                self._publish_session_result(snapshot, result)
            self._run(search)

    def _search_session(self, question, session_id, scope, start_s):
        with self.lock:
            if self._closed:
                raise RuntimeError("This replay session is closed.")
        state = self.native.status()
        if state.get("recording") and state.get("session_id") == session_id:
            self.native.flush(session_id)
        manifest = self.native.session(session_id)
        duration = manifest["duration_s"]
        if duration <= 0:
            raise ValueError("No finalized footage is available yet. Play for a few seconds and retry.")
        a = b = None
        if scope == "recent":
            a, b = max(0, duration - 25), duration
        elif scope == "window":
            if not isinstance(start_s, (int, float)) or isinstance(start_s, bool):
                raise ValueError("Enter the session timestamp in seconds.")
            a, b = start_s, min(duration, start_s + 25)
            if not 0 <= a < b:
                raise ValueError("That timestamp is outside recorded footage.")
        response = self._queries().ask(manifest, question, start_s=a, end_s=b)
        snapshot, result = response["snapshot"], response["result"]
        result.update({"session_id": session_id, "scope": response["scope"],
            "indexed_until_s": response["indexed_until_s"], "answered_at": datetime.now(timezone.utc).isoformat()})
        if snapshot:
            result["match_id"] = snapshot.match.id
            result["origin_s"] = snapshot.origin_s
            for item in result["matches"]:
                item["session_time_s"] = round(snapshot.origin_s + item["time_s"], 3)
            snapshot.path.with_name(f"answer-{time.time_ns()}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return snapshot, result

    def _publish_session_result(self, snapshot, result):
        with self.lock:
            if self._closed:
                return False
            self.match = snapshot.match if snapshot else None
            self.result = result
            return True

    def select_session(self, session_id):
        self.native.session(session_id)
        with self.lock:
            if self.busy:
                raise RuntimeError("Wait for the current question to finish.")
            self.selected_session_id, self.import_mode = session_id, False
            self.match, self.result = None, None

    def replay(self, match_id, start_s, end_s):
        match = self.store.get(match_id)
        return self.native.play(match, start_s, end_s)

    def whole_session(self, session_id):
        self.native.session(session_id)
        with self.lock:
            self._begin("preparing")
            def prepare():
                state = self.native.status()
                if state.get("recording") and state.get("session_id") == session_id:
                    self.native.flush(session_id)
                manifest = self.native.session(session_id)
                snapshot = self.native.snapshot(manifest, 0, manifest["duration_s"])
                self.native.play(snapshot.match, 0, snapshot.duration)
                with self.lock:
                    self.match, self.result = snapshot.match, None
                    self.selected_session_id, self.import_mode = session_id, False
            self._run(prepare)

    def check_model(self):
        with self.lock:
            self._begin("checking")
            self.model_status = {"state": "checking", "message": "Checking a live model request…"}

            def check():
                try:
                    model = self.model_factory()
                    model.check()
                    with self.lock:
                        self.model_status = {"state": "connected", "message": model.label}
                except Exception as exc:
                    with self.lock:
                        self.model_status = {"state": "error", "message": str(exc)[:700]}
                    raise
            self._run(check)

    def set_backend(self, backend, *, persist=True):
        if backend not in PROVIDERS:
            raise ValueError("Choose Codex, Claude or DeepSeek.")
        with self._lifecycle_lock:
            with self.lock:
                if self._closed:
                    raise RuntimeError("This replay session is closed.")
                if self.busy:
                    raise RuntimeError("Wait for the current operation to finish.")
                self._backend_changing = True
                self._model_generation += 1
            try:
                self._stop_monitor()
                with self.lock:
                    queries, self.queries = self.queries, None
                if queries:
                    queries.close()
                self._close_model_transport()
                with self.lock:
                    self.backend = backend
                    self.error = ""
                    self.model_status = {"state": "unchecked", "message": "Check this model connection before presenting."}
            finally:
                with self.lock:
                    self._backend_changing = False
                if self._monitor_requested:
                    self._ensure_monitor()
        # The demo's own picker keeps deja's setting in step.
        if persist and self.settings_file:
            self._save_provider(backend)

    def close(self):
        with self._lifecycle_lock:
            with self.lock:
                if self._closed:
                    return
                self._closed = True
                self._voice_stop.set()
                voice_request = self._active_voice_request
                recorder, capture_worker, worker = self.recorder, self.capture_worker, self.worker
            if voice_request:
                try:
                    self.native.voice_feedback(voice_request, "error", "Voice processing stopped. Restart the replay service and ask again.", self.voice.get("transcript", ""))
                except (OSError, RuntimeError, ValueError):
                    pass
            self._stop_monitor()
            if recorder:
                recorder.stop()
            if capture_worker:
                capture_worker.join(timeout=25)
            # NativeRecorder only closes a host this client owns. Never wait on a
            # network-backed question before releasing capture and its index worker.
            self.native.close()
            self._voice_worker.join(timeout=2)
            if self._settings_worker:
                self._settings_worker.join(timeout=2)
            with self.lock:
                queries, self.queries = self.queries, None
            if queries:
                queries.close()
            self._close_model_transport()
            if worker:
                worker.join(timeout=40)


class DemoServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, session: DemoSession, port=8766):
        self.session = session
        super().__init__(("127.0.0.1", port), Handler)

    @property
    def origin(self):
        return f"http://127.0.0.1:{self.server_port}"


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def _allowed(self):
        hosts = (f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}")
        return self.headers.get("Host") in hosts and self.headers.get("Origin") in (None, *(f"http://{h}" for h in hosts))

    def _headers(self, code, size, content_type, extra=None):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(size))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; media-src 'self'; style-src 'self'; frame-ancestors 'none'")
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()

    def _json(self, code, value, extra=None):
        data = json.dumps(value, allow_nan=False).encode()
        self._headers(code, len(data), "application/json; charset=utf-8", extra)
        self.wfile.write(data)

    def do_GET(self):
        if not self._allowed():
            return self._json(403, {"error": "Forbidden host or origin."})
        route = self.path.split("?", 1)[0]
        if route == "/api/status":
            return self._json(200, self.server.session.snapshot())
        media = re.fullmatch(r"/media/([0-9a-f]{32})\.mp4", route)
        if media:
            try:
                match = self.server.session.store.get(media[1])
                return self._media(match.path)
            except (FileNotFoundError, ValueError):
                return self._json(404, {"error": "Recording not found."})
        assets = {"/": ("index.html", "text/html; charset=utf-8"),
                  "/app.js": ("app.js", "application/javascript; charset=utf-8"),
                  "/app.css": ("app.css", "text/css; charset=utf-8")}
        if route not in assets:
            return self._json(404, {"error": "Not found."})
        name, kind = assets[route]
        data = (STATIC / name).read_bytes()
        self._headers(200, len(data), kind)
        self.wfile.write(data)

    def _media(self, path):
        size = path.stat().st_size
        start, end, code = 0, size - 1, 200
        requested = self.headers.get("Range")
        if requested:
            match = re.fullmatch(r"bytes=(\d*)-(\d*)", requested)
            if not match or not any(match.groups()):
                return self._json(416, {"error": "Invalid range."}, {"Content-Range": f"bytes */{size}"})
            a, b = match.groups()
            if not a:
                start = max(0, size - int(b))
            else:
                start = int(a)
                end = min(end, int(b)) if b else end
            if start > end or start >= size:
                return self._json(416, {"error": "Range outside recording."}, {"Content-Range": f"bytes */{size}"})
            code = 206
        extra = {"Accept-Ranges": "bytes"}
        if code == 206:
            extra["Content-Range"] = f"bytes {start}-{end}/{size}"
        self._headers(code, end - start + 1, "video/mp4", extra)
        with path.open("rb") as source:
            source.seek(start)
            remaining = end - start + 1
            try:
                while remaining:
                    chunk = source.read(min(65536, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass

    def do_POST(self):
        if not self._allowed():
            self.close_connection = True
            return self._json(403, {"error": "Forbidden host or origin."})
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 16384 or self.headers.get_content_type() != "application/json":
                self.close_connection = True
                return self._json(400, {"error": "Expected a small JSON request."})
            body = json.loads(self.rfile.read(length))
            if not isinstance(body, dict):
                raise ValueError("Expected a JSON object.")
            session = self.server.session
            if self.path == "/api/record":
                session.record(body.get("seconds", 25))
            elif self.path == "/api/stop":
                session.stop()
            elif self.path == "/api/ask":
                session.ask(body.get("question"), body.get("match_id"), body.get("session_id"),
                            body.get("scope", "session"), body.get("start_s"))
            elif self.path == "/api/select-session":
                session.select_session(body.get("session_id"))
            elif self.path == "/api/play":
                session.replay(body.get("match_id"), body.get("start_s"), body.get("end_s"))
            elif self.path == "/api/full-session":
                session.whole_session(body.get("session_id"))
            elif self.path == "/api/check-model":
                session.check_model()
            elif self.path == "/api/backend":
                session.set_backend(body.get("backend"))
            elif self.path == "/api/launch-game":
                if os.name != "nt":
                    raise ValueError("Open Stick Fight through Steam on your Windows PC.")
                os.startfile("steam://run/674940")
            else:
                return self._json(404, {"error": "Not found."})
            return self._json(202, session.snapshot())
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            return self._json(400, {"error": str(exc)})
        except RuntimeError as exc:
            return self._json(409, {"error": str(exc)})
