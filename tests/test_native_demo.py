"""Real archive files and IPC; no capture, game inputs, or model account required."""
from datetime import datetime, timezone, timedelta
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading
import time

import cv2
import numpy as np
import pytest

from smartreplay.demo.capture import MatchStore, ffmpeg_binary, video_duration


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(".tmp")
    pending.write_text(json.dumps(value), encoding="utf-8")
    pending.replace(path)


def archive(root, session_id="session-one"):
    directory = root / "sessions" / session_id
    directory.mkdir(parents=True)
    segments = []
    for number, color in enumerate((30, 100, 210)):
        source, target = directory / f"source-{number}.avi", directory / f"segment-{number}.mp4"
        writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"MJPG"), 8, (160, 90))
        assert writer.isOpened()
        for _ in range(16):
            writer.write(np.full((90, 160, 3), color, np.uint8))
        writer.release()
        result = subprocess.run([ffmpeg_binary(), "-hide_banner", "-loglevel", "error", "-y", "-i", str(source),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", str(target)], capture_output=True)
        assert result.returncode == 0, result.stderr
        segments.append({"path": str(target), "start_s": number * 2, "end_s": number * 2 + 2})
    manifest = {"id": session_id, "game": "Stick Fight", "created_at": datetime.now(timezone.utc).isoformat(),
                "state": "recording", "duration_s": 6, "segments": segments}
    atomic_json(directory / "session.json", manifest)
    return manifest


def status(root, session=None, instance="instance-one", age=0):
    value = {"version": 1, "instance_id": instance, "pid": os.getpid(), "recording": bool(session),
        "session_id": session["id"] if session else None, "active_session": session,
        "updated_at": (datetime.now(timezone.utc) - timedelta(seconds=age)).isoformat(), "error": "",
        "cache": {"bytes": 32, "budget_bytes": 128 * 1024 * 1024, "hits": 0, "misses": 0}}
    atomic_json(root / "native" / "status.json", value)
    return value


def test_native_snapshot_selects_old_segments_and_maps_the_session_origin(tmp_path):
    from smartreplay.demo.native import NativeRecorder
    manifest = archive(tmp_path)
    native = NativeRecorder(tmp_path, MatchStore(tmp_path / "matches"))
    assert native.sessions()[0]["duration_s"] == 6
    snapshot = native.snapshot(manifest, 2.5, 4.2)
    assert snapshot.origin_s == 2
    assert snapshot.duration == pytest.approx(4, abs=.15)
    assert snapshot.session_id == "session-one" and snapshot.match.source == "native"
    cap = cv2.VideoCapture(str(snapshot.path))
    cap.set(cv2.CAP_PROP_POS_MSEC, 2500)
    ok, frame = cap.read()
    cap.release()
    assert ok and frame.mean() > 180
    # Later native progress cannot mutate a question's pinned video.
    before = hashlib.sha256(snapshot.path.read_bytes()).hexdigest()
    manifest["duration_s"] = 8
    later = tmp_path / "sessions/session-one/segment-3.mp4"
    later.write_bytes(Path(manifest["segments"][0]["path"]).read_bytes())
    manifest["segments"].append({"path": str(later), "start_s": 6, "end_s": 8})
    atomic_json(tmp_path / "sessions/session-one/session.json", manifest)
    assert hashlib.sha256(snapshot.path.read_bytes()).hexdigest() == before
    old = native.snapshot(native.sessions()[0], 0, 1)
    assert old.origin_s == 0 and old.duration == pytest.approx(2, abs=.15)


def test_native_snapshot_refuses_missing_or_escaped_segments(tmp_path):
    from smartreplay.demo.native import NativeRecorder
    manifest = archive(tmp_path)
    native = NativeRecorder(tmp_path, MatchStore(tmp_path / "matches"))
    Path(manifest["segments"][0]["path"]).unlink()
    with pytest.raises(ValueError, match="missing"):
        native.snapshot(manifest, 0, 1)
    manifest["segments"][0]["path"] = str(tmp_path / "unrelated.mp4")
    with pytest.raises(ValueError, match="outside"):
        native.snapshot(manifest, 0, 1)


@pytest.mark.parametrize("broken", ["gap", "duration", "segment-type"])
def test_native_timeline_cannot_silently_omit_recorded_intervals(tmp_path, broken):
    from smartreplay.demo.native import NativeRecorder
    manifest = archive(tmp_path)
    if broken == "gap":
        manifest["segments"][0]["start_s"] = 1
    elif broken == "duration":
        manifest["duration_s"] = 8
    else:
        manifest["segments"][0] = "invalid"
    native = NativeRecorder(tmp_path, MatchStore(tmp_path / "matches"))
    with pytest.raises(ValueError, match="session|segment|timeline"):
        native.snapshot(manifest, 0, 6)


def test_stale_native_status_cannot_claim_recording_is_running(tmp_path):
    from smartreplay.demo.native import NativeRecorder
    manifest = archive(tmp_path)
    status(tmp_path, manifest, age=60)
    native = NativeRecorder(tmp_path, MatchStore(tmp_path / "matches"))
    state = native.status()
    assert not state["available"] and not state["recording"]
    assert "stale" in state["error"].lower()
    assert native.sessions()[0]["id"] == "session-one"


def test_bridge_rejects_reply_from_an_old_instance(tmp_path):
    from smartreplay.demo.native import NativeRecorder
    manifest = archive(tmp_path)
    status(tmp_path, manifest)
    native = NativeRecorder(tmp_path, MatchStore(tmp_path / "matches"))
    ready = threading.Event()
    def host():
        limit = time.monotonic() + 3
        while time.monotonic() < limit:
            commands = list((tmp_path / "native/commands").glob("*.json"))
            if commands:
                command = json.loads(commands[0].read_text())
                assert command["instance_id"] == "instance-one"
                atomic_json(tmp_path / "native/responses" / commands[0].name,
                            {"id": command["id"], "instance_id": "older-instance", "ok": True})
                ready.set()
                return
            time.sleep(.01)
    worker = threading.Thread(target=host, daemon=True)
    worker.start()
    with pytest.raises(RuntimeError, match="instance"):
        native.flush("session-one", timeout=2)
    assert ready.wait(1)
    worker.join(1)


def test_native_launch_selects_one_host_and_bounded_cache(tmp_path):
    from smartreplay.demo.native import NativeRecorder
    native = NativeRecorder(tmp_path, MatchStore(tmp_path / "matches"), binary=tmp_path / "host.exe")
    command = native.command()
    assert "--full-sessions" in command
    assert command[command.index("--replay-cache-mb") + 1] == "128"
    assert Path(command[command.index("--session-root") + 1]) == (tmp_path / "sessions").resolve()
    assert "ffmpeg" not in command


def test_questions_use_a_frozen_native_window_without_stopping_recording(tmp_path):
    from smartreplay.demo.server import DemoSession
    from smartreplay.demo.native import NativeRecorder
    manifest = archive(tmp_path)
    class Host(NativeRecorder):
        stopped = False
        def status(self):
            return {"available": True, "recording": True, "enabled": True, "session_id": "session-one",
                    "active_session": manifest, "cache": {"bytes": 0, "budget_bytes": 128 * 1024 * 1024}, "error": ""}
        def flush(self, session_id, timeout=10):
            return {"ok": True}
        def stop(self):
            self.stopped = True
    entered, release = threading.Event(), threading.Event()
    class Query:
        def __init__(self, root, model_factory, snapshotter):
            self.snapshotter = snapshotter
        def refresh(self, session):
            pass
        def status(self, session_id):
            return {"indexed_until_s": 0, "duration_s": 6, "indexing": False, "error": ""}
        def ask(self, session, question, start_s=None, end_s=None):
            entered.set()
            release.wait(3)
            snapshot = self.snapshotter(session, 2, 6)
            return {"snapshot": snapshot, "session_id": session["id"], "scope": {"mode": "window", "complete": True},
                    "indexed_until_s": 0, "result": {"query": question, "matches": [
                        {"time_s": 1.25, "start_s": 0, "end_s": 3.25, "verdict": "yes", "evidence": "Visible fall"}],
                        "note": "", "elapsed_s": .1}}
    session = DemoSession(tmp_path, native_factory=Host, session_search_factory=Query)
    session.ask("yellow falls", session_id="session-one", scope="recent")
    assert entered.wait(1)
    assert session.snapshot()["native"]["recording"]
    assert session.snapshot()["phase"] == "searching"
    release.set()
    limit = time.monotonic() + 4
    while session.snapshot()["busy"] and time.monotonic() < limit:
        time.sleep(.01)
    result = session.snapshot()["result"]
    assert result["matches"][0]["session_time_s"] == 3.25
    assert result["session_id"] == "session-one"
    assert not session.native.stopped
    assert session.snapshot()["native"]["recording"]


@pytest.mark.skipif(os.name != "nt" or not (Path(__file__).resolve().parents[1] /
    "experiments/gamehq-passive/out-passive/NativeSessionProbe.exe").exists(), reason="Build the native generated-video probe first")
def test_real_native_bridge_keeps_encoding_during_flush_and_cached_playback(tmp_path, monkeypatch):
    from smartreplay.demo.native import NativeRecorder
    probe = Path(__file__).resolve().parents[1] / "experiments/gamehq-passive/out-passive/NativeSessionProbe.exe"
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    native = NativeRecorder(tmp_path, MatchStore(tmp_path / "matches"), binary=probe, cache_mb=8)
    def wait_for(predicate, seconds=12):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            state = native.status()
            if predicate(state):
                return state
            if native.process and native.process.poll() is not None:
                pytest.fail((tmp_path / "native/host-stderr.log").read_text(errors="replace")[-3000:])
            time.sleep(.05)
        pytest.fail("Native probe did not reach the required state: " + str(native.status()))
    try:
        native.start()
        state = wait_for(lambda s: s.get("available") and s.get("recording") and
            (s.get("active_session") or {}).get("duration_s", 0) >= 3)
        session_id = state["session_id"]
        native.flush(session_id)
        manifest = native.session(session_id)
        before = manifest["duration_s"]
        replay = native.snapshot(manifest, 0, min(2, before))
        native.play(replay.match, 0, min(1.5, replay.duration))
        native.play(replay.match, 0, min(1.5, replay.duration))
        state = wait_for(lambda s: (s.get("active_session") or {}).get("duration_s", 0) > before + 1 and
                         (s.get("cache") or {}).get("hits", 0) >= 1)
        assert state["recording"]
        assert 0 < state["cache"]["bytes"] <= state["cache"]["budget_bytes"]
        native.stop()
        assert not native.status()["recording"]
        assert native.session(session_id)["state"] == "closed"
        assert all(Path(s["path"]).exists() for s in manifest["segments"])
        # Old footage remains playable with capture stopped; no second encoder is needed.
        native.play(replay.match, 0, min(1.5, replay.duration))
        native.start()
        state = wait_for(lambda s: s.get("recording") and s.get("session_id") != session_id)
        assert native.session(session_id)["state"] == "closed"
    finally:
        native.close()
