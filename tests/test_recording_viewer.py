import http.client
import json
import threading
import time
from types import SimpleNamespace

import pytest

from smartreplay.frontend.recordings import RecordingLibrary
from smartreplay.frontend.server import SettingsServer


def manifest(root, identity="session-one", state="closed"):
    folder = root / "sessions" / identity
    folder.mkdir(parents=True, exist_ok=True)
    video = folder / "part.mp4"
    video.write_bytes(b"fixture")
    data = {"id": identity, "game": "Stick Fight", "created_at": "2026-10-25T18:00:00Z",
            "duration_s": 5, "state": state,
            "segments": [{"path": str(video.resolve()), "start_s": 0, "end_s": 5}]}
    (folder / "session.json").write_text(json.dumps(data))
    return data


def wait_ready(library, identity):
    deadline = time.monotonic() + 4
    while time.monotonic() < deadline:
        job = library.playback(identity)
        if job["state"] != "preparing":
            return job
        time.sleep(.01)
    pytest.fail("Playback preparation did not finish")


def test_lists_sessions_without_leaking_file_paths(tmp_path):
    manifest(tmp_path)
    library = RecordingLibrary(tmp_path)
    try:
        rows = library.list()
        assert rows[0]["id"] == "session-one"
        assert rows[0]["duration_s"] == 5
        assert rows[0]["playable"] is True
        assert "segments" not in rows[0] and "path" not in rows[0]
    finally:
        library.close()


def test_bad_manifest_does_not_hide_good_recordings(tmp_path):
    manifest(tmp_path)
    broken = tmp_path / "sessions" / "broken"
    broken.mkdir()
    (broken / "session.json").write_text("{broken")
    library = RecordingLibrary(tmp_path)
    try:
        assert [x["id"] for x in library.list()] == ["session-one"]
        with pytest.raises(ValueError):
            library.prepare("../outside")
        with pytest.raises(ValueError):
            library.media("session-one", "invented")
    finally:
        library.close()


def test_missing_segment_is_reported_before_preparing(tmp_path):
    data = manifest(tmp_path)
    (tmp_path / "sessions/session-one/part.mp4").unlink()
    library = RecordingLibrary(tmp_path)
    try:
        assert library.list()[0]["playable"] is False
        with pytest.raises(ValueError, match="missing"):
            library.prepare(data["id"])
    finally:
        library.close()


def test_prepares_real_session_and_keeps_a_fixed_snapshot(tmp_path):
    from test_native_demo import archive
    from smartreplay.demo.capture import video_duration
    session = archive(tmp_path)
    library = RecordingLibrary(tmp_path)
    try:
        assert library.prepare(session["id"])["state"] in ("preparing", "ready")
        job = wait_ready(library, session["id"])
        assert job["state"] == "ready", job
        assert job["duration_s"] == pytest.approx(6, abs=.2)
        video = library.media(session["id"], job["version"])
        assert video_duration(video) == pytest.approx(6, abs=.2)
        assert library.prepare(session["id"])["url"] == job["url"]
        with pytest.raises(ValueError):
            library.media(session["id"], "wrong-version")
    finally:
        library.close()


def test_recordings_http_obeys_origin_and_media_ranges(tmp_path):
    from test_native_demo import archive
    archive(tmp_path)
    server = SettingsServer(tmp_path / "settings.json", recordings_root=tmp_path)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    def request(method, path, headers=None):
        conn = http.client.HTTPConnection(*server.server_address, timeout=10)
        conn.request(method, path, headers=headers or {})
        result = conn.getresponse()
        response = result.status, dict(result.getheaders()), result.read()
        conn.close()
        return response
    try:
        code, _, raw = request("GET", "/api/recordings")
        assert code == 200 and json.loads(raw)["recordings"][0]["id"] == "session-one"
        assert request("POST", "/api/recordings/session-one/prepare", {"Origin": "https://evil.example"})[0] == 403
        assert request("POST", "/api/recordings/session-one/prepare")[0] == 202
        job = wait_ready(server.recordings, "session-one")
        assert job["state"] == "ready", job
        code, headers, raw = request("GET", job["url"], {"Range": "bytes=0-15"})
        assert code == 206 and len(raw) == 16 and headers["Content-Type"] == "video/mp4"
        assert "Content-Range" in headers
        assert request("GET", "/api/recordings/session-one/media/wrong")[0] == 404
        assert request("GET", "/api/recordings/../media/secret")[0] == 404
    finally:
        server.shutdown()
        server.server_close()
        worker.join()


def wait_search(library, identity, token):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        job = library.search_status(identity, token)
        if job["state"] in ("ready", "error"):
            return job
        time.sleep(.01)
    pytest.fail("Search did not finish")


def test_text_search_is_bound_to_selected_session_and_handles_absence(tmp_path):
    manifest(tmp_path)
    calls = []
    def search(identity, question):
        calls.append((identity, question))
        return {"snapshot": None, "result": {"matches": [], "note": "Not found"}}
    library = RecordingLibrary(tmp_path, searcher=search)
    try:
        started = library.search("session-one", "  yellow falls  ")
        job = wait_search(library, "session-one", started["id"])
        assert calls == [("session-one", "yellow falls")]
        assert job["result"]["matches"] == [] and "url" not in job
        with pytest.raises(ValueError):
            library.search_status("other-session", started["id"])
        with pytest.raises(ValueError):
            library.search_media("session-one", started["id"])
    finally:
        library.close()


def test_voice_transcript_uses_same_selected_recording_search(tmp_path, monkeypatch):
    manifest(tmp_path)
    calls = []
    library = RecordingLibrary(tmp_path, searcher=lambda identity, q:
        calls.append((identity, q)) or {"snapshot": None, "result": {"matches": []}})
    monkeypatch.setattr(library, "_transcribe", lambda audio: "Find the snake gun" if audio == b"speech" else "")
    try:
        started = library.search("session-one", audio=b"speech")
        job = wait_search(library, "session-one", started["id"])
        assert job["question"] == "Find the snake gun"
        assert calls == [("session-one", "Find the snake gun")]
    finally:
        library.close()


def test_search_failure_never_serves_previous_match(tmp_path):
    manifest(tmp_path)
    path = tmp_path / "sessions/session-one/part.mp4"
    library = RecordingLibrary(tmp_path, searcher=lambda *a: {
        "snapshot": SimpleNamespace(path=path, duration=5), "result": {"matches": [{"time_s": 1}]}})
    try:
        first = library.search("session-one", "first")
        assert wait_search(library, "session-one", first["id"])["state"] == "ready"
        def fail(*a):
            raise RuntimeError("Model offline")
        library.searcher = fail
        second = library.search("session-one", "second")
        job = wait_search(library, "session-one", second["id"])
        assert job["state"] == "error" and "Model offline" in job["error"]
        with pytest.raises(ValueError):
            library.search_media("session-one", first["id"])
    finally:
        library.close()


def test_voice_upload_decodes_locally_before_search(tmp_path):
    import io
    import wave
    manifest(tmp_path)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(b"\x01\x01" * 8000)
    observed = []
    def transcribe(path):
        with wave.open(str(path)) as audio:
            assert audio.getframerate() == 16000 and audio.getnchannels() == 1
        observed.append(path)
        return "Find yellow falling"
    library = RecordingLibrary(tmp_path, transcriber=transcribe,
        searcher=lambda identity, q: {"snapshot": None, "result": {"matches": [], "query": q}})
    try:
        started = library.search("session-one", audio=buffer.getvalue())
        job = wait_search(library, "session-one", started["id"])
        assert job["state"] == "ready", job
        assert job["question"] == "Find yellow falling"
        assert observed and not observed[0].exists(), "Temporary microphone audio must be removed"
    finally:
        library.close()


def test_runtime_recording_search_reserves_model_without_changing_capture(tmp_path):
    from smartreplay.frontend.runtime import PlayerRuntime
    runtime = PlayerRuntime(tmp_path, tmp_path / "settings.json")
    calls = []
    session = SimpleNamespace(lock=threading.RLock(), busy=False, match=None, phase="idle")
    def begin(phase):
        assert not session.busy
        session.busy = True
        session.phase = phase
    def search(question, identity, scope, start):
        assert session.busy
        calls.append((question, identity, scope))
        return None, {"matches": []}
    session._begin, session._search_session = begin, search
    runtime._session = session
    assert runtime.search_recording("session-one", "yellow falls")["result"] == {"matches": []}
    assert calls == [("yellow falls", "session-one", "session")]
    assert not session.busy and session.phase == "idle"


def test_version_matches_package_metadata():
    import tomllib
    from pathlib import Path
    from smartreplay import __version__
    metadata = tomllib.loads((Path(__file__).resolve().parents[1] / "pyproject.toml").read_text())
    assert metadata["project"]["version"] == __version__ == "0.2.6"


def test_server_stops_capture_before_waiting_for_recording_search(tmp_path):
    server = SettingsServer(tmp_path / "settings.json")
    order = []
    server.runtime = SimpleNamespace(close=lambda: order.append("capture"))
    server._recordings = SimpleNamespace(close=lambda: order.append("search"))
    server.server_close()
    assert order == ["capture", "search"]
