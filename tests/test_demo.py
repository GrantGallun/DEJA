"""Demo boundaries: real video/storage/HTTP; only external model calls are doubled."""
from pathlib import Path
import json
import re
import threading
import time
import urllib.error
import urllib.request

import cv2
import numpy as np
import pytest


def make_video(path, seconds=4, fps=8):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (160, 90))
    assert writer.isOpened()
    for i in range(seconds * fps):
        frame = np.full((90, 160, 3), (i * 7) % 255, np.uint8)
        cv2.putText(frame, str(i), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        writer.write(frame)
    writer.release()
    return path


def test_import_pins_a_unique_copy_and_rejects_empty_video(tmp_path):
    from smartreplay.demo.capture import MatchStore
    source = make_video(tmp_path / "source.mp4")
    store = MatchStore(tmp_path / "matches")
    first = store.import_video(source)
    second = store.import_video(source)
    source.unlink()
    assert first.id != second.id
    assert first.path.exists() and second.path.exists()
    assert first.duration == pytest.approx(4)
    assert store.get(first.id).path == first.path
    empty = tmp_path / "empty.mp4"
    empty.write_bytes(b"")
    with pytest.raises(ValueError, match="video"):
        store.import_video(empty)
    with pytest.raises(ValueError):
        store.get("../source")


def test_capture_command_targets_one_window_and_has_a_time_limit(tmp_path):
    from smartreplay.demo.capture import capture_command
    cmd = capture_command("ffmpeg", 123456, tmp_path / "match.mp4", 25)
    assert cmd[cmd.index("-i") + 1] == "hwnd=123456"
    assert cmd[cmd.index("-t") + 1] == "25"
    assert "desktop" not in cmd
    assert "libx264" in cmd and "+faststart" in cmd
    for seconds in (0, 31, float("nan")):
        with pytest.raises(ValueError):
            capture_command("ffmpeg", 123456, tmp_path / "match.mp4", seconds)


class Answers:
    def __init__(self, *answers):
        self.answers = iter(answers)
        self.seen = []

    def locate(self, query, frames, duration, *, verify=False):
        self.seen.append((query, [t for t, _ in frames], duration, verify))
        return next(self.answers)


def test_search_refines_real_frames_and_returns_bounded_replay(tmp_path):
    from smartreplay.demo.search import find_moments
    video = make_video(tmp_path / "match.mp4")
    model = Answers(
        {"matches": [{"time_s": 2, "verdict": "yes", "evidence": "Yellow falls"}], "note": ""},
        {"matches": [{"time_s": 2.25, "verdict": "yes", "evidence": "Yellow crosses the bottom edge"}], "note": ""},
    )
    result = find_moments(video, "last yellow death", model)
    assert result["matches"][0]["time_s"] == 2.25
    assert result["matches"][0]["start_s"] == 0
    assert result["matches"][0]["end_s"] == 4
    assert "bottom edge" in result["matches"][0]["evidence"]
    assert len(model.seen) == 2 and model.seen[1][3]
    assert len(model.seen[1][1]) > len(model.seen[0][1])


def test_search_does_not_invent_or_promote_uncertain_matches(tmp_path):
    from smartreplay.demo.search import find_moments
    video = make_video(tmp_path / "match.mp4")
    absent = find_moments(video, "purple wins", Answers({"matches": [], "note": "No purple player"}))
    assert absent["matches"] == [] and "purple" in absent["note"]
    model = Answers(
        {"matches": [{"time_s": 2, "verdict": "yes", "evidence": "Possible fall"}]},
        {"matches": [{"time_s": 2.1, "verdict": "maybe", "evidence": "Player is obscured"}]},
    )
    result = find_moments(video, "yellow falls", model)
    assert result["matches"][0]["verdict"] == "maybe"


def test_index_selected_candidate_is_verified_without_repeating_coarse_search(tmp_path):
    from smartreplay.demo.search import find_moments
    video = make_video(tmp_path / "match.mp4")
    model = Answers({"matches": [{"time_s": 2.25, "verdict": "yes", "evidence": "Visible fall"}]})
    result = find_moments(video, "second yellow fall", model, candidate_times=[2])
    assert result["matches"][0]["time_s"] == 2.25
    assert len(model.seen) == 1 and model.seen[0][3] is True
    assert result["timings"]["model_calls"] == 1
    assert result["timings"]["coarse_frames"] == 0


def test_invalid_index_candidates_never_trigger_an_unrelated_search(tmp_path):
    from smartreplay.demo.search import find_moments
    video = make_video(tmp_path / "match.mp4")
    model = Answers()
    result = find_moments(video, "yellow falls", model,
                          candidate_times=[-1, 5, float("nan"), True])
    assert result["matches"] == [] and model.seen == []


def test_sparse_later_frame_is_not_used_to_verify_an_earlier_candidate(tmp_path, monkeypatch):
    from smartreplay.demo import search
    video = make_video(tmp_path / "match.mp4", seconds=6)
    monkeypatch.setattr(search, "sample_frames", lambda *_:
        [(0, np.zeros((8, 8, 3), np.uint8)), (5, np.ones((8, 8, 3), np.uint8))])
    model = Answers({"matches": []})
    search.find_moments(video, "yellow jumps", model, candidate_times=[1.5])
    assert model.seen[0][1] == [0]


@pytest.mark.parametrize("timestamp", [-1, 999, float("nan"), float("inf"), True, "bad"])
def test_search_rejects_ungrounded_model_timestamps(tmp_path, timestamp):
    from smartreplay.demo.search import find_moments
    video = make_video(tmp_path / "match.mp4")
    result = find_moments(video, "yellow falls", Answers({"matches": [
        {"time_s": timestamp, "verdict": "yes", "evidence": "Unusable"}]}))
    assert result["matches"] == []


def wait_ready(session):
    deadline = time.monotonic() + 5
    while session.snapshot()["busy"] and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not session.snapshot()["busy"]
    return session.snapshot()


def test_failed_search_keeps_the_selected_recording_and_allows_retry(tmp_path):
    from smartreplay.demo.server import DemoSession
    class Offline:
        def locate(self, *args, **kwargs):
            raise RuntimeError("Usage limit reached")
    session = DemoSession(tmp_path / "demo", model_factory=lambda: Offline())
    match = session.import_video(make_video(tmp_path / "source.mp4"))
    session.ask("yellow falls", match.id)
    state = wait_ready(session)
    assert "Usage limit" in state["error"]
    assert state["match"]["id"] == match.id and match.path.exists()
    session.model_factory = lambda: Answers({"matches": [], "note": "No yellow player"})
    session.ask("yellow falls", match.id)
    assert wait_ready(session)["result"]["matches"] == []
    with pytest.raises(ValueError, match="match"):
        session.ask("yellow falls", "stale")


def test_new_match_cannot_replace_a_pending_search(tmp_path):
    from smartreplay.demo.server import DemoSession
    release, entered = threading.Event(), threading.Event()
    class Slow:
        def locate(self, *args, **kwargs):
            entered.set()
            release.wait(3)
            return {"matches": [], "note": "Nothing found"}
    session = DemoSession(tmp_path / "demo", model_factory=lambda: Slow())
    source = make_video(tmp_path / "source.mp4")
    match = session.import_video(source)
    session.ask("yellow falls", match.id)
    assert entered.wait(2)
    try:
        with pytest.raises(RuntimeError, match="Wait"):
            session.import_video(source)
    finally:
        release.set()
    assert wait_ready(session)["match"]["id"] == match.id


def test_http_ranges_seek_the_pinned_video_and_foreign_origins_are_rejected(tmp_path):
    from smartreplay.demo.server import DemoSession, DemoServer
    session = DemoSession(tmp_path / "demo", model_factory=lambda: Answers({"matches": [], "note": ""}))
    match = session.import_video(make_video(tmp_path / "source.mp4"))
    server = DemoServer(session, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        origin = server.origin
        request = urllib.request.Request(origin + match.public()["url"], headers={"Range": "bytes=10-29"})
        with urllib.request.urlopen(request) as response:
            assert response.status == 206
            assert response.read() == match.path.read_bytes()[10:30]
            assert response.headers["Content-Range"] == f"bytes 10-29/{match.path.stat().st_size}"
        for headers, route, expected in [
            ({"Origin": "https://evil.example", "Content-Type": "application/json"}, "/api/record", 403),
            ({"Range": "bytes=999999999-"}, match.public()["url"], 416),
        ]:
            request = urllib.request.Request(origin + route, data=b"{}" if route.startswith("/api") else None, headers=headers)
            with pytest.raises(urllib.error.HTTPError) as caught:
                urllib.request.urlopen(request)
            assert caught.value.code == expected
        request = urllib.request.Request(origin + "/api/ask", data=json.dumps({"question": "purple wins", "match_id": match.id}).encode(),
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request) as response:
            assert response.status == 202
        assert wait_ready(session)["result"]["matches"] == []
        with urllib.request.urlopen(origin + "/api/status") as response:
            assert json.load(response)["match"]["id"] == match.id
    finally:
        server.shutdown()
        server.server_close()
        thread.join(2)


def test_codex_connection_uses_images_schema_and_read_only_execution(tmp_path, monkeypatch):
    from smartreplay.demo.search import CodexVisionModel
    import subprocess
    monkeypatch.setenv("SMARTREPLAY_CODEX_TRANSPORT", "exec")
    observed = {}
    def external_run(command, **kwargs):
        observed["command"] = command
        observed["prompt"] = kwargs["input"]
        output = Path(command[command.index("--output-last-message") + 1])
        output.write_text(json.dumps({"matches": [], "note": "No yellow player"}), encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "", "")
    monkeypatch.setattr(subprocess, "run", external_run)
    model = CodexVisionModel(tmp_path / "model", binary="codex")
    response = model.locate("yellow falls", [(1.25, np.zeros((90, 160, 3), np.uint8))], 4)
    assert response["matches"] == []
    command = observed["command"]
    assert command[command.index("--sandbox") + 1] == "read-only"
    assert "--ephemeral" in command and "--ignore-user-config" in command
    assert "--image" in command and "--output-schema" in command
    assert "1.250" in observed["prompt"] and "yellow falls" in observed["prompt"]


def test_changing_backend_preserves_match_but_rechecks_connection(tmp_path):
    from smartreplay.demo.server import DemoSession
    session = DemoSession(tmp_path / "demo")
    match = session.import_video(make_video(tmp_path / "source.mp4"))
    session.set_backend("claude")
    state = session.snapshot()
    assert state["backend"] == "claude" and state["model"]["state"] == "unchecked"
    assert state["match"]["id"] == match.id
    with pytest.raises(ValueError):
        session.set_backend("untrusted-command")


def test_import_transcodes_non_browser_video_to_h264_mp4(tmp_path):
    from smartreplay.demo.capture import MatchStore, ffmpeg_binary
    import subprocess
    source = tmp_path / "input.avi"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"MJPG"), 8, (160, 90))
    assert writer.isOpened()
    for _ in range(16):
        writer.write(np.full((90, 160, 3), (20, 80, 180), np.uint8))
    writer.release()
    match = MatchStore(tmp_path / "matches").import_video(source)
    assert match.path.read_bytes()[4:8] == b"ftyp"
    result = subprocess.run([ffmpeg_binary(), "-hide_banner", "-i", str(match.path)], capture_output=True, text=True)
    assert "Video: h264" in result.stderr
    # Full-range MJPEG sources may retain the equivalent yuvj420p range tag.
    assert re.search(r"\byuvj?420p\b", result.stderr)


def test_a_long_native_export_is_searched_only_inside_the_requested_window(tmp_path):
    from smartreplay.demo.search import find_moments
    video = make_video(tmp_path / "long.mp4", seconds=40)
    model = Answers(
        {"matches": [{"time_s": 22, "verdict": "yes", "evidence": "Yellow falls"}]},
        {"matches": [{"time_s": 22.25, "verdict": "yes", "evidence": "Yellow falls below the edge"}]})
    result = find_moments(video, "yellow falls", model, start_s=20, end_s=25)
    assert result["matches"][0]["time_s"] == 22.25
    assert result["matches"][0]["start_s"] == 20
    assert all(20 <= t < 25 for call in model.seen for t in call[1])
    with pytest.raises(ValueError, match="35"):
        find_moments(video, "yellow falls", model)
