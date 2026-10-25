"""Session retrieval uses real media; only the external model is doubled."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
import subprocess
import threading
import time
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from smartreplay.demo.capture import MatchStore, NO_WINDOW, ffmpeg_binary


def make_session(root, *, duration=70, state="closed", events=(12, 54), session_id="session-a", segment_s=10):
    folder = root / session_id
    folder.mkdir(parents=True)
    segments = []
    for start in range(0, duration, segment_s):
        end = min(duration, start + segment_s)
        path = folder / f"segment-{start:04d}.mp4"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 4, (96, 64))
        assert writer.isOpened()
        try:
            for frame_number in range((end - start) * 4):
                global_time = start + frame_number / 4
                event = any(event <= global_time < event + 1 for event in events)
                colour = (0, 0, 240) if event else (60, 40, 10)
                writer.write(np.full((64, 96, 3), colour, np.uint8))
        finally:
            writer.release()
        segments.append({"path": str(path.resolve()), "start_s": start, "end_s": end})
    return {"id": session_id, "game": "Stick Fight", "created_at": "2026-10-24T00:00:00Z",
            "state": state, "duration_s": duration, "segments": segments}


class Snapshots:
    """A real lossless snapshot callback, preserving complete overlapping segments."""

    def __init__(self, root):
        self.store = MatchStore(root)
        self.requests = []

    def __call__(self, session, start_s, end_s):
        self.requests.append((session["id"], start_s, end_s))
        selected = [segment for segment in session["segments"]
                    if segment["end_s"] > start_s and segment["start_s"] < end_s]
        assert selected
        output = self.store.allocate()
        source_list = output.with_name("segments.txt")
        source_list.write_text("".join(f"file '{Path(segment['path']).as_posix()}'\n"
                                       for segment in selected), encoding="utf-8")
        subprocess.run([ffmpeg_binary(), "-hide_banner", "-loglevel", "error", "-y", "-f", "concat",
                        "-safe", "0", "-i", str(source_list), "-c", "copy", str(output)],
                       capture_output=True, check=True, creationflags=NO_WINDOW)
        match = self.store.finish(output, "native")
        return SimpleNamespace(match=match, path=output, duration=match.duration,
                               origin_s=selected[0]["start_s"], session_id=session["id"])


class VisualAnswers:
    def __init__(self, *, candidate=54):
        self.candidate = candidate
        self.index_frames = []
        self.selection_text = ""
        self.verifications = 0
        self.question_passes = []

    def locate(self, query, frames, duration, *, verify=False):
        if not query.startswith("[SESSION INDEX]"):
            self.question_passes.append(verify)
        if query.startswith("[SESSION INDEX]"):
            self.index_frames.append([(timestamp, image.copy()) for timestamp, image in frames])
        elif verify:
            self.verifications += 1
        red = [(timestamp, image) for timestamp, image in frames
               if image[16, 16, 2] > 180 and image[16, 16, 0] < 40]
        matches = [{"time_s": timestamp, "verdict": "yes", "evidence": "Red victory screen"}
                   for timestamp, _ in red]
        return {"matches": matches[:4], "note": "" if matches else "No red victory visible"}

    def _call(self, content):
        self.selection_text = "\n".join(item["text"] for item in content)
        matches = ([] if self.candidate is None else
                   [{"time_s": self.candidate, "verdict": "maybe", "evidence": "Red victory screen"}])
        return {"matches": matches, "note": "No indexed evidence matches" if not matches else ""}


def wait_index(search, session_id, until, *, timeout=8):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = search.status(session_id)
        if status["indexed_until_s"] >= until:
            return status
        if status["error"] and not (status["pending"] or status["indexing"]):
            pytest.fail(status["error"])
        time.sleep(0.01)
    pytest.fail(f"Index did not reach {until}: {search.status(session_id)}")


def test_long_session_retrieves_old_evidence_and_keeps_snapshot_global_origin(tmp_path):
    # Losing the index offset would verify recent footage instead of the event at 54s.
    from smartreplay.demo.session_search import SessionSearch
    session = make_session(tmp_path / "native", duration=100)
    model = VisualAnswers()
    snapshots = Snapshots(tmp_path / "replays")
    search = SessionSearch(tmp_path / "index", lambda: model, snapshots)
    try:
        search.refresh(session)
        assert wait_index(search, session["id"], 100)["error"] == ""
        assert snapshots.requests == []  # Indexing decodes archives directly, with no replay copies.
        answer = search.ask(session, "red wins")
        assert answer["scope"]["mode"] == "session"
        assert answer["indexed_until_s"] == 100
        assert answer["snapshot"].origin_s == 40
        assert snapshots.requests == [(session["id"], 49, 59)]
        assert answer["snapshot"].duration == pytest.approx(20)
        assert answer["snapshot"].origin_s + answer["result"]["matches"][0]["time_s"] == pytest.approx(54, abs=0.25)
        match = answer["result"]["matches"][0]
        assert match["time_s"] - match["start_s"] == pytest.approx(3)
        assert match["end_s"] - match["time_s"] == pytest.approx(2)
        assert model.verifications == 1
        assert model.question_passes == [True], "Indexed candidates need dense verification without another coarse call"
        assert answer["result"]["timings"]["model_calls"] == 2
        assert "54" in model.selection_text and "Red victory screen" in model.selection_text
        assert all(len(frames) <= 14 for frames in model.index_frames)
        assert len(list((tmp_path / "index").glob("*.json"))) == 1
    finally:
        search.close()


def test_empty_index_candidate_returns_no_match_without_searching_unrelated_recent_clip(tmp_path):
    from smartreplay.demo.session_search import SessionSearch
    session = make_session(tmp_path / "native")
    model = VisualAnswers(candidate=None)
    snapshots = Snapshots(tmp_path / "replays")
    search = SessionSearch(tmp_path / "index", lambda: model, snapshots)
    try:
        search.refresh(session)
        wait_index(search, session["id"], 70)
        answer = search.ask(session, "purple wins")
        assert answer["result"]["matches"] == []
        assert "index" in answer["result"]["note"].lower()
        assert model.verifications == 0
        assert answer["snapshot"] is None
        assert snapshots.requests == []
        assert answer["result"]["timings"]["model_calls"] == 1
    finally:
        search.close()


def test_partial_index_is_honest_and_explicit_old_window_uses_its_own_footage(tmp_path):
    from smartreplay.demo.session_search import SessionSearch
    session = make_session(tmp_path / "native", state="recording")
    model = VisualAnswers()
    snapshots = Snapshots(tmp_path / "replays")
    search = SessionSearch(tmp_path / "index", lambda: model, snapshots)
    try:
        answer = search.ask(session, "red wins")
        assert answer["scope"]["mode"] == "recent"
        assert answer["scope"]["complete"] is False
        assert "recent" in answer["result"]["note"].lower()
        assert "0" in answer["result"]["note"]
        older = search.ask(session, "red wins", 5, 20)
        assert older["scope"]["mode"] == "window"
        assert older["snapshot"].origin_s == 0
        assert older["result"]["matches"][0]["time_s"] == pytest.approx(12, abs=0.25)
        assert snapshots.requests[-1][1:] == (5, 20)
    finally:
        search.close()


def test_recording_partial_window_waits_for_close_and_persisted_index_survives_restart(tmp_path):
    from smartreplay.demo.session_search import SessionSearch
    session = make_session(tmp_path / "native", duration=70, state="recording")
    model = VisualAnswers()
    snapshots = Snapshots(tmp_path / "replays")
    index_root = tmp_path / "index"
    search = SessionSearch(index_root, lambda: model, snapshots)
    try:
        search.refresh(session)
        assert wait_index(search, session["id"], 50)["windows"] == 2
        search.refresh(session)
        time.sleep(0.15)
        assert search.status(session["id"])["indexed_until_s"] == 50
        session = {**session, "state": "closed"}
        search.refresh(session)
        assert wait_index(search, session["id"], 70)["windows"] == 3
    finally:
        search.close()
    restored_model = VisualAnswers()
    restored = SessionSearch(index_root, lambda: restored_model, snapshots)
    try:
        restored.refresh(session)
        assert restored.status(session["id"])["indexed_until_s"] == 70
        answer = restored.ask(session, "red wins")
        assert restored_model.index_frames == []
        assert answer["snapshot"].origin_s + answer["result"]["matches"][0]["time_s"] == pytest.approx(54, abs=0.25)
        assert "54" in restored_model.selection_text
    finally:
        restored.close()


@pytest.mark.parametrize("timestamp", [-1, 25, 1000, float("nan"), float("inf"), True, "12"])
def test_index_rejects_invalid_window_relative_model_timestamps(tmp_path, timestamp):
    from smartreplay.demo.session_search import SessionSearch
    session = make_session(tmp_path / "native", duration=30, events=())

    class InvalidIndex(VisualAnswers):
        def locate(self, query, frames, duration, *, verify=False):
            return {"matches": [{"time_s": timestamp, "verdict": "yes", "evidence": "Unsupported"}], "note": ""}

    search = SessionSearch(tmp_path / "index", InvalidIndex, Snapshots(tmp_path / "replays"))
    try:
        search.refresh(session)
        wait_index(search, session["id"], 30)
        saved = json.loads(next((tmp_path / "index").glob("*.json")).read_text(encoding="utf-8"))
        assert all(window["matches"] == [] for window in saved["windows"])
    finally:
        search.close()


def test_candidate_outside_indexed_coverage_cannot_create_a_match(tmp_path):
    from smartreplay.demo.session_search import SessionSearch
    session = make_session(tmp_path / "native", state="recording")
    model = VisualAnswers(candidate=54)
    search = SessionSearch(tmp_path / "index", lambda: model, Snapshots(tmp_path / "replays"))
    try:
        search.refresh(session)
        wait_index(search, session["id"], 50)
        answer = search.ask(session, "red wins")
        assert answer["result"]["matches"] == []
        assert answer["scope"]["complete"] is False
        assert "50.0/70.0" in answer["result"]["note"]
        assert model.verifications == 0
    finally:
        search.close()


def test_explicit_window_does_not_return_a_match_from_lossless_export_padding(tmp_path):
    from smartreplay.demo.session_search import SessionSearch
    session = make_session(tmp_path / "native", duration=30, events=(12,))
    search = SessionSearch(tmp_path / "index", VisualAnswers, Snapshots(tmp_path / "replays"))
    try:
        answer = search.ask(session, "red wins", 15, 25)
        assert answer["snapshot"].origin_s == 10
        assert answer["result"]["matches"] == []
        assert answer["result"]["note"]  # Explain that no event was confirmed inside the chosen window.
    finally:
        search.close()


def test_background_failure_preserves_coverage_and_interactive_retry(tmp_path):
    from smartreplay.demo.session_search import SessionSearch
    session = make_session(tmp_path / "native", duration=30, events=(12,))

    class Offline(VisualAnswers):
        def locate(self, query, frames, duration, *, verify=False):
            if query.startswith("[SESSION INDEX]"):
                raise RuntimeError("Model connection unavailable")
            return super().locate(query, frames, duration, verify=verify)

    search = SessionSearch(tmp_path / "index", Offline, Snapshots(tmp_path / "replays"))
    try:
        search.refresh(session)
        deadline = time.monotonic() + 3
        while not search.status(session["id"])["error"] and time.monotonic() < deadline:
            time.sleep(0.01)
        status = search.status(session["id"])
        assert "unavailable" in status["error"]
        assert status["indexed_until_s"] == 0
        assert not status["pending"]
        answer = search.ask(session, "red wins", 0, 20)
        assert answer["result"]["matches"]
        assert all(Path(segment["path"]).exists() for segment in session["segments"])
    finally:
        search.close()


def test_refresh_coalesces_and_questions_take_priority_without_overlapping_model_calls(tmp_path):
    from smartreplay.demo.session_search import SessionSearch
    session = make_session(tmp_path / "native", duration=70, state="recording")
    entered, release = threading.Event(), threading.Event()
    calls, active, peak = [], [0], [0]
    guard = threading.Lock()

    class SlowModel(VisualAnswers):
        timeout = 60

        def locate(self, query, frames, duration, *, verify=False):
            kind = "index" if query.startswith("[SESSION INDEX]") else "question"
            with guard:
                active[0] += 1
                peak[0] = max(peak[0], active[0])
                calls.append(kind)
            try:
                if kind == "index" and not entered.is_set():
                    assert self.timeout <= 15
                    entered.set()
                    assert release.wait(3)
                return super().locate(query, frames, duration, verify=verify)
            finally:
                with guard:
                    active[0] -= 1

    search = SessionSearch(tmp_path / "index", SlowModel, Snapshots(tmp_path / "replays"))
    answers, errors = [], []

    def ask():
        try:
            answers.append(search.ask(session, "red wins", 0, 20))
        except Exception as exc:
            errors.append(exc)

    question = threading.Thread(target=ask)
    try:
        search.refresh(session)
        assert entered.wait(3)
        started = time.monotonic()
        for _ in range(100):
            search.refresh(session)
        assert time.monotonic() - started < 0.5
        question.start()
        time.sleep(0.05)
        release.set()
        question.join(5)
        assert not question.is_alive() and errors == []
        assert answers[0]["result"]["matches"]
        assert peak[0] == 1
        assert calls[:3] == ["index", "question", "question"]
        wait_index(search, session["id"], 50)
        assert calls.count("index") == 2
    finally:
        release.set()
        search.close()
        if question.ident:
            question.join(3)


def test_voice_priority_defers_new_background_calls_until_released(tmp_path):
    from smartreplay.demo.session_search import SessionSearch
    session = make_session(tmp_path / "native", duration=25)
    entered = threading.Event()
    class Model(VisualAnswers):
        def locate(self, *args, **kwargs):
            entered.set()
            return super().locate(*args, **kwargs)
    search = SessionSearch(tmp_path / "index", Model, Snapshots(tmp_path / "replays"))
    try:
        search.set_interactive_pending(True)
        search.refresh(session)
        assert not entered.wait(.15)
        search.set_interactive_pending(False)
        assert entered.wait(2)
        wait_index(search, session["id"], 25)
    finally:
        search.close()


def test_usable_archive_recovers_from_corrupt_index_json(tmp_path):
    from smartreplay.demo.session_search import SessionSearch
    session = make_session(tmp_path / "native", duration=30)
    index = tmp_path / "index"
    index.mkdir()
    destination = index / (hashlib.sha256(session["id"].encode()).hexdigest() + ".json")
    destination.write_text("[]", encoding="utf-8")
    search = SessionSearch(index, VisualAnswers, Snapshots(tmp_path / "replays"))
    try:
        search.refresh(session)
        assert wait_index(search, session["id"], 30)["error"] == ""
        assert json.loads(destination.read_text(encoding="utf-8"))["session_id"] == session["id"]
    finally:
        search.close()


def test_rejected_visual_candidate_does_not_promote_the_index_summary_to_a_match(tmp_path):
    from smartreplay.demo.session_search import SessionSearch
    session = make_session(tmp_path / "native", events=())

    class OverconfidentIndex(VisualAnswers):
        def locate(self, query, frames, duration, *, verify=False):
            if query.startswith("[SESSION INDEX]"):
                return {"matches": [{"time_s": 4, "verdict": "yes", "evidence": "Red victory screen"}], "note": ""}
            return super().locate(query, frames, duration, verify=verify)

    search = SessionSearch(tmp_path / "index", OverconfidentIndex, Snapshots(tmp_path / "replays"))
    try:
        search.refresh(session)
        wait_index(search, session["id"], 70)
        answer = search.ask(session, "red wins")
        assert answer["result"]["matches"] == []
        assert "No red" in answer["result"]["note"]
    finally:
        search.close()


def test_question_uses_frozen_manifest_while_new_finalized_segments_arrive(tmp_path):
    from smartreplay.demo.session_search import SessionSearch
    entire = make_session(tmp_path / "native", duration=30, state="recording", events=(12,))
    session = {**entire, "duration_s": 20, "segments": entire["segments"][:2]}
    entered, release = threading.Event(), threading.Event()

    class PausedModel(VisualAnswers):
        def locate(self, query, frames, duration, *, verify=False):
            if not query.startswith("[SESSION INDEX]") and not entered.is_set():
                entered.set()
                assert release.wait(3)
            return super().locate(query, frames, duration, verify=verify)

    search = SessionSearch(tmp_path / "index", PausedModel, Snapshots(tmp_path / "replays"))
    answers, errors = [], []

    def ask():
        try:
            answers.append(search.ask(session, "red wins"))
        except Exception as exc:
            errors.append(exc)

    thread = threading.Thread(target=ask)
    try:
        thread.start()
        assert entered.wait(3)
        session["duration_s"] = 30
        session["segments"].append(entire["segments"][2])
        search.refresh(session)
        release.set()
        thread.join(5)
        assert not thread.is_alive() and errors == []
        assert answers[0]["snapshot"].duration == 20
        assert answers[0]["scope"]["end_s"] == 20
        wait_index(search, session["id"], 25)
        assert session["duration_s"] == 30 and session["state"] == "recording"
    finally:
        release.set()
        search.close()
        thread.join(3)


def test_background_limits_both_cli_and_api_timeouts_without_changing_interactive_client(tmp_path):
    from smartreplay.demo.session_search import SessionSearch
    import anthropic
    session = make_session(tmp_path / "native", duration=30, state="recording")

    class ClientOptions(VisualAnswers):
        timeout = 60

        def __init__(self):
            super().__init__()
            self.api = anthropic.Anthropic(api_key="test-key-no-requests", timeout=60, max_retries=0)

        def locate(self, query, frames, duration, *, verify=False):
            if query.startswith("[SESSION INDEX]"):
                assert self.timeout <= 15
                assert self.api.timeout <= 15
            else:
                assert self.timeout == 60
                assert self.api.timeout == 60
            return super().locate(query, frames, duration, verify=verify)

    model = ClientOptions()
    original_api = model.api
    search = SessionSearch(tmp_path / "index", lambda: model, Snapshots(tmp_path / "replays"))
    try:
        search.refresh(session)
        wait_index(search, session["id"], 25)
        answer = search.ask(session, "red wins", 0, 20)
        assert answer["result"]["matches"]
        assert model.api is original_api and model.timeout == 60
    finally:
        search.close()
        original_api.close()


def test_asking_frozen_older_manifest_does_not_regress_current_index_progress(tmp_path):
    from smartreplay.demo.session_search import SessionSearch
    current = make_session(tmp_path / "native", duration=30, state="recording")
    frozen = {**current, "duration_s": 20, "segments": current["segments"][:2]}
    search = SessionSearch(tmp_path / "index", VisualAnswers, Snapshots(tmp_path / "replays"))
    try:
        search.refresh(current)
        wait_index(search, current["id"], 25)
        answer = search.ask(frozen, "red wins")
        assert answer["scope"]["end_s"] == 20
        assert answer["snapshot"].duration == 20
        assert search.status(current["id"])["duration_s"] == 30
        assert search.status(current["id"])["indexed_until_s"] == 25
    finally:
        search.close()


def test_background_failure_waits_before_retry_while_explicit_questions_remain_available(tmp_path):
    from smartreplay.demo.session_search import SessionSearch
    session = make_session(tmp_path / "native", duration=30, state="recording")
    now = [100.0]
    index_calls = []

    class Recovers(VisualAnswers):
        def locate(self, query, frames, duration, *, verify=False):
            if query.startswith("[SESSION INDEX]"):
                index_calls.append(now[0])
                if len(index_calls) == 1:
                    raise RuntimeError("Model usage limit reached")
            return super().locate(query, frames, duration, verify=verify)

    search = SessionSearch(tmp_path / "index", Recovers, Snapshots(tmp_path / "replays"),
                           clock=lambda: now[0])
    try:
        search.refresh(session)
        deadline = time.monotonic() + 3
        while not search.status(session["id"])["error"] and time.monotonic() < deadline:
            time.sleep(0.01)
        assert "usage limit" in search.status(session["id"])["error"]
        for _ in range(40):
            now[0] += 0.7  # Simulate the page/monitor repeatedly polling during the delay.
            search.refresh(session)
            status = search.status(session["id"])
            assert "usage limit" in status["error"]
            assert not status["pending"] and not status["indexing"]
        now[0] = 129.99
        search.refresh(session)
        assert index_calls == [100.0]
        answer = search.ask(session, "red wins", 0, 20)
        assert answer["result"]["matches"]
        assert "usage limit" in search.status(session["id"])["error"]
        now[0] = 130.0
        search.refresh(session)
        assert wait_index(search, session["id"], 25)["error"] == ""
        assert index_calls == [100.0, 130.0]
    finally:
        search.close()


def test_background_retry_delay_doubles_to_five_minutes_and_resets_after_success(tmp_path):
    from smartreplay.demo.session_search import SessionSearch
    entire = make_session(tmp_path / "native", duration=60, state="recording")
    session = {**entire, "duration_s": 30, "segments": entire["segments"][:3]}
    now, calls, offline = [0.0], [], [True]

    class Intermittent(VisualAnswers):
        def locate(self, query, frames, duration, *, verify=False):
            if query.startswith("[SESSION INDEX]"):
                calls.append(now[0])
                if offline[0]:
                    raise RuntimeError("Model quota unavailable")
            return super().locate(query, frames, duration, verify=verify)

    search = SessionSearch(tmp_path / "index", Intermittent, Snapshots(tmp_path / "replays"),
                           clock=lambda: now[0])

    def wait_failure(attempts):
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            status = search.status(session["id"])
            if len(calls) == attempts and not status["indexing"] and not status["pending"]:
                assert "quota" in status["error"]
                return status
            time.sleep(0.01)
        pytest.fail(f"Expected {attempts} failed attempts: {calls}, {search.status(session['id'])}")

    try:
        search.refresh(session)
        wait_failure(1)
        for attempts, delay in enumerate((30, 60, 120, 240, 300, 300), start=1):
            assert search.status(session["id"])["retry_after_s"] == delay
            now[0] += delay - 0.01
            search.refresh(session)
            assert not search.status(session["id"])["pending"]
            assert len(calls) == attempts
            now[0] += 0.01
            search.refresh(session)
            wait_failure(attempts + 1)
        offline[0] = False
        now[0] += 300
        search.refresh(session)
        status = wait_index(search, session["id"], 25)
        assert status["error"] == "" and status["retry_after_s"] == 0
        offline[0] = True
        search.refresh(entire)
        assert wait_failure(9)["retry_after_s"] == 30
    finally:
        search.close()


def test_short_session_question_covers_early_event_beyond_recent_twenty_five_seconds(tmp_path):
    from smartreplay.demo.session_search import SessionSearch
    session = make_session(tmp_path / "native", duration=28, segment_s=1, events=(1,))
    snapshots = Snapshots(tmp_path / "replays")
    search = SessionSearch(tmp_path / "index", VisualAnswers, snapshots)
    try:
        answer = search.ask(session, "red wins")
        assert answer["snapshot"].origin_s == 0
        assert answer["snapshot"].duration == pytest.approx(28)
        assert answer["scope"]["start_s"] == 0
        assert answer["scope"]["end_s"] == 28
        assert answer["scope"]["complete"] is True
        assert answer["result"]["matches"][0]["time_s"] == pytest.approx(1, abs=0.25)
        assert snapshots.requests == [(session["id"], 0, 28)]
        with pytest.raises(ValueError, match="25 seconds"):
            search.ask(session, "red wins", 0, 28)
    finally:
        search.close()
