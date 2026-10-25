"""Voice queue and service behavior without a microphone, model, or game input."""
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import threading
import time
import wave

import pytest

from smartreplay.demo.capture import Match, MatchStore
from smartreplay.demo.native import NativeRecorder, NativeSnapshot, atomic_json
from smartreplay.demo.server import DemoSession


def wav(path, seconds=.1, rate=16000, channels=1):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(channels)
        audio.setsampwidth(2)
        audio.setframerate(rate)
        audio.writeframes(b"\x01\x00" * int(seconds * rate) * channels)
    return path


def host_status(root, instance="instance-one", available=True):
    atomic_json(root / "native/status.json", {"version": 1, "pid": os.getpid(),
        "instance_id": instance, "updated_at": datetime.now(timezone.utc).isoformat(),
        "enabled": available, "recording": available, "session_id": "captured", "error": ""})


def queue(root, request_id="a" * 32, **changes):
    audio = wav(root / "native/voice" / f"{request_id}.wav")
    value = {"version": 1, "id": request_id, "instance_id": "instance-one", "session_id": "captured",
        "path": str(audio.resolve()), "created_at": datetime.now(timezone.utc).isoformat()}
    value.update(changes)
    atomic_json(root / "native/voice/requests" / f"{request_id}.json", value)
    return value, audio


def test_claim_is_atomic_across_clients_and_cannot_reclaim_after_restart(tmp_path):
    host_status(tmp_path)
    request, audio = queue(tmp_path)
    clients = [NativeRecorder(tmp_path, MatchStore(tmp_path / "matches")) for _ in range(2)]
    barrier = threading.Barrier(2)
    claimed = []
    def claim(client):
        barrier.wait()
        claimed.append(client.claim_voice_request())
    threads = [threading.Thread(target=claim, args=(client,)) for client in clients]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(2)
    assert len([item for item in claimed if item]) == 1
    item = next(item for item in claimed if item)
    assert item["session_id"] == "captured" and item["path"] == audio.resolve()
    assert audio.exists(), "The worker still needs the claimed WAV"
    assert clients[0].claim_voice_request() is None
    assert not list((tmp_path / "native/voice/requests").iterdir())


@pytest.mark.parametrize("broken", ["old-instance", "old-time", "future-time", "naive-time", "bad-id",
    "wrong-file-id", "relative-path", "escaped-path", "other-wav", "oversize", "too-long", "invalid-wav",
    "missing-wav", "bad-session", "bad-version", "truncated-wav"])
def test_invalid_requests_are_consumed_without_opening_or_deleting_unrelated_files(tmp_path, broken):
    host_status(tmp_path)
    request, audio = queue(tmp_path)
    unrelated = wav(tmp_path / "private.wav")
    if broken == "old-instance": request["instance_id"] = "old"
    elif broken == "old-time": request["created_at"] = (datetime.now(timezone.utc) - timedelta(seconds=121)).isoformat()
    elif broken == "future-time": request["created_at"] = (datetime.now(timezone.utc) + timedelta(seconds=6)).isoformat()
    elif broken == "naive-time": request["created_at"] = datetime.now().isoformat()
    elif broken == "bad-id": request["id"] = "../private"
    elif broken == "wrong-file-id": request["id"] = "b" * 32
    elif broken == "relative-path": request["path"] = "native/voice/" + "a" * 32 + ".wav"
    elif broken == "escaped-path": request["path"] = str(unrelated)
    elif broken == "other-wav": request["path"] = str(wav(tmp_path / "native/voice/other.wav"))
    elif broken == "oversize": audio.write_bytes(b"x" * (3 * 1024 * 1024 + 1))
    elif broken == "too-long": wav(audio, seconds=15.1)
    elif broken == "invalid-wav": audio.write_bytes(b"not audio")
    elif broken == "missing-wav": audio.unlink()
    elif broken == "bad-session": request["session_id"] = "../other"
    elif broken == "bad-version": request["version"] = True
    elif broken == "truncated-wav": audio.write_bytes(audio.read_bytes()[:44])
    atomic_json(tmp_path / "native/voice/requests" / ("a" * 32 + ".json"), request)
    native = NativeRecorder(tmp_path, MatchStore(tmp_path / "matches"))
    native.voice_feedback = lambda *args, **kwargs: None
    assert native.claim_voice_request() is None
    assert not list((tmp_path / "native/voice/requests").iterdir())
    assert unrelated.exists()


def test_maximum_native_pcm_format_is_accepted(tmp_path):
    host_status(tmp_path)
    _, audio = queue(tmp_path)
    wav(audio, seconds=15, rate=48000, channels=2)
    native = NativeRecorder(tmp_path, MatchStore(tmp_path / "matches"))
    assert native.claim_voice_request()["path"] == audio


def test_feedback_never_targets_a_replacement_recorder(tmp_path):
    host_status(tmp_path, "replacement")
    request, _ = queue(tmp_path)
    native = NativeRecorder(tmp_path, MatchStore(tmp_path / "matches"))
    assert native.voice_feedback(request, "error", "Canceled") is False
    assert not list((tmp_path / "native/commands").iterdir())


class Host(NativeRecorder):
    def __init__(self, root, store):
        super().__init__(root, store)
        self.state = {"available": False, "enabled": False, "recording": False, "instance_id": "instance-one",
            "session_id": None, "error": "", "cache": {}, "voice": {"state": "idle"}}
        self.feedback = []
        self.played = []
        self.closed = threading.Event()
        self.stopped = False

    def status(self): return dict(self.state)
    def session(self, session_id):
        assert session_id in ("captured", "browser-selected")
        return {"id": session_id, "duration_s": 20, "segments": [], "state": "closed"}
    def sessions(self): return []
    def flush(self, session_id): assert session_id == "captured"
    def voice_feedback(self, request, state, message, transcript=""):
        if request["instance_id"] != self.state["instance_id"]: return False
        self.feedback.append((state, message, transcript))
        return True
    def play(self, match, start_s, end_s, *, instance_id=None, voice_id=None, session_id=None):
        assert instance_id == self.state["instance_id"], "Playback crossed native generations"
        assert voice_id == "a" * 32 and session_id == "captured", "Native cannot fence canceled voice playback"
        self.played.append((match.id, start_s, end_s))
    def stop(self): self.stopped = True
    def close(self): self.closed.set()


class Transcriber:
    def __init__(self, failure=None):
        self.entered, self.release = threading.Event(), threading.Event()
        self.failure = failure
    def transcribe(self, path):
        assert path.exists()
        self.entered.set()
        assert self.release.wait(3)
        if self.failure: raise self.failure
        return "Show the last yellow fall"


class Queries:
    def __init__(self, matches):
        self.matches = matches
        self.seen = []
        self.closed = threading.Event()
    def refresh(self, manifest): pass
    def status(self, session_id): return {"indexing": False, "error": ""}
    def ask(self, manifest, question, start_s=None, end_s=None):
        self.seen.append((manifest["id"], question, start_s, end_s))
        match_path = self.root / "match.mp4"
        match_path.write_bytes(b"immutable-video")
        match = Match("b" * 32, match_path, 10, "2026-10-25T01:00:00Z", "native")
        return {"snapshot": NativeSnapshot(match, 10, manifest["id"]), "result": {"matches": self.matches,
            "query": question, "elapsed_s": .1}, "scope": {"mode": "session"}, "indexed_until_s": 20}
    def close(self): self.closed.set()


@pytest.fixture
def make_session(tmp_path, monkeypatch):
    import smartreplay.demo.server as server
    monkeypatch.setattr(server, "_VOICE_POLL_INTERVAL_S", .01, raising=False)
    sessions = []
    def make(matches=None, failure=None):
        transcriber = Transcriber(failure)
        queries = Queries(matches if matches is not None else [{"verdict": "yes", "start_s": 1, "end_s": 6, "time_s": 4}])
        root = tmp_path / str(len(sessions))
        root.mkdir()
        queries.root = root
        session = DemoSession(root, native_factory=Host, window_finder=lambda: None,
            voice_transcriber=transcriber, session_search_factory=lambda *args: queries,
            model_factory=lambda: pytest.fail("A live model was invoked"))
        sessions.append((session, transcriber))
        return session, transcriber, queries
    yield make
    for session, transcriber in sessions:
        transcriber.release.set()
        session.close()


def submit(session):
    request, audio = queue(session.root)
    session.native.state.update(available=True, enabled=True, recording=True, session_id="captured")
    return request, audio


def wait_done(session):
    deadline = time.monotonic() + 3
    while session.busy and time.monotonic() < deadline:
        time.sleep(.005)
    assert not session.busy
    return session.snapshot()


def test_background_voice_reserves_question_busy_uses_capture_identity_and_replays_once(make_session):
    session, transcriber, queries = make_session()
    session.selected_session_id = "browser-selected"
    _, audio = submit(session)
    assert transcriber.entered.wait(1), "Voice required an HTTP poll or recording restart"
    assert session.busy and session.phase == "transcribing"
    assert session.snapshot()["native"]["recording"]
    with pytest.raises(RuntimeError, match="Wait"):
        session.ask("Typed question", session_id="browser-selected")
    transcriber.release.set()
    state = wait_done(session)
    assert queries.seen == [("captured", "Show the last yellow fall", None, None)]
    assert session.native.played == [("b" * 32, 1, 6)]
    assert state["result"]["native_auto_played"] is True
    assert state["result"]["matches"][0]["session_time_s"] == 14
    assert state["voice"]["state"] == "replaying" and state["voice"]["transcript"] == "Show the last yellow fall"
    assert not audio.exists() and not session.native.stopped
    assert [item[0] for item in session.native.feedback] == ["transcribing", "searching", "replaying"]


@pytest.mark.parametrize("matches", [[], [{"verdict": "maybe", "start_s": 1, "end_s": 6, "time_s": 4}],
    [{"verdict": "yes", "start_s": float("nan"), "end_s": 6, "time_s": 4}]])
def test_no_confirmed_bounded_match_does_not_autoplay(make_session, matches):
    session, transcriber, _ = make_session(matches)
    _, audio = submit(session)
    assert transcriber.entered.wait(1)
    transcriber.release.set()
    state = wait_done(session)
    assert session.native.played == [] and not audio.exists()
    assert state["voice"]["state"] == "no_match"
    assert not state["result"].get("native_auto_played")


def test_transcription_failure_releases_busy_reports_error_and_removes_audio(make_session):
    session, transcriber, queries = make_session(failure=RuntimeError("Speech model unavailable"))
    _, audio = submit(session)
    assert transcriber.entered.wait(1)
    transcriber.release.set()
    state = wait_done(session)
    assert not queries.seen and not audio.exists()
    assert state["voice"]["state"] == "error" and "Speech model" in state["error"]
    assert session.native.feedback[-1][0] == "error"


def test_voice_waits_for_typed_question_before_claiming(make_session):
    session, transcriber, _ = make_session()
    with session.lock:
        session._begin("searching")
    _, audio = submit(session)
    assert not transcriber.entered.wait(.05)
    assert audio.exists() and list((session.root / "native/voice/requests").glob("*.json"))
    with session.lock:
        session.busy = False
    assert transcriber.entered.wait(1)


def test_native_instance_change_during_transcription_never_replays_into_new_host(make_session):
    session, transcriber, queries = make_session()
    _, audio = submit(session)
    assert transcriber.entered.wait(1)
    session.native.state["instance_id"] = "replacement"
    transcriber.release.set()
    state = wait_done(session)
    assert not queries.seen and not session.native.played and not audio.exists()
    assert session.voice["state"] == "error"
    assert "recorder changed" in state["error"]


def test_close_fences_transcription_completion_and_new_queue_work(make_session):
    session, transcriber, queries = make_session()
    _, audio = submit(session)
    assert transcriber.entered.wait(1)
    closer = threading.Thread(target=session.close)
    closer.start()
    assert session.native.closed.wait(1), "Capture shutdown waited for speech recognition"
    transcriber.release.set()
    closer.join(2)
    assert not closer.is_alive()
    assert not queries.seen and not session.native.played and not audio.exists()
    assert not session.busy


@pytest.mark.parametrize("phase", ["transcribing", "searching"])
@pytest.mark.parametrize("native_voice", [
    {"request_id": "a" * 32, "state": "error"},
    {"request_id": "c" * 32, "state": "transcribing"},
    {"request_id": "", "state": "idle"},
])
def test_native_cancellation_or_watchdog_fences_late_voice_work(make_session, phase, native_voice):
    session, transcriber, queries = make_session()
    entered, release = threading.Event(), threading.Event()
    original_ask = queries.ask
    def slow_ask(*args, **kwargs):
        entered.set()
        assert release.wait(2)
        return original_ask(*args, **kwargs)
    if phase == "searching": queries.ask = slow_ask
    submit(session)
    assert transcriber.entered.wait(1)
    if phase == "searching":
        transcriber.release.set()
        assert entered.wait(1)
    session.native.state["voice"] = native_voice
    transcriber.release.set()
    release.set()
    state = wait_done(session)
    assert not session.native.played
    assert phase == "searching" or not queries.seen
    assert session.voice["state"] == "error"
    assert "canceled or expired" in state["error"]


@pytest.mark.parametrize("failure_point", ["search", "play"])
def test_search_or_replay_failure_releases_busy_and_audio(make_session, failure_point):
    session, transcriber, queries = make_session()
    def fail(*args, **kwargs): raise RuntimeError(f"{failure_point} failed")
    if failure_point == "search": queries.ask = fail
    else: session.native.play = fail
    _, audio = submit(session)
    assert transcriber.entered.wait(1)
    transcriber.release.set()
    state = wait_done(session)
    assert not audio.exists() and not session.native.stopped
    assert state["voice"]["state"] == "error"
    assert f"{failure_point} failed" in state["error"]


def test_speech_warmup_is_background_work_and_does_not_own_capture_controls(make_session):
    session, transcriber, _ = make_session()
    entered, release = threading.Event(), threading.Event()
    def warmup():
        entered.set()
        assert release.wait(2)
    transcriber.warmup = warmup
    submit(session)
    try:
        assert entered.wait(1)
        state = session.snapshot()
        assert not state["busy"] and state["voice"]["model"]["state"] == "warming"
        session.stop()
        session.capture_worker.join(1)
        assert session.native.stopped and not session.capture_worker.is_alive()
    finally:
        release.set()


def test_feedback_is_nonblocking_fenced_command_with_bounded_user_text(tmp_path):
    host_status(tmp_path)
    request, _ = queue(tmp_path)
    native = NativeRecorder(tmp_path, MatchStore(tmp_path / "matches"))
    assert native.voice_feedback(request, "searching", "Checking frames", "Yellow falls") is True
    commands = list((tmp_path / "native/commands").glob("*.json"))
    assert len(commands) == 1
    command = json.loads(commands[0].read_text())
    assert {key: command[key] for key in ("type", "instance_id", "no_reply", "voice_id", "state", "message", "transcript")} == {
        "type": "voice_status", "instance_id": "instance-one", "no_reply": True, "voice_id": "a" * 32,
        "state": "searching", "message": "Checking frames", "transcript": "Yellow falls"}


def test_close_releases_the_pending_native_voice_request(make_session):
    session, transcriber, _ = make_session()
    submit(session)
    assert transcriber.entered.wait(1)
    closer = threading.Thread(target=session.close)
    closer.start()
    try:
        assert session.native.closed.wait(1)
        assert session.native.feedback[-1][0] == "error"
        assert "stopped" in session.native.feedback[-1][1].lower()
    finally:
        transcriber.release.set()
        closer.join(2)


@pytest.mark.parametrize("host_voice", [
    {"request_id": "a" * 32, "state": "error", "message": "Voice request timed out"},
    {"request_id": "", "state": "idle", "message": "Voice canceled because the game changed"},
])
def test_native_cancellation_is_visible_before_slow_transcription_finishes(make_session, host_voice):
    session, transcriber, _ = make_session()
    submit(session)
    assert transcriber.entered.wait(1)
    session.native.state["voice"] = host_voice
    state = session.snapshot()
    assert state["busy"]
    assert state["voice"]["state"] == host_voice["state"]
    assert state["voice"]["message"] == host_voice["message"]


def test_a_host_without_voice_queue_support_never_loads_the_speech_model(make_session):
    session, transcriber, _ = make_session()
    warmed = threading.Event()
    transcriber.warmup = warmed.set
    session.native.claim_voice_request = None
    submit(session)
    assert not warmed.wait(.05)
    assert not session.busy


def test_stale_crash_artifacts_remove_only_expired_voice_audio(tmp_path):
    host_status(tmp_path)
    _, audio = queue(tmp_path)
    requests = tmp_path / "native/voice/requests"
    claimed = requests / ("a" * 32 + ".json.claimed-deadworker")
    (requests / ("a" * 32 + ".json")).rename(claimed)
    marker = requests / ("a" * 32 + ".claim")
    marker.touch()
    unrelated = wav(tmp_path / "private.wav")
    fresh = wav(tmp_path / "native/voice" / ("b" * 32 + ".wav"))
    old = time.time() - 121
    for artifact in (audio, claimed, marker): os.utime(artifact, (old, old))
    native = NativeRecorder(tmp_path, MatchStore(tmp_path / "matches"))
    assert native.claim_voice_request() is None
    assert not audio.exists() and not claimed.exists() and not marker.exists()
    assert unrelated.exists() and fresh.exists()


def test_voice_playback_serializes_both_native_and_pending_request_identity(tmp_path):
    host_status(tmp_path)
    native = NativeRecorder(tmp_path, MatchStore(tmp_path / "matches"))
    media = native.store.allocate()
    media.write_bytes(b"pinned")
    match = Match(media.parent.name, media, 10, "2026-10-25T01:00:00Z", "native")
    observed = []
    def host():
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            commands = list((tmp_path / "native/commands").glob("*.json"))
            if commands:
                command = json.loads(commands[0].read_text())
                observed.append(command)
                atomic_json(tmp_path / "native/responses" / commands[0].name,
                    {"id": command["id"], "instance_id": "instance-one", "ok": True})
                return
            time.sleep(.005)
    worker = threading.Thread(target=host)
    worker.start()
    try:
        native.play(match, 1, 6, instance_id="instance-one", voice_id="a" * 32, session_id="captured")
    finally:
        worker.join(2)
    assert observed[0]["voice_id"] == "a" * 32
    assert observed[0]["session_id"] == "captured" and observed[0]["instance_id"] == "instance-one"


def test_successful_transcription_recovers_the_model_status_after_failed_warmup(make_session):
    session, transcriber, _ = make_session()
    def fail_warmup(): raise RuntimeError("Speech model could not prepare")
    transcriber.warmup = fail_warmup
    submit(session)
    assert transcriber.entered.wait(1)
    assert session.snapshot()["voice"]["model"]["state"] == "error"
    transcriber.release.set()
    assert wait_done(session)["voice"]["model"]["state"] == "ready"


def test_voice_skips_uncertain_candidates_and_opens_only_the_first_confirmed_interval(make_session):
    session, transcriber, _ = make_session([
        {"verdict": "maybe", "start_s": 0, "end_s": 2, "time_s": 1},
        {"verdict": "yes", "start_s": 2, "end_s": 4, "time_s": 3},
        {"verdict": "yes", "start_s": 5, "end_s": 8, "time_s": 6},
    ])
    submit(session)
    assert transcriber.entered.wait(1)
    transcriber.release.set()
    assert wait_done(session)["result"]["native_auto_played"] is True
    assert session.native.played == [("b" * 32, 2, 4)]

