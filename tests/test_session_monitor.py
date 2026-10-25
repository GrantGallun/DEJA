"""Background archive coordination, with native/model boundaries kept offline."""
from copy import deepcopy
import json
import threading

import cv2
import numpy as np
import pytest

from smartreplay.demo.native import NativeRecorder, atomic_json
from smartreplay.demo.server import DemoSession


def write_archive(root, session_id="current", duration=1, state="recording"):
    directory = root / "sessions" / session_id
    directory.mkdir(parents=True, exist_ok=True)
    video = directory / "segment.mp4"
    if not video.exists():
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 8, (64, 48))
        assert writer.isOpened()
        for frame in range(8):
            writer.write(np.full((48, 64, 3), frame * 20, np.uint8))
        writer.release()
    manifest = {"id": session_id, "game": "Generated footage", "created_at": "2026-10-24T12:00:00Z",
                "state": state, "duration_s": duration,
                "segments": [{"path": str(video.resolve()), "start_s": 0, "end_s": duration}]}
    atomic_json(directory / "session.json", manifest)
    return manifest


class ExternalHost(NativeRecorder):
    """Replace OS capture control, while retaining the real manifest reader."""
    def __init__(self, root, store, *, recording=False):
        super().__init__(root, store)
        self.state_lock = threading.Lock()
        self.state = {"available": True, "enabled": recording, "recording": recording,
                      "session_id": "current" if recording else None,
                      "active_session": None, "error": "", "cache": {"bytes": 0, "budget_bytes": 128 * 1024 * 1024}}
        self.start_calls = self.stop_calls = 0
        self.closed = threading.Event()
        self.bad_status = False
        self.bad_status_response = None
        self.bad_status_seen = threading.Event()
        self.start_entered = threading.Event()
        self.start_release = None

    def status(self):
        with self.state_lock:
            if self.bad_status:
                self.bad_status_seen.set()
                if isinstance(self.bad_status_response, Exception):
                    raise self.bad_status_response
                return self.bad_status_response
            state = deepcopy(self.state)
        if state["session_id"]:
            manifest = self.session(state["session_id"])
            state["active_session"] = {k: manifest[k] for k in ("id", "game", "state", "duration_s", "created_at")}
        return state

    def start(self):
        self.start_calls += 1
        self.start_entered.set()
        if self.start_release:
            assert self.start_release.wait(3)
        with self.state_lock:
            self.state.update(enabled=True, recording=True, session_id="current")
        return {"ok": True}

    def stop(self):
        self.stop_calls += 1
        manifest = self.session("current")
        write_archive(self.root, duration=manifest["duration_s"] + .25, state="closed")
        with self.state_lock:
            self.state.update(enabled=False, recording=False, session_id=None, active_session=None)
        return {"ok": True}

    def flush(self, session_id):
        assert session_id == "current"

    def close(self):
        self.closed.set()


class Queries:
    """Observe work submitted to the external model/index boundary."""
    def __init__(self, factory):
        self.factory = factory
        self.condition = threading.Condition()
        self.refreshes = []
        self.closed = threading.Event()
        self.ask_entered = threading.Event()
        self.ask_release = threading.Event()
        self.close_entered = threading.Event()
        self.close_release = None
        self.close_lock_check = None
        self.lock_available_during_close = None

    def refresh(self, manifest):
        # A compact status summary must never reach the index boundary.
        assert manifest["segments"] and all("path" in s for s in manifest["segments"])
        with self.condition:
            assert not self.closed.is_set(), "New index work was submitted after closing its worker"
            self.refreshes.append((manifest["id"], manifest["duration_s"], manifest["state"]))
            self.condition.notify_all()

    def wait_for(self, value, timeout=2):
        with self.condition:
            return self.condition.wait_for(lambda: value in self.refreshes, timeout=timeout)

    def status(self, session_id):
        with self.condition:
            latest = next((row for row in reversed(self.refreshes) if row[0] == session_id), None)
        return {"indexed_until_s": 0, "duration_s": latest[1] if latest else 0,
                "indexing": False, "pending": 0, "error": ""}

    def ask(self, manifest, question, start_s=None, end_s=None):
        self.ask_entered.set()
        assert self.ask_release.wait(3)
        return {"snapshot": None, "result": {"query": question, "matches": [], "note": "No candidate"},
                "scope": {"mode": "session", "complete": False}, "indexed_until_s": 0}

    def close(self):
        self.close_entered.set()
        if self.close_lock_check:
            acquired = threading.Event()
            def inspect_lock():
                with self.close_lock_check:
                    acquired.set()
            thread = threading.Thread(target=inspect_lock, daemon=True)
            thread.start()
            self.lock_available_during_close = acquired.wait(.5)
            thread.join(.5)
        if self.close_release:
            assert self.close_release.wait(3)
        self.closed.set()


class QueryFactory:
    def __init__(self):
        self.instances = []
        self.created = threading.Event()
        self.condition = threading.Condition()

    def __call__(self, root, model_factory, snapshotter):
        query = Queries(self)
        with self.condition:
            self.instances.append(query)
            self.created.set()
            self.condition.notify_all()
        return query

    def wait_for_count(self, count, timeout=2):
        with self.condition:
            return self.condition.wait_for(lambda: len(self.instances) >= count, timeout=timeout)


@pytest.fixture
def make_session(tmp_path, monkeypatch):
    import smartreplay.demo.server as server
    monkeypatch.setattr(server, "_SESSION_MONITOR_INTERVAL_S", .01, raising=False)
    sessions = []
    def create(*, recording=False):
        root = tmp_path / str(len(sessions))
        write_archive(root)
        queries = QueryFactory()
        session = DemoSession(root, native_factory=lambda r, s: ExternalHost(r, s, recording=recording),
                              session_search_factory=queries, window_finder=lambda: None,
                              model_factory=lambda: pytest.fail("No real model calls are allowed"))
        sessions.append(session)
        return session, queries
    yield create
    for session in sessions:
        for query in getattr(session, "_test_queries", []):
            query.ask_release.set()
            if query.close_release:
                query.close_release.set()
        session.close()


def test_record_indexes_finalized_archive_without_status_requests(make_session):
    session, factory = make_session()
    assert not factory.created.is_set()
    session.record()
    session.capture_worker.join(1)
    assert factory.created.wait(1), "Recording never started background archive indexing"
    assert factory.instances[0].wait_for(("current", 1, "recording"))
    assert session.native.start_calls == 1


def test_server_reconnect_indexes_existing_host_without_starting_capture(make_session):
    session, factory = make_session(recording=True)
    assert factory.created.wait(1), "An existing recording was ignored until the browser polled"
    assert factory.instances[0].wait_for(("current", 1, "recording"))
    assert session.native.start_calls == 0


def test_selecting_old_session_cannot_starve_the_current_recording(make_session):
    session, factory = make_session(recording=True)
    write_archive(session.root, "old", state="closed")
    session.select_session("old")
    assert factory.created.wait(1)
    query = factory.instances[0]
    assert query.wait_for(("current", 1, "recording"))
    write_archive(session.root, duration=2)
    assert query.wait_for(("current", 2, "recording"))
    assert session.selected_session_id == "old"


def test_stop_refreshes_the_final_partial_archive_and_close_stops_new_work(make_session):
    session, factory = make_session(recording=True)
    assert factory.created.wait(1)
    query = factory.instances[0]
    assert query.wait_for(("current", 1, "recording"))
    session.stop()
    session.capture_worker.join(1)
    assert query.wait_for(("current", 1.25, "closed")), "Final partial footage was never sent to the index"
    session.close()
    with query.condition:
        count = len(query.refreshes)
    write_archive(session.root, duration=3, state="closed")
    assert not query.wait_for(("current", 3, "closed"), timeout=.05)
    assert len(query.refreshes) == count
    assert len(factory.instances) == 1


@pytest.mark.parametrize("bad_status", [None, {}, RuntimeError("Native status file is unavailable")])
def test_malformed_status_is_visible_and_does_not_kill_the_monitor(make_session, bad_status):
    session, factory = make_session(recording=True)
    assert factory.created.wait(1)
    query = factory.instances[0]
    assert query.wait_for(("current", 1, "recording"))
    with session.native.state_lock:
        session.native.bad_status_response = bad_status
        session.native.bad_status = True
    assert session.native.bad_status_seen.wait(1)
    state = session.snapshot()
    assert state["index"]["error"], "The failed background read should remain visible in index state"
    assert not session.capture_error and session.native.stop_calls == 0
    with session.native.state_lock:
        session.native.bad_status = False
    write_archive(session.root, duration=2)
    assert query.wait_for(("current", 2, "recording")), "The monitor did not recover from malformed status"


def test_close_stops_native_and_index_before_waiting_for_a_question(make_session):
    session, factory = make_session(recording=True)
    assert factory.created.wait(1)
    query = factory.instances[0]
    session._test_queries = [query]
    session.ask("Any event?", session_id="current")
    assert query.ask_entered.wait(1)
    # Capture stop stays available while the independent question worker is occupied.
    session.stop()
    session.capture_worker.join(1)
    assert session.native.stop_calls == 1 and session.busy
    closer = threading.Thread(target=session.close, daemon=True)
    closer.start()
    try:
        assert session.native.closed.wait(1), "Native shutdown waited for an unrelated question"
        assert query.closed.wait(1), "Index shutdown waited for the question worker"
    finally:
        query.ask_release.set()
        closer.join(2)
    assert not closer.is_alive()


def test_close_coordinates_an_inflight_capture_start_before_native_shutdown(make_session):
    session, factory = make_session()
    session.native.start_release = threading.Event()
    session.record()
    assert session.native.start_entered.wait(1)
    closer = threading.Thread(target=session.close, daemon=True)
    closer.start()
    try:
        assert not session.native.closed.wait(.05), "Native shutdown raced an unfinished capture command"
    finally:
        session.native.start_release.set()
        closer.join(2)
    assert session.native.closed.is_set() and not closer.is_alive()
    assert not factory.created.is_set(), "A finishing capture command created an index worker during shutdown"
    with pytest.raises(RuntimeError):
        session.record()


def test_backend_switch_closes_index_outside_lock_and_fences_old_monitor(make_session):
    session, factory = make_session(recording=True)
    assert factory.created.wait(1)
    old = factory.instances[0]
    session._test_queries = [old]
    old.close_lock_check = session.lock
    old.close_release = threading.Event()
    changed = threading.Event()
    def change_backend():
        session.set_backend("claude")
        changed.set()
    changer = threading.Thread(target=change_backend, daemon=True)
    changer.start()
    try:
        assert old.close_entered.wait(1)
        # A browser poll during the change cannot revive obsolete index work.
        state = session.snapshot()
        assert state["index"]["error"]
        assert len(factory.instances) == 1
    finally:
        old.close_release.set()
        changer.join(2)
    assert changed.is_set()
    assert old.lock_available_during_close, "Closing an index worker held the lock its worker may need"
    write_archive(session.root, duration=2)
    # Wait on a model-boundary condition, without needing an HTTP poll to resume indexing.
    assert factory.wait_for_count(2)
    assert factory.instances[-1].wait_for(("current", 2, "recording"))
    assert len(factory.instances) == 2 and factory.instances[-1] is not old


def test_imported_file_does_not_start_background_model_work(make_session):
    session, factory = make_session()
    source = session.root / "sessions/current/segment.mp4"
    session.import_video(source)
    assert session.snapshot()["match"]
    assert not factory.created.wait(.05)


def test_capture_started_during_backend_change_is_monitored_after_switch(make_session):
    session, factory = make_session()
    old = session._queries()
    session._test_queries = [old]
    old.close_release = threading.Event()
    changer = threading.Thread(target=lambda: session.set_backend("claude"), daemon=True)
    changer.start()
    try:
        assert old.close_entered.wait(1)
        session.record()
        session.capture_worker.join(1)
        assert not session.capture_worker.is_alive()
    finally:
        old.close_release.set()
        changer.join(2)
    assert factory.wait_for_count(2), "A successful start during the model change lost its monitor request"
    assert factory.instances[-1].wait_for(("current", 1, "recording"))


