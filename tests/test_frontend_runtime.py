import threading
import time
from types import SimpleNamespace

import pytest

from smartreplay.frontend.runtime import PlayerRuntime, ServiceLease


def eventually(predicate):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.01)
    assert predicate()


class Session:
    def __init__(self, root, settings_file):
        self.lock = threading.RLock()
        self.state = {"available": False, "recording": False, "enabled": False}
        self.recoveries = 0
        self.native = SimpleNamespace(status=lambda: dict(self.state), recover=self.recover)
        self.capture_error = ""
        self.capture_control_busy = False
        self.model_status = {"state": "error", "message": "Model offline"}
        self.voice_model = {"state": "ready"}
        self.voice = {"state": "idle"}
        self.starts = 0
        self.closed = False
        self.model_checks = 0

    def recover(self, lease):
        assert lease.file is not None
        self.recoveries += 1
        return self.state["available"]

    def check_model(self):
        self.model_checks += 1

    def record(self):
        self.starts += 1
        self.state.update(available=True, enabled=True)

    def close(self):
        self.closed = True


def test_arms_native_detector_without_model_or_browser_and_closes(tmp_path):
    session = Session(tmp_path, None)
    runtime = PlayerRuntime(tmp_path, tmp_path / "settings.json",
                            session_factory=lambda *a, **kw: session, interval=.01, retry_delay=.02)
    runtime.start()
    try:
        eventually(lambda: runtime.snapshot()["state"] == "waiting")
        assert session.starts == 1
        assert session.model_checks == 1
        assert runtime.snapshot()["model"]["state"] == "error"
        session.state.update(recording=True, active_session={"game": "Stick Fight"})
        eventually(lambda: runtime.snapshot()["state"] == "recording")
        session.state.update(available=False, recording=False)
        eventually(lambda: session.starts == 2)
    finally:
        runtime.close()
    assert session.closed


def test_native_exit_is_reported(tmp_path):
    session = Session(tmp_path, None)
    session.native.process = SimpleNamespace(poll=lambda: 10)
    session.record = lambda: None
    runtime = PlayerRuntime(tmp_path, tmp_path / "settings.json", session_factory=lambda *a, **kw: session,
                            interval=.01, retry_delay=30)
    runtime.start()
    try:
        eventually(lambda: runtime.snapshot()["state"] == "error")
        assert "10" in runtime.snapshot()["message"]
    finally:
        runtime.close()


@pytest.mark.parametrize("enabled", [True, False])
def test_existing_orphan_is_recovered_under_the_service_lease(tmp_path, enabled):
    session = Session(tmp_path, None)
    session.state.update(available=True, enabled=enabled)
    runtime = PlayerRuntime(tmp_path, tmp_path / "settings.json", session_factory=lambda *a, **kw: session,
                            interval=.01)
    runtime.start()
    try:
        eventually(lambda: runtime.snapshot()["state"] == "waiting")
        assert session.recoveries == 1
        assert session.starts == 1
    finally:
        runtime.close()
    assert session.closed


def test_second_service_never_touches_the_first_owners_recorder(tmp_path):
    created = []
    with ServiceLease(tmp_path):
        runtime = PlayerRuntime(tmp_path, tmp_path / "settings.json",
            session_factory=lambda *a, **kw: created.append(True), interval=.01)
        runtime.start()
        eventually(lambda: runtime.snapshot()["state"] == "error")
        runtime.close()
    assert not created


def test_model_check_scheduling_failure_does_not_block_capture(tmp_path):
    session = Session(tmp_path, None)
    def fail():
        raise RuntimeError("Model check unavailable")
    session.check_model = fail
    runtime = PlayerRuntime(tmp_path, tmp_path / "settings.json", session_factory=lambda *a, **kw: session,
                            interval=.01)
    runtime.start()
    try:
        eventually(lambda: runtime.snapshot()["state"] == "waiting")
        assert session.starts == 1
        assert runtime.snapshot()["model"]["message"] == "Model check unavailable"
    finally:
        runtime.close()


def test_settings_endpoints_include_runtime_and_close_it(tmp_path):
    import http.client
    import json
    from smartreplay.frontend.server import SettingsServer
    server = SettingsServer(tmp_path / "settings.json")
    closed = []
    server.runtime = SimpleNamespace(snapshot=lambda: {"state": "waiting", "message": "Ready"},
                                     close=lambda: closed.append(True))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        for route in ("/api/status", "/api/settings"):
            conn = http.client.HTTPConnection(*server.server_address)
            conn.request("GET", route)
            response = conn.getresponse()
            assert response.status == 200
            data = json.loads(response.read())
            status = data["status"] if route == "/api/settings" else data
            assert status["runtime"]["state"] == "waiting"
            conn.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    assert closed == [True]


def test_desktop_lifecycle_starts_and_closes_service_even_if_window_fails(monkeypatch, tmp_path):
    import sys
    from smartreplay.frontend import window
    monkeypatch.setattr(window, "DEMO_MATCHES", tmp_path / "matches")
    calls = []
    server = SimpleNamespace(origin="http://127.0.0.1:12345",
        start_player=lambda: calls.append("player"), serve_forever=lambda: None,
        shutdown=lambda: calls.append("shutdown"), server_close=lambda: calls.append("closed"))
    monkeypatch.setattr(window, "make_server", lambda *a: server)
    def fail(*a, **kw):
        raise RuntimeError("window failed")
    monkeypatch.setitem(sys.modules, "webview", SimpleNamespace(create_window=fail))
    with pytest.raises(RuntimeError, match="window failed"):
        window.run()
    assert calls == ["player", "shutdown", "closed"]


def test_native_command_uses_deja_settings_file(tmp_path):
    from smartreplay.demo.native import NativeRecorder
    from smartreplay.demo.capture import MatchStore
    native = NativeRecorder(tmp_path, MatchStore(tmp_path / "matches"))
    native.settings_file = tmp_path / "custom settings.json"
    command = native.command()
    assert command[command.index("--settings") + 1] == str(native.settings_file.resolve())


def test_startup_failure_is_visible_and_releases_ownership(tmp_path):
    def fail(*args, **kwargs):
        raise ImportError("Missing capture dependency")
    runtime = PlayerRuntime(tmp_path, tmp_path / "settings.json", session_factory=fail, interval=.01)
    runtime.start()
    try:
        eventually(lambda: runtime.snapshot()["state"] == "error")
        assert "Missing capture dependency" in runtime.snapshot()["message"]
    finally:
        runtime.close()
    with ServiceLease(tmp_path):
        pass


def test_exclusive_service_owner_and_release(tmp_path):
    with ServiceLease(tmp_path):
        with pytest.raises(RuntimeError, match="already"):
            with ServiceLease(tmp_path):
                pass
    with ServiceLease(tmp_path):
        pass


def test_capture_failure_is_visible_without_retry_storm(tmp_path):
    session = Session(tmp_path, None)
    def fail():
        session.starts += 1
        session.capture_error = "Native executable missing"
    session.record = fail
    runtime = PlayerRuntime(tmp_path, tmp_path / "settings.json", session_factory=lambda *a, **kw: session,
                            interval=.01, retry_delay=30)
    runtime.start()
    try:
        eventually(lambda: runtime.snapshot()["state"] == "error")
        assert "Native executable missing" in runtime.snapshot()["message"]
        assert session.starts == 1
    finally:
        runtime.close()
