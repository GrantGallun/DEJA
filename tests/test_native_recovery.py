from datetime import datetime, timezone, timedelta
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest

from smartreplay.demo.capture import MatchStore
from smartreplay.demo.native import NativeRecorder, atomic_json
from smartreplay.frontend.runtime import ServiceLease


class Process:
    def __init__(self):
        self.pid = 123
        self.terminated = self.released = False
    def poll(self): return 1 if self.terminated else None
    def terminate(self): self.terminated = True
    kill = terminate
    def wait(self, timeout=None): return 1
    def close_handle(self): self.released = True


def recorder(tmp_path, monkeypatch, *, healthy=True):
    from smartreplay.demo import native
    n = NativeRecorder(tmp_path, MatchStore(tmp_path / "matches"))
    state = {"version": 1, "pid": 123, "instance_id": "old-host", "recording": True,
        "session_id": "same-session", "enabled": True,
        "updated_at": datetime.now(timezone.utc).isoformat()}
    atomic_json(n.control_dir / "status.json", state)
    (n.control_dir / "native.lock").write_text("123\nGameHQPassiveReplay\n")
    process = Process()
    monkeypatch.setattr(native, "attach_native_process", lambda *args: process, raising=False)
    n.status = lambda: {**state, "available": healthy}
    calls = []
    n.request = lambda kind, **kw: calls.append((kind, kw)) or {"ok": True}
    return n, process, calls


def test_healthy_orphan_is_reused_and_owned_through_shutdown(tmp_path, monkeypatch):
    n, process, calls = recorder(tmp_path, monkeypatch)
    with ServiceLease(tmp_path) as lease:
        assert n.recover(lease) is True
        assert n.process is process and not process.terminated
        assert calls[0] == ("start", {"expected_instance": "old-host"})
        n.close()
    assert process.terminated and process.released
    assert calls[1][0] == "stop"


def test_stale_verified_orphan_is_retired_before_new_launch(tmp_path, monkeypatch):
    n, process, calls = recorder(tmp_path, monkeypatch, healthy=False)
    with ServiceLease(tmp_path) as lease:
        assert n.recover(lease) is False
    assert process.terminated and process.released
    assert n.process is None


def test_recovery_requires_held_lease_for_same_recording_root(tmp_path, monkeypatch):
    n, process, _ = recorder(tmp_path, monkeypatch)
    with pytest.raises(RuntimeError, match="ownership"):
        n.recover(ServiceLease(tmp_path))
    with ServiceLease(tmp_path / "other") as lease:
        with pytest.raises(RuntimeError, match="ownership"):
            n.recover(lease)
    assert not process.terminated


def test_mismatched_native_lock_never_terminates_a_process(tmp_path, monkeypatch):
    n, process, _ = recorder(tmp_path, monkeypatch)
    (n.control_dir / "native.lock").write_text("999\n")
    with ServiceLease(tmp_path) as lease:
        with pytest.raises(RuntimeError, match="verify"):
            n.recover(lease)
    assert not process.terminated


def test_failed_process_identity_check_never_adopts_or_kills(tmp_path, monkeypatch):
    from smartreplay.demo import native
    n, process, _ = recorder(tmp_path, monkeypatch)
    def refuse(*args): raise RuntimeError("Cannot verify recorder executable or start time")
    monkeypatch.setattr(native, "attach_native_process", refuse)
    with ServiceLease(tmp_path) as lease:
        with pytest.raises(RuntimeError, match="verify"):
            n.recover(lease)
    assert n.process is None and not process.terminated


@pytest.mark.skipif(os.name != "nt", reason="Windows process handles")
def test_process_identity_rejects_wrong_binary_and_reused_pid():
    from smartreplay.demo.native_process import attach_native_process
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"],
                             creationflags=subprocess.CREATE_NO_WINDOW)
    try:
        now = datetime.now(timezone.utc).isoformat()
        with pytest.raises(RuntimeError, match="verify"):
            attach_native_process(child.pid, Path(sys.executable).with_name("wrong.exe"), now)
        before_start = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        with pytest.raises(RuntimeError, match="reused"):
            attach_native_process(child.pid, sys.executable, before_start)
        assert child.poll() is None
        attached = attach_native_process(child.pid, sys.executable, now)
        assert attached.pid == child.pid and attached.poll() is None
        attached.close_handle()
    finally:
        child.terminate()
        child.wait(timeout=5)


PROBE = Path(__file__).resolve().parents[1] / "experiments/gamehq-passive/out-passive/NativeSessionProbe.exe"


@pytest.mark.skipif(os.name != "nt" or not PROBE.exists(), reason="Windows native probe required")
def test_real_crashed_owner_reconnects_same_recording_and_closes_host(tmp_path):
    from smartreplay.demo.native_process import attach_native_process
    script = textwrap.dedent('''
        import json, os, sys, time
        from pathlib import Path
        from smartreplay.demo.native import NativeRecorder
        from smartreplay.demo.capture import MatchStore
        from smartreplay.frontend.runtime import ServiceLease
        root = Path(sys.argv[1])
        with ServiceLease(root):
            n = NativeRecorder(root, MatchStore(root / 'matches'), binary=Path(sys.argv[2]))
            n.start()
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                state = n.status()
                if state.get('available') and (state.get('active_session') or {}).get('duration_s', 0) >= 1:
                    print(json.dumps(state), flush=True)
                    os._exit(17)  # No close(), finally block, or recorder shutdown.
                time.sleep(.05)
            os._exit(2)
    ''')
    n = NativeRecorder(tmp_path, MatchStore(tmp_path / "matches"), binary=PROBE)
    try:
        owner = subprocess.run([sys.executable, "-c", script, str(tmp_path), str(PROBE)],
            capture_output=True, text=True, timeout=25, creationflags=subprocess.CREATE_NO_WINDOW)
        assert owner.returncode == 17, owner.stderr
        before = json.loads(owner.stdout)
        with ServiceLease(tmp_path) as lease:
            assert n.recover(lease)
            after = n.status()
            assert n.process.pid == before["pid"]
            assert after["session_id"] == before["session_id"]
            assert after["recording"] and after["available"]
            n.flush(after["session_id"])
            assert n.session(after["session_id"])["duration_s"] >= before["active_session"]["duration_s"]
            attached = n.process
            n.close()
            assert attached.poll() is not None
            assert n.session(after["session_id"])["state"] == "closed"
    finally:
        # Even a failed assertion must not leave this generated-media fixture running.
        state_path = n.control_dir / "status.json"
        if state_path.exists():
            state = json.loads(state_path.read_text())
            process = attach_native_process(state["pid"], PROBE, state["updated_at"])
            if process:
                process.terminate()
                process.wait(timeout=5)
                process.close_handle()
