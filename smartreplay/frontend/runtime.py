"""Deja's headless replay service. The native host owns game detection/capture."""
from __future__ import annotations

import os
from pathlib import Path
import threading
import time


class ServiceAlreadyRunning(RuntimeError):
    pass


class ServiceLease:
    """One service per recording directory; the OS releases the lease on exit."""

    def __init__(self, root, filename="service.lock"):
        self.path = Path(root) / filename
        self.file = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+b")
        try:
            if not self.path.stat().st_size:
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            handle.close()
            raise ServiceAlreadyRunning("Deja's player service is already running. Close the other deja or demo instance.") from exc
        self.file = handle
        return self

    def __exit__(self, *args):
        if self.file:
            self.file.close()
            self.file = None


class PlayerRuntime:
    def __init__(self, root, settings_file, *, session_factory=None, interval=1.0, retry_delay=10.0):
        self.root, self.settings_file = Path(root).resolve(), Path(settings_file)
        self.session_factory = session_factory
        self.interval, self.retry_delay = interval, retry_delay
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread = None
        self._session = None
        self._status = {"state": "starting", "message": "Starting the player…"}

    def start(self):
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="deja-player", daemon=True)
            self._thread.start()

    def snapshot(self):
        with self._lock:
            return dict(self._status)

    def _publish(self, state, message, **extra):
        with self._lock:
            self._status = {"state": state, "message": message, **extra}

    def _run(self):
        try:
            with ServiceLease(self.root) as lease:
                factory = self.session_factory
                if factory is None:
                    from ..demo.server import DemoSession
                    session = DemoSession(self.root, settings_file=self.settings_file,
                        native_recovery=lambda native: native.recover(lease))
                else:
                    session = factory(self.root, settings_file=self.settings_file)
                try:
                    if factory is not None:
                        session.native.recover(lease)
                    with self._lock:
                        self._session = session
                    # This schedules a background check; capture never waits for the model.
                    try:
                        session.check_model()
                    except Exception as exc:
                        with session.lock:
                            session.model_status = {"state": "error", "message": str(exc)}
                    if session.native.status().get("available"):
                        session.record()  # Resume session indexing for an adopted host.
                    retry_at = 0
                    while not self._stop.is_set():
                        native = session.native.status()
                        with session.lock:
                            busy, error = session.capture_control_busy, session.capture_error
                            extra = {"model": dict(session.model_status),
                                     "voice_model": dict(session.voice_model), "voice": dict(session.voice),
                                     "native": native}
                        now = time.monotonic()
                        if not native.get("available"):
                            error = error or native.get("error", "")
                            process = getattr(session.native, "process", None)
                            exit_code = process.poll() if process is not None else None
                            if exit_code is not None and not error:
                                error = f"Player exited (code {exit_code}). See out/demo/native/host-stderr.log. Retrying…"
                            if not busy and now >= retry_at:
                                retry_at = now + self.retry_delay
                                session.record()
                            if error:
                                self._publish("error", error, **extra)
                            else:
                                self._publish("starting", "Starting the player…", **extra)
                        elif native.get("error"):
                            self._publish("error", native["error"], **extra)
                        elif native.get("recording"):
                            game = (native.get("active_session") or {}).get("game") or "gameplay"
                            self._publish("recording", f"Recording {game} · replay is ready", **extra)
                        elif native.get("enabled"):
                            self._publish("waiting", "Ready · waiting for gameplay in a supported game", **extra)
                        else:
                            self._publish("paused", "Recording paused", **extra)
                        self._stop.wait(self.interval)
                finally:
                    with self._lock:
                        self._session = None
                    session.close()
        except Exception as exc:
            self._publish("error", str(exc) or "The player could not start.")

    def close(self):
        self._stop.set()
        if self._thread:
            self._thread.join()

    def _service(self):
        with self._lock:
            session = self._session
        if session is None or self._stop.is_set():
            raise RuntimeError("The player service is not ready. Check deja's status above.")
        return session

    def search_recording(self, identity, question):
        session = self._service()
        with session.lock:
            session._begin("searching")
        try:
            snapshot, result = session._search_session(question, identity, "session", None)
            return {"snapshot": snapshot, "result": result}
        finally:
            with session.lock:
                session.busy = False
                session.phase = "ready" if session.match else "idle"

    def transcribe_question(self, path):
        session = self._service()
        with session.lock:
            if session.voice_transcriber is None:
                from ..demo.voice import VoiceTranscriber
                session.voice_transcriber = VoiceTranscriber()
            transcriber = session.voice_transcriber
        return transcriber.transcribe(path)
