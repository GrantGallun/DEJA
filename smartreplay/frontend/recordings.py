"""Read saved sessions and prepare playback/search without controlling capture."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile
import threading
import uuid


class RecordingLibrary:
    def __init__(self, root, *, searcher=None, transcriber=None):
        from ..demo.capture import MatchStore
        from ..demo.native import NativeRecorder
        self.root = Path(root).resolve()
        self.native = NativeRecorder(self.root, MatchStore(self.root / "matches"))
        self.searcher, self.transcriber = searcher, transcriber
        self._lock = threading.Lock()
        self._jobs, self._searches = {}, {}
        self._prepare_worker = self._search_worker = None
        self._closed = False

    def _session(self, identity):
        if not isinstance(identity, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", identity):
            raise ValueError("Invalid recording.")
        directory = (self.native.session_root / identity).resolve()
        if not directory.is_relative_to(self.native.session_root.resolve()):
            raise ValueError("Invalid recording directory.")
        return self.native.session(identity)

    @staticmethod
    def _playable(session):
        return bool(session["segments"]) and all(Path(s["path"]).is_file() for s in session["segments"])

    def list(self):
        rows = []
        for path in self.native.session_root.glob("*/session.json"):
            try:
                session = self._session(path.parent.name)
                rows.append({**{k: session.get(k) for k in ("id", "game", "created_at", "duration_s", "state")},
                             "playable": self._playable(session)})
            except (ValueError, KeyError, TypeError, OSError):
                continue
        return sorted(rows, key=lambda r: r.get("created_at") or "", reverse=True)

    def prepare(self, identity):
        session = self._session(identity)
        if not session["segments"]:
            raise ValueError("No saved footage yet. Play for a few seconds and refresh.")
        if not self._playable(session):
            raise ValueError("Part of this recording is missing.")
        signature = hashlib.sha256(json.dumps(session, sort_keys=True).encode()).hexdigest()
        with self._lock:
            if self._closed:
                raise RuntimeError("The recording viewer is closing.")
            prior = self._jobs.get(identity)
            if prior and prior["signature"] == signature and prior["state"] in ("preparing", "ready"):
                return self._public(prior)
            if self._prepare_worker and self._prepare_worker.is_alive():
                raise RuntimeError("Another recording is being prepared. Try again in a moment.")
            job = {"state": "preparing", "signature": signature, "version": uuid.uuid4().hex,
                   "session_id": identity, "duration_s": session["duration_s"]}
            self._jobs[identity] = job
            self._prepare_worker = threading.Thread(target=self._prepare, args=(session, job), daemon=True)
            self._prepare_worker.start()
            return self._public(job)

    def _prepare(self, session, job):
        try:
            snapshot = self.native.snapshot(session, 0, session["duration_s"])
            with self._lock:
                job.update(state="ready", path=snapshot.path, duration_s=snapshot.duration,
                           url=f"/api/recordings/{session['id']}/media/{job['version']}")
        except Exception as exc:
            with self._lock:
                job.update(state="error", error=str(exc) or "Could not prepare this recording.")

    @staticmethod
    def _public(job):
        return {k: v for k, v in job.items() if k not in ("path", "signature")}

    def playback(self, identity):
        with self._lock:
            return self._public(self._jobs.get(identity, {"state": "idle"}))

    def media(self, identity, version):
        with self._lock:
            job = self._jobs.get(identity)
            if not job or job["state"] != "ready" or version != job["version"]:
                raise ValueError("This playback is not available. Open the recording again.")
            path = job["path"]
        if not path.is_file():
            raise ValueError("This playback file is no longer available.")
        return path

    def search(self, identity, question=None, audio=None):
        self._session(identity)
        if audio is None and (not isinstance(question, str) or not question.strip() or len(question) > 800):
            raise ValueError("Enter a question of 1–800 characters.")
        if audio is not None and not 1 <= len(audio) <= 2 * 1024 * 1024:
            raise ValueError("Record a voice question of up to 15 seconds.")
        if self.searcher is None:
            raise RuntimeError("The player service is not ready for search.")
        with self._lock:
            if self._closed:
                raise RuntimeError("The recording viewer is closing.")
            if self._search_worker and self._search_worker.is_alive():
                raise RuntimeError("A recording search is still running. Please wait.")
            # Only the latest completed search is retained; never serve arbitrary exports.
            self._searches.clear()
            job = {"id": uuid.uuid4().hex, "session_id": identity,
                   "state": "transcribing" if audio else "searching", "question": question or ""}
            self._searches[job["id"]] = job
            self._search_worker = threading.Thread(target=self._search, args=(job, audio), daemon=True)
            self._search_worker.start()
            return self._public(job)

    def _transcribe(self, audio):
        from ..demo.capture import ffmpeg_binary, NO_WINDOW
        if self.transcriber is None:
            raise RuntimeError("Local speech recognition is not ready.")
        with tempfile.TemporaryDirectory(prefix="deja-question-") as directory:
            source, target = Path(directory) / "question.webm", Path(directory) / "question.wav"
            source.write_bytes(audio)
            result = subprocess.run([ffmpeg_binary(), "-hide_banner", "-loglevel", "error", "-y",
                "-i", str(source), "-t", "15", "-vn", "-ar", "16000", "-ac", "1", str(target)],
                capture_output=True, timeout=25, creationflags=NO_WINDOW)
            if result.returncode:
                raise ValueError("Could not read the microphone recording. Please try again.")
            return self.transcriber(target)

    def _search(self, job, audio):
        try:
            question = self._transcribe(audio) if audio else job["question"].strip()
            with self._lock:
                job.update(state="searching", question=question)
            response = self.searcher(job["session_id"], question)
            result = response["result"]
            snapshot = response.get("snapshot")
            with self._lock:
                job.update(state="ready", result=result)
                if snapshot is not None:
                    job.update(path=snapshot.path,
                               url=f"/api/recordings/{job['session_id']}/search-media/{job['id']}",
                               duration_s=snapshot.duration)
        except Exception as exc:
            with self._lock:
                job.update(state="error", error=str(exc) or "The recording search failed.")

    def search_status(self, identity, job_id):
        with self._lock:
            job = self._searches.get(job_id)
            if not job or job["session_id"] != identity:
                raise ValueError("That search is no longer available.")
            return self._public(job)

    def search_media(self, identity, job_id):
        with self._lock:
            job = self._searches.get(job_id)
            if not job or job["session_id"] != identity or job["state"] != "ready" or "path" not in job:
                raise ValueError("No playable search result.")
            return job["path"]

    def close(self):
        with self._lock:
            self._closed = True
        for worker in (self._prepare_worker, self._search_worker):
            if worker:
                worker.join(timeout=130)
