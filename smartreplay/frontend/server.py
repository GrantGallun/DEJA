"""Local web server for the replay player settings app.

    python -m smartreplay.cli settings              # opens the browser
    python -m smartreplay.cli settings --no-browser --port 8765

The page is static (static/); everything it changes goes through a small JSON
API, and player_settings does all validation, so the browser never decides
what is valid:

    GET  /api/settings   current settings, defaults, themes, issues, file path
    POST /api/validate   issues for a draft, without saving
    PUT  /api/settings   validate and save; 422 with issues if anything is wrong
    GET  /api/status     what the running player last reported
    GET  /api/footage    a recorded clip for the preview to play (byte ranges)

The footage is the newest .mp4 under --footage (default: the demo's finished
replays in out/demo/matches), so the preview shows the user's own game rather
than a drawing. The server picks the file; the page never names a path.

The server listens on loopback only. Because it writes a file, it also refuses
requests whose Host or Origin is not itself, so another web page cannot change
someone's hotkeys by posting to localhost (or by DNS rebinding onto it).
"""

from __future__ import annotations

import json
import os
import re
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .. import player_settings as ps
from .. import __version__
from ..providers import deepseek_key

STATIC = Path(__file__).with_name("static")
MAX_BODY = 256 * 1024
# Where `python -m smartreplay.demo` keeps finished replays by default.
DEMO_MATCHES = Path(__file__).resolve().parents[2] / "out" / "demo" / "matches"
CHUNK = 256 * 1024
# Not mimetypes: on Windows it reads the registry, which often maps .js to
# text/plain, and nosniff would then refuse to run the script.
TYPES = {".html": "text/html; charset=utf-8", ".css": "text/css; charset=utf-8",
         ".js": "text/javascript; charset=utf-8", ".svg": "image/svg+xml"}


class SettingsServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, settings_file: Path, port: int = 0, footage: Path | None = None,
                 recordings_root: Path | None = None):
        self.settings_file = Path(settings_file)
        self.footage = Path(footage) if footage else None
        self.runtime = None
        self.recordings_root = Path(recordings_root or DEMO_MATCHES.parent)
        self._recordings = None
        self._recordings_lock = threading.Lock()
        # One save at a time: two tabs saving together must not interleave
        # their read-modify-write of the revision counter.
        self.save_lock = threading.Lock()
        super().__init__(("127.0.0.1", port), Handler)

    @property
    def origin(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}"

    def start_player(self):
        from .runtime import PlayerRuntime
        if self.runtime is None:
            self.runtime = PlayerRuntime(DEMO_MATCHES.parent, self.settings_file)
            self.runtime.start()

    def player_status(self):
        status = ps.player_status(self.settings_file)
        if self.runtime is not None:
            status["runtime"] = self.runtime.snapshot()
        return status

    @property
    def recordings(self):
        from .recordings import RecordingLibrary
        with self._recordings_lock:
            if self._recordings is None:
                self._recordings = RecordingLibrary(self.recordings_root,
                    searcher=self.search_recording, transcriber=self.transcribe_question)
            return self._recordings

    def search_recording(self, identity, question):
        if self.runtime is None:
            raise RuntimeError("Start deja's player service before searching recordings.")
        return self.runtime.search_recording(identity, question)

    def transcribe_question(self, path):
        if self.runtime is None:
            raise RuntimeError("Start deja's player service before using voice search.")
        return self.runtime.transcribe_question(path)

    def server_close(self):
        try:
            if self.runtime is not None:
                self.runtime.close()
            if self._recordings is not None:
                self._recordings.close()
        finally:
            super().server_close()

    def footage_file(self) -> Path | None:
        """The clip the preview plays: the footage file, or the newest .mp4 under it."""
        root = self.footage
        if root is None:
            return None
        if root.is_file():
            return root if root.suffix.lower() == ".mp4" else None
        try:
            clips = [p for p in root.rglob("*.mp4") if p.is_file()]
        except OSError:
            return None
        return max(clips, key=lambda p: p.stat().st_mtime, default=None)


class Handler(BaseHTTPRequestHandler):
    server: SettingsServer

    def log_message(self, fmt, *args):  # keep the console for the URL
        pass

    # ------------------------------------------------------------ plumbing

    def _allowed(self) -> bool:
        port = self.server.server_address[1]
        hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
        if self.headers.get("Host") not in hosts:
            return False
        origin = self.headers.get("Origin")
        return origin is None or origin in {f"http://{h}" for h in hosts}

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy",
                         "default-src 'self'; style-src 'self' 'unsafe-inline'; frame-ancestors 'none'")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, payload) -> None:
        self._send(status, json.dumps(payload).encode("utf-8"), "application/json; charset=utf-8")

    def _body(self):
        if "application/json" not in (self.headers.get("Content-Type") or ""):
            raise ValueError("Expected application/json.")
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY:
            raise ValueError("Request too large.")
        return json.loads(self.rfile.read(length) or b"null")

    def _send_file(self, path: Path, content_type: str) -> None:
        """Stream a file, honouring one byte range so video can seek and loop."""
        size = path.stat().st_size
        start, end = 0, size - 1
        m = re.fullmatch(r"bytes=(\d*)-(\d*)", self.headers.get("Range") or "")
        ranged = bool(m and (m.group(1) or m.group(2)))
        if ranged:
            if m.group(1):
                start = int(m.group(1))
                end = min(int(m.group(2)), size - 1) if m.group(2) else size - 1
            else:  # suffix range: the last N bytes
                start = max(0, size - int(m.group(2)))
            if start > end:
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
        self.send_response(HTTPStatus.PARTIAL_CONTENT if ranged else HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Accept-Ranges", "bytes")
        if ranged:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Cache-Control", "private, max-age=600")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        try:
            with path.open("rb") as fh:
                fh.seek(start)
                remaining = end - start + 1
                while remaining > 0:
                    chunk = fh.read(min(CHUNK, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass  # the video element moved on; nothing to report

    def _snapshot(self) -> dict:
        settings, issues = ps.load(self.server.settings_file)
        clip = self.server.footage_file()
        # The version keeps a newer recording from being served from cache.
        footage = f"/api/footage?v={clip.stat().st_mtime_ns}" if clip else None
        return {"version": __version__, "settings": settings, "issues": issues, "defaults": ps.defaults(),
                "themes": ps.THEMES, "path": str(self.server.settings_file),
                "exists": self.server.settings_file.exists(),
                "status": self.server.player_status(),
                "footage": footage,
                # Whether a key exists, never the key: the page only says if DeepSeek can sign in.
                "providers": {"deepseek": {"key": deepseek_key() is not None}}}

    # ------------------------------------------------------------ routes

    def do_GET(self):
        if not self._allowed():
            return self._json(HTTPStatus.FORBIDDEN, {"error": "Forbidden host or origin."})
        route = self.path.split("?", 1)[0]
        if route == "/api/recordings":
            return self._json(HTTPStatus.OK, {"recordings": self.server.recordings.list()})
        recording = re.fullmatch(r"/api/recordings/([A-Za-z0-9_-]{1,100})/(playback|media|search|search-media)(?:/([a-f0-9]{32}))?", route)
        if recording:
            identity, action, token = recording.groups()
            try:
                library = self.server.recordings
                if action == "playback" and token is None:
                    return self._json(HTTPStatus.OK, library.playback(identity))
                if action == "search" and token:
                    return self._json(HTTPStatus.OK, library.search_status(identity, token))
                if action in ("media", "search-media") and token:
                    path = library.media(identity, token) if action == "media" else library.search_media(identity, token)
                    return self._send_file(path, "video/mp4")
            except (ValueError, OSError) as exc:
                return self._json(HTTPStatus.NOT_FOUND, {"error": str(exc)})
            return self._json(HTTPStatus.NOT_FOUND, {"error": "Not found."})
        if route == "/api/settings":
            return self._json(HTTPStatus.OK, self._snapshot())
        if route == "/api/status":
            return self._json(HTTPStatus.OK, self.server.player_status())
        if route == "/api/footage":
            clip = self.server.footage_file()
            if clip is None:
                return self._json(HTTPStatus.NOT_FOUND, {"error": "No recorded footage yet."})
            return self._send_file(clip, "video/mp4")
        name = "index.html" if route == "/" else route.lstrip("/")
        target = (STATIC / name).resolve()
        if STATIC.resolve() not in target.parents or not target.is_file():
            return self._json(HTTPStatus.NOT_FOUND, {"error": "Not found."})
        kind = TYPES.get(target.suffix, "application/octet-stream")
        return self._send(HTTPStatus.OK, target.read_bytes(), kind)

    def do_POST(self):
        if not self._allowed():
            return self._json(HTTPStatus.FORBIDDEN, {"error": "Forbidden host or origin."})
        recording = re.fullmatch(r"/api/recordings/([A-Za-z0-9_-]{1,100})/(prepare|search|voice)", self.path)
        if recording:
            identity, action = recording.groups()
            try:
                library = self.server.recordings
                if action == "prepare":
                    result = library.prepare(identity)
                elif action == "search":
                    body = self._body()
                    if not isinstance(body, dict):
                        raise ValueError("Enter a search question.")
                    result = library.search(identity, body.get("question"))
                else:
                    length = int(self.headers.get("Content-Length") or 0)
                    if not self.headers.get("Content-Type", "").startswith("audio/") or not 1 <= length <= 2 * 1024 * 1024:
                        raise ValueError("Record a voice question of up to 15 seconds.")
                    result = library.search(identity, audio=self.rfile.read(length))
                return self._json(HTTPStatus.ACCEPTED, result)
            except (ValueError, OSError) as exc:
                return self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            except RuntimeError as exc:
                return self._json(HTTPStatus.CONFLICT, {"error": str(exc)})
        if self.path != "/api/validate":
            return self._json(HTTPStatus.NOT_FOUND, {"error": "Not found."})
        try:
            draft = self._body()
        except ValueError as exc:
            return self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
        settings, issues = ps.normalize(draft)
        return self._json(HTTPStatus.OK, {"settings": settings, "issues": issues})

    def do_PUT(self):
        if not self._allowed():
            return self._json(HTTPStatus.FORBIDDEN, {"error": "Forbidden host or origin."})
        if self.path != "/api/settings":
            return self._json(HTTPStatus.NOT_FOUND, {"error": "Not found."})
        try:
            draft = self._body()
        except ValueError as exc:
            return self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
        with self.server.save_lock:
            try:
                ps.save(draft, self.server.settings_file)
            except ps.SettingsError as exc:
                return self._json(HTTPStatus.UNPROCESSABLE_ENTITY, {"issues": exc.issues})
            except OSError as exc:
                return self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": f"Could not save: {exc}"})
        return self._json(HTTPStatus.OK, self._snapshot())


def make_server(settings_file: Path | None = None, port: int = 8765,
                footage: Path | None = None) -> SettingsServer:
    settings_file = Path(settings_file or ps.default_path())
    footage = Path(footage or os.environ.get("SMARTREPLAY_FOOTAGE") or DEMO_MATCHES)
    try:
        return SettingsServer(settings_file, port, footage)
    except OSError:
        return SettingsServer(settings_file, 0, footage)  # preferred port taken


def serve(settings_file: Path | None = None, port: int = 8765, open_browser: bool = True,
          footage: Path | None = None) -> int:
    server = make_server(settings_file, port, footage)
    server.start_player()
    url = server.origin + "/"
    print(f"Smart Replay settings: {url}")
    print(f"  file: {server.settings_file}")
    print(f"  preview footage: {server.footage_file() or f'none yet (looked in {server.footage})'}")
    print("  Ctrl+C to stop")
    if open_browser:
        threading.Timer(0.3, webbrowser.open, (url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0
