"""Bounded, window-only recording and immutable match storage."""
from __future__ import annotations

import ctypes
from ctypes import wintypes
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import uuid

import cv2


NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def video_duration(path: Path) -> float:
    cap = cv2.VideoCapture(str(path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps if fps > 0 else 0
        ok, _ = cap.read()
        if not ok or not math.isfinite(duration) or duration <= 0:
            raise ValueError("The video is empty or cannot be decoded.")
        return duration
    finally:
        cap.release()


@dataclass(frozen=True)
class Match:
    id: str
    path: Path
    duration: float
    created_at: str
    source: str

    def public(self):
        return {"id": self.id, "duration": self.duration, "created_at": self.created_at,
                "source": self.source, "url": f"/media/{self.id}.mp4"}


class MatchStore:
    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def allocate(self) -> Path:
        folder = self.root / uuid.uuid4().hex
        folder.mkdir()
        return folder / "match.mp4"

    def finish(self, path: Path, source="live") -> Match:
        path = Path(path).resolve()
        if path.parent.parent != self.root or path.name != "match.mp4":
            raise ValueError("Recording is outside the match store.")
        match = Match(path.parent.name, path, video_duration(path),
                      datetime.now(timezone.utc).isoformat(), source)
        data = asdict(match)
        data.pop("path")
        temporary = path.with_name("match.json.tmp")
        temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
        temporary.replace(path.with_suffix(".json"))
        return match

    def import_video(self, source: Path) -> Match:
        source = Path(source).resolve()
        if video_duration(source) > 35:
            raise ValueError("Use a short video of at most 35 seconds.")
        path = self.allocate()
        # OpenCV can read containers/codecs that Chromium cannot play. Normalize
        # imports to the same known browser format as actual game capture.
        command = [ffmpeg_binary(), "-hide_banner", "-loglevel", "error", "-y", "-i", str(source),
                   "-an", "-vf", "scale=trunc(min(1280\\,iw)/2)*2:-2", "-c:v", "libx264",
                   "-preset", "ultrafast", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)]
        try:
            result = subprocess.run(command, capture_output=True, timeout=90, creationflags=NO_WINDOW)
        except subprocess.TimeoutExpired as exc:
            raise ValueError("The video could not be prepared in time.") from exc
        if result.returncode:
            raise ValueError("The video could not be converted to a playable replay.")
        return self.finish(path, "imported")

    def get(self, match_id: str) -> Match:
        if not re.fullmatch(r"[0-9a-f]{32}", match_id):
            raise ValueError("Invalid match ID.")
        path = self.root / match_id / "match.mp4"
        data = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        return Match(match_id, path, float(data["duration"]), data["created_at"], data["source"])


def ffmpeg_binary() -> str:
    binary = os.environ.get("SMARTREPLAY_FFMPEG") or shutil.which("ffmpeg")
    if binary:
        return binary
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError as exc:
        raise RuntimeError("Install the demo dependencies: python -m pip install -r requirements-demo.txt") from exc


def capture_command(binary: str, hwnd: int, output: Path, seconds: float) -> list[str]:
    if isinstance(seconds, bool) or not math.isfinite(seconds) or not 1 <= seconds <= 30:
        raise ValueError("Record between 1 and 30 seconds.")
    if not hwnd:
        raise ValueError("A game window is required.")
    return [binary, "-hide_banner", "-loglevel", "warning", "-y", "-f", "gdigrab",
            "-framerate", "30", "-draw_mouse", "1", "-i", f"hwnd={hwnd}",
            "-t", f"{seconds:g}", "-an", "-vf", "scale=trunc(min(1280\\,iw)/2)*2:-2",
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "20", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(output)]


def game_window() -> dict | None:
    """Only accept a visible window belonging to the actual StickFight.exe."""
    if os.name != "nt":
        return None
    user = ctypes.WinDLL("user32", use_last_error=True)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    user.IsWindowVisible.argtypes = [wintypes.HWND]
    user.IsIconic.argtypes = [wintypes.HWND]
    user.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
    user.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.QueryFullProcessImageNameW.argtypes = [wintypes.HANDLE, wintypes.DWORD,
                                                 wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD)]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    found = []

    @callback_type
    def visit(hwnd, _):
        if not user.IsWindowVisible(hwnd):
            return True
        pid = wintypes.DWORD()
        user.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        handle = kernel.OpenProcess(0x1000, False, pid.value)
        if not handle:
            return True
        try:
            size = wintypes.DWORD(32768)
            image = ctypes.create_unicode_buffer(size.value)
            if not kernel.QueryFullProcessImageNameW(handle, 0, image, ctypes.byref(size)):
                return True
            if Path(image.value).name.lower() != "stickfight.exe":
                return True
            title = ctypes.create_unicode_buffer(1024)
            user.GetWindowTextW(hwnd, title, len(title))
            if title.value:
                found.append({"hwnd": int(hwnd), "title": title.value,
                              "minimized": bool(user.IsIconic(hwnd))})
        finally:
            kernel.CloseHandle(handle)
        return True

    user.EnumWindows.argtypes = [callback_type, wintypes.LPARAM]
    user.EnumWindows(visit, 0)
    return found[0] if found else None


class WindowRecorder:
    def __init__(self, binary: str | None = None):
        self.binary = binary or ffmpeg_binary()
        self.process = None
        self.log = None

    def start(self, window: dict, output: Path, seconds=25):
        if self.process is not None:
            raise RuntimeError("A recording is already running.")
        if os.name == "nt":
            user = ctypes.WinDLL("user32", use_last_error=True)
            user.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
            user.SetForegroundWindow.argtypes = [wintypes.HWND]
            user.ShowWindow(window["hwnd"], 9)
            user.SetForegroundWindow(window["hwnd"])
        self.log = output.with_suffix(".capture.log").open("wb")
        try:
            self.process = subprocess.Popen(capture_command(self.binary, window["hwnd"], output, seconds),
                stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self.log, creationflags=NO_WINDOW)
        except Exception:
            self.log.close()
            raise

    def stop(self):
        proc = self.process
        if proc is not None and proc.poll() is None:
            try:
                proc.stdin.write(b"q\n")
                proc.stdin.flush()
            except (BrokenPipeError, OSError):
                pass

    def wait(self, timeout=40):
        try:
            code = self.process.wait(timeout=timeout)
            if code != 0:
                raise RuntimeError("Game capture failed. Keep Stick Fight visible in windowed/borderless mode; see the capture log.")
        except subprocess.TimeoutExpired as exc:
            self.process.kill()
            self.process.wait()
            raise RuntimeError("Recording timed out. Please try again.") from exc
        finally:
            if self.process and self.process.stdin:
                self.process.stdin.close()
            if self.log:
                self.log.close()
