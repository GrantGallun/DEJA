"""One desktop window per recording directory; repeat launches focus its owner."""
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import time

from .runtime import ServiceLease, ServiceAlreadyRunning


def focus_window(pid):
    if os.name != "nt" or not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        return False
    user = ctypes.WinDLL("user32", use_last_error=True)
    callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    user.EnumWindows.argtypes = [callback_type, wintypes.LPARAM]
    user.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
    user.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    user.ShowWindowAsync.argtypes = [wintypes.HWND, ctypes.c_int]
    user.SetForegroundWindow.argtypes = [wintypes.HWND]
    found = []
    def visit(window, _):
        owner = wintypes.DWORD()
        user.GetWindowThreadProcessId(window, ctypes.byref(owner))
        if owner.value == pid:
            title = ctypes.create_unicode_buffer(256)
            user.GetWindowTextW(window, title, len(title))
            if title.value == "deja":
                found.append(window)
                return False
        return True
    user.EnumWindows(callback_type(visit), 0)
    if not found:
        return False
    user.ShowWindowAsync(found[0], 9)  # Restore a minimized window.
    user.SetForegroundWindow(found[0])
    return True


class DesktopInstance:
    def __init__(self, root, *, focus=focus_window, wait_seconds=5):
        self.root = Path(root)
        self.lease = ServiceLease(self.root, "desktop.lock")
        self.focus, self.wait_seconds = focus, wait_seconds
        self.primary = False

    def __enter__(self):
        deadline = time.monotonic() + self.wait_seconds
        while True:
            try:
                self.lease.__enter__()
            except ServiceAlreadyRunning:
                try:
                    owner = json.loads((self.root / "desktop-owner.json").read_text())
                    if self.focus(owner["pid"]):
                        return self
                except (OSError, ValueError, KeyError, TypeError):
                    pass
                if time.monotonic() >= deadline:
                    return self  # Owner is still starting; never create a duplicate.
                time.sleep(.05)
            else:
                try:
                    pending = self.root / "desktop-owner.tmp"
                    pending.write_text(json.dumps({"pid": os.getpid()}))
                    pending.replace(self.root / "desktop-owner.json")
                    self.primary = True
                    return self
                except Exception:
                    self.lease.__exit__()
                    raise

    def __exit__(self, *args):
        if self.primary:
            self.lease.__exit__(*args)
