"""A retained Windows process handle prevents PID reuse during orphan recovery."""
import ctypes
from ctypes import wintypes
from datetime import datetime
import os
from pathlib import Path
import subprocess


class AttachedProcess:
    def __init__(self, kernel, handle, pid):
        self.kernel, self.handle, self.pid = kernel, handle, pid
        self.returncode = None

    def poll(self):
        if not self.handle:
            return self.returncode
        result = self.kernel.WaitForSingleObject(self.handle, 0)
        if result == 258:
            return None
        if result != 0:
            raise ctypes.WinError(ctypes.get_last_error())
        code = wintypes.DWORD()
        if not self.kernel.GetExitCodeProcess(self.handle, ctypes.byref(code)):
            raise ctypes.WinError(ctypes.get_last_error())
        self.returncode = code.value
        return self.returncode

    def terminate(self):
        if self.poll() is None and not self.kernel.TerminateProcess(self.handle, 1):
            raise ctypes.WinError(ctypes.get_last_error())

    kill = terminate

    def wait(self, timeout=None):
        milliseconds = 0xFFFFFFFF if timeout is None else max(0, int(timeout * 1000))
        if self.kernel.WaitForSingleObject(self.handle, milliseconds) == 258:
            raise subprocess.TimeoutExpired("recovered native recorder", timeout)
        return self.poll()

    def close_handle(self):
        if self.handle:
            self.kernel.CloseHandle(self.handle)
            self.handle = None

    def __del__(self):
        self.close_handle()


def attach_native_process(pid, binary, updated_at):
    if os.name != "nt":
        raise RuntimeError("Native recorder recovery requires Windows.")
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        raise RuntimeError("Cannot verify the old recorder's process identity.")
    updated = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
    if updated.tzinfo is None:
        raise RuntimeError("Cannot verify the old recorder's timestamp.")
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.QueryFullProcessImageNameW.argtypes = [wintypes.HANDLE, wintypes.DWORD,
        wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD)]
    kernel.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
    kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    kernel.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.OpenProcess(0x1000 | 0x100000 | 0x0001, False, pid)
    if not handle:
        if ctypes.get_last_error() == 87:  # Process already exited.
            return None
        raise RuntimeError("Cannot verify the old recorder's process permissions.")
    process = AttachedProcess(kernel, handle, pid)
    try:
        if process.poll() is not None:
            process.close_handle()
            return None
        name, size = ctypes.create_unicode_buffer(32768), wintypes.DWORD(32768)
        if not kernel.QueryFullProcessImageNameW(handle, 0, name, ctypes.byref(size)):
            raise RuntimeError("Cannot verify the old recorder's executable.")
        created, exited, cpu, user = (wintypes.FILETIME() for _ in range(4))
        if not kernel.GetProcessTimes(handle, ctypes.byref(created), ctypes.byref(exited),
                                      ctypes.byref(cpu), ctypes.byref(user)):
            raise RuntimeError("Cannot verify the old recorder's start time.")
        started = ((created.dwHighDateTime << 32) | created.dwLowDateTime) / 10_000_000 - 11644473600
        if (Path(name.value).resolve() != Path(binary).resolve()
                or started > updated.timestamp() + .002):
            raise RuntimeError("Cannot verify the old recorder: executable changed or process ID was reused.")
        return process
    except Exception:
        process.close_handle()
        raise
