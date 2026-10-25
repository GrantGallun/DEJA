"""deja as a desktop app: the settings page in a window of its own.

    python -m smartreplay.frontend --window        # pythonw for no console

The settings server runs inside this process on a free loopback port and the
page shows in a WebView2 window (pywebview), so closing the window ends both.
The process claims its own AppUserModelID; that is what gives the window a
taskbar button with deja's icon instead of grouping it under Python.
Install-Deja.ps1 makes a Start menu shortcut with the same ID, so a pinned
deja keeps its icon and relaunches this.
"""

from __future__ import annotations

import ctypes
import os
import threading
from pathlib import Path

from .server import STATIC, make_server

APP_ID = "SmartReplay.Deja"
ICON = STATIC / "deja.ico"


def run(settings_file: Path | None = None, footage: Path | None = None) -> int:
    try:
        import webview
    except ImportError:
        print("deja's window needs pywebview:  python -m pip install pywebview")
        return 2
    if os.name == "nt":
        # Before any window exists, or the taskbar files it under python.exe.
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_ID)
    server = make_server(settings_file, 0, footage)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    webview.create_window("deja", server.origin + "/", width=1360, height=900, min_size=(880, 600),
                          background_color="#0e0c14")
    try:
        webview.start(icon=str(ICON))
    finally:
        server.shutdown()
        server.server_close()
    return 0
