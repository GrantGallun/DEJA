"""python -m smartreplay.frontend [--file PATH] [--port N] [--no-browser]

Same as `python -m smartreplay.cli settings`, without importing the discovery
pipeline (and so without needing OpenCV) just to edit player settings.
"""

from __future__ import annotations

import argparse

from .server import serve


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="smartreplay.frontend", description="Replay player settings app")
    ap.add_argument("--file", default=None,
                    help="settings file (default: %%APPDATA%%/SmartReplay/player-settings.json)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true", help="print the URL instead of opening it")
    args = ap.parse_args(argv)
    return serve(args.file, args.port, open_browser=not args.no_browser)


if __name__ == "__main__":
    raise SystemExit(main())
