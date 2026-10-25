"""python -m smartreplay.frontend [--file PATH] [--port N] [--no-browser] [--window]

Same as `python -m smartreplay.cli settings`, without importing the discovery
pipeline (and so without needing OpenCV) just to edit player settings.
`--window` opens it as the deja desktop app instead of a browser tab.
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
    ap.add_argument("--footage", default=None,
                    help="clip or folder of clips for the preview (default: out/demo/matches; "
                         "SMARTREPLAY_FOOTAGE also sets it)")
    ap.add_argument("--window", action="store_true",
                    help="open as the deja desktop app, in its own window (needs pywebview)")
    args = ap.parse_args(argv)
    if args.window:
        from .window import run
        return run(args.file, args.footage)
    return serve(args.file, args.port, open_browser=not args.no_browser, footage=args.footage)


if __name__ == "__main__":
    raise SystemExit(main())
