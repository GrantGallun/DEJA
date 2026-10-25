"""Run with python -m smartreplay.demo, then open the printed local URL."""
import argparse
from pathlib import Path
import webbrowser

from .. import player_settings
from .server import DemoServer, DemoSession


def main():
    parser = argparse.ArgumentParser(description="Native full sessions -> concurrent questions -> matching replay")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--data-dir", type=Path, default=Path("out/demo"))
    parser.add_argument("--video", type=Path, help="Import a short recording, clearly labelled as imported")
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--skip-model-check", action="store_true")
    args = parser.parse_args()
    session = DemoSession(args.data_dir, settings_file=player_settings.default_path())
    if args.video:
        session.import_video(args.video)
    server = DemoServer(session, args.port)
    if not args.skip_model_check:
        session.check_model()
    print(f"Stick Fight demo: {server.origin}", flush=True)
    print(f"Full sessions are preserved in {session.native.session_root}", flush=True)
    print(f"Replay snapshots are preserved in {session.store.root}", flush=True)
    if not args.no_browser:
        webbrowser.open(server.origin)
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        session.close()


if __name__ == "__main__":
    main()
