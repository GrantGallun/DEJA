"""Build a ground-truth file for Minecraft footage from the game's own log.

Minecraft timestamps every death in logs/latest.log, which makes it the best
realistic test substrate available without hand labelling: record some play,
die a few times, and the game tells us exactly when. The death screen itself
is a textbook event overlay - full screen, fixed text, holds until the player
clicks respawn.

    python tools/minecraft_truth.py footage/mc.mp4
    python tools/minecraft_truth.py footage/mc.mp4 --start "2026-10-24 14:33:12"
    python tools/minecraft_truth.py footage/mc.mp4 --list

The only fiddly part is aligning the log's wall clock to the video's timeline.
In order of preference: an explicit --start, a timestamp in the file name (as
OBS and ShadowPlay write), or the file's modification time minus its duration.
Check the alignment with --list before trusting the result.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime, timedelta
from pathlib import Path

import cv2

LOG_LINE = re.compile(r"^\[(\d{2}):(\d{2}):(\d{2})\]\s+\[(?P<thread>[^\]]*)\]:\s*(?P<body>.*)$")

# Minecraft's death messages, as they appear in chat.
DEATH_PATTERNS = [
    "was slain by", "was shot by", "was killed by", "was blown up by",
    "was fireballed by", "was pummeled by", "was impaled by", "was squashed by",
    "was skewered by", "was pricked to death", "was struck by lightning",
    "was roasted in dragon breath", "was stung to death", "was obliterated by",
    "was squished too much", "was doomed to fall", "was shot off", "was blown up",
    "fell from a high place", "fell out of the world", "fell off",
    "hit the ground too hard", "drowned", "burned to death", "went up in flames",
    "tried to swim in lava", "discovered the floor was lava", "walked into fire",
    "walked into a cactus", "suffocated in a wall", "starved to death",
    "withered away", "blew up", "experienced kinetic energy",
    "died from dehydration", "froze to death", "was frozen to death",
    "didn't want to live in the same world as", "left the confines of this world",
    "died",
]
ADVANCEMENT = "has made the advancement"
CHALLENGE = "has completed the challenge"

FILENAME_STAMPS = [
    # OBS default, ShadowPlay, and common manual namings.
    (re.compile(r"(\d{4})-(\d{2})-(\d{2})[ _T](\d{2})[-:.](\d{2})[-:.](\d{2})"), True),
    (re.compile(r"(\d{4})(\d{2})(\d{2})[ _T](\d{2})(\d{2})(\d{2})"), True),
]


def video_duration(path: Path) -> float:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    cap.release()
    return float(n) / float(fps) if n else 0.0


def guess_start(path: Path, duration: float) -> tuple[datetime, str]:
    for rx, _ in FILENAME_STAMPS:
        m = rx.search(path.name)
        if m:
            y, mo, d, hh, mm, ss = (int(g) for g in m.groups())
            try:
                return datetime(y, mo, d, hh, mm, ss), "filename timestamp"
            except ValueError:
                pass
    mtime = datetime.fromtimestamp(path.stat().st_mtime)
    return mtime - timedelta(seconds=duration), "file mtime minus duration"


def default_log() -> Path:
    appdata = os.environ.get("APPDATA", "")
    return Path(appdata) / ".minecraft" / "logs" / "latest.log"


SET_USER = re.compile(r"Setting user:\s*(?P<name>\S+)")


def detect_player(lines: list[str]) -> str | None:
    """The local player's name, which the launcher logs at startup.

    This matters: on a server the log records everyone's deaths, but the death
    screen only appears for your own, so counting other players would inflate
    the truth with events the footage never shows.
    """
    for line in lines:
        m = SET_USER.search(line)
        if m:
            return m.group("name")
    return None


def _starts_with_player(chat: str, player: str) -> bool:
    """True when the line is about exactly this player, not one whose name
    merely begins the same way ("Alex" must not match "Alexander")."""
    return re.match(rf"{re.escape(player)}(?![A-Za-z0-9_])", chat) is not None


def parse_log(log_path: Path, start: datetime, duration: float, player: str | None = None):
    """Return (deaths, advancements, player) as offsets in seconds from `start`."""
    deaths: list[tuple[float, str]] = []
    advancements: list[tuple[float, str]] = []
    text = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    player = player or detect_player(text)

    # latest.log carries no date. Rather than guessing the day the log began
    # on and rolling it forward at midnight, place each line on whichever of
    # the days around the recording puts it inside the video window. A clip
    # is far shorter than a day, so at most one choice ever fits - and a
    # recording that starts just after midnight still picks up lines the log
    # stamped 23:59.
    days = [start.date() + timedelta(days=k) for k in (-1, 0, 1)]
    for line in text:
        m = LOG_LINE.match(line)
        if not m:
            continue
        hh, mm, ss = int(m.group(1)), int(m.group(2)), int(m.group(3))
        clock = timedelta(hours=hh, minutes=mm, seconds=ss)
        offset = None
        for day in days:
            o = (datetime.combine(day, datetime.min.time()) + clock - start).total_seconds()
            if -1.0 <= o <= duration + 1.0:
                offset = o
                break
        if offset is None:
            continue

        body = m.group("body")
        if "[CHAT]" not in body:
            continue
        chat = body.split("[CHAT]", 1)[1].strip()
        low = chat.lower()
        # Player chat is "<Name> text"; only system lines report deaths, so a
        # player typing "I died" must not count.
        if chat.startswith("<"):
            continue
        if player and not _starts_with_player(chat, player):
            continue
        if any(pat in low for pat in DEATH_PATTERNS):
            deaths.append((offset, chat))
        elif ADVANCEMENT in low or CHALLENGE in low:
            advancements.append((offset, chat))
    return deaths, advancements, player


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--log", default=None, help="defaults to %%APPDATA%%/.minecraft/logs/latest.log")
    ap.add_argument("--start", default=None, help='video start wall clock, "YYYY-MM-DD HH:MM:SS"')
    ap.add_argument("--death-duration", type=float, default=6.0,
                    help="nominal time the death screen stays up")
    ap.add_argument("--player", default=None,
                    help="local player name; auto-detected from the log by default")
    ap.add_argument("--list", action="store_true", help="print matched lines and exit")
    args = ap.parse_args(argv)

    video = Path(args.video)
    log_path = Path(args.log) if args.log else default_log()
    if not log_path.exists():
        print(f"log not found: {log_path}")
        return 1

    duration = video_duration(video)
    if args.start:
        start, how = datetime.fromisoformat(args.start), "--start"
    else:
        start, how = guess_start(video, duration)

    deaths, advancements, player = parse_log(log_path, start, duration, args.player)
    print(f"video    {video}  ({duration:.0f}s)")
    print(f"log      {log_path}")
    print(f"start    {start:%Y-%m-%d %H:%M:%S}  (from {how})")
    print(f"player   {player if player else 'not detected - counting every chat line'}")
    print(f"matched  {len(deaths)} death(s), {len(advancements)} advancement(s)\n")

    for off, text in deaths:
        print(f"  death  {off:8.1f}s  {text}")
    for off, text in advancements[:10]:
        print(f"  advn   {off:8.1f}s  {text}")

    if args.list:
        return 0
    if not deaths:
        print("\nNo deaths matched inside the video window. Either the alignment is off "
              "(try --start, or --list to inspect) or the clip contains no deaths.")
        return 1

    events = {
        "death": {
            "onsets_s": [round(o, 2) for o, _ in deaths],
            "duration_s": args.death_duration,
            "messages": [t for _, t in deaths],
        }
    }
    if len(advancements) >= 3:
        events["advancement"] = {
            "onsets_s": [round(o, 2) for o, _ in advancements],
            "duration_s": 5.0,
        }

    truth = {
        "video": str(video),
        "seconds": round(duration, 2),
        "start_wallclock": start.isoformat(),
        "alignment": how,
        "player": player,
        "notes": "onsets from Minecraft logs/latest.log chat lines",
        "events": events,
    }
    out = video.with_suffix(".truth.json")
    out.write_text(json.dumps(truth, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    print(f"next:  python -m smartreplay.cli discover {video}")
    print(f"       python tools/verify.py {out} out/{video.stem}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
