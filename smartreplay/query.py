"""Answer a free-form replay request from the event index, then cut the reel.

    python -m smartreplay.query out/x "every time I burned to death"
    python -m smartreplay.query out/x "burned to death" --matcher lexical
    python -m smartreplay.query out/x "burned to death" --ids g2#26 --reason "went up in flames"

There is no code here for any particular game or request. The request and
the index (index.py: when each discovered HUD element fired, plus the text on
screen around it) go to a language model, which decides which occurrences the
request means - "burned to death" covers Minecraft's "went up in flames" and
"walked into fire", and a model can say whether "tried to swim in lava"
counts. Its answer is a list of occurrence ids with a reason for each; the
reel is cut from those.

Matchers:
    claude   (default when ANTHROPIC_API_KEY is set) - semantic, any phrasing
    lexical  word overlap; offline fallback, only finds literal wording
    --ids    a selection made elsewhere (another planner, or a person)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import shutil
from pathlib import Path

import cv2
import numpy as np

PROMPT = """You are the search step of a gameplay replay tool. A recording was analysed
without any game-specific knowledge: recurring HUD elements were discovered
("groups"), and every time one appeared ("occurrence") the text on screen
around that moment was read by OCR. OCR is noisy - expect misspellings such as
"Greathiller" for "GreatKiller" or "suin" for "swim".

The player's request: {query!r}

Decide which occurrences the request refers to.
- First work out which group is the relevant kind of event (for "deaths",
  the group whose text reads like death messages).
- Then keep only occurrences whose evidence matches the request's meaning,
  not its exact words.
- One real moment can appear as several occurrences seconds apart; keep one.
- Put genuinely debatable cases in "maybe" with the reason, rather than
  guessing.

Index:
{index}

Reply with JSON only:
{{"group": <int or null>, "matches": [{{"id": "...", "why": "..."}}],
  "maybe": [{{"id": "...", "why": "..."}}], "notes": "..."}}"""


def compact_index(index: dict, max_lines: int = 6) -> str:
    rows = ["groups: " + "; ".join(
        f"g{g['group']} roi={[round(v, 2) for v in g['roi']]} n={g['n']} ~{g['mean_duration']:.1f}s"
        for g in index["groups"])]
    for o in index["occurrences"]:
        text = " | ".join(o["text"][:max_lines])
        rows.append(f"{o['id']} t={o['t']:.1f}s: {text}")
    return "\n".join(rows)


def match_claude(index: dict, query: str, model: str) -> dict:
    import anthropic  # pip install anthropic

    client = anthropic.Anthropic()
    msg = client.messages.create(
        model=model, max_tokens=2000,
        messages=[{"role": "user", "content": PROMPT.format(query=query, index=compact_index(index))}],
    )
    text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
    m = re.search(r"\{.*\}", text, re.S)
    return json.loads(m.group(0)) if m else {"matches": [], "notes": text}


def match_lexical(index: dict, query: str) -> dict:
    stop = {"the", "a", "an", "to", "of", "i", "me", "my", "every", "time", "times", "when", "all",
            "clip", "show", "only"}
    q = [w for w in re.findall(r"[a-z]+", query.lower()) if w not in stop]
    hits = []
    for o in index["occurrences"]:
        blob = " ".join(o["text"]).lower()
        if q and all(w in blob for w in q):
            hits.append({"id": o["id"], "why": "all query words present"})
    return {"matches": hits, "notes": f"lexical match on {q}"}


def dedupe(index: dict, ids: list[str], window: float = 8.0) -> list[dict]:
    by_id = {o["id"]: o for o in index["occurrences"]}
    picked = sorted((by_id[i] for i in ids if i in by_id), key=lambda o: o["t"])
    out: list[dict] = []
    for o in picked:
        if not out or o["t"] - out[-1]["t"] > window:
            out.append(o)
    return out


def cut_reel(video: str, moments: list[dict], reasons: dict[str, str], out: Path, query: str,
             pre: float = 5.0, post: float = 1.0, width: int = 854) -> Path:
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    sw, sh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w, h = width, int(round(width * sh / sw / 2)) * 2
    tmp = out.with_suffix(".tmp.avi")
    wr = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))
    bar = max(28, h // 14)
    s = bar / 40.0

    card = np.full((h, w, 3), 24, np.uint8)
    for i, line in enumerate([f'"{query}"', f"{len(moments)} moment(s), {pre:.0f}s before each"]):
        cv2.putText(card, line, (24, int(h * 0.42) + i * int(bar * 1.4)), cv2.FONT_HERSHEY_SIMPLEX,
                    s * (1.0 if i == 0 else 0.7), (235, 235, 235), 2 if i == 0 else 1, cv2.LINE_AA)
    for _ in range(int(fps * 2.5)):
        wr.write(card)

    for n, o in enumerate(moments):
        start = max(0.0, o["t"] - pre)
        cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000.0)
        for k in range(int((pre + post) * fps)):
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
            t = start + k / fps
            cv2.rectangle(frame, (0, 0), (w, bar), (40, 110, 200), -1)
            cv2.putText(frame, f"[{n + 1}/{len(moments)}] {reasons.get(o['id'], '')}"[:70],
                        (8, int(bar * 0.72)), cv2.FONT_HERSHEY_SIMPLEX, s * 0.8, (255, 255, 255),
                        max(1, int(s * 1.6)), cv2.LINE_AA)
            tag = f"{t - o['t']:+.1f}s"
            cv2.putText(frame, tag, (w - 90, int(bar * 0.72)), cv2.FONT_HERSHEY_SIMPLEX, s * 0.8,
                        (255, 255, 255), max(1, int(s * 1.6)), cv2.LINE_AA)
            wr.write(frame)
    wr.release()
    cap.release()
    if shutil.which("ffmpeg"):
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp), "-c:v", "libx264",
                        "-preset", "veryfast", "-crf", "26", "-pix_fmt", "yuv420p",
                        "-movflags", "+faststart", str(out)], check=True)
        tmp.unlink()
        return out
    return tmp


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", help="discovery output dir containing index.json")
    ap.add_argument("query")
    ap.add_argument("--matcher", choices=("claude", "lexical"), default=None)
    ap.add_argument("--model", default=os.environ.get("SMARTREPLAY_MODEL", "claude-sonnet-4-5"))
    ap.add_argument("--ids", default=None, help="comma-separated occurrence ids, skipping the matcher")
    ap.add_argument("--reason", default="", help="banner text when --ids is used")
    ap.add_argument("--include-maybe", action="store_true")
    ap.add_argument("--pre", type=float, default=5.0)
    ap.add_argument("--post", type=float, default=1.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--print-prompt", action="store_true", help="show what the model would be sent")
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir)
    index = json.loads((out_dir / "index.json").read_text(encoding="utf-8"))
    if args.print_prompt:
        print(PROMPT.format(query=args.query, index=compact_index(index)))
        return 0

    if args.ids:
        ids = [i.strip() for i in args.ids.split(",") if i.strip()]
        result = {"matches": [{"id": i, "why": args.reason} for i in ids], "notes": "ids given"}
    else:
        matcher = args.matcher or ("claude" if os.environ.get("ANTHROPIC_API_KEY") else "lexical")
        result = (match_claude(index, args.query, args.model) if matcher == "claude"
                  else match_lexical(index, args.query))
    picks = list(result.get("matches", []))
    if args.include_maybe:
        picks += result.get("maybe", [])
    reasons = {p["id"]: p.get("why", "") for p in picks}
    moments = dedupe(index, [p["id"] for p in picks])

    print(json.dumps(result, indent=1))
    if not moments:
        print("no matching moments")
        return 1
    slug = re.sub(r"[^a-z0-9]+", "_", args.query.lower()).strip("_")[:40]
    out = Path(args.out or (out_dir / f"reel_{slug}.mp4"))
    path = cut_reel(index["video"], moments, reasons, out, args.query, args.pre, args.post)
    print(f"{len(moments)} moment(s) -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
