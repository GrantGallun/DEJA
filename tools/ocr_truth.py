"""Turn an OCR word index (tools/ocr_scan.py) into a ground-truth file.

Each game gets a few rules: an event name, a regular expression over the text
OCR found in a frame (optionally only inside a screen region), and a minimum
number of consecutive matching samples. Matching samples are joined into
occurrences, bridging short OCR dropouts.

    python tools/ocr_truth.py footage/bench/minecraft_30ways.mp4 --game minecraft
    python tools/ocr_truth.py footage/bench/x.mp4 --game minecraft --sheet out/x_truth.jpg

The result is a *draft*: always check the contact sheet before trusting it.
OCR misses some frames and misreads others, so the per-event onset is the
first sample it read, which can lag the real onset by a sample or two.
Writes <video>.truth.json in the format tools/verify.py reads, plus the text
each occurrence was matched on (useful later for semantic checks: who killed
you, what the death message said).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# name: (regex over joined frame text, region (x0,y0,x1,y1) or None, min samples, duration)
RULES: dict[str, dict[str, tuple]] = {
    "minecraft": {
        "death": (r"\b(you|yeu|tou|vou)\s*died\b|\bdied!", (0.2, 0.05, 0.8, 0.5), 1, None),
        "advancement": (r"advancement\s+made|goal\s+reached|challenge\s+complete", (0.6, 0.0, 1.0, 0.25), 1, None),
    },
    "valorant": {
        # Top-centre result banner after every round.
        "round_result": (r"\b(won|lost|w0n|l0st)\b", (0.35, 0.0, 0.65, 0.25), 1, None),
        "buy_phase": (r"\bphase\b", (0.35, 0.05, 0.65, 0.25), 2, None),
        # The combat report opens on death; mid-round it reads ROUND ONGOING.
        "death": (r"\bround\s+ongo", (0.7, 0.45, 1.0, 0.65), 2, None),
        "combat_report": (r"\bcombat\b", (0.7, 0.25, 1.0, 0.55), 2, None),
    },
    "league": {
        "enemy_slain": (r"\b(you\s+have\s+slain|enemy\s+has\s+been\s+slain|double\s+kill|triple\s+kill|first\s+blood|shut\s*down)\b",
                        (0.2, 0.05, 0.8, 0.45), 1, None),
        "you_died": (r"\b(you\s+have\s+been\s+slain|executed)\b", (0.2, 0.05, 0.8, 0.45), 1, None),
    },
    "deadlock": {
        "you_died": (r"\b(respawn(ing)?\s*in|killed\s+by|you\s+died)\b", None, 2, None),
    },
}


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def frame_text(words: list, region) -> str:
    if region is None:
        return " ".join(w[0] for w in words)
    x0, y0, x1, y1 = region
    keep = [w for w in words if x0 <= w[2] + w[4] / 2 <= x1 and y0 <= w[3] + w[5] / 2 <= y1]
    return " ".join(w[0] for w in keep)


def occurrences(rows: list[dict], rx: re.Pattern, region, min_n: int, dt: float,
                bridge: int = 2) -> list[dict]:
    hits = [(r["t"], frame_text(r["words"], region)) for r in rows]
    hits = [(t, txt) for t, txt in hits if rx.search(txt)]
    occ: list[dict] = []
    for t, txt in hits:
        if occ and t - occ[-1]["end_s"] <= dt * (bridge + 1) + 1e-6:
            occ[-1]["end_s"] = t
            occ[-1]["n"] += 1
        else:
            occ.append({"start_s": t, "end_s": t, "n": 1, "text": txt})
    return [o for o in occ if o["n"] >= min_n]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--game", required=True, choices=sorted(RULES))
    ap.add_argument("--ocr", default=None, help="defaults to <video>.ocr.jsonl")
    ap.add_argument("--max-seconds", type=float, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    video = Path(args.video)
    rows = load(Path(args.ocr) if args.ocr else video.with_suffix(".ocr.jsonl"))
    if args.max_seconds:
        rows = [r for r in rows if r["t"] <= args.max_seconds]
    dt = rows[1]["t"] - rows[0]["t"] if len(rows) > 1 else 1.0

    events = {}
    for name, (pattern, region, min_n, dur) in RULES[args.game].items():
        occ = occurrences(rows, re.compile(pattern, re.I), region, min_n, dt)
        print(f"{name:12s} {len(occ):3d} occurrence(s)")
        for o in occ:
            print(f"    {o['start_s']:8.1f}s  {o['end_s'] - o['start_s'] + dt:5.1f}s  {o['text'][:70]}")
        if occ:
            durs = [o["end_s"] - o["start_s"] + dt for o in occ]
            events[name] = {
                "onsets_s": [o["start_s"] for o in occ],
                "durations_s": [round(d, 2) for d in durs],
                "duration_s": dur or round(sorted(durs)[len(durs) // 2], 2),
                "texts": [o["text"] for o in occ],
            }

    truth = {
        "video": str(video),
        "seconds": rows[-1]["t"] if rows else 0,
        "source": f"ocr_truth rules={args.game} (draft - verify against the contact sheet)",
        "events": events,
    }
    out = Path(args.out) if args.out else video.with_suffix(".truth.json")
    out.write_text(json.dumps(truth, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
