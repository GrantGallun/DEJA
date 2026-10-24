"""OCR a video at a fixed rate and store every word with its position.

This is a ground-truth helper, not part of discovery. Most game events that
matter come with text - "You died!", "SPECTATING", "ROUND WIN", "An enemy has
been slain" - so a word index over the footage turns labelling into keyword
search plus a visual check of the hits, instead of scrubbing video by hand.

    python tools/ocr_scan.py footage/bench/valorant.mp4             # writes .ocr.jsonl
    python tools/ocr_scan.py footage/bench/valorant.mp4 --fps 1 --workers 2

Output, one line per sampled frame:
    {"t": 12.0, "words": [["SPECTATING", 91, 0.41, 0.82, 0.18, 0.03], ...]}
with [text, confidence, x, y, w, h] in normalised frame coordinates.

Two passes per frame, because game text is drawn both as bright glyphs on a
busy background and as dark text on light panels: the plain grey frame, and
an inverted threshold that isolates bright text.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np


def _words(img: np.ndarray, scale: float, w: int, h: int, min_conf: int) -> list[list]:
    import pytesseract

    d = pytesseract.image_to_data(img, config="--psm 11", output_type=pytesseract.Output.DICT)
    out = []
    for i, text in enumerate(d["text"]):
        text = text.strip()
        try:
            conf = int(float(d["conf"][i]))
        except ValueError:
            continue
        if not text or conf < min_conf or not any(c.isalnum() for c in text):
            continue
        out.append([text, conf,
                    round(d["left"][i] / scale / w, 4), round(d["top"][i] / scale / h, 4),
                    round(d["width"][i] / scale / w, 4), round(d["height"][i] / scale / h, 4)])
    return out


def ocr_frame(frame: np.ndarray, min_conf: int = 50) -> list[list]:
    h, w = frame.shape[:2]
    # Small HUD text needs more pixels than a 480p frame gives it.
    scale = max(1.0, 1280.0 / w)
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if scale != 1.0:
        g = cv2.resize(g, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    words = _words(g, scale, w, h, min_conf)
    _, bright = cv2.threshold(g, 190, 255, cv2.THRESH_BINARY_INV)
    seen = {(x[0].lower(), round(x[2], 2), round(x[3], 2)) for x in words}
    for x in _words(bright, scale, w, h, min_conf):
        if (x[0].lower(), round(x[2], 2), round(x[3], 2)) not in seen:
            words.append(x)
    return words


def _work(args) -> list[dict]:
    video, times, min_conf = args
    cap = cv2.VideoCapture(video)
    rows = []
    for t in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        if not ok:
            continue
        rows.append({"t": round(float(t), 3), "words": ocr_frame(frame, min_conf)})
    cap.release()
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--min-conf", type=int, default=50)
    ap.add_argument("--max-seconds", type=float, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    dur = (cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) / fps
    cap.release()
    if args.max_seconds:
        dur = min(dur, args.max_seconds)
    times = np.arange(0.0, dur, 1.0 / args.fps)

    # Contiguous chunks per worker keep seeks short.
    n = max(1, args.workers) * 8
    chunks = [c for c in np.array_split(times, n) if len(c)]
    out = Path(args.out) if args.out else Path(args.video).with_suffix(".ocr.jsonl")
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, part in enumerate(pool.map(_work, [(args.video, c, args.min_conf) for c in chunks])):
            rows.extend(part)
            print(f"  chunk {i + 1}/{len(chunks)}", flush=True)
    rows.sort(key=lambda r: r["t"])
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {out} ({len(rows)} frames)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
