"""Build a searchable event index for one recording.

Discovery says *when* each HUD element appeared, not what it meant. The
index attaches evidence to every occurrence - the text on screen around it -
so a free-form request ("every time I burned to death", "the round I clutched")
can be answered later by reading the index, without game-specific code and
without anyone watching the footage.

Nothing here knows about any game. An occurrence is:

    {"id": "g2#14", "group": 2, "t": 1305.0, "duration": 4.0,
     "roi": [x0, y0, x1, y1],
     "text": ["GreatKiller was killed by magic", "Your game mode has been updated"]}

Text comes from OCR of a few frames around the onset, read over the whole
frame: what an event *means* is often written somewhere other than the
element itself (Minecraft prints the cause of death in chat, not on the
death screen). OCR runs only on those frames, not the whole video, so the
cost scales with the number of events, not the length of the recording.

    python -m smartreplay.index footage/bench/x.mp4 out/x
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


def _lines(words: list, y_tol: float = 0.012) -> list[tuple[float, float, str]]:
    """Join OCR words into text lines: (x, y, text) in normalised coords."""
    words = sorted(words, key=lambda w: (round(w[3] / y_tol), w[2]))
    lines: list[list] = []
    for w in words:
        for ln in lines:
            if abs(ln[0][3] - w[3]) <= y_tol and 0 <= w[2] - (ln[-1][2] + ln[-1][4]) <= 0.06:
                ln.append(w)
                break
        else:
            lines.append([w])
    return [(ln[0][2], ln[0][3], " ".join(x[0] for x in ln)) for ln in lines]


def _ocr_lookup(ocr_path: Path | None):
    if not ocr_path or not ocr_path.exists():
        return None
    rows = [json.loads(line) for line in ocr_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {round(r["t"]): r["words"] for r in rows}


def build(video: str | Path, out_dir: str | Path, before: float = 1.0, after: float = 3.0,
          step: float = 1.0, min_conf: int = 60) -> dict:
    from tools.ocr_scan import ocr_frame  # noqa: WPS433 - optional dependency (pytesseract)

    out_dir = Path(out_dir)
    groups = json.loads((out_dir / "events.json").read_text(encoding="utf-8"))
    cached = _ocr_lookup(Path(video).with_suffix(".ocr.jsonl"))
    cap = cv2.VideoCapture(str(video))

    def words_at(t: float) -> list:
        if cached is not None and round(t) in cached:
            return [w for w in cached[round(t)] if w[1] >= min_conf]
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        return ocr_frame(frame, min_conf) if ok else []

    occurrences = []
    for g in groups:
        for k, o in enumerate(g["occurrences"]):
            t0 = o["start_s"]
            seen: dict[str, int] = {}
            t = max(0.0, t0 - before)
            while t <= t0 + after + 1e-6:
                for _, _, text in _lines(words_at(t)):
                    if sum(c.isalpha() for c in text) >= 4:
                        seen[text] = seen.get(text, 0) + 1
                t += step
            # Lines seen in several frames are real text; one-off reads are
            # mostly OCR noise. Keep the steadiest first.
            text = [s for s, n in sorted(seen.items(), key=lambda kv: (-kv[1], -len(kv[0])))][:12]
            occurrences.append({
                "id": f"g{g['gid']}#{k}", "group": g["gid"], "t": round(t0, 2),
                "duration": round(o["duration_s"], 2), "roi": g["roi_norm"], "text": text,
            })
    cap.release()

    index = {
        "video": str(video),
        "groups": [{"group": g["gid"], "roi": g["roi_norm"], "n": len(g["occurrences"]),
                    "mean_duration": g["mean_duration_s"]} for g in groups],
        "occurrences": sorted(occurrences, key=lambda o: o["t"]),
    }
    (out_dir / "index.json").write_text(json.dumps(index, indent=1), encoding="utf-8")
    return index


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("out_dir")
    args = ap.parse_args(argv)
    idx = build(args.video, args.out_dir)
    print(f"{len(idx['occurrences'])} occurrence(s) across {len(idx['groups'])} group(s) "
          f"-> {Path(args.out_dir) / 'index.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
