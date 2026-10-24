"""Cut a review reel: every hit, miss and false alarm as a short labelled clip.

Numbers like "F1 0.91" are easy to misread. A reel makes it checkable by eye:
each truth event is matched to the discovered group that best explains it
(the same matching tools/verify.py scores), and every occurrence becomes a
clip with a banner saying what it is:

    HIT          truth and detection agree (onset offset shown)
    MISSED       truth event with no detection near it
    FALSE ALARM  detection with no truth event near it
    EXTRA        groups that matched no truth event, a few occurrences each -
                 often real HUD the truth file doesn't cover (chat, menus)

The group's region is outlined so you can see *what* fired.

    python tools/review_video.py footage/bench/x.mp4 out/x footage/bench/x.truth.json
    python tools/review_video.py VIDEO OUT_DIR TRUTH --pre 2 --post 4 --out review.mp4
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import verify  # noqa: E402

GREEN, RED, AMBER, BLUE = (80, 200, 90), (60, 60, 230), (40, 170, 240), (220, 160, 60)


def _banner(frame: np.ndarray, text: str, sub: str, color, t: float) -> np.ndarray:
    h, w = frame.shape[:2]
    bar = max(28, h // 14)
    out = frame.copy()
    cv2.rectangle(out, (0, 0), (w, bar), color, -1)
    scale = bar / 40.0
    cv2.putText(out, text, (8, int(bar * 0.72)), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255),
                max(1, int(scale * 2)), cv2.LINE_AA)
    tag = f"t={t:7.1f}s"
    (tw, _), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, scale * 0.8, 1)
    cv2.putText(out, tag, (w - tw - 8, int(bar * 0.72)), cv2.FONT_HERSHEY_SIMPLEX, scale * 0.8,
                (255, 255, 255), max(1, int(scale * 1.5)), cv2.LINE_AA)
    if sub:
        cv2.rectangle(out, (0, h - bar), (w, h), (20, 20, 20), -1)
        cv2.putText(out, sub, (8, h - int(bar * 0.3)), cv2.FONT_HERSHEY_SIMPLEX, scale * 0.75,
                    (235, 235, 235), max(1, int(scale * 1.5)), cv2.LINE_AA)
    return out


def _card(w: int, h: int, lines: list[str]) -> np.ndarray:
    img = np.full((h, w, 3), 24, np.uint8)
    scale = h / 480.0
    y = int(h * 0.35)
    for i, line in enumerate(lines):
        s = scale * (1.1 if i == 0 else 0.7)
        (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, s, 2)
        cv2.putText(img, line, ((w - tw) // 2, y), cv2.FONT_HERSHEY_SIMPLEX, s, (235, 235, 235),
                    2 if i == 0 else 1, cv2.LINE_AA)
        y += int(th * 2.2)
    return img


def _active_at(group: dict, t: float) -> bool:
    return any(o["start_s"] <= t <= o["start_s"] + o["duration_s"] for o in group["occurrences"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("out_dir", help="discovery output directory (events.json, meta.json)")
    ap.add_argument("truth", help='truth file, or "-" to review discovered groups only')
    ap.add_argument("--tol", type=float, default=2.0)
    ap.add_argument("--pre", type=float, default=2.0)
    ap.add_argument("--post", type=float, default=4.0)
    ap.add_argument("--extra-per-group", type=int, default=3)
    ap.add_argument("--width", type=int, default=854, help="reel width (height keeps aspect)")
    ap.add_argument("--names", default="", help='labels for groups, e.g. "0=kill feed,5=combat report"')
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    names = {int(k): v for k, v in (kv.split("=", 1) for kv in args.names.split(",") if "=" in kv)}

    out_dir = Path(args.out_dir)
    groups = json.loads((out_dir / "events.json").read_text(encoding="utf-8"))
    meta = json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))
    truth = (json.loads(Path(args.truth).read_text(encoding="utf-8")).get("events", {})
             if args.truth != "-" else {})
    crop = meta.get("crop")

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    sw, sh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = args.width
    h = int(round(w * sh / sw / 2)) * 2

    # clips: (start_t, label, sub, color, group or None)
    clips: list[tuple] = []
    sections: list[tuple[int, list[str]]] = []
    claimed: set[int] = set()
    summary = []
    for name, ev in truth.items():
        gt = ev["onsets_s"]
        best = None
        for g in groups:
            found = [o["start_s"] for o in g["occurrences"]]
            tp, pairs = verify.match_onsets(gt, found, args.tol)
            score = verify.f1(tp, len(gt), len(found))[2]
            if best is None or score > best[0]:
                best = (score, g, pairs, found)
        sections.append((len(clips), [f"{name}", f"{len(gt)} in the truth file"]))
        if best is None or best[0] == 0:
            for t in gt:
                clips.append((t, f"MISSED  {name}", "no group matches this event", RED, None))
            summary.append(f"{name}: no matching group, {len(gt)} missed")
            continue
        score, g, pairs, found = best
        claimed.add(g["gid"])
        matched_t = {p[0] for p in pairs}
        matched_f = {p[1] for p in pairs}
        for t in gt:
            if t in matched_t:
                f = next(p[1] for p in pairs if p[0] == t)
                clips.append((min(t, f), f"HIT  {name}", f"group {g['gid']}  detected {f - t:+.1f}s from truth",
                              GREEN, g))
            else:
                clips.append((t, f"MISSED  {name}", f"group {g['gid']} did not fire here", RED, g))
        for f in found:
            if f not in matched_f:
                clips.append((f, f"FALSE ALARM  {name}?", f"group {g['gid']} fired, no truth event",
                              AMBER, g))
        clips[sections[-1][0]:] = sorted(clips[sections[-1][0]:], key=lambda c: c[0])
        tp = len(pairs)
        summary.append(f"{name}: group {g['gid']}, {tp}/{len(gt)} hit, "
                       f"{len(found) - tp} false alarm, F1 {score:.2f}")

    extras = [g for g in groups if g["gid"] not in claimed]
    if extras and args.extra_per_group > 0:
        sections.append((len(clips), ["Discovered groups" if not truth else "Extra groups",
                                      f"{len(extras)} group(s) not matched to any truth event",
                                      f"first {args.extra_per_group} occurrence(s) of each"]))
        for g in extras:
            for o in g["occurrences"][:args.extra_per_group]:
                tag = f"  ({names[g['gid']]})" if g["gid"] in names else ""
                clips.append((o["start_s"], f"EXTRA  group {g['gid']}{tag}",
                              f"{len(g['occurrences'])} occurrences, ~{g['mean_duration_s']:.1f}s each",
                              BLUE, g))

    tmp = Path(args.out or (out_dir / "review.mp4")).with_suffix(".tmp.avi")
    writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))
    title = _card(w, h, ["Smart Replay review", Path(args.video).name, *summary])
    for _ in range(int(fps * 4)):
        writer.write(title)

    sec_at = {i: lines for i, lines in sections}
    for i, (t0, label, sub, color, g) in enumerate(clips):
        if i in sec_at:
            card = _card(w, h, sec_at[i])
            for _ in range(int(fps * 1.5)):
                writer.write(card)
        start = max(0.0, t0 - args.pre)
        cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000.0)
        n = int((args.pre + args.post) * fps)
        for k in range(n):
            ok, frame = cap.read()
            if not ok:
                break
            t = start + k / fps
            if g is not None and crop:
                x0, y0, x1, y1 = g["roi_norm"]
                cx0, cy0, cx1, cy1 = crop
                bw, bh = cx1 - cx0, cy1 - cy0
                p0 = (int(cx0 + x0 * bw), int(cy0 + y0 * bh))
                p1 = (int(cx0 + x1 * bw), int(cy0 + y1 * bh))
                live = _active_at(g, t)
                cv2.rectangle(frame, p0, p1, (0, 255, 255) if live else (140, 140, 140),
                              2 if live else 1)
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
            writer.write(_banner(frame, f"[{i + 1}/{len(clips)}] {label}", sub, color, t))
    writer.release()
    cap.release()

    out = Path(args.out or (out_dir / "review.mp4"))
    if shutil.which("ffmpeg"):
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp), "-c:v", "libx264",
                        "-preset", "veryfast", "-crf", "26", "-pix_fmt", "yuv420p",
                        "-movflags", "+faststart", str(out)], check=True)
        tmp.unlink()
    else:
        out = tmp
    print("\n".join(summary))
    print(f"{len(clips)} clip(s) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
