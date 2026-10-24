"""Synthesise a fake gameplay clip with known event timings.

The point is a ground truth to check discovery against: a world layer whose
pixels take a broad spread of values, an always-on HUD that barely moves, and
three event overlays that appear at recorded times for recorded durations.
Writes <out>.mp4 plus <out>.truth.json.

    python tools/make_test_clip.py footage/synthetic.mp4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX

DEATHS = [15.0, 38.0, 62.0, 85.0, 103.0]
DEATH_DUR = 5.0
KILLS = [10.0, 22.0, 30.0, 48.0, 55.0, 70.0, 78.0, 92.0]
KILL_DUR = 1.5
ROUNDS = [5.0, 45.0, 75.0]
ROUND_DUR = 3.0

KILLERS = ["Vex", "Harrow", "Mint"]


def _world_texture(h: int, w: int, seed: int = 7) -> np.ndarray:
    """Smooth large-scale colour noise, big enough to scroll a window over."""
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 255, size=(24, 40, 3), dtype=np.uint8)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)


def _draw_static_hud(frame: np.ndarray) -> None:
    h, w = frame.shape[:2]
    # Health bar, bottom left: fixed geometry and colour.
    cv2.rectangle(frame, (40, h - 70), (300, h - 40), (30, 30, 30), -1)
    cv2.rectangle(frame, (44, h - 66), (240, h - 44), (80, 200, 90), -1)
    # Ammo readout, bottom right.
    cv2.rectangle(frame, (w - 220, h - 80), (w - 40, h - 36), (24, 24, 24), -1)
    cv2.putText(frame, "24 / 90", (w - 205, h - 48), FONT, 0.9, (235, 235, 235), 2, cv2.LINE_AA)
    # Minimap frame, top right: constant border, noisy interior.
    cv2.rectangle(frame, (w - 230, 30), (w - 40, 220), (200, 200, 200), 3)


def _draw_death(frame: np.ndarray, killer: str) -> None:
    h, w = frame.shape[:2]
    panel = frame[h // 2 - 110:h // 2 + 110, :]
    panel[:] = (18, 16, 15)
    cv2.putText(frame, "YOU DIED", (w // 2 - 250, h // 2 - 10), FONT, 2.6,
                (60, 60, 230), 6, cv2.LINE_AA)
    cv2.putText(frame, f"killed by {killer}", (w // 2 - 150, h // 2 + 60), FONT, 1.0,
                (210, 210, 210), 2, cv2.LINE_AA)


def _draw_kill(frame: np.ndarray, victim: str) -> None:
    w = frame.shape[1]
    cv2.rectangle(frame, (w // 2 - 190, 60), (w // 2 + 190, 110), (20, 20, 20), -1)
    cv2.putText(frame, f"ELIMINATED {victim}", (w // 2 - 170, 95), FONT, 0.8,
                (250, 230, 120), 2, cv2.LINE_AA)


def _draw_round(frame: np.ndarray, n: int) -> None:
    w = frame.shape[1]
    cv2.rectangle(frame, (0, 170), (w, 240), (40, 30, 70), -1)
    cv2.putText(frame, f"ROUND {n}", (w // 2 - 120, 220), FONT, 1.6,
                (230, 230, 255), 3, cv2.LINE_AA)


def _active(t: float, starts: list[float], dur: float) -> int | None:
    for i, s in enumerate(starts):
        if s <= t < s + dur:
            return i
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--seconds", type=float, default=120.0)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    w, h = args.width, args.height

    tex = _world_texture(h * 2, w * 2)
    n_frames = int(args.seconds * args.fps)
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))
    if not writer.isOpened():
        print("could not open VideoWriter")
        return 1

    rng = np.random.default_rng(11)
    # Aperiodic camera and exposure. Earlier versions scrolled linearly and
    # breathed on a sine, which made world pixels quasi-periodic: they dwelt
    # near one value for seconds at a time and so looked exactly like event
    # HUD to the classifier. Real gameplay is not periodic, so neither is this.
    cam = np.array([w / 2.0, h / 2.0])
    vel = np.array([6.0, 4.0])
    gain = 1.0
    shapes = [
        {"pos": rng.uniform([0, 0], [w, h]),
         "vel": rng.uniform(-7, 7, size=2),
         "color": tuple(int(c) for c in rng.integers(40, 230, size=3)),
         "r": int(60 + 25 * k)}
        for k in range(4)
    ]

    for i in range(n_frames):
        t = i / args.fps
        vel += rng.normal(0, 1.6, size=2)
        vel = np.clip(vel, -14, 14)
        cam += vel
        cam[0] = np.clip(cam[0], 0, w - 1)
        cam[1] = np.clip(cam[1], 0, h - 1)
        if cam[0] in (0, w - 1):
            vel[0] = -vel[0]
        if cam[1] in (0, h - 1):
            vel[1] = -vel[1]

        ox, oy = int(cam[0]), int(cam[1])
        frame = tex[oy:oy + h, ox:ox + w].copy()
        gain = float(np.clip(gain + rng.normal(0, 0.03), 0.55, 1.25))
        frame = np.clip(frame.astype(np.float32) * gain, 0, 255).astype(np.uint8)

        # Objects with persistent colour but aperiodic motion.
        for s in shapes:
            s["vel"] += rng.normal(0, 0.9, size=2)
            s["vel"] = np.clip(s["vel"], -9, 9)
            s["pos"] += s["vel"]
            for ax, lim in ((0, w), (1, h)):
                if not (0 <= s["pos"][ax] < lim):
                    s["pos"][ax] = np.clip(s["pos"][ax], 0, lim - 1)
                    s["vel"][ax] = -s["vel"][ax]
            cv2.circle(frame, (int(s["pos"][0]), int(s["pos"][1])), s["r"], s["color"], -1)

        # Grain decorrelates adjacent samples the way real footage does, but it
        # belongs to the world layer only: a game composites its interface
        # after post-processing, and codecs keep flat HUD panels flat.
        frame = np.clip(frame.astype(np.int16) + rng.normal(0, 5, frame.shape), 0, 255).astype(np.uint8)

        _draw_static_hud(frame)

        ri = _active(t, ROUNDS, ROUND_DUR)
        if ri is not None:
            _draw_round(frame, ri + 1)
        ki = _active(t, KILLS, KILL_DUR)
        if ki is not None:
            _draw_kill(frame, KILLERS[ki % len(KILLERS)])
        di = _active(t, DEATHS, DEATH_DUR)
        if di is not None:
            _draw_death(frame, KILLERS[di % len(KILLERS)])

        writer.write(frame)
    writer.release()

    truth = {
        "video": str(out),
        "fps": args.fps,
        "seconds": args.seconds,
        "size": [w, h],
        "events": {
            "death": {"onsets_s": DEATHS, "duration_s": DEATH_DUR,
                      "roi_norm": [0.0, 0.5 - 110 / h, 1.0, 0.5 + 110 / h]},
            "kill": {"onsets_s": KILLS, "duration_s": KILL_DUR,
                     "roi_norm": [0.5 - 190 / w, 60 / h, 0.5 + 190 / w, 110 / h]},
            "round": {"onsets_s": ROUNDS, "duration_s": ROUND_DUR,
                      "roi_norm": [0.0, 170 / h, 1.0, 240 / h]},
        },
    }
    tp = out.with_suffix(".truth.json")
    tp.write_text(json.dumps(truth, indent=2), encoding="utf-8")
    print(f"wrote {out} ({n_frames} frames) and {tp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
