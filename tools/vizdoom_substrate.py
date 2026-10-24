"""Record real game footage with frame-exact ground truth, using ViZDoom.

This is the autonomous test substrate: an actual 3D game, rendered offscreen,
whose engine tells us precisely when each event happened. No hand labelling,
no screen recording, no human in the loop.

The event used is Doom's item-pickup message, which is a textbook example of
the thing discovery is built to find: a short line of text that appears at a
fixed place on screen, holds for a couple of seconds, and recurs. Ground truth
comes from the engine's HEALTH variable - a medikit pickup is a jump upward,
and the message is drawn on the same tic.

    python tools/vizdoom_substrate.py footage/doom.mp4 --seconds 240

Writes <out>.mp4 and <out>.truth.json, ready for:

    python -m smartreplay.cli discover footage/doom.mp4 --width 640
    python tools/verify.py footage/doom.truth.json out/doom
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import vizdoom as vzd

DOOM_TICRATE = 35.0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--scenario", default="health_gathering_supreme.cfg")
    ap.add_argument("--seconds", type=float, default=240.0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--message-duration", type=float, default=2.0,
                    help="nominal on-screen life of a pickup message")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    game = vzd.DoomGame()
    game.load_config(os.path.join(vzd.scenarios_path, args.scenario))
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
    game.set_screen_format(vzd.ScreenFormat.BGR24)
    game.set_render_hud(True)
    game.set_render_messages(True)       # the overlay under test
    game.set_render_weapon(True)
    game.set_window_visible(False)
    game.set_episode_timeout(int(args.seconds * DOOM_TICRATE) + 200)
    game.set_available_game_variables([vzd.GameVariable.HEALTH])
    game.init()

    n_buttons = game.get_available_buttons_size()
    rng = np.random.default_rng(args.seed)
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"),
                             DOOM_TICRATE, (args.width, args.height))
    if not writer.isOpened():
        print("could not open VideoWriter")
        return 1

    n_frames = int(args.seconds * DOOM_TICRATE)
    onsets: list[float] = []
    episode_starts: list[float] = []
    prev_health = None
    written = 0

    game.new_episode()
    episode_starts.append(0.0)

    while written < n_frames:
        if game.is_episode_finished() or game.is_player_dead():
            # Restarting makes a hard cut in the footage; record where, so the
            # scorer and any analysis can account for it.
            game.new_episode()
            episode_starts.append(written / DOOM_TICRATE)
            prev_health = None
            continue

        state = game.get_state()
        if state is None:
            game.advance_action()
            continue

        frame = state.screen_buffer
        if frame.shape[1] != args.width or frame.shape[0] != args.height:
            frame = cv2.resize(frame, (args.width, args.height), interpolation=cv2.INTER_AREA)
        writer.write(np.ascontiguousarray(frame))

        health = game.get_game_variable(vzd.GameVariable.HEALTH)
        if prev_health is not None and health - prev_health > 5.0:
            onsets.append(written / DOOM_TICRATE)
        prev_health = health
        written += 1

        # Wander and turn so world content keeps changing; the agent only has
        # to generate pickups, not play well.
        action = [False] * n_buttons
        for i in rng.choice(n_buttons, size=max(1, n_buttons // 2), replace=False):
            action[int(i)] = True
        game.make_action(action, 1)

    writer.release()
    game.close()

    truth = {
        "video": str(out),
        "fps": DOOM_TICRATE,
        "seconds": written / DOOM_TICRATE,
        "size": [args.width, args.height],
        "notes": "onsets are engine tics where HEALTH jumped, i.e. item pickups",
        "episode_starts_s": [round(s, 3) for s in episode_starts],
        "events": {
            "pickup_message": {
                "onsets_s": [round(t, 3) for t in onsets],
                "duration_s": args.message_duration,
            }
        },
    }
    tp = out.with_suffix(".truth.json")
    tp.write_text(json.dumps(truth, indent=2), encoding="utf-8")
    print(f"wrote {out} ({written} frames, {written / DOOM_TICRATE:.0f}s)")
    print(f"wrote {tp}: {len(onsets)} pickup(s), {len(episode_starts)} episode(s)")
    if len(onsets) < 4:
        print("warning: very few pickups - raise --seconds or try another scenario")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
