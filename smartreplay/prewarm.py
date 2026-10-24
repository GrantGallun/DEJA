"""Do the slow work before anyone asks: warm the model, label, describe.

A request is slow for two reasons: starting the model (seconds per process)
and looking at moments nobody has looked at yet (one vision call each). Both
can happen in the background - while the game is still being played, or as
soon as discovery finishes - so that answering a request later is just the
plan call plus text matching.

    1 start warm workers for every call shape         (llm.preload)
    2 label the recording's HUD groups                 1 call
    3 describe every event moment once: the messages   1 vision call per moment,
      on screen and short facts ("cause: fire")        in parallel
    4 pre-start a plan worker for this recording's names

The facts are written into index.json next to the OCR text, so every later
request - however it is worded - is matched against what a vision model
already read, with no vision calls at request time.

    python -m smartreplay.prewarm out/x --game Minecraft
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2

from .fulfil import Fulfiller, Memory, _collapse, _slug
from .llm import LLM, plan_tool


def frames_at(video: str, t: float, offsets=(-0.5, 0.5, 1.5)) -> list:
    cap = cv2.VideoCapture(video)
    out = []
    for d in offsets:
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t + d) * 1000.0)
        ok, f = cap.read()
        if ok:
            out.append(f)
    cap.release()
    return out


def prewarm(out_dir: str | Path, game: str, llm: LLM, profiles: str | Path = "profiles",
            roles=("event",), workers: int = 4) -> dict:
    t0 = time.perf_counter()
    if hasattr(llm, "preload"):
        llm.preload()
    out_dir = Path(out_dir)
    index_path = out_dir / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    mem = Memory(Path(profiles) / f"{_slug(game)}.json", out_dir / "memory.json")
    f = Fulfiller(index, game, llm, mem)
    f.resolve_anchor("")
    labels = mem.rec["labels"]
    catalog = sorted({f"{v['concept']} ({v.get('role', 'event')})" for v in labels.values()
                      if v["concept"] != "unknown"})
    if getattr(llm, "pool", None) is not None:
        llm.pool.start([plan_tool(catalog)["input_schema"]], [1])

    targets = []
    for gid, lab in labels.items():
        if lab.get("role", "event") not in roles or lab["concept"] == "unknown":
            continue
        occs = [o for o in index["occurrences"] if o["group"] == int(gid)]
        for o in _collapse(occs):
            if "facts" not in o:
                targets.append((o, lab["concept"]))

    def describe(item):
        o, kind = item
        frames = frames_at(index["video"], o["t"])
        return o["id"], (llm.describe(frames, game, kind) if frames else None)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        facts = dict(pool.map(describe, targets))
    for o in index["occurrences"]:
        if facts.get(o["id"]):
            o["facts"] = facts[o["id"]]
    index_path.write_text(json.dumps(index, indent=1), encoding="utf-8")
    mem.save()
    return {"described": len(facts), "groups_labelled": len(labels),
            "seconds": round(time.perf_counter() - t0, 1)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--game", required=True)
    ap.add_argument("--profiles", default="profiles")
    args = ap.parse_args(argv)
    from .llm import default_llm

    print(prewarm(args.out_dir, args.game, default_llm(), args.profiles))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
