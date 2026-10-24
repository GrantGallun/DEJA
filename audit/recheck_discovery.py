"""Recompute discovery from the existing sample stacks, saving audit-only output."""

from __future__ import annotations

import contextlib
import gc
import io
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE / "snapshot"))
from smartreplay import discovery, grouping, sampling
from tools import verify

rows = []
for name in ("synthetic", "doom"):
    folder = HERE / "results" / "recomputed" / name
    folder.mkdir(parents=True, exist_ok=True)
    print(f"Recomputing {name} from sampled frames...", flush=True)
    start = time.perf_counter()
    gray, color, times, meta = sampling.load_samples(ROOT / "out" / name)
    result = discovery.discover(gray, meta["sample_fps"])
    groups = grouping.group_events(result.on, result.event_mask, times, meta["sample_fps"],
                                   grouping.GroupParams(min_pixels=10))
    grouping.save(groups, folder)
    row = {"name": name, "n_frames": len(gray), "groups": len(groups),
           "classification": result.summary(), "elapsed_s": round(time.perf_counter() - start, 2),
           "sampling_note": "Existing samples reused; video decoding not re-tested."}
    if name == "synthetic":
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            row["verify_exit_code"] = verify.main([str(ROOT / "footage/synthetic.truth.json"), str(folder)])
        row["verification"] = captured.getvalue()
    else:
        row["expected_visible_event_groups"] = 0
        row["negative_control_passed"] = len(groups) == 0
        row["truth_note"] = "Engine pickup events in doom.truth.json are not visible overlays; not a positive detection test."
    rows.append(row)
    print(json.dumps(row), flush=True)
    del gray, color, times, result, groups
    gc.collect()

(HERE / "results" / "recomputed.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
