"""Score discovered events against a ground-truth file.

Truth format (as written by make_test_clip.py, minecraft_truth.py, or any
other source):

    {"events": {"<name>": {"onsets_s": [...], "duration_s": 5.0,
                           "roi_norm": [x0, y0, x1, y1]}}}

`roi_norm` is optional. For each truth event the best-matching discovered
group is chosen by F1 over onset times, so a correct answer does not depend on
discovery having found the events in any particular order or having named
them - naming is a later step.

    python tools/verify.py footage/synthetic.truth.json out/synthetic
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


def match_onsets(truth: list[float], found: list[float], tol: float) -> tuple[int, list]:
    """Maximum one-to-one matching of found onsets to truth onsets within tol.

    Solved as an assignment problem: pairs farther apart than tol cost more
    than every valid pair combined, so the solver first maximises the number
    of matches and only then minimises their total offset. Greedy nearest
    matching undercounts, e.g. truth=[2, 3], found=[1, 2.4], tol=1.1.
    """
    if not truth or not found:
        return 0, []
    t = np.asarray(truth, dtype=np.float64)
    f = np.asarray(found, dtype=np.float64)
    d = np.abs(t[:, None] - f[None, :])
    big = tol * (len(t) + len(f)) + 1.0
    rows, cols = linear_sum_assignment(np.where(d <= tol, d, big))
    pairs = [(float(t[r]), float(f[c]), float(d[r, c]))
             for r, c in zip(rows, cols) if d[r, c] <= tol]
    pairs.sort()
    return len(pairs), pairs


def f1(tp: int, n_truth: int, n_found: int) -> tuple[float, float, float]:
    prec = tp / n_found if n_found else 0.0
    rec = tp / n_truth if n_truth else 0.0
    return prec, rec, (2 * prec * rec / (prec + rec) if prec + rec else 0.0)


def iou(a, b) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("truth")
    ap.add_argument("out_dir")
    ap.add_argument("--tol", type=float, default=1.5, help="onset tolerance in seconds")
    ap.add_argument("--min-f1", type=float, default=0.8, help="per-event pass threshold")
    args = ap.parse_args(argv)

    truth = json.loads(Path(args.truth).read_text(encoding="utf-8"))
    groups = json.loads((Path(args.out_dir) / "events.json").read_text(encoding="utf-8"))
    truth_events = truth.get("events") or {}

    # A negative control: footage known to contain no events. The only
    # correct output is no groups at all, and any group is a false positive.
    if not truth_events:
        if groups:
            print(f"negative control: truth has no events but {len(groups)} group(s) "
                  f"were discovered - FAIL")
            return 1
        print("negative control: truth has no events and none were discovered - PASS")
        return 0
    if not groups:
        print(f"no groups discovered, {len(truth_events)} truth event(s) expected - FAIL")
        return 1

    print(f"{len(groups)} group(s) discovered, {len(truth_events)} truth event(s), "
          f"tolerance {args.tol}s\n")
    print(f"{'event':8s} {'gid':>4} {'truth':>6} {'found':>6} {'tp':>4} "
          f"{'prec':>6} {'rec':>6} {'f1':>6} {'roi_iou':>8} {'mean|dt|':>9}")

    rows = []
    claimed = {}
    for name, ev in truth_events.items():
        gt = ev["onsets_s"]
        best = None
        for g in groups:
            found = [o["start_s"] for o in g.get("occurrences") or []]
            tp, pairs = match_onsets(gt, found, args.tol)
            _, _, score = f1(tp, len(gt), len(found))
            if best is None or score > best[0]:
                best = (score, g, tp, pairs, found)
        score, g, tp, pairs, found = best
        prec, rec, _ = f1(tp, len(gt), len(found))
        roi_iou = iou(ev["roi_norm"], g["roi_norm"]) if "roi_norm" in ev else float("nan")
        mean_dt = sum(d for _, _, d in pairs) / len(pairs) if pairs else float("nan")
        print(f"{name:8s} {g['gid']:>4} {len(gt):>6} {len(found):>6} {tp:>4} "
              f"{prec:>6.2f} {rec:>6.2f} {score:>6.2f} {roi_iou:>8.2f} {mean_dt:>8.2f}s")
        rows.append((name, score))
        claimed.setdefault(g["gid"], []).append(name)

    dupes = {gid: names for gid, names in claimed.items() if len(names) > 1}
    print()
    for gid, names in dupes.items():
        print(f"warning: group {gid} is the best match for several events: {names}")

    passed = [n for n, s in rows if s >= args.min_f1]
    failed = [n for n, s in rows if s < args.min_f1]
    extra = len(groups) - len(set(claimed))
    print(f"pass {len(passed)}/{len(rows)} events at f1>={args.min_f1}"
          + (f"   failed: {', '.join(failed)}" if failed else ""))
    print(f"{extra} unmatched group(s) (false positives or genuinely extra elements)")
    return 0 if not failed and not dupes else 1


if __name__ == "__main__":
    raise SystemExit(main())
