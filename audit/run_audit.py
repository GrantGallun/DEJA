"""Independent, non-mutating checks of the SmartReplay prototype.

Run with a Python environment containing requirements.txt. Production modules
are copied to snapshot/ before importing them. All fixtures and results stay
under audit/; failures intentionally document unresolved product defects.
"""

from __future__ import annotations

import contextlib
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
SNAPSHOT = HERE / "snapshot"
RESULTS = HERE / "results"
RESULTS.mkdir(parents=True, exist_ok=True)
manifest = {}
for directory in ("smartreplay", "tools"):
    target = SNAPSHOT / directory
    target.mkdir(parents=True, exist_ok=True)
    for source in (ROOT / directory).glob("*.py"):
        payload = source.read_bytes()
        (target / source.name).write_bytes(payload)
        manifest[str(source.relative_to(ROOT))] = hashlib.sha256(payload).hexdigest()

sys.path.insert(0, str(SNAPSHOT))
import cv2
import numpy as np
from smartreplay import discovery, grouping, profile, sampling, variants
from tools import minecraft_truth, verify

checks = []


def check(name, source, expected, action):
    before = time.perf_counter()
    try:
        observed, details = action()
        passed = observed == expected
    except Exception:
        observed, details, passed = "exception", traceback.format_exc(), False
    row = {"name": name, "source": source, "passed": passed,
           "expected": expected, "observed": observed, "details": details,
           "elapsed_s": round(time.perf_counter() - before, 3)}
    checks.append(row)
    print(f"{'PASS' if passed else 'FAIL'} {name}: {observed!r} (expected {expected!r})", flush=True)


def score_case(name, truth, groups):
    folder = RESULTS / name
    folder.mkdir(exist_ok=True)
    truth_path = folder / "truth.json"
    truth_path.write_text(json.dumps(truth), encoding="utf-8")
    (folder / "events.json").write_text(json.dumps(groups), encoding="utf-8")
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        code = verify.main([str(truth_path), str(folder)])
    return code, output.getvalue()


def baseline_score():
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        code = verify.main([str(ROOT / "footage/synthetic.truth.json"), str(ROOT / "out/synthetic")])
    return code, output.getvalue()


def player_prefix():
    log = RESULTS / "prefix.log"
    log.write_text("[12:00:00] [Render thread/INFO]: Setting user: Alex\n"
                   "[12:00:10] [Render thread/INFO]: [CHAT] Alexander was slain by Zombie\n", encoding="utf-8")
    deaths, _, player = minecraft_truth.parse_log(log, datetime(2026, 10, 24, 12), 60)
    return len(deaths), {"detected_player": player, "deaths": deaths}


def midnight():
    log = RESULTS / "midnight.log"
    log.write_text("[23:59:50] [Render thread/INFO]: Setting user: Alex\n"
                   "[00:00:30] [Render thread/INFO]: [CHAT] Alex was slain by Zombie\n", encoding="utf-8")
    deaths, _, _ = minecraft_truth.parse_log(log, datetime(2026, 10, 24, 0, 0, 10), 60)
    return [offset for offset, _ in deaths], {"expected_death_offset_s": 20, "deaths": deaths}


def boundary_duration():
    on = np.zeros((60, 24, 32), dtype=bool)
    mask = np.zeros((24, 32), dtype=bool)
    mask[5:15, 5:15] = True
    for start, end in ((4, 8), (20, 24), (56, 60)):
        on[start:end, mask] = True
    groups = grouping.group_events(on, mask, np.arange(60) / 2, 2)
    return [occ.duration_s for occ in groups[0].occurrences], groups[0].to_dict()


def onset_matching():
    count, pairs = verify.match_onsets([2.0, 3.0], [1.0, 2.4], 1.1)
    return count, {"greedy_pairs": pairs, "valid_two_match_assignment": [[2.0, 1.0], [3.0, 2.4]]}


def crop_geometry():
    on = np.zeros((60, 24, 32), dtype=bool)
    mask = np.zeros((24, 32), dtype=bool)
    mask[6:18, 8:24] = True
    for start, end in ((4, 8), (20, 24), (40, 44)):
        on[start:end, mask] = True
    groups = grouping.group_events(on, mask, np.arange(60) / 2, 2)
    fake_result = type("Result", (), {"static_mask": np.zeros_like(mask), "event_mask": mask})()
    meta = {"source": "letterboxed.mp4", "size": [32, 24], "native_size": [96, 72],
            "crop": [16, 12, 112, 84], "sample_fps": 2, "duration_s": 29.5}
    gv = [variants.GroupVariants(g.gid, [], []) for g in groups]
    learned, _ = profile.build(meta, fake_result, groups, gv, np.zeros((60, 24, 32), dtype=np.uint8))
    has_transform = "crop" in learned or "crop" in learned["learned_at"]
    return has_transform, {"input_crop": meta["crop"], "profile": learned,
                           "roi_on_original_128x96_frame": [0.3125, 0.3125, 0.6875, 0.6875]}


def stale_templates():
    folder = RESULTS / "templates"
    folder.mkdir(exist_ok=True)
    profile.save({"detectors": [{"id": "d0"}]}, {"d0_e0": np.ones((4, 4), dtype=np.uint8)}, folder)
    profile.save({"detectors": []}, {}, folder)
    return (folder / "templates.npz").exists(), "Second profile has zero detectors; the previous template archive remains."


def candidate_budget():
    on = np.zeros((1106, 60, 60), dtype=bool)
    mask = np.zeros((60, 60), dtype=bool)
    for index in range(24):
        y, x = 1 + (index // 6) * 10, 1 + (index % 6) * 10
        mask[y:y + 8, x:x + 8] = True
        base = index * 44
        # These candidates pass pixel-stage recurrence/mean-duration rules,
        # but grouping rejects their inconsistent occurrence durations.
        for start, end in ((base + 2, base + 6), (base + 8, base + 12), (base + 14, base + 38)):
            on[start:end, y:y + 8, x:x + 8] = True
    mask[51:57, 1:7] = True
    for start, end in ((1060, 1064), (1078, 1082), (1098, 1102)):
        on[start:end, 51:57, 1:7] = True
    times = np.arange(1106) / 2
    limited = grouping.group_events(on, mask, times, 2)
    control = grouping.group_events(on, mask, times, 2, grouping.GroupParams(max_groups=25))
    return len(limited), {"default_max_groups_24": [g.to_dict() for g in limited],
                           "control_max_groups_25": [g.to_dict() for g in control]}


def generate_small_video(path, seed):
    path.parent.mkdir(exist_ok=True)
    codec = "MJPG" if path.suffix == ".avi" else "mp4v"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), 10, (64, 48))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create {path}")
    rng = np.random.default_rng(seed)
    for _ in range(100):
        writer.write(rng.integers(32, 255, (48, 64, 3), dtype=np.uint8))
    writer.release()


def run_cli(*args):
    completed = subprocess.run([sys.executable, "-m", "smartreplay.cli", *map(str, args)],
                               cwd=SNAPSHOT, capture_output=True, text=True, timeout=90)
    if completed.returncode:
        raise RuntimeError(completed.stdout + completed.stderr)
    return completed.stdout


fixtures = RESULTS / "same_stem_inputs"
fixtures.mkdir(exist_ok=True)
generate_small_video(fixtures / "match.avi", 1)
generate_small_video(fixtures / "match.mp4", 2)
batch_out = RESULTS / "same_stem_out"


def same_stem():
    output = run_cli("batch", fixtures, "--out", batch_out, "--width", 64, "--no-fullres")
    index = json.loads((batch_out / "index.json").read_text(encoding="utf-8"))
    return len({row["out_dir"] for row in index}), {"index": index, "stdout": output}


def wrong_source_reuse():
    requested = fixtures / "match.avi"
    output = run_cli("discover", requested, "--out", batch_out, "--width", 64,
                     "--no-fullres", "--reuse-samples")
    meta = json.loads((batch_out / "match/meta.json").read_text(encoding="utf-8"))
    return Path(meta["source"]).resolve() == requested.resolve(), {
        "requested_source": str(requested), "actual_cached_source": meta["source"], "stdout": output}


check("Existing synthetic timing score", "tools/verify.py", 0, baseline_score)
check("No-event clip with no detections passes", "tools/verify.py:69", 0,
      lambda: score_case("negative_empty", {"events": {}}, []))
check("No-event clip with a false detection fails", "tools/verify.py:109", 1,
      lambda: score_case("negative_false_positive", {"events": {}},
                         [json.loads((ROOT / "out/synthetic/events.json").read_text(encoding="utf-8"))[0]]))
check("Minecraft player name matches exactly", "tools/minecraft_truth.py:133", 0, player_prefix)
check("Minecraft recording after midnight aligns earlier log lines", "tools/minecraft_truth.py:110", [20.0], midnight)
check("Final occurrence keeps a full sample duration", "smartreplay/grouping.py:295", [2.0, 2.0, 2.0], boundary_duration)
check("Onset matching maximizes valid matches", "tools/verify.py:25", 2, onset_matching)
check("Standalone profile preserves letterbox transform", "smartreplay/profile.py:137", True, crop_geometry)
check("Empty profile removes stale template archive", "smartreplay/profile.py:159", False, stale_templates)
check("Rejected candidate does not exhaust confirmed-group limit", "smartreplay/grouping.py:167", 1, candidate_budget)
check("Batch files with the same stem keep distinct output folders", "smartreplay/cli.py:44", 2, same_stem)
check("Sample reuse rejects a cache from another input video", "smartreplay/cli.py:50", True, wrong_source_reuse)

current = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
           for directory in ("smartreplay", "tools") for path in (ROOT / directory).glob("*.py")}
report = {"created_at": datetime.now(timezone.utc).isoformat(), "snapshot_sha256": manifest,
          "production_sources_still_match_snapshot": current == manifest,
          "summary": {"checks": len(checks), "passed": sum(row["passed"] for row in checks),
                      "failed": sum(not row["passed"] for row in checks)}, "checks": checks}
(RESULTS / "audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report["summary"]), flush=True)
print(f"Full evidence: {RESULTS / 'audit.json'}", flush=True)
raise SystemExit(1 if report["summary"]["failed"] else 0)
