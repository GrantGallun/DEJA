"""Regression tests for the defects found by the audit (audit/findings.json).

Each test names the finding it pins down. Run with:  python -m pytest tests
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from smartreplay import cli, grouping, profile, sampling, variants  # noqa: E402
from tools import minecraft_truth, verify  # noqa: E402


# --------------------------------------------------------------------- verify

def _score(tmp_path, truth, groups):
    (tmp_path / "truth.json").write_text(json.dumps(truth), encoding="utf-8")
    (tmp_path / "events.json").write_text(json.dumps(groups), encoding="utf-8")
    return verify.main([str(tmp_path / "truth.json"), str(tmp_path)])


def _group(gid, onsets):
    return {"gid": gid, "roi_norm": [0, 0, 1, 1],
            "occurrences": [{"start_s": o, "duration_s": 1.0} for o in onsets]}


def test_negative_control_empty_output_passes(tmp_path):
    """P1: truth with no events and no detections is the correct answer."""
    assert _score(tmp_path, {"events": {}}, []) == 0


def test_negative_control_false_detection_fails(tmp_path):
    """P1: any detection on a no-event clip is a false positive."""
    assert _score(tmp_path, {"events": {}}, [_group(0, [1.0, 5.0, 9.0])]) == 1


def test_positive_truth_with_no_detections_fails(tmp_path):
    assert _score(tmp_path, {"events": {"death": {"onsets_s": [1.0, 5.0]}}}, []) == 1


def test_positive_truth_scored(tmp_path):
    truth = {"events": {"death": {"onsets_s": [10.0, 20.0, 30.0]}}}
    assert _score(tmp_path, truth, [_group(0, [10.2, 20.0, 29.5])]) == 0


def test_onset_matching_is_maximal():
    """Greedy nearest matching found 1 pair here; a valid assignment has 2."""
    n, pairs = verify.match_onsets([2.0, 3.0], [1.0, 2.4], 1.1)
    assert n == 2
    assert all(d <= 1.1 for _, _, d in pairs)


def test_onset_matching_empty():
    assert verify.match_onsets([], [1.0], 1.0) == (0, [])
    assert verify.match_onsets([1.0], [], 1.0) == (0, [])


# ------------------------------------------------------------------- grouping

def test_final_occurrence_keeps_full_duration():
    """An occurrence reaching the last sample was shortened by one interval."""
    on = np.zeros((60, 24, 32), dtype=bool)
    mask = np.zeros((24, 32), dtype=bool)
    mask[5:15, 5:15] = True
    for s, e in ((4, 8), (20, 24), (56, 60)):
        on[s:e, mask] = True
    groups = grouping.group_events(on, mask, np.arange(60) / 2, 2)
    assert [o.duration_s for o in groups[0].occurrences] == [2.0, 2.0, 2.0]


def test_rejected_candidates_do_not_exhaust_group_limit():
    """P2: 24 rejected candidates ahead of a valid one used to hide it."""
    on = np.zeros((1106, 60, 60), dtype=bool)
    mask = np.zeros((60, 60), dtype=bool)
    for index in range(24):
        y, x = 1 + (index // 6) * 10, 1 + (index % 6) * 10
        mask[y:y + 8, x:x + 8] = True
        base = index * 44
        # Recurring, but with inconsistent durations, so validation rejects them.
        for s, e in ((base + 2, base + 6), (base + 8, base + 12), (base + 14, base + 38)):
            on[s:e, y:y + 8, x:x + 8] = True
    mask[51:57, 1:7] = True
    for s, e in ((1060, 1064), (1078, 1082), (1098, 1102)):
        on[s:e, 51:57, 1:7] = True
    groups = grouping.group_events(on, mask, np.arange(1106) / 2, 2)
    assert len(groups) == 1
    assert [o.start_s for o in groups[0].occurrences] == [530.0, 539.0, 549.0]


def test_max_groups_caps_confirmed_output():
    on = np.zeros((200, 40, 120), dtype=bool)
    mask = np.zeros((40, 120), dtype=bool)
    for k in range(5):
        x = 2 + k * 23
        mask[5:25, x:x + 18] = True
        for rep in range(3):
            s = 10 + rep * 60 + k * 8
            on[s:s + 4, 5:25, x:x + 18] = True
    times = np.arange(200) / 2
    assert len(grouping.group_events(on, mask, times, 2)) == 5
    capped = grouping.group_events(on, mask, times, 2, grouping.GroupParams(max_groups=2))
    assert len(capped) == 2


# -------------------------------------------------------------------- profile

def _one_group():
    on = np.zeros((60, 24, 32), dtype=bool)
    mask = np.zeros((24, 32), dtype=bool)
    mask[6:18, 8:24] = True
    for s, e in ((4, 8), (20, 24), (40, 44)):
        on[s:e, mask] = True
    groups = grouping.group_events(on, mask, np.arange(60) / 2, 2)
    result = type("R", (), {"static_mask": np.zeros_like(mask), "event_mask": mask})()
    return groups, result


def test_profile_records_letterbox_crop():
    groups, result = _one_group()
    meta = {"source": "letterboxed.mp4", "size": [32, 24], "native_size": [96, 72],
            "crop": [16, 12, 112, 84], "frame_size": [128, 96],
            "sample_fps": 2, "duration_s": 29.5}
    gv = [variants.GroupVariants(g.gid, [], []) for g in groups]
    prof, _ = profile.build(meta, result, groups, gv, np.zeros((60, 24, 32), np.uint8))
    la = prof["learned_at"]
    assert la["crop"] == [16, 12, 112, 84]
    assert la["frame_size"] == [128, 96]
    assert prof["detectors"][0]["roi_norm_frame"] == pytest.approx(
        [0.3125, 0.3125, 0.6875, 0.6875], abs=1e-4)


def test_empty_profile_removes_stale_templates(tmp_path):
    profile.save({"detectors": [{"id": "d0"}]}, {"d0_e0": np.ones((4, 4), np.uint8)}, tmp_path)
    assert (tmp_path / "templates.npz").exists()
    profile.save({"detectors": []}, {}, tmp_path)
    assert not (tmp_path / "templates.npz").exists()


# ------------------------------------------------------------ cache identity

def _write_video(path: Path, seed: int) -> None:
    codec = "MJPG" if path.suffix == ".avi" else "mp4v"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), 10, (64, 48))
    rng = np.random.default_rng(seed)
    for _ in range(200):
        w.write(rng.integers(32, 255, (48, 64, 3), dtype=np.uint8))
    w.release()


def test_same_stem_inputs_get_distinct_names(tmp_path):
    """P2: match.avi and match.mp4 used to share one output directory."""
    _write_video(tmp_path / "match.avi", 1)
    _write_video(tmp_path / "match.mp4", 2)
    _write_video(tmp_path / "solo.mp4", 3)
    names = {cli.clip_name(tmp_path / "match.avi"), cli.clip_name(tmp_path / "match.mp4")}
    assert names == {"match-avi", "match-mp4"}
    assert cli.clip_name(tmp_path / "solo.mp4") == "solo"


def test_cache_rejects_other_source(tmp_path):
    """P2: --reuse-samples used to load another video's stack."""
    a, b = tmp_path / "a.mp4", tmp_path / "b.mp4"
    _write_video(a, 1)
    _write_video(b, 2)
    out = tmp_path / "out"
    params = sampling.sampling_params(2.0, 64, 192, 600.0, 0.0, 0.0)
    sampling.sample_video(a, out, sample_fps=2.0, width=64)
    assert sampling.cache_matches(out, a, params)[0]
    ok, why = sampling.cache_matches(out, b, params)
    assert not ok and "not this file" in why
    other = sampling.sampling_params(2.0, 128, 192, 600.0, 0.0, 0.0)
    assert not sampling.cache_matches(out, a, other)[0]


def test_cli_reuse_resamples_on_mismatch(tmp_path, capsys):
    a = tmp_path / "clip.mp4"
    _write_video(a, 1)
    out = tmp_path / "out"
    base = ["discover", str(a), "--out", str(out), "--width", "64", "--no-fullres",
            "--min-event-s", "0.5"]
    assert cli.main(base) == 0
    capsys.readouterr()
    assert cli.main(base + ["--reuse-samples"]) == 0
    out1 = capsys.readouterr().out
    assert "[sample]   reusing" in out1 and "not reusing" not in out1
    assert cli.main(base + ["--reuse-samples", "--fps", "4"]) == 0
    assert "not reusing: cache was sampled with different settings" in capsys.readouterr().out
    meta = json.loads((out / "clip" / "meta.json").read_text(encoding="utf-8"))
    assert Path(meta["fingerprint"]["path"]) == a.resolve()


# ------------------------------------------------------------ minecraft truth

def _log(tmp_path, lines):
    p = tmp_path / "latest.log"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def test_minecraft_player_name_matches_exactly(tmp_path):
    """P2: local player Alex used to collect Alexander's deaths."""
    log = _log(tmp_path, [
        "[12:00:00] [Render thread/INFO]: Setting user: Alex",
        "[12:00:10] [Render thread/INFO]: [CHAT] Alexander was slain by Zombie",
        "[12:00:20] [Render thread/INFO]: [CHAT] Alex was slain by Skeleton",
    ])
    deaths, _, player = minecraft_truth.parse_log(log, datetime(2026, 10, 24, 12), 60)
    assert player == "Alex"
    assert [t for _, t in deaths] == ["Alex was slain by Skeleton"]


def test_minecraft_ignores_player_chat(tmp_path):
    log = _log(tmp_path, [
        "[12:00:00] [Render thread/INFO]: Setting user: Alex",
        "[12:00:10] [Render thread/INFO]: [CHAT] <Alex> lol I almost died",
    ])
    deaths, _, _ = minecraft_truth.parse_log(log, datetime(2026, 10, 24, 12), 60)
    assert deaths == []


def test_minecraft_recording_after_midnight(tmp_path):
    """P2: a clip starting 00:00:10 lost deaths when the log began before midnight."""
    log = _log(tmp_path, [
        "[23:59:50] [Render thread/INFO]: Setting user: Alex",
        "[00:00:30] [Render thread/INFO]: [CHAT] Alex was slain by Zombie",
    ])
    deaths, _, _ = minecraft_truth.parse_log(log, datetime(2026, 10, 24, 0, 0, 10), 60)
    assert [o for o, _ in deaths] == [20.0]


def test_minecraft_recording_spanning_midnight(tmp_path):
    log = _log(tmp_path, [
        "[23:59:00] [Render thread/INFO]: Setting user: Alex",
        "[23:59:40] [Render thread/INFO]: [CHAT] Alex drowned",
        "[00:00:20] [Render thread/INFO]: [CHAT] Alex was slain by Zombie",
    ])
    deaths, _, _ = minecraft_truth.parse_log(log, datetime(2026, 10, 23, 23, 59, 30), 120)
    assert [o for o, _ in deaths] == [10.0, 50.0]
