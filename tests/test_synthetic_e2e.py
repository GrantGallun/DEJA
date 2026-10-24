"""End-to-end regression on the synthetic clip: generate, discover, score.

Slow (generates and decodes a 120 s 720p clip), so opt in with RUN_SLOW=1:

    RUN_SLOW=1 python -m pytest tests/test_synthetic_e2e.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools import verify  # noqa: E402

pytestmark = pytest.mark.skipif(os.environ.get("RUN_SLOW") != "1", reason="set RUN_SLOW=1")


def test_synthetic_clip_scores(tmp_path):
    clip = tmp_path / "synthetic.mp4"
    subprocess.run([sys.executable, str(ROOT / "tools/make_test_clip.py"), str(clip)],
                   check=True, cwd=ROOT)
    subprocess.run([sys.executable, "-m", "smartreplay.cli", "discover", str(clip),
                    "--out", str(tmp_path / "out")], check=True, cwd=ROOT)
    groups = json.loads((tmp_path / "out/synthetic/events.json").read_text(encoding="utf-8"))
    truth = json.loads(clip.with_suffix(".truth.json").read_text(encoding="utf-8"))

    for name, floor in (("death", 1.0), ("kill", 1.0), ("round", 0.8)):
        gt = truth["events"][name]["onsets_s"]
        best = 0.0
        for g in groups:
            found = [o["start_s"] for o in g["occurrences"]]
            tp, _ = verify.match_onsets(gt, found, 1.5)
            best = max(best, verify.f1(tp, len(gt), len(found))[2])
        assert best >= floor, f"{name}: f1 {best:.2f} < {floor}"
