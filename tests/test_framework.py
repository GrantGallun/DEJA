"""Tests for request fulfilment (fulfil.py) and exemplar verification (verify_images.py)."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from smartreplay import fulfil, verify_images as vi  # noqa: E402
from smartreplay.llm import FakeLLM  # noqa: E402


# ------------------------------------------------------------ text matching

def test_phrase_in_tolerates_ocr_errors():
    assert fulfil.phrase_in("tried to swim in lava", "GreatKiller tried to suin in lava")
    assert fulfil.phrase_in("went up in flames", "Greathiller went up in flarnes")
    assert not fulfil.phrase_in("burned to death", "GreatKiller fell out of the world")


def test_text_verdict_three_valued():
    yes, other = ["went up in flames"], ["fell out of the world", "drowned"]
    assert fulfil.text_verdict(["X went up in flames"], yes, other)[0] == "yes"
    assert fulfil.text_verdict(["X drowned"], yes, other)[0] == "no"
    assert fulfil.text_verdict(["Respawn", "Score: 10"], yes, other)[0] == "unknown"


# ------------------------------------------------------------- fulfilment

def _index(tmp_path: Path) -> dict:
    video = tmp_path / "clip.avi"
    w = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
    for _ in range(600):
        w.write(np.zeros((48, 64, 3), np.uint8))
    w.release()
    occ = [
        {"id": "g1#0", "group": 1, "t": 5.0, "duration": 3, "roi": [0, 0, 1, 1], "text": ["P went up in flames"]},
        {"id": "g1#1", "group": 1, "t": 20.0, "duration": 3, "roi": [0, 0, 1, 1], "text": ["P drowned"]},
        {"id": "g1#2", "group": 1, "t": 35.0, "duration": 3, "roi": [0, 0, 1, 1], "text": ["Respawn"]},
        {"id": "g1#3", "group": 1, "t": 50.0, "duration": 3, "roi": [0, 0, 1, 1], "text": ["P burned to a crisp"]},
        {"id": "g0#0", "group": 0, "t": 10.0, "duration": 2, "roi": [0, 0, .2, .1], "text": ["chat"]},
    ]
    return {"video": str(video), "groups": [{"group": 0, "roi": [0, 0, .2, .1], "n": 1, "mean_duration": 2},
                                            {"group": 1, "roi": [0, 0, 1, 1], "n": 4, "mean_duration": 3}],
            "occurrences": occ}


PLAN = {"anchor": "death", "before_s": 5, "after_s": 1, "limit": None, "ambiguity": "",
        "anchor_text_terms": ["went up in flames", "drowned", "fell out of the world", "burned to a crisp"],
        "filters": [{"concept": "fire_death", "description": "Did the player die by fire?",
                     "include": True, "text_terms": ["went up in flames"], "visual": True}]}
LABELS = {"labels": [{"group": 1, "concept": "death", "confidence": 0.9, "why": ""},
                     {"group": 0, "concept": "chat", "confidence": 0.6, "why": ""}]}


def test_ladder_uses_vision_only_for_undecided(tmp_path):
    def vision(images, q):
        return {"answer": "no", "box": None, "why": "no fire visible"}

    llm = FakeLLM(PLAN, LABELS, vision)
    mem = fulfil.Memory(tmp_path / "game.json", tmp_path / "rec.json")
    f = fulfil.Fulfiller(_index(tmp_path), "TestGame", llm, mem)
    res = f.fulfil(fulfil.RequestSpec.from_plan(PLAN))
    verdicts = {m.id: m.verdict for m in res.moments}
    assert res.anchor_group == 1
    assert verdicts["g1#0"] == "yes"           # text: the filter's own phrase
    assert verdicts["g1#1"] == "no"            # text: a different death message
    assert verdicts["g1#3"] == "no"            # text: anchor phrase not in the filter list
    assert verdicts["g1#2"] == "no"            # nothing read -> vision said no
    assert [c[0] for c in llm.calls].count("check") == 1
    assert [c[0] for c in llm.calls].count("label") == 1


def test_second_request_is_free_and_vision_teaches_text(tmp_path):
    calls = {"n": 0}

    def vision(images, q):
        calls["n"] += 1
        return {"answer": "yes", "box": None, "why": "flames on screen"}

    idx = _index(tmp_path)
    idx["occurrences"][2]["text"] = ["P was roasted by a blaze"]   # unknown to the plan's vocabulary
    plan = dict(PLAN, anchor_text_terms=PLAN["anchor_text_terms"] + ["was roasted by"])
    plan["filters"] = [dict(PLAN["filters"][0])]
    llm = FakeLLM(plan, LABELS, vision)
    mem = fulfil.Memory(tmp_path / "game.json", tmp_path / "rec.json")
    fulfil.Fulfiller(idx, "TestGame", llm, mem).fulfil(fulfil.RequestSpec.from_plan(plan))
    # "was roasted by" is an anchor phrase, so text said "no" and vision was never asked.
    assert calls["n"] == 0

    idx["occurrences"][2]["text"] = ["Respawn"]                    # nothing readable this time
    mem = fulfil.Memory(tmp_path / "game.json", tmp_path / "rec2.json")
    f = fulfil.Fulfiller(idx, "TestGame", llm, mem)
    res = f.fulfil(fulfil.RequestSpec.from_plan(plan))
    assert calls["n"] == 1 and {m.id: m.verdict for m in res.moments}["g1#2"] == "yes"

    # Same recording again: labels and the check come from memory - no model calls.
    mem2 = fulfil.Memory(tmp_path / "game.json", tmp_path / "rec2.json")
    f2 = fulfil.Fulfiller(idx, "TestGame", llm, mem2)
    f2.fulfil(fulfil.RequestSpec.from_plan(plan))
    assert f2.calls == 0 and calls["n"] == 1


def test_missing_anchor_is_reported_not_guessed(tmp_path):
    llm = FakeLLM(dict(PLAN, anchor="round_end"), LABELS)
    mem = fulfil.Memory(tmp_path / "g.json", tmp_path / "r.json")
    res = fulfil.Fulfiller(_index(tmp_path), "TestGame", llm, mem).fulfil(
        fulfil.RequestSpec.from_plan(dict(PLAN, anchor="round_end")))
    assert res.anchor_group is None and res.moments == []


# ---------------------------------------------------------- verify_images

def _scene(rng, w=1280, h=720, element=True, style="real", scale=1.0):
    img = cv2.resize(rng.integers(0, 255, (9, 16, 3), dtype=np.uint8), (w, h), interpolation=cv2.INTER_CUBIC)
    for _ in range(40):  # scenery: every screenshot's background is different
        p1 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        p2 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        cv2.line(img, p1, p2, tuple(int(v) for v in rng.integers(0, 255, 3)), int(rng.integers(1, 6)))
    if element:
        cx, cy = w // 2, int(h * 0.3)
        bw, bh = int(360 * scale), int(90 * scale)
        cv2.rectangle(img, (cx - bw // 2, cy - bh // 2), (cx + bw // 2, cy + bh // 2), (30, 30, 30), -1)
        text = "YOU DIED" if style == "real" else "GAME OVER"
        font = cv2.FONT_HERSHEY_SIMPLEX if style == "real" else cv2.FONT_HERSHEY_SCRIPT_COMPLEX
        cv2.putText(img, text, (cx - int(150 * scale), cy + int(18 * scale)), font, 1.6 * scale,
                    (240, 240, 240), max(2, int(4 * scale)), cv2.LINE_AA)
    return img


def test_verify_images_gates(tmp_path):
    rng = np.random.default_rng(3)
    good = [_scene(rng, scale=s) for s in (0.95, 1.0, 1.05, 1.0, 0.9, 1.1)]
    wrong_style = [_scene(rng, style="mod"), _scene(rng, style="mod")]
    tiny = [cv2.resize(good[0], (320, 180))]
    meme = [_scene(rng, element=False)]
    images = good + wrong_style + tiny + meme + [good[1].copy()]
    box = [0.33, 0.22, 0.67, 0.38]

    says_yes = {id(im) for im in good + wrong_style + tiny}

    def vision(imgs, q):
        # A fallible vision model: yes to the real element *and* to the
        # modded look-alike (gate 3 must catch that), no to the meme.
        has = id(imgs[0]) in says_yes or any(imgs[0] is g for g in good)
        return {"answer": "yes" if has else "no", "box": box if has else None, "why": ""}

    negatives = [_scene(rng, element=False) for _ in range(40)]
    det = vi.build_detector(images, "TestGame", "death_screen", "the death screen", FakeLLM(checks=vision),
                            negatives)
    rej = det.report["rejected"]
    assert "too small" in rej[8] and rej[10] == "duplicate"
    assert rej[9].startswith("vision: no")
    assert rej[6] == rej[7] == "doesn't look like the other exemplars"
    assert det.trusted, det.report
    held_out = [_scene(rng, scale=1.0) for _ in range(5)]
    assert all(vi.score(im, det) >= det.threshold for im in held_out)
    assert all(vi.score(im, det) < det.threshold for im in [_scene(rng, element=False) for _ in range(10)])


def test_untrustworthy_detector_is_flagged():
    rng = np.random.default_rng(4)
    images = [_scene(rng, element=False) for _ in range(5)]
    llm = FakeLLM(checks=lambda imgs, q: {"answer": "yes", "box": [0.3, 0.2, 0.7, 0.4], "why": ""})
    det = vi.build_detector(images, "TestGame", "x", "x", llm, [_scene(rng, element=False) for _ in range(30)])
    assert not det.trusted


def test_same_fact_inclusions_mean_or():
    plan = dict(PLAN, filters=[
        {"concept": "fire", "description": "fire", "include": True, "text_terms": ["went up in flames"],
         "visual": True, "fact_key": "cause", "fact_values": ["fire"], "null_is_no": True},
        {"concept": "lava", "description": "lava", "include": True, "text_terms": ["tried to swim in lava"],
         "visual": True, "fact_key": "cause", "fact_values": ["lava"], "null_is_no": True}])
    spec = fulfil.RequestSpec.from_plan(plan)
    assert len(spec.filters) == 1 and spec.filters[0].fact_values == ["fire", "lava"]
    facts = {"facts": {"cause": "lava"}}
    assert fulfil.fact_verdict(facts, spec.filters[0])[0] == "yes"
    assert fulfil.fact_verdict({"facts": {"cause": "fall"}}, spec.filters[0])[0] == "no"
    assert fulfil.fact_verdict({"facts": {"cause": None}}, spec.filters[0])[0] == "no"
