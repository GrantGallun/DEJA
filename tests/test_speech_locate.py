"""Tests for the speech channel (speech.py) and description lookup (locate.py)."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from smartreplay import locate, speech  # noqa: E402
from smartreplay.llm import FakeLLM  # noqa: E402


def _seg(t0: float, words: list[str], step: float = 1.0) -> dict:
    ws = [[w, round(t0 + i * step, 2), round(t0 + i * step + 0.5, 2)] for i, w in enumerate(words)]
    return {"t0": t0, "t1": ws[-1][2], "text": " ".join(words), "words": ws}


# ----------------------------------------------------------------- speech

def test_transcript_splits_long_unpunctuated_segments():
    # Twelve seconds of play-by-play with no punctuation must not collapse to
    # one timestamp: a reader has to be able to place what is said in it.
    seg = _seg(100.0, "throws deep caught at the thirty breaking a tackle still going touchdown".split())
    lines = speech.transcript([seg], resolution_s=4.0).splitlines()
    starts = [float(ln[1:ln.index("]")]) for ln in lines]
    assert len(lines) >= 3
    assert all(b - a <= 5.0 for a, b in zip(starts, starts[1:]))


def test_transcript_breaks_at_sentence_ends():
    seg = _seg(10.0, ["Intercepted!", "What", "a", "play."])
    assert speech.transcript([seg]).splitlines() == ["[10.0] Intercepted!", "[11.0] What a play."]


def test_said_between_cuts_at_word_times():
    seg = _seg(0.0, ["one", "two", "three", "four"])
    assert speech.said_between([seg], 0.9, 2.2) == "two three"


# ----------------------------------------------------------------- locate

def _clip(tmp_path: Path) -> str:
    video = tmp_path / "clip.avi"
    w = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
    for _ in range(600):
        w.write(np.zeros((48, 64, 3), np.uint8))
    w.release()
    return str(video)


SEGS = [_seg(5.0, ["Pass", "intercepted!"]), _seg(30.0, ["Touchdown", "Chicago."])]


def test_clear_speech_needs_no_vision(tmp_path):
    spans = {"spans": [{"t0": 5.0, "t1": 7.0, "clear": True, "quote": "Pass intercepted!", "why": ""}],
             "visual_question": "Is a pass intercepted?"}
    llm = FakeLLM(speech=spans)
    loc = locate.locate(_clip(tmp_path), "the pick", "Test", llm, segs=SEGS)
    assert [(f.t0, f.verdict, f.rung) for f in loc.found] == [(5.0, "yes", "speech")]
    assert [c[0] for c in llm.calls] == ["speech"]


def test_unclear_spans_are_settled_by_vision_with_its_question(tmp_path):
    spans = {"spans": [{"t0": 5.0, "t1": 7.0, "clear": False, "quote": "Pass", "why": ""},
                       {"t0": 30.0, "t1": 32.0, "clear": False, "quote": "Touchdown", "why": ""}],
             "visual_question": "Does the score graphic read 3rd & 17?"}
    asked = []

    def vision(images, q):
        asked.append(q)
        # The two spans' frames differ only in time; answer by call order.
        return {"answer": "yes" if len(asked) == 2 else "no", "box": None, "why": ""}

    llm = FakeLLM(speech=spans, checks=vision)
    loc = locate.locate(_clip(tmp_path), "the 3rd and 17", "Test", llm, segs=SEGS, workers=1)
    assert [(f.t0, f.rung) for f in loc.found] == [(30.0, "vision")]
    assert all("3rd & 17" in q for q in asked)


def test_mention_is_looked_at_and_dropped_when_not_on_screen(tmp_path):
    # "47-yard kick return" said as a stat: the words are clear about what
    # happened, but not that it is being shown.
    spans = {"spans": [{"t0": 30.0, "t1": 32.0, "clear": True, "narrated": False,
                        "quote": "Touchdown Chicago.", "why": ""}],
             "visual_question": "Is a touchdown being scored?"}
    llm = FakeLLM(speech=spans, checks=lambda images, q: {"answer": "no", "box": None, "why": ""})
    loc = locate.locate(_clip(tmp_path), "the Chicago touchdown", "Test", llm, segs=SEGS)
    assert loc.found == []
    assert [c[0] for c in llm.calls] == ["speech", "check"]
    assert any("not on screen" in n for n in loc.notes)


def test_narrated_span_wins_over_mentions_without_looking(tmp_path):
    spans = {"spans": [{"t0": 5.0, "t1": 7.0, "clear": True, "narrated": True, "quote": "Pass intercepted!", "why": ""},
                       {"t0": 30.0, "t1": 32.0, "clear": True, "narrated": False, "quote": "Touchdown", "why": ""}],
             "visual_question": "?"}
    llm = FakeLLM(speech=spans)
    loc = locate.locate(_clip(tmp_path), "the pick", "Test", llm, segs=SEGS)
    assert [(f.t0, f.rung) for f in loc.found] == [(5.0, "speech")]
    assert [c[0] for c in llm.calls] == ["speech"]


def test_frames_are_placed_on_the_called_lines():
    # Five called events a second or so apart inside a 12 s span that also
    # holds the replay chatter: the frames go where the calls are, a beat early.
    segs = [_seg(865.6, ["Look", "out!"], 0.7), _seg(867.1, ["Escapes", "the", "sack."], 0.3),
            _seg(868.3, ["Throws."]), _seg(869.0, ["Caught", "at", "the", "31."], 0.4),
            _seg(872.6, ["Talking", "about", "Ellis", "on", "the", "replay."], 0.5)]
    span = {"t0": 865.6, "t1": 877.0, "quote": "Look out! Escapes the sack. Throws. Caught at the 31."}
    times = locate.look_times(segs, span, call_lag_s=1.0)
    assert times == [864.6, 866.1, 867.3, 868.0]


def test_nothing_said_and_no_windows_is_reported_not_guessed(tmp_path):
    llm = FakeLLM(speech={"spans": [], "visual_question": "?"})
    loc = locate.locate(_clip(tmp_path), "a penalty on the kickoff", "Test", llm, segs=SEGS)
    assert loc.found == []
    assert any("nothing said or seen" in n for n in loc.notes)
