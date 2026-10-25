"""Find the moment a free-form description refers to: by what was said first,
by what is seen only where the words leave it open.

fulfil.py answers requests whose moment the HUD marks - a death screen, a
round banner - by filtering that element's occurrences. Many requests have no
such mark: "the missed sack that turned into a huge gain", "the onside kick
nobody wanted", "when I nearly fell off the bridge". Nothing on screen fires
for them, but somebody usually says something - a commentator, a teammate,
the player - within seconds of it happening.

So the ladder starts from speech (speech.py):

    1 speech   the request vs. the timestamped transcript      1 text call
               -> spans: clear or unclear, narrated or mentioned
    2 vision   frames from each span the words cannot settle    1 call per span
               alone, checked against a question the speech
               step wrote for it
    3 vision   when nothing said fits, frames from candidate     1 call per window,
               windows supplied by the caller (HUD occurrences)  budgeted

Two properties of speech decide where vision is spent:

  Being talked about is not being shown. Play-by-play is spoken while the
  action is on screen, so a clear, narrated span is answered by the words
  alone. A mention - "47-yard kick return" said as a stat while the next play
  lines up - says the moment happened, not that it is on screen; it gets one
  look, and if the picture does not show it, it is dropped with a note
  rather than clipped.

  The words say when to look. Each line of a called quote names one event as
  it happens ("Look out!" / "Escapes the sack." / "Throws." / "Caught at the
  31"), so a check's frames are taken where those lines are spoken, a beat
  earlier for the caller's lag - not spread evenly over a span that also
  covers the replay and the chat after it, where a one-second event falls
  between frames.

Every moment says which rung decided it, and spans nobody could settle come
back as "maybe", not dropped.

    python -m smartreplay.locate footage/bench/x.mp4 "the pick in the end zone" --game "NFL: Patriots at Bears"
"""

from __future__ import annotations

import difflib
import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

from . import speech
from .llm import LLM


@dataclass
class Found:
    t0: float
    t1: float
    verdict: str                 # yes | maybe
    rung: str                    # speech | vision
    evidence: list[str] = field(default_factory=list)


@dataclass
class Location:
    query: str
    found: list[Found]
    model_calls: int
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def called_lines(segs: list[dict], quote: str, t0: float, t1: float) -> list[tuple[float, float, list[str]]]:
    """The transcript lines inside [t0, t1] that say the quote.

    Matched word by word, in order, so a quote the model trimmed or
    paraphrased still lands on the lines it came from.
    """
    ls = [ln for ln in speech.lines(segs, t0, t1) if ln[1] >= t0 and ln[0] <= t1]
    toks =[(tk, i) for i, (_, _, ws) in enumerate(ls) for w in ws for tk in _tokens(w)]
    sm = difflib.SequenceMatcher(None, _tokens(quote), [tk for tk, _ in toks], autojunk=False)
    hit = {toks[b.b + k][1] for b in sm.get_matching_blocks() for k in range(b.size)}
    return [ls[i] for i in sorted(hit)]


def look_times(segs: list[dict], span: dict, call_lag_s: float = 1.0, n_min: int = 3,
               n_max: int = 6) -> list[float]:
    """When to look for a span's moment: where its called lines are spoken.

    call_lag_s is how far a caller trails the action - a free parameter. n_max
    is the image budget of one check. With fewer than n_min called lines (a
    quote that matched nothing, or one long line), the frames spread over
    what was matched, or over the span.
    """
    called = called_lines(segs, span.get("quote", ""), span["t0"], span["t1"])
    times = [st - call_lag_s for st, _, _ in called]
    if len(times) < n_min:
        a = called[0][0] if called else span["t0"]
        b = called[-1][1] if called else span["t1"]
        times = [float(t) for t in np.linspace(a - call_lag_s, max(a, b), n_min)]
    if len(times) > n_max:
        times = [times[i] for i in np.linspace(0, len(times) - 1, n_max).round().astype(int)]
    return times


def frames_at(video: str, times: list[float]) -> list[np.ndarray]:
    cap = cv2.VideoCapture(video)
    out = []
    for t in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, float(t)) * 1000.0)
        ok, f = cap.read()
        if ok:
            out.append(f)
    cap.release()
    return out


def locate(video: str, query: str, game: str, llm: LLM, segs: list[dict] | None = None,
           windows: list[tuple[float, float]] | None = None, max_checks: int = 12,
           workers: int = 4) -> Location:
    segs = speech.load(video) if segs is None else segs
    calls, notes, found = 0, [], []
    question = f"Do these screenshots show this moment: {query}?"

    def ask(times: list[float]) -> dict:
        frames = frames_at(video, times)
        if not frames:
            return {"answer": "unsure", "why": "could not read frames"}
        return llm.check(frames, f"Consecutive screenshots from {game}. {question} "
                                 "Answer yes, no, or unsure.")

    def run_checks(times: list[list[float]]) -> list[dict]:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(ask, times))

    open_: list[dict] = []
    if segs:
        calls += 1
        res = llm.find_in_speech(query, game, speech.transcript(segs))
        question = res.get("visual_question") or question
        for s in res.get("spans", []):
            if s.get("clear") and s.get("narrated", True):
                found.append(Found(s["t0"], s["t1"], "yes", "speech", [f'said: "{s["quote"]}"']))
            else:
                open_.append(s)
    else:
        notes.append("no transcript: speech rung skipped")

    # Spans the words could not settle - unclear ones, and clear mentions that
    # may not be on screen - are alternatives to a narrated answer, not
    # additions, so they are looked at only when no narrated span settled it.
    if open_ and not found:
        todo = open_[:max_checks]
        answers = run_checks([look_times(segs, s) for s in todo])
        calls += len(todo)
        for s, a in zip(todo, answers):
            ev = [f'said: "{s["quote"]}"', f"vision: {a.get('why', '')}"]
            if a.get("answer") == "yes":
                found.append(Found(s["t0"], s["t1"], "yes", "vision", ev))
            elif a.get("answer") == "unsure":
                found.append(Found(s["t0"], s["t1"], "maybe", "vision", ev))
            elif s.get("clear"):
                notes.append(f'mentioned at {s["t0"]:.1f}s ("{s["quote"]}") but not on screen there')
        if len(open_) > max_checks:
            notes.append(f"{len(open_) - max_checks} open span(s) left unchecked (budget)")

    if not found and windows:
        budget = max(0, max_checks - (calls - 1))
        todo = windows[:budget]
        answers = run_checks([list(np.linspace(w[0], w[1], 4)) for w in todo])
        calls += len(todo)
        for w, a in zip(todo, answers):
            if a.get("answer") in ("yes", "unsure"):
                found.append(Found(w[0], w[1], "yes" if a["answer"] == "yes" else "maybe", "vision",
                                   [f"vision (not mentioned in speech): {a.get('why', '')}"]))
        if len(windows) > budget:
            notes.append(f"{len(windows) - budget} candidate window(s) left unchecked (budget)")

    if not found:
        notes.append("nothing said or seen matches")
    if hasattr(llm, "live_calls"):
        calls = llm.live_calls  # cache hits cost nothing; count only real calls
    return Location(query, sorted(found, key=lambda f: f.t0), calls, notes)


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Find the moment a description refers to.")
    ap.add_argument("video")
    ap.add_argument("query")
    ap.add_argument("--game", required=True, help="what the recording is, in a few words")
    ap.add_argument("--out", default=None, help="write the result as JSON here")
    args = ap.parse_args(argv)

    from .llm import default_llm

    loc = locate(args.video, args.query, args.game, default_llm())
    for f in loc.found:
        print(f"  {f.verdict:5s} {f.t0:7.1f}-{f.t1:7.1f}s  [{f.rung}]  {'; '.join(f.evidence)[:110]}")
    print(f"{len(loc.found)} moment(s), {loc.model_calls} model call(s)")
    for n in loc.notes:
        print(f"note: {n}")
    if args.out:
        Path(args.out).write_text(json.dumps(loc.to_dict(), indent=1), encoding="utf-8")
    return 0 if loc.found else 1


if __name__ == "__main__":
    raise SystemExit(main())
