"""Transcribe what is said in a recording, with word timings.

What happened is often *said* before it is shown anywhere: a commentator
calls "picked off at the goal line", a teammate on voice chat shouts "he's
one shot", a streamer narrates "I'm burning". Words spoken within seconds of
a moment are the cheapest evidence a free-form request can be matched
against - text against text, no frames decoded, no vision call - so speech
is read first and the picture is looked at only where the words leave a
request open.

Nothing here knows what is being said or by whom. The transcript is a list
of segments with per-word times, cached next to the video like OCR's
.ocr.jsonl:

    {"t0": 81.4, "t1": 84.9, "text": "Intercepted at the goal line!",
     "words": [["Intercepted", 81.4, 82.0], ["at", 82.0, 82.1], ...]}

    python -m smartreplay.speech footage/bench/x.mp4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

DEFAULT_MODEL = os.environ.get("SMARTREPLAY_SPEECH_MODEL", "large-v3-turbo")


def speech_path(video: str | Path) -> Path:
    return Path(video).with_suffix(".speech.jsonl")


def _cuda_dlls() -> None:
    """CTranslate2 needs cuBLAS and cuDNN at load time; on Windows the usual
    place a machine already has them is a CUDA build of torch."""
    if sys.platform != "win32":
        return
    try:
        import torch  # noqa: F401 - only for the DLLs it ships
    except ImportError:
        return
    lib = Path(torch.__file__).parent / "lib"
    if lib.is_dir():
        os.add_dll_directory(str(lib))
        os.environ["PATH"] = str(lib) + os.pathsep + os.environ.get("PATH", "")


def _model(name: str):
    from faster_whisper import WhisperModel  # pip install faster-whisper

    _cuda_dlls()
    try:
        return WhisperModel(name, device="cuda", compute_type="float16")
    except (RuntimeError, ValueError, OSError):
        return WhisperModel(name, device="cpu", compute_type="int8")


def transcribe(video: str | Path, model: str = DEFAULT_MODEL, out: str | Path | None = None) -> list[dict]:
    from faster_whisper.audio import decode_audio
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    m = _model(model)
    # Speech is found first and each stretch of it is decoded where it is.
    # Decoding the stretches glued end to end (faster-whisper's vad_filter)
    # lets one utterance run across a join, so a two-second voice line
    # ("Spike down B") comes back timed to two places 47 s apart. Game audio
    # is mostly short lines between long silences, so joins are everywhere.
    # Music beds and crowd-only stretches, where Whisper hallucinates
    # captions, are never decoded at all.
    sr = 16000
    audio = decode_audio(str(video), sampling_rate=sr)
    clips = [x for r in get_speech_timestamps(audio, VadOptions()) for x in (r["start"] / sr, r["end"] / sr)]
    parts = (m.transcribe(audio, word_timestamps=True, clip_timestamps=clips,
                          condition_on_previous_text=False)[0] if clips else [])
    segs = []
    for s in parts:
        words = [[w.word.strip(), round(w.start, 2), round(w.end, 2)] for w in (s.words or []) if w.word.strip()]
        segs.append({"t0": round(s.start, 2), "t1": round(s.end, 2), "text": s.text.strip(), "words": words})
    path = Path(out) if out else speech_path(video)
    with path.open("w", encoding="utf-8") as f:
        for s in segs:
            f.write(json.dumps(s) + "\n")
    return segs


def load(video: str | Path) -> list[dict] | None:
    path = speech_path(video)
    if not path.exists():
        return None
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def said_between(segs: list[dict], t0: float, t1: float) -> str:
    """The words spoken inside [t0, t1], cut at word boundaries."""
    words = [w[0] for s in segs if s["t1"] >= t0 and s["t0"] <= t1
             for w in s["words"] if w[2] >= t0 and w[1] <= t1]
    return " ".join(words)


def lines(segs: list[dict], t0: float = 0.0, t1: float = float("inf"),
          resolution_s: float = 4.0) -> list[tuple[float, float, list[str]]]:
    """What was said, as (start, end, words) lines.

    A recogniser segment can run for half a minute of unbroken play-by-play,
    often with no punctuation at all, and one start time for all of it says
    nothing about where inside it a moment is. So lines are rebuilt from word
    timings: a line ends at a sentence end, or at the next word once it
    covers `resolution_s` - the one free parameter, how finely a reader of
    the transcript can place a moment.
    """
    out: list[tuple[float, float, list[str]]] = []
    for s in segs:
        if s["t1"] < t0 or s["t0"] > t1:
            continue
        words = s["words"] or [[s["text"], s["t0"], s["t1"]]]
        cur: list[str] = []
        start = words[0][1]
        for w, ws, we in words:
            if not cur:
                start = ws
            cur.append(w)
            if w[-1:] in ".?!" or ws - start >= resolution_s:
                out.append((start, we, cur))
                cur = []
        if cur:
            out.append((start, words[-1][2], cur))
    return out


def transcript(segs: list[dict], t0: float = 0.0, t1: float = float("inf"),
               resolution_s: float = 4.0) -> str:
    """Timestamped lines for a prompt: '[81.4] Intercepted at the goal line!'"""
    return "\n".join(f"[{st:.1f}] {' '.join(ws)}" for st, _, ws in lines(segs, t0, t1, resolution_s))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    segs = transcribe(args.video, args.model, args.out)
    n = sum(len(s["words"]) for s in segs)
    print(f"{len(segs)} segment(s), {n} word(s) -> {args.out or speech_path(args.video)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
