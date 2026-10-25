"""The only place SmartReplay talks to a language model.

Everything model-shaped goes through three narrow calls, each returning a
fixed structure (via a forced tool call, so the model cannot answer in prose):

    plan(query, game, catalog)      -> RequestSpec fields   (text in, ~1 call per request)
    label_groups(game, groups)      -> {group id: concept}  (once per game, cached)
    check(images, question)         -> yes / no / unsure + box  (vision, per candidate)
    find_in_speech(query, game, transcript) -> time spans + how clear  (text, per request)

Keeping the model's job this small is what makes the rest deterministic,
testable and fast: the model turns a request into structure, and plain code
executes the structure. All three are cached on disk by a hash of their
inputs, so a repeated request or a re-run costs nothing.

The default model is the fastest one that can still read a request and an
image well: Claude Haiku 5.5 (`claude-haiku-5-5`). Override with
SMARTREPLAY_MODEL. Tests use FakeLLM, which replays canned answers.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np

DEFAULT_MODEL = os.environ.get("SMARTREPLAY_MODEL", "claude-haiku-5-5")
CACHE_DIR = Path(os.environ.get("SMARTREPLAY_CACHE", Path.home() / ".smartreplay" / "llm"))

PLAN_TOOL = {
    "name": "request_spec",
    "description": "The structured form of a replay request.",
    "input_schema": {
        "type": "object",
        "properties": {
            "anchor": {"type": "string",
                       "description": "The kind of moment to clip, as a short snake_case concept: "
                                      "death, kill, round_end, round_start, item_pickup, level_up, ..."},
            "filters": {
                "type": "array",
                "description": "ALL filters must hold (AND). Alternatives (fire OR lava) go in ONE "
                               "filter's fact_values and text_terms, never in separate filters.",
                "items": {
                    "type": "object",
                    "properties": {
                        "concept": {"type": "string", "description": "snake_case, e.g. fire_damage"},
                        "description": {"type": "string",
                                        "description": "One sentence a person (or a vision model) could check "
                                                       "against a screenshot."},
                        "include": {"type": "boolean", "description": "false = exclude these moments"},
                        "text_terms": {"type": "array", "items": {"type": "string"},
                                       "description": "Exact phrases this game prints on screen when the "
                                                      "filter is true (death messages, kill-feed wording, "
                                                      "banners). Empty if the game shows no text for it."},
                        "visual": {"type": "boolean",
                                   "description": "Whether the filter can be seen in the frame without text."},
                        "fact_key": {"type": ["string", "null"],
                                     "enum": ["cause", "enemy", "weapon", "item", "place", "outcome", None],
                                     "description": "Which recorded fact of a moment decides this filter "
                                                    "(e.g. cause for 'burned to death'), or null."},
                        "fact_values": {"type": "array", "items": {"type": "string"},
                                        "description": "Values of that fact that satisfy the filter. Copy "
                                                       "them from the recorded values listed in the prompt "
                                                       "when possible; '<any>' means any recorded value."},
                        "null_is_no": {"type": "boolean",
                                       "description": "true if a moment with no value for fact_key fails "
                                                      "the filter (no enemy recorded = not killed by a mob)."},
                    },
                    "required": ["concept", "description", "include", "text_terms", "visual",
                                 "fact_key", "fact_values", "null_is_no"],
                },
            },
            "anchor_text_terms": {"type": "array", "items": {"type": "string"},
                                  "description": "Case-specific phrases this game prints for the anchor, "
                                                 "covering all its variants (for a death: every death-"
                                                 "message form, e.g. 'drowned', 'was slain by'), not "
                                                 "generic words like 'died'. Lets code tell 'a different "
                                                 "case was read' apart from 'nothing was read'."},
            "before_s": {"type": "number", "description": "seconds to include before each moment"},
            "after_s": {"type": "number", "description": "seconds to include after each moment"},
            "limit": {"type": ["integer", "null"], "description": "only the most recent N, or null"},
            "ambiguity": {"type": "string",
                          "description": "Anything in the request that could reasonably mean two things."},
        },
        "required": ["anchor", "filters", "anchor_text_terms", "before_s", "after_s", "limit",
                     "ambiguity"],
    },
}

LABEL_TOOL = {
    "name": "group_labels",
    "description": "Name each discovered HUD element.",
    "input_schema": {
        "type": "object",
        "properties": {
            "labels": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "group": {"type": "integer"},
                        "concept": {"type": "string", "description": "snake_case event concept, or 'unknown'"},
                        "role": {"type": "string", "enum": ["event", "feed", "state", "menu", "other"],
                                 "description": "event: each appearance is one moment of one kind "
                                                "(death screen, round banner). feed: a place where "
                                                "messages of many kinds appear (chat, kill feed). "
                                                "state: an ongoing condition (boss bar, spectating). "
                                                "menu: a menu or pause screen."},
                        "confidence": {"type": "number"},
                        "why": {"type": "string"},
                    },
                    "required": ["group", "concept", "role", "confidence", "why"],
                },
            }
        },
        "required": ["labels"],
    },
}

CHECK_TOOL = {
    "name": "visual_check",
    "description": "Answer a yes/no question about the screenshots.",
    "input_schema": {
        "type": "object",
        "properties": {
            "answer": {"type": "string", "enum": ["yes", "no", "unsure"]},
            "box": {"type": ["array", "null"], "items": {"type": "number"},
                    "description": "[x0, y0, x1, y1] in 0..1 of the evidence in the first image, or null"},
            "why": {"type": "string"},
            "evidence_text": {"type": ["string", "null"],
                              "description": "The on-screen phrase your answer rests on, minus player "
                                             "names (e.g. 'was shot by arrow'), or null if none."},
        },
        "required": ["answer", "box", "why", "evidence_text"],
    },
}


FACT_KEYS = ["cause", "enemy", "weapon", "item", "place", "outcome"]

DESCRIBE_TOOL = {
    "name": "moment_facts",
    "description": "What happened at one moment of gameplay, for later search.",
    "input_schema": {
        "type": "object",
        "properties": {
            "what_happened": {"type": "string", "description": "One sentence."},
            "on_screen_text": {"type": "array", "items": {"type": "string"},
                               "description": "Every message, banner or kill-feed line about this "
                                              "moment, verbatim (fix obvious misreads)."},
            "facts": {
                "type": "object",
                "description": "Short lowercase values; null when not shown or not applicable.",
                "properties": {
                    "cause": {"type": ["string", "null"], "description": "what caused it, e.g. fire, "
                              "lava, fall, void, arrow, explosion, drowning, melee"},
                    "enemy": {"type": ["string", "null"], "description": "who or what did it"},
                    "weapon": {"type": ["string", "null"]},
                    "item": {"type": ["string", "null"], "description": "item gained or used"},
                    "place": {"type": ["string", "null"]},
                    "outcome": {"type": ["string", "null"], "description": "e.g. death, kill, win, loss"},
                },
                "required": FACT_KEYS,
            },
        },
        "required": ["what_happened", "on_screen_text", "facts"],
    },
}


SPEECH_TOOL = {
    "name": "speech_spans",
    "description": "Where in the transcript a request's moment is talked about.",
    "input_schema": {
        "type": "object",
        "properties": {
            "spans": {
                "type": "array",
                "description": "Every stretch of speech the request could refer to, best first. "
                               "Empty if nothing said fits.",
                "items": {
                    "type": "object",
                    "properties": {
                        "t0": {"type": "number", "description": "seconds: when the moment begins "
                                                                "being described"},
                        "t1": {"type": "number", "description": "seconds: when that description ends"},
                        "clear": {"type": "boolean",
                                  "description": "true only if the words alone settle that this is "
                                                 "the requested moment; false if it is a plausible "
                                                 "candidate the words cannot confirm"},
                        "narrated": {"type": "boolean",
                                     "description": "true if the words call the moment while it is "
                                                    "being shown - play-by-play, or talking over its "
                                                    "replay; false if they only refer to it (a stat, a "
                                                    "recap, a mention of something earlier or "
                                                    "off-screen)"},
                        "quote": {"type": "string", "description": "the words the span rests on"},
                        "why": {"type": "string"},
                    },
                    "required": ["t0", "t1", "clear", "narrated", "quote", "why"],
                },
            },
            "visual_question": {
                "type": "string",
                "description": "One yes/no question a person could answer from screenshots of a "
                               "candidate moment to tell whether it is the requested one. Name what "
                               "would be visible (an action, a score or clock graphic, a banner).",
            },
        },
        "required": ["spans", "visual_question"],
    },
}


def plan_tool(catalog: list[str]) -> dict:
    """The plan schema, with the anchor constrained to this recording's names."""
    tool = json.loads(json.dumps(PLAN_TOOL))
    names = [c.split(" (")[0] for c in catalog]
    if names:
        tool["input_schema"]["properties"]["anchor"]["enum"] = sorted(set(names)) + ["none"]
    return tool


class LLM(Protocol):
    def plan(self, query: str, game: str, catalog: list[str], facts: str = "") -> dict: ...
    def label_groups(self, game: str, groups: list[dict]) -> dict: ...
    def check(self, images: list[np.ndarray], question: str) -> dict: ...


def _jpeg_b64(img: np.ndarray, max_side: int = 768) -> str:
    h, w = img.shape[:2]
    s = min(1.0, max_side / max(h, w))
    if s < 1.0:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf.tobytes()).decode("ascii")


class ClaudeLLM:
    """Anthropic API client with an on-disk cache keyed by the full request."""

    def __init__(self, model: str = DEFAULT_MODEL, cache_dir: Path = CACHE_DIR):
        import anthropic  # pip install anthropic

        self.client = anthropic.Anthropic()
        self.live_calls = 0
        self.model = model
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _call(self, tool: dict, content: list[dict], max_tokens: int = 1500) -> dict:
        key = hashlib.sha256(json.dumps([self.model, tool, content], sort_keys=True)
                             .encode()).hexdigest()
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        self.live_calls += 1
        msg = self.client.messages.create(
            model=self.model, max_tokens=max_tokens, tools=[tool],
            tool_choice={"type": "tool", "name": tool["name"]},
            messages=[{"role": "user", "content": content}],
        )
        out = next(b.input for b in msg.content if getattr(b, "type", "") == "tool_use")
        path.write_text(json.dumps(out), encoding="utf-8")
        return out

    def plan(self, query: str, game: str, catalog: list[str], facts: str = "") -> dict:
        """catalog: "concept (role)" strings; the anchor is constrained to them.
        facts: the values actually recorded at this recording's moments."""
        tool = plan_tool(catalog)
        text = (f"Game: {game}\nPlayer request: {query!r}\n"
                f"HUD elements found in this recording, with their role: {catalog or 'none yet'}.\n"
                "Pick the anchor from those names: prefer an 'event' whose every appearance is the "
                "kind of moment asked for (a death screen for deaths); use a 'feed' only when no event "
                "fits; 'none' if nothing does. Text anywhere on screen near the moment is read "
                "anyway, so the anchor does not need to be where the text appears. Default window: "
                "5 s before and 1 s after each moment unless the request asks for something else. "
                + (f"\nFacts recorded at this recording's moments (value x count):\n{facts}\n"
                   "Prefer deciding filters by fact_key/fact_values chosen from these exact values. "
                   if facts else "")
                + 
                "For text_terms, list the game's own on-screen wording (exact phrases), including "
                "every variant that should count.")
        return self._call(tool, [{"type": "text", "text": text}])

    def label_groups(self, game: str, groups: list[dict]) -> dict:
        text = (f"Game: {game}\nThese HUD elements were discovered in a recording without any "
                "game knowledge. For each: screen region (0..1), how often and how long it appeared, "
                "and OCR text seen around its appearances (noisy).\n"
                + json.dumps(groups, indent=1))
        return self._call(LABEL_TOOL, [{"type": "text", "text": text}])

    def describe(self, images: list[np.ndarray], game: str, kind: str) -> dict:
        content: list[dict[str, Any]] = [
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                         "data": _jpeg_b64(im)}} for im in images]
        content.append({"type": "text", "text":
                        f"Consecutive screenshots from {game} around one '{kind}' moment. Record what "
                        "happened: read every on-screen message about it and list searchable facts."})
        return self._call(DESCRIBE_TOOL, content, max_tokens=600)

    def check(self, images: list[np.ndarray], question: str) -> dict:
        content: list[dict[str, Any]] = [
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                         "data": _jpeg_b64(im)}} for im in images]
        content.append({"type": "text", "text": question})
        return self._call(CHECK_TOOL, content, max_tokens=400)

    def find_in_speech(self, query: str, game: str, transcript: str) -> dict:
        text = (f"Recording: {game}\nRequest: {query!r}\n"
                "Below is what is said in the recording (commentary, voice chat or narration), one "
                "line per utterance, each with its start time in seconds. Speech recognition is "
                "noisy: names are often misspelled or split, so match by sound and by meaning, not "
                "spelling. A moment is usually described as it happens and sometimes again on a "
                "replay; one span should cover both. The request may be worded as a fan or player "
                "would say it, as a reaction or opinion rather than a description - work out which "
                "moment it is about. Mark a span clear only if the words identify the moment "
                "beyond reasonable doubt; when several moments could fit and the words cannot tell "
                "them apart, return each as unclear. Being talked about is not being shown: mark a "
                "span narrated only when the words call the action as it plays (or over its "
                "replay), not when they cite it as a stat or recap. Quote the words verbatim from "
                "the transcript.\n\n" + transcript)
        return self._call(SPEECH_TOOL, [{"type": "text", "text": text}], max_tokens=1500)


class _Worker:
    """One long-lived `claude -p` process speaking stream-json.

    Starting `claude` costs ~3-4 s before the model sees a token (process
    start, auth, loading Claude Code's own prompt). A worker pays that once;
    after that a text call takes ~2-3 s and an image check ~1.5-2 s, because
    images go inline as base64 content instead of through file reads.
    """

    def __init__(self, binary: str, model: str, schema: dict):
        import subprocess

        self.calls = 0
        self.proc = subprocess.Popen(
            [binary, "-p", "--input-format", "stream-json", "--output-format", "stream-json",
             "--verbose", "--model", model, "--json-schema", json.dumps(schema), "--tools", "",
             "--no-session-persistence", "--disable-slash-commands"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, bufsize=1)

    def ask(self, content: list[dict]) -> dict | None:
        self.calls += 1
        msg = {"type": "user", "message": {"role": "user", "content": content}}
        self.proc.stdin.write(json.dumps(msg) + "\n")
        self.proc.stdin.flush()
        while True:
            line = self.proc.stdout.readline()
            if not line:
                return None
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("type") == "result":
                return d.get("structured_output")

    def close(self) -> None:
        try:
            self.proc.stdin.close()
            self.proc.terminate()
        except Exception:
            pass


class WarmPool:
    """Pre-started workers per answer shape, recycled before context builds up.

    A worker's conversation grows with each call, which costs tokens and lets
    one moment's screenshots linger into the next question. So each worker
    answers at most `max_calls` questions and is then replaced - by a worker
    that was already started in the background, so no caller ever waits on a
    cold start. Call start() when the app launches and the first request is
    as fast as the hundredth.
    """

    def __init__(self, binary: str, model: str, size: int = 4, max_calls: int = 6):
        import queue
        import threading

        self.binary, self.model, self.size, self.max_calls = binary, model, size, max_calls
        self._idle: dict[str, "queue.Queue[_Worker]"] = {}
        self._schemas: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._queue_mod = queue

    def _key(self, schema: dict) -> str:
        return hashlib.sha1(json.dumps(schema, sort_keys=True).encode()).hexdigest()[:12]

    def _spawn(self, key: str) -> None:
        import threading

        def go():
            self._idle[key].put(_Worker(self.binary, self.model, self._schemas[key]))
        threading.Thread(target=go, daemon=True).start()

    def start(self, schemas: list[dict], sizes: list[int] | None = None) -> None:
        for i, schema in enumerate(schemas):
            key = self._key(schema)
            with self._lock:
                if key in self._idle:
                    continue
                self._idle[key] = self._queue_mod.Queue()
                self._schemas[key] = schema
            for _ in range((sizes or [self.size] * len(schemas))[i]):
                self._spawn(key)

    def ask(self, schema: dict, content: list[dict], timeout: float = 180.0) -> dict | None:
        key = self._key(schema)
        if key not in self._idle:
            self.start([schema], [1])
        w = self._idle[key].get(timeout=timeout)
        try:
            out = w.ask(content)
        except Exception:
            out = None
        if out is None or w.calls >= self.max_calls:
            w.close()
            self._spawn(key)          # replacement starts now, in the background
        else:
            self._idle[key].put(w)
        return out

    def close(self) -> None:
        for q in self._idle.values():
            while not q.empty():
                q.get_nowait().close()


class ClaudeCLILLM(ClaudeLLM):
    """Same three calls, through the `claude` command line instead of an API key.

    Uses whoever is signed in to Claude Code on this machine, so it needs no
    key - convenient for development. Each call is a fresh `claude -p`
    process with tools disabled (or only Read, to look at images) and the
    tool's input schema passed as --json-schema, so the answer comes back
    validated, as `structured_output`.

    It is slower than the API: every process start carries Claude Code's own
    system prompt, so a call takes ~5-9 s instead of ~1-2 s. Checks run in
    parallel and every answer is cached, which hides most of that. For a
    shipped product, use the API (each user signing in their own Claude Code
    is not a distribution model).
    """

    def __init__(self, model: str = DEFAULT_MODEL, cache_dir: Path = CACHE_DIR,
                 binary: str = "claude", timeout: float = 180.0, warm: bool = True,
                 pool_size: int = 4):
        import shutil

        self.binary = shutil.which(binary) or binary
        self.model, self.timeout = model, timeout
        self.live_calls = 0
        self.pool = WarmPool(self.binary, model, size=pool_size) if warm else None
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _call(self, tool: dict, content: list[dict], max_tokens: int = 1500) -> dict:
        import subprocess
        import tempfile

        key = hashlib.sha256(json.dumps(["cli", self.model, tool, content], sort_keys=True)
                             .encode()).hexdigest()
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))

        if self.pool is not None:
            # Each question stands alone even though the worker remembers
            # earlier ones; say so, so earlier screenshots are not consulted.
            fresh = [{"type": "text", "text": "New, independent question. Ignore all earlier "
                                              "messages and images."}] + content
            for _ in range(2):
                self.live_calls += 1
                out = self.pool.ask(tool["input_schema"], fresh, timeout=self.timeout)
                if out is not None:
                    path.write_text(json.dumps(out), encoding="utf-8")
                    return out

        with tempfile.TemporaryDirectory(prefix="smartreplay_") as tmp:
            texts, images = [], []
            for part in content:
                if part["type"] == "text":
                    texts.append(part["text"])
                else:
                    img = Path(tmp) / f"image_{len(images)}.jpg"
                    img.write_bytes(base64.b64decode(part["source"]["data"]))
                    images.append(str(img))
            prompt = "\n\n".join(texts)
            if images:
                prompt = ("Look at these screenshots, in order: " + ", ".join(images) + "\n\n" + prompt)
            # The prompt goes straight after -p: the list-valued flags below
            # would otherwise swallow it as one more value.
            cmd = [self.binary, "-p", prompt, "--model", self.model, "--output-format", "json",
                   "--json-schema", json.dumps(tool["input_schema"])]
            cmd += ["--tools", "Read", "--allowedTools", "Read"] if images else ["--tools", ""]
            out, err = None, ""
            for _ in range(2):  # one retry: transient CLI/auth/network hiccups happen
                self.live_calls += 1
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout, cwd=tmp)
                try:
                    payload = json.loads(proc.stdout)
                    out = payload.get("structured_output")
                    err = payload.get("result", "")
                except json.JSONDecodeError:
                    err = (proc.stdout + proc.stderr)[:300]
                if out is not None:
                    break
        if out is None:
            raise RuntimeError(f"claude CLI returned no structured output: {err[:300]}")
        path.write_text(json.dumps(out), encoding="utf-8")
        return out


    def preload(self) -> None:
        """Start warm workers for every call shape now, before any request."""
        if self.pool is not None:
            self.pool.start([PLAN_TOOL["input_schema"], LABEL_TOOL["input_schema"],
                             CHECK_TOOL["input_schema"], DESCRIBE_TOOL["input_schema"],
                             SPEECH_TOOL["input_schema"]],
                            [1, 1, self.pool.size, self.pool.size, 1])


class FakeLLM:
    """Replays canned answers; records every call. For tests and offline runs."""

    def __init__(self, plan: dict | None = None, labels: dict | None = None, checks=None,
                 speech: dict | None = None):
        self._plan, self._labels, self._speech = plan, labels or {"labels": []}, speech
        self._checks = checks or (lambda images, q: {"answer": "unsure", "box": None, "why": "fake"})
        self.calls: list[tuple] = []

    def plan(self, query, game, catalog, facts=""):
        self.calls.append(("plan", query))
        return self._plan

    def label_groups(self, game, groups):
        self.calls.append(("label", game))
        return self._labels

    def check(self, images, question):
        self.calls.append(("check", question))
        return self._checks(images, question)

    def describe(self, images, game, kind):
        self.calls.append(("describe", kind))
        return {"what_happened": "", "on_screen_text": [], "facts": {}}

    def find_in_speech(self, query, game, transcript):
        self.calls.append(("speech", query))
        return self._speech or {"spans": [], "visual_question": query}


def default_llm() -> LLM:
    """API when a key is set (fastest); otherwise the signed-in `claude` CLI."""
    import shutil

    backend = os.environ.get("SMARTREPLAY_BACKEND", "")
    if backend == "api" or (not backend and os.environ.get("ANTHROPIC_API_KEY")):
        return ClaudeLLM()
    if backend == "cli" or shutil.which("claude"):
        return ClaudeCLILLM()
    raise RuntimeError("no model backend: set ANTHROPIC_API_KEY or install and sign in to `claude`")
