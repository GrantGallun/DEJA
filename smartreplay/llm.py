"""The only place SmartReplay talks to a language model.

Everything model-shaped goes through three narrow calls, each returning a
fixed structure (via a forced tool call, so the model cannot answer in prose):

    plan(query, game, catalog)      -> RequestSpec fields   (text in, ~1 call per request)
    label_groups(game, groups)      -> {group id: concept}  (once per game, cached)
    check(images, question)         -> yes / no / unsure + box  (vision, per candidate)

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
                    },
                    "required": ["concept", "description", "include", "text_terms", "visual"],
                },
            },
            "anchor_text_terms": {"type": "array", "items": {"type": "string"},
                                  "description": "Phrases this game prints for the anchor in general, "
                                                 "covering all its variants (for a death: every death-"
                                                 "message form). Lets code tell 'a different cause was "
                                                 "read' apart from 'nothing was read'."},
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
                        "confidence": {"type": "number"},
                        "why": {"type": "string"},
                    },
                    "required": ["group", "concept", "confidence", "why"],
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
        },
        "required": ["answer", "box", "why"],
    },
}


class LLM(Protocol):
    def plan(self, query: str, game: str, catalog: list[str]) -> dict: ...
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
        self.model = model
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _call(self, tool: dict, content: list[dict], max_tokens: int = 1500) -> dict:
        key = hashlib.sha256(json.dumps([self.model, tool["name"], content], sort_keys=True)
                             .encode()).hexdigest()
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        msg = self.client.messages.create(
            model=self.model, max_tokens=max_tokens, tools=[tool],
            tool_choice={"type": "tool", "name": tool["name"]},
            messages=[{"role": "user", "content": content}],
        )
        out = next(b.input for b in msg.content if getattr(b, "type", "") == "tool_use")
        path.write_text(json.dumps(out), encoding="utf-8")
        return out

    def plan(self, query: str, game: str, catalog: list[str]) -> dict:
        text = (f"Game: {game}\nPlayer request: {query!r}\n"
                f"Event concepts this recording already has detectors for: {catalog or 'none yet'}.\n"
                "Prefer one of those as the anchor when it fits. Turn the request into a request_spec. "
                "For text_terms, list the game's own on-screen wording (exact phrases), including "
                "every variant that should count.")
        return self._call(PLAN_TOOL, [{"type": "text", "text": text}])

    def label_groups(self, game: str, groups: list[dict]) -> dict:
        text = (f"Game: {game}\nThese HUD elements were discovered in a recording without any "
                "game knowledge. For each: screen region (0..1), how often and how long it appeared, "
                "and OCR text seen around its appearances (noisy).\n"
                + json.dumps(groups, indent=1))
        return self._call(LABEL_TOOL, [{"type": "text", "text": text}])

    def check(self, images: list[np.ndarray], question: str) -> dict:
        content: list[dict[str, Any]] = [
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                         "data": _jpeg_b64(im)}} for im in images]
        content.append({"type": "text", "text": question})
        return self._call(CHECK_TOOL, content, max_tokens=400)


class FakeLLM:
    """Replays canned answers; records every call. For tests and offline runs."""

    def __init__(self, plan: dict | None = None, labels: dict | None = None, checks=None):
        self._plan, self._labels = plan, labels or {"labels": []}
        self._checks = checks or (lambda images, q: {"answer": "unsure", "box": None, "why": "fake"})
        self.calls: list[tuple] = []

    def plan(self, query, game, catalog):
        self.calls.append(("plan", query))
        return self._plan

    def label_groups(self, game, groups):
        self.calls.append(("label", game))
        return self._labels

    def check(self, images, question):
        self.calls.append(("check", question))
        return self._checks(images, question)


def default_llm() -> LLM:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("set ANTHROPIC_API_KEY (or pass an LLM explicitly)")
    return ClaudeLLM()
