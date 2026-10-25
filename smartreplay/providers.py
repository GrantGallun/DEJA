"""Which model answers replay questions, and where DeepSeek's key comes from.

The provider is a setting (`model.provider` in player-settings.json, edited in
deja) because the best one depends on the network: DeepSeek answered fastest in
our tests, but some networks, campus Wi-Fi especially, block it. Standard
library only, so the settings app can use it without the demo's dependencies.
"""

from __future__ import annotations

import os
from pathlib import Path

PROVIDERS = ("codex", "claude", "deepseek")
DEEPSEEK_KEY_NAMES = ("DEEPSEEK_API_KEY", "DEEPSEEKAPI_KEY")
ENV_FILE = Path(__file__).resolve().parents[1] / ".env"


def deepseek_key(env_file: Path = ENV_FILE) -> str | None:
    """DeepSeek's API key from the environment, else the checkout's .env. Never logged."""
    for name in DEEPSEEK_KEY_NAMES:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    try:
        lines = Path(env_file).read_text(encoding="utf-8-sig").splitlines()
    except OSError:
        return None
    for line in lines:
        name, sep, value = line.partition("=")
        name = name.strip()
        if name.startswith("export "):
            name = name[len("export "):].strip()
        value = value.strip().strip("'\"")
        if sep and name in DEEPSEEK_KEY_NAMES and value:
            return value
    return None
