"""The model provider is a setting: deja picks it, the demo follows it, DeepSeek is one choice."""
from __future__ import annotations

import io
import json
import time
import urllib.error
from types import SimpleNamespace

import numpy as np
import pytest

from smartreplay import player_settings as ps
from smartreplay import providers
from smartreplay.demo import search, server


# ---------------------------------------------------------------- setting

def test_provider_defaults_to_codex_and_rejects_unknown_names():
    assert ps.defaults()["model"]["provider"] == "codex"
    raw = ps.defaults()
    raw["model"]["provider"] = "deepseek"
    assert ps.normalize(raw) == (raw, [])
    raw["model"]["provider"] = "some-other-api"
    settings, issues = ps.normalize(raw)
    assert [i["path"] for i in issues] == ["model.provider"]
    assert settings["model"]["provider"] == "codex"


# ---------------------------------------------------------------- DeepSeek key

def test_deepseek_key_prefers_the_environment_then_the_env_file(tmp_path, monkeypatch):
    for name in providers.DEEPSEEK_KEY_NAMES:
        monkeypatch.delenv(name, raising=False)
    env = tmp_path / ".env"
    assert providers.deepseek_key(env) is None
    env.write_text('# local secrets\nOTHER=1\nexport DEEPSEEKAPI_KEY="sk-file"\n', encoding="utf-8")
    assert providers.deepseek_key(env) == "sk-file"
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-env")
    assert providers.deepseek_key(env) == "sk-env"


# ---------------------------------------------------------------- DeepSeek model

def _reply(content, finish="stop"):
    body = {"choices": [{"finish_reason": finish, "message": {"content": json.dumps(content)}}]}
    return io.BytesIO(json.dumps(body).encode())


def test_deepseek_sends_frames_as_images_and_returns_the_answer(tmp_path, monkeypatch):
    sent = {}

    def urlopen(request, timeout):
        sent.update(url=request.full_url, auth=request.get_header("Authorization"),
                    body=json.loads(request.data), timeout=timeout)
        return _reply({"matches": [{"time_s": 1.25, "verdict": "yes", "evidence": "yellow falls"}], "note": ""})

    monkeypatch.setattr(search.urllib.request, "urlopen", urlopen)
    model = search.DeepSeekVisionModel(tmp_path / "model", key="sk-test")
    answer = model.locate("yellow falls", [(1.25, np.zeros((90, 160, 3), np.uint8))], 4)
    assert answer["matches"][0]["time_s"] == 1.25
    assert sent["url"] == "https://api.deepseek.com/chat/completions" and sent["auth"] == "Bearer sk-test"
    user = sent["body"]["messages"][1]["content"]
    assert any(p["type"] == "image_url" and p["image_url"]["url"].startswith("data:image/jpeg;base64,") for p in user)
    assert sent["body"]["response_format"] == {"type": "json_object"}


@pytest.mark.parametrize("failure, words", [
    (urllib.error.URLError("blocked"), "campus Wi-Fi"),
    (urllib.error.HTTPError("u", 401, "no", {}, None), "HTTP 401"),
])
def test_deepseek_failures_point_to_switching_provider(tmp_path, monkeypatch, failure, words):
    def urlopen(request, timeout):
        raise failure
    monkeypatch.setattr(search.urllib.request, "urlopen", urlopen)
    model = search.DeepSeekVisionModel(tmp_path / "model", key="sk-test")
    with pytest.raises(RuntimeError) as caught:
        model.check()
    assert words in str(caught.value) and "deja" in str(caught.value)
    assert "sk-test" not in str(caught.value)


def test_deepseek_without_a_key_says_where_to_put_one(tmp_path, monkeypatch):
    monkeypatch.setattr(search, "deepseek_key", lambda: None)
    with pytest.raises(RuntimeError, match="DEEPSEEK_API_KEY"):
        search.DeepSeekVisionModel(tmp_path / "model")


def test_create_model_knows_every_provider_setting():
    for name in providers.PROVIDERS:
        assert name in ("codex", "claude", "deepseek")
    with pytest.raises(ValueError, match="DeepSeek"):
        search.create_model("nope", None)


# ---------------------------------------------------------------- demo follows the setting

class IdleNative:
    def __init__(self, *args):
        pass

    def status(self):
        return {"available": False, "enabled": False, "recording": False}

    def close(self):
        pass


@pytest.fixture
def fake_models(monkeypatch):
    built = []

    def factory(backend, directory, *, transport=None):
        built.append(backend)
        return SimpleNamespace(check=lambda: None, label=f"fake {backend}", transport=None, _owns_transport=False)

    monkeypatch.setattr(server, "create_model", factory)
    monkeypatch.delenv("SMARTREPLAY_DEMO_BACKEND", raising=False)
    return built


def _wait(condition, seconds=5.0):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.05)
    return False


def test_demo_starts_on_the_saved_provider_and_follows_new_saves(tmp_path, fake_models):
    settings_file = tmp_path / "player-settings.json"
    saved = ps.defaults()
    saved["model"]["provider"] = "claude"
    ps.save(saved, settings_file)
    session = server.DemoSession(tmp_path / "demo", native_factory=IdleNative, settings_file=settings_file)
    try:
        assert session.backend == "claude"
        saved["model"]["provider"] = "deepseek"
        time.sleep(0.05)  # a distinct modification time
        ps.save(saved, settings_file)
        assert _wait(lambda: session.backend == "deepseek")
        assert _wait(lambda: session.snapshot()["model"]["state"] == "connected")  # rechecked after the switch
        assert "deepseek" in fake_models
    finally:
        session.close()


def test_the_demo_picker_writes_the_setting_back(tmp_path, fake_models):
    settings_file = tmp_path / "player-settings.json"
    ps.save(ps.defaults(), settings_file)
    session = server.DemoSession(tmp_path / "demo", native_factory=IdleNative, settings_file=settings_file)
    try:
        session.set_backend("deepseek")
        assert ps.load(settings_file)[0]["model"]["provider"] == "deepseek"
    finally:
        session.close()


def test_without_a_settings_file_the_demo_keeps_its_old_default(tmp_path, fake_models):
    session = server.DemoSession(tmp_path / "demo", native_factory=IdleNative)
    try:
        assert session.backend == "codex"
        session.set_backend("deepseek")  # nothing to write
        assert session.backend == "deepseek"
    finally:
        session.close()
