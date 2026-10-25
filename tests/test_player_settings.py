"""Replay player settings: validation, the file format, and the settings server.

    python -m pytest tests/test_player_settings.py
"""

from __future__ import annotations

import http.client
import json
import os
import re
import sys
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from smartreplay import player_settings as ps  # noqa: E402
from smartreplay.frontend.server import SettingsServer  # noqa: E402


def _paths(issues, level="error"):
    return sorted(i["path"] for i in issues if i["level"] == level)


# --------------------------------------------------------------------- chords

@pytest.mark.parametrize("text, chord", [
    ("Ctrl+Alt+R", "Ctrl+Alt+R"),
    ("alt+ctrl+r", "Ctrl+Alt+R"),
    ("Shift+Ctrl+Alt+f12", "Ctrl+Alt+Shift+F12"),
    ("Ctrl+7", "Ctrl+7"),
])
def test_global_chords_are_canonical(text, chord):
    assert ps.parse_chord(text, global_hotkey=True) == (chord, None)


@pytest.mark.parametrize("text", ["R", "F8", "Win+R", "Alt+F4", "Ctrl+Space", "Ctrl+Alt", "Hyper+R", ""])
def test_global_chords_rejected(text):
    chord, why = ps.parse_chord(text, global_hotkey=True)
    assert chord is None and why


@pytest.mark.parametrize("text, chord", [
    ("Space", "Space"), ("escape", "Esc"), ("ArrowLeft", "Left"), ("shift+left", "Shift+Left"),
    ("[", "["), ("pageup", "PgUp"), ("m", "M"),
])
def test_player_keys_allow_plain_keys(text, chord):
    assert ps.parse_chord(text, global_hotkey=False) == (chord, None)


# --------------------------------------------------------------------- validation

def test_defaults_are_clean():
    settings, issues = ps.normalize(ps.defaults())
    assert issues == []
    assert settings == ps.defaults()


def test_bad_fields_fall_back_and_are_reported():
    raw = ps.defaults()
    raw["layout"]["width"] = 99999
    raw["input"]["mode"] = "telepathy"
    raw["appearance"]["colors"]["accent"] = "red"
    raw["buttons"]["speeds"] = [0.5, 7]
    raw["appearance"]["show_header"] = "yes"
    raw["mystery"] = 1
    settings, issues = ps.normalize(raw)
    assert _paths(issues) == ["appearance.colors.accent", "appearance.show_header", "buttons.speeds",
                              "input.mode", "layout.width"]
    assert _paths(issues, "warning") == ["mystery"]
    assert settings["layout"]["width"] == 536
    assert settings["input"]["mode"] == "hybrid"
    assert "mystery" not in settings


def test_duplicate_and_shadowed_keys_are_errors():
    raw = ps.defaults()
    raw["hotkeys"]["quit"] = "ctrl+alt+r"               # same as toggle_replay
    raw["player_keys"]["mute"] = "Ctrl+Alt+F"           # a global hotkey always wins
    raw["player_keys"]["restart"] = "Space"             # same as play_pause
    _, issues = ps.normalize(raw)
    assert _paths(issues) == ["hotkeys.quit", "player_keys.mute", "player_keys.restart"]


def test_required_hotkeys_depend_on_input_mode():
    raw = ps.defaults()
    raw["hotkeys"]["toggle_controls"] = ""
    assert _paths(ps.normalize(raw)[1]) == ["hotkeys.toggle_controls"]
    raw["input"]["mode"] = "mouse"                      # clicking takes control instead
    assert _paths(ps.normalize(raw)[1]) == []
    raw["hotkeys"]["toggle_replay"] = ""
    assert _paths(ps.normalize(raw)[1]) == ["hotkeys.toggle_replay"]


def test_editing_a_colour_makes_the_theme_custom():
    raw = ps.defaults()
    raw["appearance"]["colors"]["accent"] = "#FF0000"
    settings, issues = ps.normalize(raw)
    assert settings["appearance"]["theme"] == "custom"
    assert settings["appearance"]["colors"]["accent"] == "#ff0000"
    assert _paths(issues) == []


def test_clipped_buttons_and_unreadable_text_warn():
    raw = ps.defaults()
    raw["layout"]["width"] = 360
    raw["appearance"]["colors"]["text"] = raw["appearance"]["colors"]["panel"]
    _, issues = ps.normalize(raw)
    assert _paths(issues) == []
    assert _paths(issues, "warning") == ["appearance.colors.text", "layout.width"]


def test_every_theme_is_readable():
    for name, colors in ps.THEMES.items():
        assert ps.contrast(colors["text"], colors["panel"]) >= 4.5, name


# --------------------------------------------------------------------- file I/O

def test_missing_file_means_defaults(tmp_path):
    assert ps.load(tmp_path / "none.json") == (ps.defaults(), [])


def test_corrupt_file_is_reported_not_raised(tmp_path):
    path = tmp_path / "player-settings.json"
    path.write_text("{ nope", encoding="utf-8")
    settings, issues = ps.load(path)
    assert settings == ps.defaults()
    assert _paths(issues) == [""]


def test_save_bumps_revision_and_refuses_errors(tmp_path):
    path = tmp_path / "SmartReplay" / "player-settings.json"
    first, _ = ps.save(ps.defaults(), path)
    second, _ = ps.save(first, path)
    assert (first["revision"], second["revision"]) == (1, 2)
    before = path.read_bytes()
    bad = ps.defaults()
    bad["hotkeys"]["toggle_replay"] = "R"
    with pytest.raises(ps.SettingsError):
        ps.save(bad, path)
    assert path.read_bytes() == before
    assert [p.name for p in path.parent.iterdir()] == ["player-settings.json"]  # no temp files left


def test_player_status_checks_the_process(tmp_path):
    settings = tmp_path / "player-settings.json"
    status = ps.status_path(settings)
    assert ps.player_status(settings) == {"running": False}
    status.write_text(json.dumps({"pid": os.getpid(), "running": True, "revision": 3}), encoding="utf-8")
    assert ps.player_status(settings)["running"] is True
    status.write_text(json.dumps({"pid": 2 ** 31 - 2, "running": True}), encoding="utf-8")
    assert ps.player_status(settings)["running"] is False


PLAYER_SETTINGS_CPP = ROOT / "experiments" / "gamehq-passive" / "spike" / "passive-replay" / "PlayerSettings.cpp"


@pytest.mark.skipif(not PLAYER_SETTINGS_CPP.exists(), reason="replay player checkout not present")
def test_player_defaults_match():
    """The player falls back to its own copy of the defaults; keep them equal."""
    source = PLAYER_SETTINGS_CPP.read_text(encoding="utf-8")
    literal = re.search(r'kDefaults = R"\((.*?)\)";', source, re.S).group(1)
    assert json.loads(literal) == ps.DEFAULTS


# --------------------------------------------------------------------- server

@pytest.fixture
def server(tmp_path):
    srv = SettingsServer(tmp_path / "player-settings.json", 0)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield srv
    srv.shutdown()
    srv.server_close()


def _request(srv, method, path, body=None, headers=None):
    port = srv.server_address[1]
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    hdrs = {"Host": f"127.0.0.1:{port}"}
    if body is not None:
        hdrs["Content-Type"] = "application/json"
    hdrs.update(headers or {})
    conn.request(method, path, body=None if body is None else json.dumps(body), headers=hdrs)
    res = conn.getresponse()
    data = res.read()
    conn.close()
    kind = res.getheader("Content-Type") or ""
    return res.status, kind, (json.loads(data) if "json" in kind else data)


def test_server_round_trip(server):
    status, _, snap = _request(server, "GET", "/api/settings")
    assert status == 200 and snap["settings"] == ps.defaults() and not snap["exists"]
    draft = snap["settings"]
    draft["input"]["mode"] = "keyboard"
    draft["hotkeys"]["toggle_replay"] = "ctrl+shift+k"
    status, _, checked = _request(server, "POST", "/api/validate", draft)
    assert status == 200 and checked["issues"] == []
    status, _, saved = _request(server, "PUT", "/api/settings", draft)
    assert status == 200
    assert saved["settings"]["revision"] == 1
    assert saved["settings"]["hotkeys"]["toggle_replay"] == "Ctrl+Shift+K"
    on_disk = json.loads(server.settings_file.read_text(encoding="utf-8"))
    assert on_disk["input"]["mode"] == "keyboard"


def test_server_rejects_invalid_saves(server):
    draft = ps.defaults()
    draft["hotkeys"]["toggle_replay"] = "R"
    status, _, body = _request(server, "PUT", "/api/settings", draft)
    assert status == 422
    assert _paths(body["issues"]) == ["hotkeys.toggle_replay"]
    assert not server.settings_file.exists()


def test_server_refuses_other_sites(server):
    """A page elsewhere must not rewrite someone's hotkeys via localhost."""
    assert _request(server, "GET", "/api/settings", headers={"Host": "evil.example:80"})[0] == 403
    assert _request(server, "PUT", "/api/settings", ps.defaults(),
                    headers={"Origin": "https://evil.example"})[0] == 403
    status, _, _ = _request(server, "PUT", "/api/settings", ps.defaults(),
                            headers={"Content-Type": "text/plain"})
    assert status == 400
    assert not server.settings_file.exists()


def test_server_serves_only_its_static_files(server):
    status, kind, body = _request(server, "GET", "/")
    assert status == 200 and kind.startswith("text/html") and b"Smart Replay" in body
    assert _request(server, "GET", "/app.js")[1].startswith("text/javascript")
    assert _request(server, "GET", "/../player_settings.py")[0] == 404
    assert _request(server, "GET", "/%2e%2e/server.py")[0] == 404
