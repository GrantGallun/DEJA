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
    ("7", "7"),
    ("8", "8"),
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


def test_push_to_talk_defaults_to_nine_and_can_be_unbound(tmp_path):
    raw = ps.defaults()
    assert raw["hotkeys"]["push_to_talk"] == "9"
    raw["hotkeys"]["push_to_talk"] = ""
    path = tmp_path / "settings.json"
    ps.save(raw, path)
    assert ps.load(path)[0]["hotkeys"]["push_to_talk"] == ""


def test_push_to_talk_conflicts_are_rejected():
    raw = ps.defaults()
    raw["hotkeys"]["push_to_talk"] = raw["hotkeys"]["toggle_controls"]
    assert "hotkeys.push_to_talk" in _paths(ps.normalize(raw)[1])


def test_replay_takes_controls_on_open_by_default_and_can_be_disabled(tmp_path):
    raw = ps.defaults()
    assert raw["input"]["take_controls_on_open"] is True
    raw["input"]["take_controls_on_open"] = False
    path = tmp_path / "settings.json"
    ps.save(raw, path)
    assert ps.load(path)[0]["input"]["take_controls_on_open"] is False


@pytest.mark.parametrize("value", [1, 0, "true", None])
def test_replay_auto_controls_requires_boolean(value):
    raw = ps.defaults()
    raw["input"]["take_controls_on_open"] = value
    clean, issues = ps.normalize(raw)
    assert "input.take_controls_on_open" in _paths(issues)
    assert clean["input"]["take_controls_on_open"] is True


def test_recent_replay_start_defaults_and_persists(tmp_path):
    raw = ps.defaults()
    assert raw["playback"]["start_behind_live_s"] == 5
    raw["playback"]["start_behind_live_s"] = 12
    path = tmp_path / "settings.json"
    ps.save(raw, path)
    assert ps.load(path)[0]["playback"]["start_behind_live_s"] == 12


@pytest.mark.parametrize("value", [0, -1, 301, 1.5, True, "5"])
def test_recent_replay_start_rejects_invalid_values(value):
    raw = ps.defaults()
    raw["playback"] = {"start_behind_live_s": value}
    settings, issues = ps.normalize(raw)
    assert "playback.start_behind_live_s" in _paths(issues)
    assert settings["playback"]["start_behind_live_s"] == 5


def test_number_row_replay_shortcuts_round_trip(tmp_path):
    raw = ps.defaults()
    raw["hotkeys"]["toggle_replay"] = "7"
    raw["hotkeys"]["toggle_controls"] = "8"
    path = tmp_path / "player-settings.json"
    saved, _ = ps.save(raw, path)
    loaded, issues = ps.load(path)
    assert _paths(issues) == []
    assert loaded["hotkeys"] == saved["hotkeys"] == raw["hotkeys"]


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


def test_defaults_are_minimal():
    settings = ps.defaults()
    assert settings["appearance"]["controls"] == "minimal"
    assert settings["appearance"]["show_header"] is False
    assert settings["layout"]["width"] * 9 / 16 == pytest.approx(settings["layout"]["height"], abs=0.5)


def test_minimal_controls_skip_the_button_width_check_but_warn_in_mouse_mode():
    raw = ps.defaults()
    raw["layout"]["width"] = 320                        # far too narrow for a button row
    assert ps.normalize(raw)[1] == []
    raw["input"]["mode"] = "mouse"
    assert _paths(ps.normalize(raw)[1], "warning") == ["appearance.controls"]


def test_clipped_buttons_and_unreadable_text_warn():
    raw = ps.defaults()
    raw["appearance"]["controls"] = "full"
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
    assert json.loads(literal) == {k: v for k, v in ps.DEFAULTS.items() if k not in ps.HOST_ONLY}


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


def test_settings_page_and_api_expose_recent_start(server):
    status, _, page = _request(server, "GET", "/")
    assert status == 200 and b'data-path="playback.start_behind_live_s"' in page
    assert b'data-path="input.take_controls_on_open"' in page
    _, _, snapshot = _request(server, "GET", "/api/settings")
    assert snapshot["defaults"]["playback"]["start_behind_live_s"] == 5
    assert snapshot["defaults"]["input"]["take_controls_on_open"] is True
    draft = snapshot["settings"]
    draft["playback"]["start_behind_live_s"] = 9
    draft["input"]["take_controls_on_open"] = False
    status, _, saved = _request(server, "PUT", "/api/settings", draft)
    assert status == 200 and saved["settings"]["playback"]["start_behind_live_s"] == 9
    assert ps.load(server.settings_file)[0]["playback"]["start_behind_live_s"] == 9
    assert ps.load(server.settings_file)[0]["input"]["take_controls_on_open"] is False


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


def test_preview_footage_is_the_newest_clip_served_in_ranges(tmp_path):
    clips = tmp_path / "matches"
    (clips / "old").mkdir(parents=True)
    (clips / "new").mkdir()
    old = clips / "old" / "match.mp4"
    old.write_bytes(b"stale" * 20)
    os.utime(old, (1, 1))
    data = bytes(range(256)) * 4
    (clips / "new" / "match.mp4").write_bytes(data)
    srv = SettingsServer(tmp_path / "player-settings.json", 0, clips)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        _, _, snapshot = _request(srv, "GET", "/api/settings")
        assert snapshot["footage"].startswith("/api/footage?v=")
        status, kind, body = _request(srv, "GET", snapshot["footage"])
        assert (status, kind, body) == (200, "video/mp4", data)
        ranged = lambda r: _request(srv, "GET", "/api/footage", headers={"Range": r})[::2]
        assert ranged("bytes=10-19") == (206, data[10:20])
        assert ranged("bytes=1000-") == (206, data[1000:])
        assert ranged("bytes=-4") == (206, data[-4:])
        assert ranged("bytes=5000-")[0] == 416
    finally:
        srv.shutdown()
        srv.server_close()


def test_without_footage_the_preview_keeps_its_drawing(server):
    assert _request(server, "GET", "/api/settings")[2]["footage"] is None
    assert _request(server, "GET", "/api/footage")[0] == 404


def test_server_serves_only_its_static_files(server):
    status, kind, body = _request(server, "GET", "/")
    assert status == 200 and kind.startswith("text/html") and b'alt="deja"' in body
    assert _request(server, "GET", "/app.js")[1].startswith("text/javascript")
    assert _request(server, "GET", "/logo.svg")[1] == "image/svg+xml"
    assert _request(server, "GET", "/../player_settings.py")[0] == 404
    assert _request(server, "GET", "/%2e%2e/server.py")[0] == 404
