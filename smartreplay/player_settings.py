"""Replay player settings: the file the settings app writes and the player reads.

One JSON document holds everything a user can change about the replay player:
global hotkeys, the keys that work while the player has controls, which input
device drives it, its size and placement, its look, and which buttons it shows.

    %APPDATA%\\SmartReplay\\player-settings.json     (SMARTREPLAY_SETTINGS overrides)

The file is the interface between processes. The settings app validates and
writes it whole; the player watches it, re-reads it on change and reports what
it applied - including hotkeys Windows refused - in player-status.json beside
it. Neither side imports the other, so a different player host only has to
honour this format. docs/PLAYER_SETTINGS.md describes it field by field.

Validation here is the user-facing gate: errors block saving, warnings do not.
The player still re-checks every field, because a hand-edited file must never
stop it from starting.
"""

from __future__ import annotations

import copy
import ctypes
import json
import os
import re
import tempfile
import time
from pathlib import Path

VERSION = 1

THEMES = {
    "slate": {"background": "#111721", "panel": "#19222e", "header": "#1b2533",
              "header_active": "#193d48", "text": "#d7e2e8", "muted": "#99b0bf",
              "accent": "#46b6cc"},
    "midnight": {"background": "#05070a", "panel": "#0d1117", "header": "#0d1117",
                 "header_active": "#10283a", "text": "#e6edf3", "muted": "#8b98a5",
                 "accent": "#58a6ff"},
    "ember": {"background": "#140f0d", "panel": "#1e1714", "header": "#241a16",
              "header_active": "#4a2414", "text": "#f3e6dd", "muted": "#b59c8f",
              "accent": "#ff8a3d"},
    "paper": {"background": "#eef1f4", "panel": "#ffffff", "header": "#e3e8ee",
              "header_active": "#cfe6f0", "text": "#18202a", "muted": "#5b6876",
              "accent": "#0b7fa6"},
    "contrast": {"background": "#000000", "panel": "#000000", "header": "#000000",
                 "header_active": "#00343f", "text": "#ffffff", "muted": "#e0e0e0",
                 "accent": "#ffd400"},
}

DEFAULTS = {
    "version": VERSION,
    "revision": 0,
    "hotkeys": {
        "toggle_replay": "Ctrl+Alt+R",
        "toggle_controls": "Ctrl+Alt+F",
        "quit": "Ctrl+Alt+Shift+R",
    },
    "player_keys": {
        "play_pause": "Space",
        "seek_back": "Left",
        "seek_forward": "Right",
        "close": "Esc",
        "speed_down": "Down",
        "speed_up": "Up",
        "mute": "M",
        "expand": "F",
        "restart": "Home",
    },
    "input": {"mode": "hybrid", "seek_step_s": 5},
    "layout": {"width": 536, "height": 374, "anchor": "top_right",
               "margin_x": 24, "margin_y": 24, "screen": "game", "fill": "fit"},
    "appearance": {
        "theme": "slate",
        "colors": dict(THEMES["slate"]),
        "passive_opacity": 1.0,
        "active_opacity": 1.0,
        "font_scale": 1.0,
        "show_header": True,
        "controls": "full",
        "show_status": True,
        "show_key_hints": False,
        "focus_outline": True,
    },
    "buttons": {"seek": True, "play_pause": True, "mute": True, "expand": False,
                "close": False, "speeds": [0.25, 0.5, 1, 1.5, 2]},
}

INPUT_MODES = ("hybrid", "keyboard", "mouse")
ANCHORS = ("top_left", "top_center", "top_right", "middle_left", "center",
           "middle_right", "bottom_left", "bottom_center", "bottom_right")
SCREENS = ("game", "cursor", "primary")
FILLS = ("fit", "crop", "stretch")
CONTROLS = ("full", "compact", "timeline", "none")
SPEED_CHOICES = (0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 3, 4)

# (min, max) for every number; ints stay ints.
RANGES = {
    "input.seek_step_s": (1, 60),
    "layout.width": (320, 1920),
    "layout.height": (200, 1200),
    "layout.margin_x": (0, 400),
    "layout.margin_y": (0, 400),
    "appearance.passive_opacity": (0.2, 1.0),
    "appearance.active_opacity": (0.2, 1.0),
    "appearance.font_scale": (0.75, 1.5),
}
ENUMS = {
    "input.mode": INPUT_MODES,
    "layout.anchor": ANCHORS,
    "layout.screen": SCREENS,
    "layout.fill": FILLS,
    "appearance.controls": CONTROLS,
}

# Chords other overlays ship as defaults. Registering one of them is allowed
# (the user may have turned that overlay off), but whichever app starts first
# owns it, so it is worth a warning.
KNOWN_CONFLICTS = {
    "Alt+Z": "the NVIDIA app overlay",
    "Alt+F1": "NVIDIA screenshot",
    "Alt+F9": "NVIDIA recording",
    "Alt+F10": "NVIDIA instant replay",
    "Alt+R": "the AMD Adrenalin overlay",
    "Ctrl+Shift+E": "AMD recording",
    "Ctrl+Shift+S": "AMD instant replay",
}

MODIFIERS = ("Ctrl", "Alt", "Shift")
_MOD_ALIASES = {"ctrl": "Ctrl", "control": "Ctrl", "alt": "Alt", "shift": "Shift"}
_RESERVED_MODS = {"win", "windows", "meta", "cmd", "super"}

# Key names as Qt's portable key-sequence text spells them, so the player can
# hand them straight to QKeySequence. Aliases cover the browser's names.
NAMED_KEYS = ("Space", "Esc", "Tab", "Backspace", "Return", "Enter", "Ins", "Del",
              "Home", "End", "PgUp", "PgDown", "Left", "Up", "Right", "Down")
_KEY_ALIASES = {"escape": "Esc", "delete": "Del", "insert": "Ins", "pageup": "PgUp",
                "pagedown": "PgDown", "arrowleft": "Left", "arrowright": "Right",
                "arrowup": "Up", "arrowdown": "Down", "spacebar": "Space"}
PUNCTUATION = tuple(",./;'[]\\-=`")

HOTKEY_LABELS = {"toggle_replay": "Show / hide replay",
                 "toggle_controls": "Take / return controls",
                 "quit": "Quit player"}
PLAYER_KEY_LABELS = {"play_pause": "Play / pause", "seek_back": "Seek back",
                     "seek_forward": "Seek forward", "close": "Close replay",
                     "speed_down": "Slower", "speed_up": "Faster", "mute": "Mute",
                     "expand": "Expand to screen", "restart": "Jump to start"}


def default_path() -> Path:
    override = os.environ.get("SMARTREPLAY_SETTINGS")
    if override:
        return Path(override)
    base = os.environ.get("APPDATA") or str(Path.home() / ".config")
    return Path(base) / "SmartReplay" / "player-settings.json"


def status_path(settings_file: Path) -> Path:
    return Path(settings_file).with_name("player-status.json")


def defaults() -> dict:
    return copy.deepcopy(DEFAULTS)


# ----------------------------------------------------------------- key chords

def _split(text: str) -> list[str]:
    return [p.strip() for p in text.split("+") if p.strip()]


def _canonical_key(name: str) -> str | None:
    if len(name) == 1:
        ch = name.upper()
        if ch.isascii() and (ch.isalnum() or ch in PUNCTUATION):
            return ch
        return None
    low = name.lower()
    if low in _KEY_ALIASES:
        return _KEY_ALIASES[low]
    for key in NAMED_KEYS:
        if key.lower() == low:
            return key
    m = re.fullmatch(r"f(\d{1,2})", low)
    if m and 1 <= int(m.group(1)) <= 24:
        return f"F{int(m.group(1))}"
    return None


def parse_chord(text: str, *, global_hotkey: bool) -> tuple[str | None, str | None]:
    """Canonicalise one key chord. Returns (chord, None) or (None, reason).

    Global hotkeys follow what the player registers with RegisterHotKey: at
    least one of Ctrl/Alt/Shift plus a letter, digit or F1-F24, never the
    Windows key, never Alt+F4. Player keys only fire while the player has
    controls, so a bare key such as Space is fine there.
    """
    parts = _split(text)
    if not parts:
        return None, "Empty shortcut."
    *mods, key_name = parts
    seen = set()
    for m in mods:
        low = m.lower()
        if low in _RESERVED_MODS:
            return None, "The Windows key is reserved for system shortcuts."
        if low not in _MOD_ALIASES:
            return None, f'Unknown modifier "{m}".'
        seen.add(_MOD_ALIASES[low])
    if key_name.lower() in _MOD_ALIASES or key_name.lower() in _RESERVED_MODS:
        return None, "A shortcut needs a key as well as modifiers."
    key = _canonical_key(key_name)
    if key is None:
        return None, f'Unsupported key "{key_name}".'
    if global_hotkey:
        if not seen:
            return None, "A global hotkey needs Ctrl, Alt or Shift so it does not steal a key from the game."
        if not (len(key) == 1 and key.isalnum()) and not re.fullmatch(r"F\d+", key):
            return None, "Global hotkeys use a letter, digit or F1-F24."
        if seen == {"Alt"} and key == "F4":
            return None, "Alt+F4 is reserved by Windows."
    chord = "+".join([m for m in MODIFIERS if m in seen] + [key])
    return chord, None


# ----------------------------------------------------------------- validation

def _issue(issues: list, level: str, path: str, message: str) -> None:
    issues.append({"level": level, "path": path, "message": message})


def _get(tree: dict, path: str):
    node = tree
    for part in path.split("."):
        node = node[part]
    return node


def _set(tree: dict, path: str, value) -> None:
    *head, last = path.split(".")
    node = tree
    for part in head:
        node = node[part]
    node[last] = value


def _merge(base: dict, raw: dict, prefix: str, issues: list) -> None:
    """Copy raw values onto base where the key exists; report unknown keys."""
    for key, value in raw.items():
        path = f"{prefix}{key}"
        if key not in base:
            _issue(issues, "warning", path, "Unknown setting; it will be dropped on save.")
            continue
        if isinstance(base[key], dict):
            if isinstance(value, dict):
                _merge(base[key], value, path + ".", issues)
            else:
                _issue(issues, "error", path, "Expected a group of settings.")
        else:
            base[key] = value


def _luminance(hex_colour: str) -> float:
    rgb = [int(hex_colour[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def contrast(a: str, b: str) -> float:
    la, lb = sorted((_luminance(a), _luminance(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


# Pixel metrics shared with ReplayWindow.qml: button widths at font scale 1.
BUTTON_WIDTHS = {"seek": 53, "play_pause": 58, "speed": 39, "mute": 57,
                 "expand": 40, "close": 34}
BUTTON_SPACING = 4
CONTROLS_MARGIN = 8
COMPACT_EXTRA = 6 + 80 + 6 + 44  # compact row: gap, shortest timeline, gap, time


def button_row_width(settings: dict) -> int:
    b = settings["buttons"]
    widths = []
    if b["seek"]:
        widths += [BUTTON_WIDTHS["seek"]] * 2
    if b["play_pause"]:
        widths.append(BUTTON_WIDTHS["play_pause"])
    widths += [BUTTON_WIDTHS["speed"]] * len(b["speeds"])
    for name in ("mute", "expand", "close"):
        if b[name]:
            widths.append(BUTTON_WIDTHS[name])
    if not widths:
        return 0
    scale = settings["appearance"]["font_scale"]
    return round(sum(widths) * scale + BUTTON_SPACING * (len(widths) - 1)) + 2 * CONTROLS_MARGIN


def normalize(raw) -> tuple[dict, list[dict]]:
    """Merge raw settings onto the defaults, canonicalise, and check them.

    Returns the settings to use and a list of issues. A malformed field falls
    back to its default, so the result is always well-formed; callers decide
    whether errors (which also cover clashes between valid fields) block a save.
    """
    issues: list[dict] = []
    out = defaults()
    if not isinstance(raw, dict):
        _issue(issues, "error", "", "Settings must be a JSON object.")
        return out, issues
    raw = copy.deepcopy(raw)
    version = raw.pop("version", VERSION)
    if version != VERSION:
        _issue(issues, "error", "version", f"Unsupported settings version {version!r}.")
        return out, issues
    revision = raw.pop("revision", 0)
    out["revision"] = revision if isinstance(revision, int) and revision >= 0 else 0

    _merge(out, raw, "", issues)
    fallback = defaults()

    def reject(path, message):
        _issue(issues, "error", path, message)
        _set(out, path, _get(fallback, path))

    # Scalars.
    for path, (lo, hi) in RANGES.items():
        value = _get(out, path)
        is_int = isinstance(_get(fallback, path), int)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            reject(path, "Expected a number.")
        elif is_int and value != int(value):
            reject(path, "Expected a whole number.")
        elif not lo <= value <= hi:
            reject(path, f"Must be between {lo} and {hi}.")
        else:
            _set(out, path, int(value) if is_int else float(value))
    for path, options in ENUMS.items():
        if _get(out, path) not in options:
            reject(path, f"Must be one of: {', '.join(options)}.")
    for group in ("appearance", "buttons"):
        for key, default in fallback[group].items():
            if isinstance(default, bool) and not isinstance(out[group][key], bool):
                reject(f"{group}.{key}", "Expected true or false.")

    # Colours.
    colors = out["appearance"]["colors"]
    for key in fallback["appearance"]["colors"]:
        value = colors.get(key)
        if not (isinstance(value, str) and re.fullmatch(r"#[0-9a-fA-F]{6}", value)):
            reject(f"appearance.colors.{key}", "Expected a colour like #1a2b3c.")
        else:
            colors[key] = value.lower()
    theme = out["appearance"]["theme"]
    if not isinstance(theme, str) or (theme not in THEMES and theme != "custom"):
        reject("appearance.theme", f"Must be one of: {', '.join(THEMES)}, custom.")
    elif theme != "custom" and colors != THEMES[theme]:
        out["appearance"]["theme"] = "custom"
    if contrast(colors["text"], colors["panel"]) < 4.5:
        _issue(issues, "warning", "appearance.colors.text",
               f"Text on the control panel has contrast {contrast(colors['text'], colors['panel']):.1f}:1; "
               "4.5:1 or more stays readable over bright games.")

    # Speeds.
    speeds = out["buttons"]["speeds"]
    if not isinstance(speeds, list) or any(isinstance(s, bool) or s not in SPEED_CHOICES for s in speeds):
        reject("buttons.speeds", f"Speeds come from: {', '.join(map(_fmt_speed, SPEED_CHOICES))}.")
    else:
        out["buttons"]["speeds"] = sorted({float(s) if s != int(s) else int(s) for s in speeds})

    # Hotkeys.
    mode = out["input"]["mode"]
    owner: dict[str, str] = {}
    for action in fallback["hotkeys"]:
        path = f"hotkeys.{action}"
        value = out["hotkeys"][action]
        if value in ("", None):
            out["hotkeys"][action] = ""
            continue
        if not isinstance(value, str):
            reject(path, "Expected a shortcut such as Ctrl+Alt+R.")
            continue
        chord, why = parse_chord(value, global_hotkey=True)
        if why:
            reject(path, why)
            continue
        out["hotkeys"][action] = chord
        if chord in owner:
            _issue(issues, "error", path, f"Already used by “{HOTKEY_LABELS[owner[chord]]}”.")
        else:
            owner[chord] = action
        if chord in KNOWN_CONFLICTS:
            _issue(issues, "warning", path,
                   f"{chord} is the default for {KNOWN_CONFLICTS[chord]}; whichever app starts first gets it.")
    if not out["hotkeys"]["toggle_replay"]:
        _issue(issues, "error", "hotkeys.toggle_replay", "Required: it is the only way to open a replay.")
    if not out["hotkeys"]["toggle_controls"] and mode != "mouse":
        _issue(issues, "error", "hotkeys.toggle_controls",
               "Required unless the input mode is Mouse: nothing else gives the player controls.")
    if not out["hotkeys"]["quit"]:
        _issue(issues, "warning", "hotkeys.quit", "With no quit hotkey the player can only be closed from Task Manager.")

    # Player keys.
    used: dict[str, str] = {}
    for action in fallback["player_keys"]:
        path = f"player_keys.{action}"
        value = out["player_keys"][action]
        if value in ("", None):
            out["player_keys"][action] = ""
            continue
        if not isinstance(value, str):
            reject(path, "Expected a key such as Space.")
            continue
        chord, why = parse_chord(value, global_hotkey=False)
        if why:
            reject(path, why)
            continue
        out["player_keys"][action] = chord
        if chord in used:
            _issue(issues, "error", path, f"Already used by “{PLAYER_KEY_LABELS[used[chord]]}”.")
        elif chord in owner:
            _issue(issues, "error", path,
                   f"Also a global hotkey (“{HOTKEY_LABELS[owner[chord]]}”), which always wins.")
        used[chord] = action
    if mode == "keyboard" and not used:
        _issue(issues, "warning", "player_keys", "Keyboard mode with no player keys: playback cannot be controlled.")

    # Combinations that leave the player uncontrollable or clipped.
    controls = out["appearance"]["controls"]
    b = out["buttons"]
    has_buttons = b["seek"] or b["play_pause"] or b["mute"] or b["speeds"]
    if mode == "mouse" and (controls in ("none", "timeline") or not has_buttons):
        _issue(issues, "warning", "appearance.controls",
               "Mouse mode with no on-screen buttons: only the timeline (or nothing) can be clicked.")
    if controls in ("full", "compact"):
        need = button_row_width(out) + (COMPACT_EXTRA if controls == "compact" else 0)
        if need > out["layout"]["width"]:
            _issue(issues, "warning", "layout.width",
                   f"The buttons need {need} px; at {out['layout']['width']} px the row is cut off.")
    return out, issues


def _fmt_speed(s: float) -> str:
    return f"{s:g}×"


def errors(issues: list[dict]) -> list[dict]:
    return [i for i in issues if i["level"] == "error"]


# ----------------------------------------------------------------- file I/O

def load(path: Path | None = None) -> tuple[dict, list[dict]]:
    """Read the settings file. A missing file means defaults, without issues."""
    path = Path(path or default_path())
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return defaults(), []
    except (OSError, ValueError) as exc:
        out = defaults()
        return out, [{"level": "error", "path": "", "message": f"Could not read {path.name}: {exc}"}]
    return normalize(raw)


class SettingsError(ValueError):
    def __init__(self, issues: list[dict]):
        super().__init__("; ".join(f"{i['path']}: {i['message']}" for i in errors(issues)))
        self.issues = issues


def save(settings: dict, path: Path | None = None) -> tuple[dict, list[dict]]:
    """Validate and write the whole file atomically, bumping its revision.

    The revision lets the player report which save it has applied. Raises
    SettingsError, leaving the file untouched, when there are errors.
    """
    path = Path(path or default_path())
    current, _ = load(path)
    clean, issues = normalize(settings)
    if errors(issues):
        raise SettingsError(issues)
    clean["revision"] = max(current["revision"], clean["revision"]) + 1
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".player-settings-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(clean, fh, indent=2)
            fh.write("\n")
        # The player may hold the file open for a moment while reading it;
        # Windows refuses the replace until it lets go.
        for attempt in range(20):
            try:
                os.replace(tmp, path)
                break
            except PermissionError:
                if attempt == 19:
                    raise
                time.sleep(0.05)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return clean, issues


def _process_alive(pid: int) -> bool:
    if os.name != "nt":
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        return True
    # os.kill(pid, 0) sends CTRL_C_EVENT on Windows, so ask the kernel instead.
    kernel32 = ctypes.windll.kernel32
    handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        return False
    try:
        code = ctypes.c_ulong()
        ok = kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
        return bool(ok) and code.value == 259  # STILL_ACTIVE
    finally:
        kernel32.CloseHandle(handle)


def player_status(settings_file: Path | None = None) -> dict:
    """What the player last reported, with whether that process still runs."""
    path = status_path(Path(settings_file or default_path()))
    status = None
    for attempt in range(3):  # the player may be mid-replace
        try:
            status = json.loads(path.read_text(encoding="utf-8"))
            break
        except FileNotFoundError:
            break
        except (OSError, ValueError):
            time.sleep(0.03)
    if not isinstance(status, dict):
        return {"running": False}
    pid = status.get("pid")
    status["running"] = bool(status.get("running")) and isinstance(pid, int) and _process_alive(pid)
    return status
