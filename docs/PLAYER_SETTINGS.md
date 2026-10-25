# Replay player settings

How the replay player looks and takes input is one JSON file. The settings app
edits it; the player reads it at start and again whenever it changes.

```bash
python -m smartreplay.frontend                 # opens the settings app in the browser
python -m smartreplay.cli settings             # same, through the main CLI
python -m smartreplay.frontend --file other.json --port 8800 --no-browser
```

`smartreplay.frontend` needs only the standard library; the `cli` route also
imports the discovery pipeline and so needs OpenCV.

| File | Written by | Read by |
|---|---|---|
| `%APPDATA%\SmartReplay\player-settings.json` | settings app (`SMARTREPLAY_SETTINGS` overrides the path) | player (`--settings <path>` overrides) |
| `player-status.json`, same folder | player | settings app |

Neither side imports the other. A different player host only has to honour
this format, which is the point: the current player is the GameHQ experiment
(GPL, kept separate), and its replacement can read the same file.

## Rules both sides follow

- The settings app validates (`smartreplay/player_settings.py`) and writes the
  **whole** file atomically: a sibling temp file, then a replace. Errors block
  saving; warnings do not. Each save increments `revision`.
- The player re-checks every field. A bad value falls back to its default and
  is listed in `player-status.json`; it never stops the player starting. An
  unreadable file keeps the last good settings. A missing file means defaults.
- The player watches the folder (an atomic replace swaps the file out from
  under a file watch) and applies changes live: hotkeys are re-registered,
  placement and passive hit-testing update, and the QML rebinds its look.

## Fields

| Field | Default | Notes |
|---|---|---|
| `version` | `1` | Anything else: the player uses defaults. |
| `revision` | `0` | Save counter; the player reports the one it applied. |
| `hotkeys.toggle_replay` | `Ctrl+Alt+R` | Global (`RegisterHotKey`). Required. |
| `hotkeys.toggle_controls` | `Ctrl+Alt+F` | Global. Required unless `input.mode` is `mouse`. |
| `hotkeys.quit` | `Ctrl+Alt+Shift+R` | Global. May be `""` (warning). |
| `player_keys.*` | see below | Only while the player has controls; off in `mouse` mode. `""` unbinds. |
| `input.mode` | `hybrid` | `keyboard`, `mouse` or `hybrid` (below). |
| `input.seek_step_s` | `5` | 1-60. |
| `layout.width`, `layout.height` | `536`, `374` | 320-1920, 200-1200, logical px; clamped to the screen. |
| `layout.anchor` | `top_right` | `top_left` … `bottom_right`, `center` (3x3 grid). |
| `layout.margin_x`, `layout.margin_y` | `24`, `24` | 0-400 px from the anchored edges. |
| `layout.screen` | `game` | `game` (the captured window's monitor), `cursor`, `primary`. |
| `layout.fill` | `fit` | `fit`, `crop`, `stretch`. |
| `appearance.theme` | `slate` | Name of the preset the colours came from, or `custom`. Informational. |
| `appearance.colors.*` | slate | `background`, `panel`, `header`, `header_active`, `text`, `muted`, `accent` as `#rrggbb`. |
| `appearance.passive_opacity`, `active_opacity` | `1.0` | 0.2-1.0. |
| `appearance.font_scale` | `1.0` | 0.75-1.5; scales text and buttons. |
| `appearance.controls` | `full` | `full`, `compact`, `timeline`, `none`. |
| `appearance.show_header`, `show_status`, `show_key_hints`, `focus_outline` | `true`, `true`, `false`, `true` | |
| `buttons.seek`, `play_pause`, `mute`, `expand`, `close` | `true`, `true`, `true`, `false`, `false` | |
| `buttons.speeds` | `[0.25, 0.5, 1, 1.5, 2]` | From 0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 3, 4. |

Player keys default to `Space` play/pause, `Left`/`Right` seek, `Down`/`Up`
slower/faster, `M` mute, `Home` jump to start, `F` expand, `Esc` close. Key
names are Qt's portable key-sequence text, so the player hands them straight
to `QKeySequence`.

Global hotkeys need Ctrl, Alt or Shift plus a letter, digit or F1-F24; never
the Windows key, never Alt+F4. Two actions cannot share a chord, and a player
key cannot equal a global hotkey (Windows delivers the global one first).

## Input modes

| | Passive replay | Take controls | Give them back | Playback |
|---|---|---|---|---|
| `keyboard` | click-through | controls hotkey | controls hotkey, or click the game | player keys; on-screen buttons ignore the mouse |
| `mouse` | clickable, never activated by the click | click the player (or the hotkey) | click the game (or the hotkey) | buttons and timeline; player keys off |
| `hybrid` | click-through | controls hotkey | controls hotkey, or click the game | player keys, buttons and timeline |

In `mouse` mode a showing replay blocks clicks on the part of the game it
covers; that is what makes it clickable.

## player-status.json

```json
{"pid": 1234, "running": true, "revision": 5, "input_mode": "hybrid",
 "hotkeys": {"toggle_replay": {"chord": "Ctrl+Alt+R", "state": "registered"},
             "quit": {"chord": "Alt+Z", "state": "failed",
                      "detail": "Windows refused it; another app has probably registered it. Still using Ctrl+Alt+Shift+R."}},
 "issues": [{"level": "error", "path": "layout.width", "message": "Ignored: ..."}],
 "settings_file": "C:\\Users\\...\\player-settings.json", "updated_utc": "2026-10-24T19:30:00.000Z"}
```

Hotkey `state` is `registered`, `failed` (Windows refused; the previous chord
is kept when it can be), `invalid`, `unbound`, or `override` (`--function-keys`).
The settings app treats the player as running only while `pid` is alive.
