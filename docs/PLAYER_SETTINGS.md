# Replay player settings

How the replay player looks and takes input is one JSON file. The settings app
edits it; the player reads it at start and again whenever it changes.

```bash
python -m smartreplay.frontend                 # opens the settings app in the browser
python -m smartreplay.frontend --window        # the same app as deja, in its own window
python -m smartreplay.cli settings             # same, through the main CLI
python -m smartreplay.frontend --file other.json --port 8800 --no-browser
python -m smartreplay.frontend --footage clip.mp4   # preview footage (default: newest out/demo/matches clip)
```

`smartreplay.frontend` needs only the standard library; `--window` also needs
pywebview (WebView2 on Windows), and the `cli` route imports the discovery
pipeline and so needs OpenCV.

The app is branded **deja**. `./Install-Deja.ps1` adds a Start menu shortcut
and opens it; the window claims the AppUserModelID `SmartReplay.Deja` and the
shortcut carries the same ID and `deja.ico`, so its taskbar button shows the
logo and "Pin to taskbar" pins deja rather than Python. Windows offers no API
to pin for you. `tools/make_deja_icon.py` redraws `deja.ico` from `icon.svg`'s
shapes.

The preview plays the newest recorded clip (`SMARTREPLAY_FOOTAGE` or
`--footage`; a file, or a folder searched for `.mp4`), with the same recording
blurred behind it as the game. Without footage it shows a drawing.

| File | Written by | Read by |
|---|---|---|
| `%APPDATA%\SmartReplay\player-settings.json` | settings app (`SMARTREPLAY_SETTINGS` overrides the path); the demo's model picker writes `model.provider` | player (`--settings <path>` overrides); the demo service reads `model.provider` |
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
| `hotkeys.push_to_talk` | `9` | Hold while speaking, release to ask about the active recorded session. Requires the native demo service; `""` disables. |
| `hotkeys.quit` | `Ctrl+Alt+Shift+R` | Global. May be `""` (warning). |
| `player_keys.*` | see below | Only while the player has controls; off in `mouse` mode. `""` unbinds. |
| `input.mode` | `hybrid` | `keyboard`, `mouse` or `hybrid` (below). |
| `input.seek_step_s` | `5` | 1-60. |
| `input.take_controls_on_open` | `true` | Quick replay takes input controls once its first playable frame is ready. Set false for passive opening. The controls shortcut can always summon a hidden replay and request controls. |
| `playback.start_behind_live_s` | `5` | 1-300 seconds behind the latest recorded moment when quick replay opens. Changes apply on the next opening. |
| `layout.width`, `layout.height` | `536`, `302` | 320-1920, 200-1200, logical px; clamped to the screen. The default is a 16:9 video with no chrome; the settings app's size presets add the title bar and control panel heights when those are on. |
| `layout.anchor` | `top_right` | `top_left` … `bottom_right`, `center` (3x3 grid). |
| `layout.margin_x`, `layout.margin_y` | `24`, `24` | 0-400 px from the anchored edges. |
| `layout.screen` | `game` | `game` (the captured window's monitor), `cursor`, `primary`. |
| `layout.fill` | `fit` | `fit`, `crop`, `stretch`. |
| `appearance.theme` | `slate` | Name of the preset the colours came from, or `custom`. Informational. |
| `appearance.colors.*` | slate | `background`, `panel`, `header`, `header_active`, `text`, `muted`, `accent` as `#rrggbb`. |
| `appearance.passive_opacity`, `active_opacity` | `1.0` | 0.2-1.0. |
| `appearance.font_scale` | `1.0` | 0.75-1.5; scales text and buttons. |
| `appearance.controls` | `minimal` | `minimal`, `full`, `compact`, `timeline`, `none`. `minimal` fills the window with video and lays a 3 px timeline over its bottom edge (a 12 px strip to drag); a corner badge appears only when playback is paused, not at 1×, or loading history. The others put a control panel beneath the video. |
| `appearance.show_header`, `show_status`, `show_key_hints`, `focus_outline` | `false`, `true`, `false`, `true` | The title bar shows the passive/controls hint and the full history button. `show_status` applies to `full` controls only. |
| `buttons.seek`, `play_pause`, `mute`, `expand`, `close` | `true`, `true`, `true`, `false`, `false` | |
| `buttons.speeds` | `[0.25, 0.5, 1, 1.5, 2]` | From 0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 3, 4. |
| `model.provider` | `codex` | `codex`, `claude` or `deepseek`: which model answers replay questions. The player ignores it; the demo service starts on it (unless `SMARTREPLAY_DEMO_BACKEND` is set), switches within a second of a save and re-checks the connection, and its own picker writes back here. DeepSeek reads `DEEPSEEK_API_KEY` (or `DEEPSEEKAPI_KEY`) from the environment or the checkout's `.env`; deja shows whether one exists, never the key. |

Player keys default to `Space` play/pause, `Left`/`Right` seek, `Down`/`Up`
slower/faster, `M` mute, `Home` jump to start, `H` full session history, `F` expand, `Esc` close. Key
names are Qt's portable key-sequence text, so the player hands them straight
to `QKeySequence`.

Global hotkeys accept a number-row digit alone, or Ctrl, Alt or Shift plus a letter, digit or F1-F24; never
the Windows key, never Alt+F4. Two actions cannot share a chord, and a player
key cannot equal a global hotkey (Windows delivers the global one first).

The native demo's controls mode now routes input without activating the player window: the game keeps foreground. A virtual pointer can move outside the player; an outside press releases controls and is consumed. The configured controls shortcut also releases them. Playback actions are delivered directly to the inactive Qt interface. Game-specific raw-input/cursor behavior still requires rehearsal; this is not a universal input-isolation guarantee.

## Input modes

| | Passive replay | Take controls | Give them back | Playback |
|---|---|---|---|---|
| `keyboard` | click-through | controls hotkey | controls hotkey, or click the game | player keys; on-screen buttons ignore the mouse |
| `mouse` | clickable, never activated by the click | click the player (or the hotkey) | click the game (or the hotkey) | buttons and timeline; player keys off |
| `hybrid` | click-through | controls hotkey | controls hotkey, or click the game | player keys, buttons and timeline |

In `mouse` mode a showing replay blocks clicks on the part of the game it
covers; that is what makes it clickable.

With the mouse (`mouse` and `hybrid`, while the player has controls): scroll
over the video to magnify around the pointer (up to 4×; a new replay starts
whole), and drag to pan a magnified picture. Drag any edge or corner to resize,
like a window; minimal controls leave the bottom edge to the timeline. A
resize keeps the picture's shape when the video fit is `fit`, so it never adds
letterbox bars. Without a title bar, dragging the video moves the player.

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
