# Claude handoff for the replay miniplayer

Prepared 2026-10-24. Continue integration without starting live capture or desktop tests. The user explicitly asked to leave live testing for later. This handoff changes documentation only.

Smart Replay's main repository is `C:\Code\HACKTX`. The working miniplayer prototype is a separate GameHQ checkout at `C:\Code\HACKTX\experiments\gamehq-passive`, based on commit `f8a8ebc6593bc04bdba6a5d8dfca81cfe9e07637`. Its changes are uncommitted, and most prototype files are untracked. The parent repository ignores `experiments/`; committing this handoff does not preserve the prototype. Inspect both modified and untracked files before moving or packaging anything. Other work has been happening in the checkout, so reconcile the current files rather than replacing them wholesale.

## Behavior to preserve

| Input | Behavior |
|---|---|
| Ctrl+Alt+R | Show a new recent replay, or hide/cancel the current one. Showing it preserves game focus and makes the player click-through. |
| Ctrl+Alt+F | Give the player controls, or return controls to the original game while leaving replay visible. It does not show or hide the player. |
| Ctrl+Alt+Shift+R | Exit the experiment. In the main app, use the normal application shutdown path. |
| Space and Left/Right, while interactive | Pause/play and seek five seconds within the selected interval. |

Keep these letter chords as the defaults. The user's laptop toggles airplane mode on F8. Legacy F8/F9 bindings exist only behind `--function-keys`; the on-screen hints derive from the registered bindings.

**User settings (added 2026-10-24).** Every chord above, the in-player keys, an input mode (keyboard only, mouse only, or the original hybrid), size and placement, colours, opacity and visible buttons now come from `%APPDATA%\SmartReplay\player-settings.json`, edited with `python -m smartreplay.frontend` ([PLAYER_SETTINGS.md](PLAYER_SETTINGS.md)). The experiment reads it through `spike/passive-replay/PlayerSettings.cpp`, applies saves live, and reports registration results in `player-status.json`. Defaults reproduce the behaviour above; the `--benchmark` and `--verify-catchup` modes ignore the user's file. Startup now exits with code 6 only when the show/hide hotkey cannot be registered.

Normal launch now uses live capture. Once GameHQ detects an eligible foreground game, it records automatically. Opening review plays the latest 30 seconds, or the available history while the buffer warms up. Capture continues against the original game while review is preparing, passive, paused, or focused.

Each replay uses an immutable temporary MP4. Its end represents the recording cutoff taken when review is requested; playback closes when it reaches that fixed end. Pausing keeps it open, and rewinding or changing speed changes when it reaches the end. Do not replace this with a 30-second wall-clock timeout or a moving live endpoint. Closing returns focus only if the player currently owns foreground focus, preserving another app the user selected.

Temporary review is separate from saving a clip: no gallery insertion, `clipSaved`/`clipSaving` notification, or saved-clip sound. Audio starts muted. Explicit `--video <file>` mode still plays a whole file for comparison.

## Where the implementation lives

All paths below are in the isolated GameHQ checkout.

| File | Responsibility |
|---|---|
| [PassiveReplay.cpp](C:/Code/HACKTX/experiments/gamehq-passive/spike/passive-replay/PassiveReplay.cpp) | `ReplayController`, separate visibility/focus actions, request IDs, review ownership, temporary-media lifetime, fixed playback bounds, logging, and experiment startup. |
| [ReplayMediaSession.h](C:/Code/HACKTX/experiments/gamehq-passive/spike/passive-replay/ReplayMediaSession.h) and [ReplayMediaSession.cpp](C:/Code/HACKTX/experiments/gamehq-passive/spike/passive-replay/ReplayMediaSession.cpp) | Per-replay `QMediaPlayer`/audio output: created on open, retired with a deferred delete on close, then `disposed(source)` reports that the backend released that file. Speed and volume persist across replays. |
| [ReplayWindow.qml](C:/Code/HACKTX/experiments/gamehq-passive/spike/passive-replay/ReplayWindow.qml) | Qt Multimedia view and controls. Waits for loaded metadata, seeks to the selected start before playing, and presents a timeline relative to the selected interval. |
| [FramePumpService.h](C:/Code/HACKTX/experiments/gamehq-passive/src/capture/FramePumpService.h) and [FramePumpService.cpp](C:/Code/HACKTX/experiments/gamehq-passive/src/capture/FramePumpService.cpp) | `requestTemporaryReplay(requestId, seconds)`, ready/failure signals, background remux, `setReviewActive`, target-loss handling, and frame-progress reporting. |
| [SegmentRecorder.h](C:/Code/HACKTX/experiments/gamehq-passive/src/capture/SegmentRecorder.h) and [SegmentRecorder.cpp](C:/Code/HACKTX/experiments/gamehq-passive/src/capture/SegmentRecorder.cpp) | `snapshotForSave(requestedSeconds = 0)` finalizes the current segment and immediately reopens recording. Retains/selects segments by recorded duration and probes restored segment durations on re-arm. |
| [ReplayBufferOwners.h](C:/Code/HACKTX/experiments/gamehq-passive/src/capture/ReplayBufferOwners.h) | Adds a Review owner so foreground changes to the player do not disarm capture or retarget it to another window. Also blocks HDR operations from retargeting that session. |
| [TemporaryReplay.h](C:/Code/HACKTX/experiments/gamehq-passive/src/capture/TemporaryReplay.h) and [TemporaryReplay.cpp](C:/Code/HACKTX/experiments/gamehq-passive/src/capture/TemporaryReplay.cpp) | Shared ownership of a private temporary directory and its replay MP4. The last owner removes the media. |
| [ReplayWindowPolicy.h](C:/Code/HACKTX/experiments/gamehq-passive/src/capture/ReplayWindowPolicy.h) | Thirty-second constant, bounded playback range, seek clamping, and retention margin. |
| [ReplayFocusPolicy.h](C:/Code/HACKTX/experiments/gamehq-passive/spike/passive-replay/ReplayFocusPolicy.h) | Uses actual native foreground ownership so a delayed Qt inactive notification cannot immediately undo successful focus acquisition. |
| [HotkeyManager.cpp](C:/Code/HACKTX/experiments/gamehq-passive/src/input/HotkeyManager.cpp) | Handles both `windows_generic_MSG` and `windows_dispatcher_MSG`; thread-delivered `RegisterHotKey(nullptr, ...)` events need the latter. Preserve this fix. |
| [src/main.cpp](C:/Code/HACKTX/experiments/gamehq-passive/src/main.cpp) and [src/CMakeLists.txt](C:/Code/HACKTX/experiments/gamehq-passive/src/CMakeLists.txt) | Add `--passive-replay` to the full GameHQ executable and compile the experiment/resources. This is a separate entry mode, not integration into the ordinary gallery UI. |
| [launch.ps1](C:/Code/HACKTX/experiments/gamehq-passive/spike/passive-replay/launch.ps1) and [build.ps1](C:/Code/HACKTX/experiments/gamehq-passive/spike/passive-replay/build.ps1) | Default live mode, optional file/full-app mode, launch-only dry run, standalone and full builds. |

The snapshot path is:

```text
show request
  -> acquire Review owner and remember original target identity
  -> queue requestTemporaryReplay(requestId, 30) to the capture worker
  -> finalize current segment, lease a duration-selected suffix, reopen recording
  -> ReplayExportTask remuxes compressed samples into a private MP4
  -> accept result only if requestId still matches the visible review
  -> load metadata, select [max(0, duration - 30000), duration], seek, play
  -> close at fixed endpoint, retire per-replay player, release media on disposed(source)
  -> release Review owner; ordinary capture ownership continues independently
```

Keep capture/encoder work on its worker thread and remux work off the GUI/capture path. Segment leases protect source files during remux; the resulting private MP4 survives later ring rotation. Existing helpers are [SegmentLease.cpp](C:/Code/HACKTX/experiments/gamehq-passive/src/capture/SegmentLease.cpp), [ReplayExportTask.h](C:/Code/HACKTX/experiments/gamehq-passive/src/capture/ReplayExportTask.h), and [ReplayExporter.cpp](C:/Code/HACKTX/experiments/gamehq-passive/src/capture/ReplayExporter.cpp). Hiding cancels acceptance of a pending result; it does not destroy a running remux thread. Shutdown joins that work.

## Fixes that must travel with integration

1. **Retain time, not a fixed number of files.** Every replay closes the current segment early. Seven files can therefore contain seven seconds after repeated opens. The regression reproduced exactly that failure. Duration-based retention and selection now produce 30 seconds after repeated opens and after re-arming. Do not restore the old seven-file cutoff. The experiment keeps a 35-second ring at the default five-second segment size, then bounds playback to the final 30 seconds.
2. **Pin capture ownership to the original target.** Review must remain an owner while preparing and playing, including when Qt has foreground focus. Hide, failed preparation, target loss, and shutdown must release that owner. A separate save finishing must not release review ownership.
3. **Preserve cancellation and lifetime guards.** A result from an earlier request must not reopen a hidden player or replace a newer request. Hold private media until the multimedia backend releases it; clearing a URL and deleting a file immediately is insufficient on Windows.
4. **Start only after metadata is ready.** Seek to the selected start before playback. Clamp slider/keyboard seeks to that interval and show elapsed time relative to its start. Ordinary file mode keeps its full duration.
5. **Treat visibility and focus independently.** Passive mode uses no-activation and transparent-input flags. Return to the remembered target only when its identity is still valid and the player owns focus. Preserve bounded native focus acquisition and the delayed-Qt-activation fix.

Known retry limitation: hiding and reopening while a remux is still running can return “A temporary replay is still being prepared.” Stale results are discarded safely; the newer request is not queued for automatic retry.

## Integrating with Smart Replay

The Python repository currently provides offline discovery, profiles, event indexing, and query/reel tools. This prototype does not connect detection or AI to live replay, and it does not add a Python IPC bridge.

Follow the existing repository policy in [ROADMAP.md](C:/Code/HACKTX/docs/ROADMAP.md): GameHQ is kept as a separate GPL experiment; Smart Replay's own implementation is independent rather than copied or translated from GameHQ. Reusing substantial GameHQ code requires a deliberate change in product direction. Do not silently copy this checkout into the Python package or assume that a process boundary changes that policy.

For the current independent implementation direction:

1. Keep Python responsible for profiles, event evidence, and queries. Build the Windows capture/player host independently using the behavior contract above. It should own native window focus, capture, the encoder ring, and media lifetime. Keep the replay hot path independent of OCR/model completion.
2. Define a narrow local interface before connecting processes. Proposed commands are request replay with a request ID and interval, hide/cancel, set interaction mode, and shutdown. Proposed events are capture state, replay ready/failed/closed, and recording progress. Include a session/generation ID and a monotonic time origin so late responses and event timestamps cannot be confused across capture restarts. These messages are proposed integration work, not an existing API.
3. Land manual “last 30 seconds” first. Use generated frames, fake responses, and file playback to check request routing, cancellation, duration selection, concurrent recording, and cleanup without desktop input. Avoid feeding an actively written ring file straight to a media player; use a stable snapshot with explicit release ownership.
4. Add profile-driven placement from [profile.py](C:/Code/HACKTX/smartreplay/profile.py). Its HUD-safe geometry includes crop/frame coordinate information; map through that crop into the game client rectangle and handle DPI. The experiment currently uses a fixed top-right position, so automatic HUD-safe placement is still integration work.
5. Later connect event-selected intervals from [index.py](C:/Code/HACKTX/smartreplay/index.py) and [query.py](C:/Code/HACKTX/smartreplay/query.py). Their current workflows analyze recorded files; continuous live event indexing is not implemented by the miniplayer changes. An event outside retained footage must report unavailable, not play an unrelated interval.

If work instead continues inside the GameHQ product, extract the controller from experiment startup, inject the app's existing `FramePumpService`, and register actions through its existing hotkey service. Use the running app's settings/event loop and one capture service. Do not invoke `runPassiveReplay` from the already-running app, create a second application object/recorder, or force every user's retention setting to the experiment's 35 seconds. Review-ready/failure/target-loss signals and temporary-media ownership provide the integration points. The current full-app flag only proves that the experiment can be built into that executable.

## Verification and remaining work

Last verified integration run on 2026-10-24 (17:41–17:44, after the decoder-teardown change):

- Standalone `out-passive\GameHQPassiveReplay.exe` and full `out-full\GameHQ.exe` rebuilt from the current tree.
- All five CTest suites passed: `temporary_replay`, `replay_owners`, `recorder_snapshots`, `live_replay`, and `replay_media` (39.73 seconds total). Encoder tests use generated D3D WARP textures, not screen capture.
- `replay_media` covers the new [ReplayMediaSession](C:/Code/HACKTX/experiments/gamehq-passive/spike/passive-replay/ReplayMediaSession.cpp): the player and audio output are destroyed on close; repeated close/reopen releases only that replay's file and keeps speed/volume; closing from a playback signal is safe; closing after a load error still releases the backend.
- The continued-recording test keeps an MP4 reader open while the recording ring advances and rotates; the snapshot size/duration stay fixed and it remains readable.
- The repeated-open regression failed with 7000 ms before the fix and passed with 30000 ms after it, including restoration after re-arm.
- Dry-run launch checks covered default live mode, explicit file paths containing spaces, and full-app routing. QML lint exited successfully with context-property warnings.
- A read-only follow-up review found the duration-retention issue resolved.

After the settings change (2026-10-24, 19:29–19:33): both builds rebuilt; the five suites plus the new `player_settings` suite passed (38.65 seconds). `player_settings` covers per-field fallback, canonical chords and keys, placement for each anchor, live reload across repeated atomic replaces, keeping the last good settings when the file is corrupt, the status file, and `ReplayWindow.qml` following each setting offscreen (`REPLAY_SNAPSHOT_DIR` saves PNGs of the rendered player). Live hotkey re-registration, mouse-mode click-to-take-controls, opacity and placement on a real desktop are not yet exercised.

Evidence is in [out-passive-verify.log](C:/Code/HACKTX/experiments/gamehq-passive/out-passive-verify.log) (UTF-16), [LastTest.log](C:/Code/HACKTX/experiments/gamehq-passive/out-passive/Testing/Temporary/LastTest.log), [retention-red.txt](C:/Code/HACKTX/experiments/gamehq-passive/out-passive/retention-red.txt), and [retention-green.txt](C:/Code/HACKTX/experiments/gamehq-passive/out-passive/retention-green.txt). These are local build artifacts, not committed evidence.

**Decoder teardown is reconciled.** [ReplayWindow.qml](C:/Code/HACKTX/experiments/gamehq-passive/spike/passive-replay/ReplayWindow.qml) no longer keeps a persistent `MediaPlayer`; `ReplayMediaSession` creates the player and audio output per replay and retires them with a deferred delete, so closing from inside a playback or end-of-media signal is safe. Its `disposed(source)` signal calls `ReplayController::mediaReleased`, which is the point where the temporary MP4 may be released. Both builds compile the class.

Build note: on Windows, a broken `HKCU\Software\Microsoft\Command Processor\Autorun` value makes every `cmd /C` exit 1 silently, and Ninja's automoc steps then fail with no message. If builds stop at "Automatic MOC" without an error, check that value first. A failed build leaves old test binaries in place, so ctest can report stale passes; confirm binary timestamps.

Safe commands for the five suites and command-generation checks (also wrapped in `experiments\verify-miniplayer-offline.ps1`, which is git-ignored):

```powershell
$replayRoot = 'C:\Code\HACKTX\experiments\gamehq-passive'
& "$replayRoot\spike\passive-replay\build.ps1"
& "$replayRoot\spike\passive-replay\build.ps1" -FullApp
$env:PATH = "$replayRoot\tools\Qt\Tools\mingw1310_64\bin;$replayRoot\tools\Qt\6.8.3\mingw_64\bin;$env:PATH"
& "$replayRoot\.tooling\python\cmake\data\bin\ctest.exe" --test-dir "$replayRoot\out-passive" -R '^(temporary_replay|replay_owners|recorder_snapshots|live_replay|replay_media|player_settings)$' --output-on-failure
& "$replayRoot\spike\passive-replay\launch.ps1" -PrintCommand
& "$replayRoot\spike\passive-replay\launch.ps1" -FullApp -PrintCommand
```

Use per-command `git -c safe.directory=C:/Code/HACKTX/experiments/gamehq-passive ...` in that checkout if ownership checks prevent inspection; do not change global Git settings.

Do not run the launcher without `-PrintCommand`, `Run-LiveReplay.ps1`, `--verify-catchup`, `--benchmark`, the native hotkey probe, or a desktop test target now. Those can start capture, register hotkeys, or move focus. Real-game playback, physical hotkeys, uninterrupted focus restoration, fullscreen/HDR/multi-monitor behavior, capture recursion, performance, and anti-cheat compatibility remain deferred. Existing benchmark material is separate evidence and is not a live-acceptance signoff for these changes.
