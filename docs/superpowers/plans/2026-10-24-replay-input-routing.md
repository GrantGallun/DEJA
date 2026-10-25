# Foreground-preserving replay input implementation plan

**Goal:** 8 diverts input to replay controls while the game retains foreground/fullscreen.
**Architecture:** Worker-thread input gate, bounded input mailbox, direct Qt action/event delivery and a virtual replay pointer. No OS focus switching.
**Tech stack:** Existing C++20/Qt 6.8 player and Win32 low-level hooks.
**Spec:** ../specs/2026-10-24-replay-input-routing.md

Work in place to preserve required uncommitted changes in both repositories. The user has specified the interaction; implement it without another approval round. No live desktop testing or automatic game inputs.

- [x] Gate owner: add ReplayInputRouter/ReplayInputPolicy and pure policy regressions first; worker hook lifecycle, owner/generation/heartbeat guards, held-input transitions and bounded queue. Build tests in an isolated directory, never install real hooks in tests.
- [x] Root: regressions for game-owned input and offscreen action/mouse delivery, then replace ReplayController activation with gate ownership. Keep nonactivation styles, 7/8/configurable controls and capture independent. Draw virtual pointer and preserve local drag/resize without system move loops.
- [x] Root: integrate native CMake targets, build and run affected/native checks, independent source review, update runbook. Preserve the active archive during a necessary host restart; do not run computer-use tools.

Review focus: stale queued input after exit; originally-held game releases; missing exit shortcut; GUI stalls; external focus loss; raw-input/cursor limitations; no foreground or global-cursor mutations.

## Evidence and runtime

- Router policy/lifecycle suite: 13 QtTest cases passed; tests never installed global hooks. Fixed 128-event queue, coalesced pointer updates, 25ms worker checks, 500ms GUI watchdog and 200ms bounded release drain.
- Observed failing then passing regressions for direct inactive QML action/mouse delivery, game-owned foreground in the benchmark contract, eligible game selection for stopped-session archive controls, outside-click release, and keyboard-only outside-click release.
- Independent source review found the stopped-session target gap and keyboard-mode outside-click gap; both fixed with regressions. The pointer companion was reviewed for nonactivation and the outside press for cancellation/drain behavior.
- Final standalone native build succeeded. All four affected CTest suites passed: temporary_replay, replay_media, player_settings and replay_input_router (18.58s). Existing unrelated capture suites were not rerun for this input-only change.
- Updated local server restarted at port 8766; new native PID 16824, generation 90966f881e9b496cb1a9fc288d37f5f0. Player status confirms saved 7/8 bindings registered without issues. Recording armed; hooks remain inactive until the user explicitly takes controls.
- The old launcher was stopped before the asynchronous recording stop published its final entry. The intact unlisted last MP4 decoded successfully; its duration was read from the video metadata and appended after validating the full manifest. The original manifest is preserved as session.before-recovery.json. The closed saved session has 46 segments and 207.431565s. No session media was deleted.
- Manual Stick Fight test requested asynchronously. The user's live gestures produced input_capture_started at 04:43:39.692Z and outside_click_returned_input at 04:43:41.133Z, both with foreground/target 0x30d78; recording continued and no focus-target disappearance followed. This confirms actual hook startup, the outside-exit path and preserved native foreground. It does not by itself establish that every game input backend was isolated or that the virtual pointer looked correct; the user's visual/input confirmation remains pending. No computer-use tools or automated game inputs ran for this change.
- The user then confirmed **Works as expected** to the combined live Stick Fight check: stays fullscreen, replay receives the temporary controls, game stops responding to those inputs, and outside-click returns input. This completes the requested interaction rehearsal for Stick Fight. Broader engine/raw-input compatibility and comparative performance remain separate, unmeasured work.
