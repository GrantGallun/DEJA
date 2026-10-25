# Replay controls without changing foreground

The user clarified that 8 must temporarily divert mouse/keyboard to the replay while the fullscreen game stays foreground. Returning focus after minimizing is not the requested behavior. Keep 7 show/hide and the existing configurable player actions.

Replace activation with a dedicated, temporary Win32 low-level keyboard/mouse gate. The replay always retains nonactivating window styles. A worker message loop consumes supported input only while the original game HWND/PID is foreground, the replay is available and its GUI heartbeat remains fresh. It queues small input events to Qt; keys dispatch configured replay actions directly, and mouse events go to the inactive Qt window with a replay-drawn virtual pointer. No SendInput, global cursor warping, ClipCursor fighting, or foreground API calls in the new path.

User steering: an outside click must return game input. A tiny nonactivating, click-through cursor companion keeps the virtual pointer visible beyond the player. The first outside press releases routing and is consumed, avoiding a simultaneous shot. This applies to keyboard-only input mode as well as mouse/hybrid.

Pass releases of pre-existing game-held controls through; drain releases belonging to replay-held presses on normal exit. Fail open on target/focus loss, stale GUI, hide, shutdown or hook failure. System navigation must remain available. Exit/toggle shortcuts must work inside the gate because normal registered hotkeys may be consumed. The gate never records key contents to disk and is not active during questions alone.

Scope: the standalone native experiment; preserve unrelated work and full-session capture. Tests must use pure policy and offscreen Qt event delivery, without installing hooks, launching games, desktop automation or changing live focus. A real-game manual check is required before claiming Stick Fight input is fully isolated. Hook suppression does not establish universal Raw Input/DirectInput isolation, and constrained/recentered mice may require a different backend.
