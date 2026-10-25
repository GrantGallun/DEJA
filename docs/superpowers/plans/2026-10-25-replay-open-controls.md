# Replay opening controls

## Approved scope

The user wants 7 to take replay controls automatically, with an adjustable passive/active opening setting. They confirmed that 8 should also summon the hidden player and take controls. Keep OS foreground/fullscreen unchanged: these actions use the existing input gate. Outside click and 8 still release controls.

Default input.take_controls_on_open=true. Sample it when 7 opens; 8 explicitly requests controls regardless of that setting. Acquire only once a valid replay frame/range is ready and the game still owns foreground. Hide, failure and release cancel pending automatic acquisition. Repeated 8 while loading toggles the pending intent rather than taking input for an unusable player. Explicit question-selected intervals keep their existing control behavior.

## Tasks

- [x] Tests first: mirrored Boolean default/validation, persistent settings/API; open-control intent state (active/passive, explicit demand, cancellation, one-shot readiness).
- [x] Implement settings checkbox and native state/helper integration; preserve media request fencing, current replay timing and foreground guards.
- [x] Build/test affected native paths, source review, then finalize recording before host replacement; reload settings helper and verify runtime. No Computer Use or automated game inputs.

Completed together with push to talk. Final combined verification: 205 Python tests passed, 1 skipped; all 9 native suites and 8 JavaScript tests passed. Native PID 35244 is running with 7/8/9 registered and no settings issues; settings report automatic controls=true and lookback=5. Recording was finalized and verified closed before replacing the native executable. Manual live testing of the new automatic opening behavior remains with the user.
