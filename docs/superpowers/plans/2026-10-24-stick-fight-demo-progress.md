# Progress — plan: 2026-10-24-stick-fight-demo.md

The user approved and requested implementation of the demo discussed in chat.

Ruling: Work in place on player-settings-app using new demo files — current model experiments are uncommitted and the separate native checkout is ignored; a new worktree would omit both. Leave existing changes intact.
Ruling: Use bounded game-window FFmpeg capture and browser replay for the first demo — no native IPC exists, and repository policy keeps the GPL experiment separate. The cost is that this demo does not exercise the native rolling-buffer mini-player.
Pre-flight: capture produces immutable Match objects; search consumes their exact path and duration; session retains their identity throughout jobs; HTTP serves only these objects. All three use seconds relative to the finalized MP4.

Tasks 1–4: implementation complete. Unique retained match files, bounded game-window recording, dense visual verification, isolated session jobs, local HTTP/range serving, demo page, launcher and runbook are implemented.

Task 5: automated verification complete; actual online-match capture and native browser seeking remain a manual rehearsal item.

Ruling: Default to Codex after the user authorized using their larger Codex allowance. The installed model cache advertises GPT-6-Luna with image input; a real Codex connection succeeded. Keep Claude selectable without changing either account's sign-in. The cost is using Codex subscription usage for this local prototype; it is not a distribution/authentication plan for a shipped product.

Environment evidence: restricted runner cannot enumerate desktop windows (EnumWindows false) or open outbound sockets (WinError 10013). Normal-access readiness sees Stick Fight and reaches Anthropic. The standalone Claude account reports its weekly limit, although the user has a different account signed into the desktop app. No credentials were extracted or changed. The local server was restarted with normal access.

Final review: independent read-only review found three Important defects, all fixed with failing-then-passing regressions: prior replay surviving an absent/uncertain answer, request errors erased by polling, and imported video accepted without browser-compatible encoding. Imports now normalize to H.264/4:2:0 MP4. No Minor findings were reported.

Verification: full Python suite 86 passed, 1 skipped (17.21 s); four page-script regressions passed; JS syntax and git diff whitespace checks passed. Real GPT-6-Luna image search over a generated 25-second clip returned the falling yellow object with a grounded replay interval in 11.83 s; an absent purple-player request returned no match in 5.69 s. These prove live image transport and result plumbing, not Stick Fight accuracy. Evidence is under ignored out/demo/verification/.

Desktop control was stopped by the user. No subsequent game inputs or capture were attempted. Actual multiplayer footage needs the user's online match or the borrowed controller. Do not describe this as a verified live-game demo until that rehearsal passes.
