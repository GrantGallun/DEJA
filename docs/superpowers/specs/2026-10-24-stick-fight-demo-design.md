# Stick Fight demo

The user approved the complete fresh-match -> question -> matching-replay flow in this chat. One borrowed controller and keyboard/mouse let two judges play on one PC. The presentation lasts 3–5 minutes.

## Approach

Use a small local browser demo with its own game-window capture, the existing Claude model adapter, and a browser video player. This is the smallest independent implementation of the approved flow. Extending the GPL GameHQ experiment would require native IPC, rebuilds and a deliberate integration decision; replaying a prepared recording would not demonstrate a fresh answer. Leave both Claude's ongoing analysis work and the GameHQ experiment intact.

Record only the visible window belonging to StickFight.exe, for at most 25 seconds by default. Bring the game forward when recording starts. Finalize an H.264 MP4 in a unique match directory. Keep that immutable recording through Q&A and across subsequent recordings. Never substitute another recording when capture fails.

The question box accepts free text. Sample the actual recording, ask the configured model for visible matching moments, and refine around candidates with denser frames. Speech is optional and is not a dependency. Enforce timestamp bounds and distinguish confirmed, uncertain and absent matches. Each result has evidence, a timestamp and a directly playable interval. No canned answers in production; injected model answers are confined to tests.

The page presents game/backend readiness, record/finish controls, processing state, question input, results and replay controls. Model failure keeps the video usable and explains the failure. A file-input launch mode supports reproducible checks and clearly labels imported footage. The server binds to loopback, rejects foreign origins, and serves only its own recorded media with byte ranges so browser seeking works.

## Acceptance

- A newly captured match survives player close and later captures.
- The question is evaluated against that match, and selecting a result seeks to its interval.
- Invalid or absent results cannot silently play unrelated footage.
- Recording, finalizing and searching serialize safely; failures remain recoverable.
- Generated-video/HTTP integration checks and the existing Python suite pass.
- Actual capture, browser behavior and model availability are checked separately and reported honestly.
