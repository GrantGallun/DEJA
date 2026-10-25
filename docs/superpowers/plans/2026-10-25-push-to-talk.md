# Push to talk implementation plan

**Goal:** Hold an adjustable shortcut (default 9), ask about gameplay, release, and automatically open a visually supported matching replay while native session recording continues.

**Architecture:** Native Qt microphone capture writes a bounded WAV and atomic request manifest. The demo service claims each current-instance request once, transcribes locally using the installed faster-whisper model, invokes the existing session vision search, and plays the first confirmed match. Speech capture never changes OS foreground or installs input hooks. No speech is captured before the key is held.

**Approaches:** Browser microphone capture requires browser focus/permission during fullscreen gameplay. Windows dictation would couple the feature to OS UI. Use native Qt capture and the existing local speech engine, which fit the current player and keep the game foreground. This is an implementation choice under the user's request to start implementing push to talk.

## Interface

- New mirrored setting `hotkeys.push_to_talk`, default `9`, optional/unbindable, validated for collisions like other global hotkeys.
- Native queue: `<control_dir>/voice/requests/<32-hex-id>.json`, version 1, `id`, `instance_id`, `session_id`, absolute `path` to `<control_dir>/voice/<id>.wav`, and UTC `created_at`. WAV fully committed before manifest. Maximum 15 seconds, bounded bytes; release submits, focus/session change cancels, no queue during a pending request.
- Native status includes `voice` with `state`, `message`, `request_id`, `transcript`. Feedback command `voice_status` carries `voice_id`, `state`, `message`, `transcript`; accept only the current request id. Terminal states `replaying`, `no_match`, `error`, `idle` release the pending voice request. Processing states `transcribing`, `searching`. Listening is native-only. A watchdog releases abandoned requests.
- Python `NativeRecorder.claim_voice_request()` returns a validated current-instance request plus a Path or None, consumes invalid/stale records, rejects traversal/oversize/stale identities. `voice_feedback(request, state, message, transcript='')` fences feedback to the original instance. Remove microphone audio after use.
- Python `VoiceTranscriber.transcribe(path: Path) -> str`: model lazy-loaded once, cached model available locally, silence rejected, 800-character maximum. Test with deterministic WAV and injected fake model; verify real decoding separately using synthesized speech, without recording the user's microphone.
- Reserve the existing query busy state before transcription. Use the request's session identity, never a browser selection. Transcribe/search/replay off HTTP and Qt capture threads. Do not auto-play uncertain/no matches. Mark successful voice auto-play in results to prevent the web UI issuing duplicate native playback.

## Tasks

- [x] Native worker: tests first for WAV limits/release/cancel/fencing, Qt microphone source, key hold/release polling, bridge feedback, passive visible status, CMake wiring. Preserve pending 7/8 changes.
- [x] Service worker: tests first for exactly once claim, stale/path validation, voice/query concurrency, session identity, confirmed replay, no-match/error/cleanup/shutdown; voice status in demo UI.
- [x] Main: mirrored Python settings/UI and tests, local transcriber tests and implementation, speech dependency/runbook, integration review.
- [x] Build and run affected suites; verify local speech on synthesized audio; restart services and recorder, inspect registered keys/status without automating user input. User performs live microphone/game rehearsal.

## Constraints and review focus

Work in both dirty checkouts in place; no commits or resets. No Computer Use, focus changes, synthetic keys, live microphone tests, or test input hooks. Preserve recording ownership and finalized sessions. No account/key changes. Failures must release voice busy state and leave the recorder running. Check held-key release, lost focus, stale native generations, duplicate result playback, service restart and missing microphone/model.

## Implementation and checks

The user confirmed 9. Native capture supports PCM16 mono/stereo, 8–48 kHz, with a 15-second / 2,880,044-byte WAV bound. Speech is local and the model loads once; generated speech recognized the exact question, with 15.98 seconds cold and 0.38 seconds warm. The service warms the model in its background worker when the native recorder becomes available.

Feedback uses `no_reply=true`; automatic playback also carries `voice_id` and `session_id`, checked at native execution and during media preparation. Native cancellation, watchdog expiry and recorder replacement fence later answers. Changing replay input ownership cancels an in-progress microphone capture to avoid ambiguity around swallowed key releases. The existing gate publishes a bounded atomic physical-key snapshot; no new input hook is installed for speech.

Independent review found and verified fixes for delayed playback after cancellation, status races, duplicate browser playback, and shutdown callback lifetimes. Frequent status polling exposed a Windows atomic-file replacement race. Deterministic tests hold a reader over publication, then release it; a shared bounded commit retry now protects both heartbeat and session manifests without changing their atomic replacement semantics. Persistent failures remain bounded and preserve the prior complete document.

Final parent verification: **205 Python tests passed, 1 skipped in 49.15 seconds**, including the rebuilt real native bridge/generated-recording probe; **all 9 native CTest suites passed in 86.52 seconds**; **8 JavaScript tests passed**. The native voice suite includes 13 checks, including transient/persistent reader contention. Both checkout tracked diffs passed whitespace checks.

Runtime restarted and checked on 2026-10-25: settings service on 8765, demo service on 8766, native PID 35244. Settings report voice=9, automatic quick-replay controls=true and lookback=5 seconds. Player status reports 7/8/9 registered with no settings issues. Speech model is ready, GPT-6-Luna via Codex is connected, and the native recorder is enabled/armed with no error; the foreground app is currently outside the game so no fresh capture has begun. No microphone or game input was automated. Previous session e7687502a086443c8a8ac4b37aabb8f1 remains closed at 156.25 seconds. Live microphone pickup and end-to-end gameplay timing remain for the user's rehearsal.
