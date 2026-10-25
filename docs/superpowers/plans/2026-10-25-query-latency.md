# Query latency implementation plan

**Goal:** Reduce time from a spoken question to its supported replay, retaining visual verification and continuous native recording. No language port is authorized or needed for this change.

**Evidence:** Live quick replay logs show 304–484 ms to first frame. Two actual spoken questions recorded 17.03 and 20.25 seconds of session search. A controlled search on saved footage took 19.06 seconds: 7.31 seconds coarse model, 6.11 seconds verification model, with the remainder mostly video sampling. Fifty random-seek samples take 3.5–3.8 seconds; 24 dense samples take 1.6–1.8 seconds. OpenCV and FFmpeg already execute native code.

**Approach:** Keep the existing C++ capture/player and Python coordinator. Correct EOF rounding in segment index sampling, use monotonic timestamp-based decoding for nearby samples, and skip the repeated coarse scan after an indexed candidate has already been selected. Keep the wider search's ordinal selection and dense visual rejection of unsupported events. Add stage timings to search results for future measurements.

## Tasks

- [x] Reproduce and fix the healthy near-EOF segment index failure, with truthful actual frame timestamps and corrupt-file rejection (service agent owns `_frames` and tests).
- [x] Test first, then optimize `search.sample_frames` with timestamp-based sequential decoding; retain bounded resolution, sorted unique requested positions, missing/empty-file errors and variable-rate correctness. Benchmark saved clips before/after.
- [x] Test first, then allow `find_moments` to accept validated candidate timestamps for dense verification. Integrate indexed selection without a second coarse call; preserve first/second/last selection and no-match behavior. Add timing breakdown.
- [x] Warm native audio input/output metadata in a background startup worker, without opening a microphone. Re-run the generated voice/real replay diagnostic after confirming readiness.
- [x] Run affected and full Python/page checks, compare a controlled real-model search, review changes, safely restart the demo service and verify native capture/shortcuts/speech readiness. Preserve recorded footage and do not automate game input or microphone capture.

## Choices

The user asks for speed improvements and is considering C++ or Rust, not requesting a port yet. The native recorder/player are already C++. A rewrite cannot remove remote inference time; first remove extra work in the existing pipeline. C++/Rust remains a future choice if a measured CPU component justifies it. No model/account changes, no skipped visual verification, no fabricated cached answers.

## Follow-up: persistent model transport

User authorized further autonomous optimization while away. Keep Luna low at Standard usage and the existing recorder/player. Reuse a supported Codex app-server connection, with a fresh ephemeral thread for each request and unchanged structured visual verification. Retain the exec transport as an explicit environment rollback, without retrying failed inference automatically.

- [x] Add a serialized, timeout-bounded transport in `smartreplay/demo/codex_transport.py`; test fresh thread identity, standard tier, disabled tools, malformed replies, failure restart and close during a request.
- [x] Integrate fresh lightweight model wrappers sharing one DemoSession-owned transport. Test independent timeout state, backend switch disposal and shutdown ordering. Keep injected test factories unchanged.
- [x] Reduce indexed snapshot requests from 25 seconds to candidate +/-5 seconds. Dense verification may shift by +/-1.5 seconds, so this retains the full 3-second lead-in and 2-second tail. Real saved-footage assembly measured 1.651s/49.7MB versus .925s/31.8MB; all 24 session retrieval tests pass, including global origin and full replay-context checks.
- [x] Run the Python suite, an actual saved-footage benchmark and a generated voice replay; keep the faster tested route, document measured limits, and safely restart/arm the demo.

Review focus: no cross-question history; no model/tool/tier drift from user config; no old process replies accepted after restart; no leaked subprocess after close; no network wait before native recorder shutdown. The transport agent owns new client/tests, the main agent owns integration, and a separate agent reviews lifecycle.

## Measured diagnostics and research

First full simulated voice call used generated speech, actual local transcription, the real Codex vision model, saved native gameplay and actual visible native playback. Only microphone/voice-token admission was simulated. `out/demo/voice-smoke/simulation-result.json`: 22.489 seconds to first displayed frame, plus 15.061 seconds startup before the request. Speech 1.000s, coarse vision 6.857s, dense vision 5.318s, frame extraction 1.750s, snapshot .921s, native first frame 6.484s. Sampling was faster than the original ~5.5 seconds, but a cold audio enumeration cost dominated native opening. Isolated native diagnostics traced this to the first output-device enumeration (~5.3s); subsequent first frames were ~.337s. Input-device enumeration has an independent ~5.3s cold cost.

A four-call, same-image/Standard-tier transport spike observed app-server requests 3.504/3.640s plus .180/.098s thread creation, versus exec 4.398/6.118s. App-server initialization .287s once. This small sample suggests a persistent connection may help (~29% lower mean), but app-server and exec configuration differ because app-server cannot ignore user configuration. No transport, model, account or speed tier was changed. Exact safe metadata is under `out/demo/transport-bench/`.

Research supports keeping Luna low initially; it is the lowest reasoning effort advertised by this sign-in. [Official speed guidance](https://learn.chatgpt.com/docs/agent-configuration/speed) offers Fast at 2.5x included subscription usage / 2x purchased credits, left disabled. Community/wiki research produced visible question families (deaths, recoil/knockback, weapons/snakes, recovery, blocks, hazards, winner colors). The index prompt now asks for these broad visible cues without asserting hidden causes; source-linked rehearsal examples are in the existing demo runbook. Video metadata was inspected, not video footage/comments.

## Final verification and repeated simulation

Background audio metadata warmup now runs on one native worker, inputs then outputs; getters only, no audio stream. Qt's backend serializes the two device caches; parallel enumeration offered no improvement. An isolated diagnostic measured metadata-ready startup ~10.5 seconds while GUI timers remained responsive. Deterministic tests inject enumerators to verify nonblocking construction, completion/error reporting and joined shutdown.

The repeated generated question, “Show the last time yellow jumped,” used the real speech engine, real Luna requests, saved gameplay, lossless snapshot and actual native first frame. After 11.109 seconds of startup preparation (audio metadata 10.924 seconds), request-to-first-frame was **17.504 seconds**, versus **22.489 seconds** before background audio warmup. Speech 1.101s; coarse vision 7.962s; dense vision 5.109s; sampling 1.484s; snapshot .829s; native first frame .879s. Different valid model outputs/latency between runs mean this is an observed comparison, not a guaranteed response budget. This simulation had no existing visual index, so it used the two-pass fallback; the indexed route's removal of its extra coarse pass is covered by regression tests.

Final checks: **218 Python tests passed, 1 skipped in 45.45s; 8 JavaScript tests passed; all 5 affected native suites passed in 16.43s** (audio warmup, voice, replay media, settings, input routing). Independent review found and fixed sparse-frame leakage outside verification neighborhoods and a model-call timing undercount. Whitespace checks passed in both checkouts.

Clean production service restarted at 8766 (execution session 96168), existing settings service at 8765 retained (48410), native PID 9860. Speech ready, Codex connected, 7/8/9 registered with no issues, recorder armed. Live index advanced to 25 seconds with no error on the previously stuck archived session. The prior user session 457626552bb247ea952eb9d1a09cdbcc was finalized and retained at 123.6537658 seconds before testing. Simulation hosts exited cleanly. No credentials, model, subscription speed tier or production transport changed; no OS focus change, synthetic key or live microphone capture was used.

## Autonomous follow-up results

Persistent transport is now the default. The actual installed server verified `gpt-6-luna`, low reasoning, Standard/default tier, read-only execution and no approvals. Apps/plugins/hooks/tools are disabled. Empty MCP-table overrides merge with user configuration, so the client discovers server names, restarts with individual disable flags and verifies disabled/empty inventories before inference. The CLI's dotted override parser does not accept quoted server keys; unsupported names fail closed. Fresh ephemeral threads unsubscribe after completion; request deadlines, malformed output, full response queues and concurrent close are covered by tests. One session owns the connection, while each operation has its own timeout settings and obsolete index generations are fenced off.

Six counterbalanced production requests used identical 24-frame input and the same prompt/schema. Persistent times were 6.752s (cold), 4.509s and 3.498s; exec times were 5.360s, 6.182s and 8.507s. Medians **4.509s versus 6.182s (~27% lower)**, including setup on the first persistent request. Outputs varied between supported, uncertain and no-match results; this is a small latency sample, not a quality equivalence claim. No higher-usage speed tier was enabled. Evidence: `out/demo/transport-bench/production-results.json`.

The first generated voice test (last yellow jump) correctly returned no verified match in 4.823s and did not play a clip. The saved footage is mainly the join screen, and the sampled index did not establish that ordinal jump. A position question found a verified candidate, but the existing player hid because its game target was minimized (`target_unavailable`); this was not counted as successful playback. Both diagnostic reports retain these outcomes.

After safely finalizing session `9682c172eb504d54941448c8059e1fcf` at 12.1558999s and restarting the service, the generated question “Show yellow high in the upper left corner” reached a real native first frame in **12.610s** on a warmed system. Speech 1.213s, indexed selection 3.297s, dense verification 5.329s, sampling .453s, snapshot 1.219s, native first frame .924s. Startup preparation 12.097s was measured separately. This uses a different question and the completed index, so it is not a controlled comparison with the earlier 17.504s fallback run. Evidence: `out/demo/voice-smoke/simulation-persistent-position.json`.

Final full suite: **244 passed, 1 skipped in 51.86s**. Independent review's early-notification, thread cleanup and full-queue wakeup issues are resolved; whitespace checks pass. Production service execution session **73364**, native PID **6932**, settings service **48410** retained. Codex persistent connection connected, speech ready, 7/8/9 registered, recorder armed. No real microphone, synthetic input or focus manipulation was used; all saved recordings remain intact.

## DeepSeek API diagnostic

The user supplied `DEEPSEEKAPI_KEY` in the local `.env` and explicitly requested an API test. The probe reads that value without logging it or placing it in shell arguments, calls only `https://api.deepseek.com`, disables thinking, requests bounded JSON output and validates the existing replay schema. `.env` is now ignored by Git. The diagnostic lives under ignored `out/demo/deepseek-bench/`; no production backend or recorder configuration was changed.

The account lists `deepseek-flash` and `deepseek-v4-pro`; the test used Flash. Three 24-frame requests took **2.212, 2.174 and 2.314 seconds**, versus counterbalanced persistent Codex requests of **8.985, 3.693 and 4.322 seconds**. Medians **2.212 versus 4.322 seconds (~49% lower)**. The ordered JPEGs/query were identical; DeepSeek JSON mode receives the schema in a system instruction, whereas Codex uses native structured output. Two DeepSeek repeats found no supported second jump and one returned a jump candidate; Codex returned a candidate each time, but this cropped lobby sequence cannot establish the global ordinal independently. This is latency evidence, not proof of equal accuracy. The absent purple/rocket-launcher control returned no match in 1.627s.

Generated speech asking “Show yellow high in the upper left corner” reached a prepared matching replay in **7.390s**: transcription 1.008s, index selection 1.484s, dense verification 2.891s, frame sampling .390s and snapshot 1.469s. **Player opening was not performed or included.** The first selected frame was inspected and visibly contains yellow in the upper-left area of the join screen. Compared with the prior complete 12.610s test, native opening must be accounted for separately; these are not identical end-to-end measurements. Reports are `check.json`, `compare.json`, and `session.json` in the diagnostic directory. A real combat/ordinal evaluation remains necessary before switching the live backend.
