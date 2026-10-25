# Progress — plan: 2026-10-24-native-sessions.md

User approved native full-session architecture and explicitly requested implementation.
Ruling: Modify the separate GPL native experiment and connect it as the selected demo host; do not copy native source into the Python project. Prior independent-host direction is superseded for this requested demo integration.
Ruling: Work in place with disjoint native, session-search and root integration file ownership, preserving both repositories' existing uncommitted work.
Baseline: prior Python suite 86 passed/1 skipped, four page-script tests; prior native checkpoint six passing suites. Fresh baseline native checks belong to the native task.

## Implementation

- Native owner completed persistent full-session manifests and segments, separate recording ownership, flush/start/stop/play file bridge, 128 MiB compressed replay cache and generated-media regressions. The original game target remains owned across browser focus and minimized/hidden visibility, and releases on explicit Stop, target destruction or process replacement.
- Session-search owner completed progressive 25-second sampled indexing, bounded queue, interactive priority, retry backoff, candidate selection and visual verification. A separate monitor follows finalized footage even when no question is asked or another session is selected.
- Root completed the native client, generation/status/path/range validation, lossless immutable replay assembly and browser/server integration. Capture controls remain available during questions and native replay loading. Imported short-video fallback remains available.
- Independent source review found the implicit short-session search omitted the beginning of 25–30-second sessions. Fixed and covered by a regression. Root also added bounded-range sampling so a short question window works inside a longer lossless export without a second encoder.

## Verification

- Final Python project suite: **132 passed, 1 skipped**, 46.53 seconds (`python -m pytest tests -q`). The optional slow test remains opt-in. The real native probe ran, including encoding, bridge commands, lossless export, repeat RAM-backed play, recording progress during replay, Stop and a new session after Start.
- Page-script suite: **6 passed**, including stale-answer clearing, persistent request errors, recording independent of questions and Stop remaining enabled during native replay loading.
- Native standalone build succeeded after the capture-ownership fix. Final native CTest run: **7/7 suites passed**, 80.98 seconds, including full-session persistence, ownership, snapshots, live replay, media, settings and native session cache/bridge checks.
- Earlier real Codex checks established text connectivity and generated-image event/no-match answers; the final local launcher checks the connection again. Automated session/model tests use boundary doubles and do not establish real-game recognition quality.
- Both repositories pass `git diff --check`. Existing uncommitted work is preserved; no commit, push or merge was requested.

## Demo state and remaining rehearsal

The updated local server has been restarted on port 8766 with recording off. HTTP checks confirmed the new native controls/session picker, available Stop controls during replay requests, idle recording state and a connected GPT-6-Luna backend. No desktop focus changes, game launch, game input or live capture were performed for this implementation. Generated WARP textures exercise the production native recorder/player without desktop capture.

Actual Stick Fight capture, target behavior, model accuracy, response time and comparative game overhead remain a manual rehearsal. The RAM limit accounts for compressed replay bytes, including pinned retiring-player leases; it does not cap total process memory. The first uncached replay can require a lossless disk remux. Full recordings and replay exports persist on disk without automatic deletion.
