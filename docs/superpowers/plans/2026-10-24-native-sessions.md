# Native Sessions Implementation Plan

**Goal:** Full native recording with bounded replay memory and concurrent questions.
**Architecture:** Persistent native segments and manifests; an instance-tagged local file bridge; native cached playback; Python query/index jobs independent of recording.
**Spec:** ../specs/2026-10-24-native-sessions-design.md
**Tech Stack:** Existing Qt/C++/WGC/Media Foundation host, Python/FFmpeg lossless remux, existing Codex/Claude adapters, existing local page.

## Constraints

Preserve uncommitted work and existing ring/player tests. Native source stays in experiments/gamehq-passive. No desktop actions or live capture during automated checks. Work in place because both codebases contain required uncommitted changes. The user has approved this implementation direction; no additional design confirmation is needed.

## Tasks

- [x] Native storage/cache/bridge: native owner implements session archive, persistent recording ownership, explicit flush/play/stop bridge and bounded RAM-backed media; adds generated native regressions and rebuilds/tests the standalone host.
- [x] Session search: separate owner implements progressive indexing and older-session candidate retrieval in smartreplay/demo/session_search.py with model doubles only; uses the fixed manifest and snapshot interfaces.
- [x] Python native client: root implements manifest/status validation, launch/stop/flush/play generation fences, lossless snapshots, and real-media/IPC tests in native.py/test_native_demo.py.
- [x] App integration: root replaces timed capture with native session controls; serializes question jobs independently of recording; session/window selection and native replay; maintains imported-file fallback.
- [x] Verify and review: full Python/JS/native suites, source review, corrected regressions, dry-run launch checks and accurate runbook. Keep live-game rehearsal distinct.

## Review focus

Recording must survive browser/model focus and errors. Never delete session media on cache eviction. Reject stale native status/requests and paths escaping archives/media root. Cache leases cannot outlive the accounted memory budget. Manifest timestamps and lossless-remux offsets must match what the player displays. Progress/index coverage must not imply an unindexed session was exhaustively searched.
