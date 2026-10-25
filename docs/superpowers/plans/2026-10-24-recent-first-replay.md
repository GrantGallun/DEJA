# Recent-first replay implementation plan

**Goal:** 7 opens five seconds behind live by default, adjustable, with full history on demand.
**Architecture:** Separate initial cursor from playback range; short quick exports; explicit lazy history export with timeline metadata and preserved playhead.
**Spec:** ../specs/2026-10-24-recent-first-replay.md
**Stack:** Existing Python/settings page and separate native Qt/Media Foundation experiment.

Work in place; preserve both repositories' uncommitted changes. The user has specified this behavior; no additional confirmation is needed. No desktop automation or automatic game inputs.

- [x] Native capture owner: tests first for full-session snapshot beyond ring bounds and timeline metadata, then worker negative-seconds full-history branch and immutable TemporaryReplay metadata. Build affected test targets only until coordinated shutdown.
- [x] Root: settings tests, initial-position policy regressions and offscreen player tests first; implement mirrored validators, adjustable UI, initial seek, lazy history action and preserved session cursor. Maintain explicit AI interval bounds and existing input routing.
- [x] Root: review, affected checks/build, cleanly finalize current native session before replacing the host; start the updated local services and verify defaults/runtime readiness. Keep generated verification separate from real-game rehearsal.

## Evidence

- Settings tests: 46 passed, including the adjustable page/API round trip, the default of 5, persistence and invalid-value rejection. Native/Python defaults agree.
- Final standalone native build succeeded; all 8 CTest suites passed in 92.13s. Offscreen media tests observed the old incorrect initial cursor before the fix and verified the separate cursor and retained seek bounds afterward. Generated tests cover full versus bounded history, frozen offsets, continued encoding and cooperative cancellation/preemption.
- Six page-script tests passed. Final full Python suite: **143 passed, 1 skipped in 48.51s**. An earlier generated-probe failure was reproduced before fixing its COM apartment layout: the fixture now puts encoding on an MTA worker and Qt playback on the STA GUI thread, matching production. The formerly failing focused integration passed twice before the final full run. Heartbeat freshness and timeouts were not weakened; only the test probe was rebuilt, leaving the running demo untouched.
- Independent review identified a long-history export blocking a reopened quick replay. Fixed with one latest pending request and cooperative full-export interruption. Reviewer audited the fix and found no additional concrete issue. Cancellation cannot interrupt an in-progress Media Foundation call.
- The prior active session f58ac4a027ee4a0b985097554dfb7f63 was cleanly closed before replacing its host: 48 segments, 214.3299647s. The demo server stayed running. New native host PID 14172/generation f295cf2fbaaf4bc4b06cbd2beefb03df reports registered 7/8 with no settings issues. Settings service 8765 reloaded; HTTP confirms default/current start 5, history key H and the visible control.
- The user confirmed the live five-second and full-history check works. Native logs also show quick cutoffs 33628/30881 with initial cursors 28628/25881 (exactly 5000ms behind), plus full-history length 51963 with preserved initial cursor 14375. Foreground remains the game.
- The requested 30 seconds of footage was only a test that history extends beyond the quick window. There is no 30-second admission gate: Recording and Ready states both accept a replay, including a partial first segment. A fresh recorder session can show only the footage it has captured; prior sessions stay on disk.
