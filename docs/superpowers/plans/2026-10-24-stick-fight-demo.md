# Stick Fight demo implementation

Spec: ../specs/2026-10-24-stick-fight-demo-design.md

Implement inline in the existing feature branch, using new demo modules so Claude's uncommitted modules remain untouched. The user has authorized execution of the proposed demo. No push or merge is requested.

1. Capture and storage (`smartreplay/demo/capture.py`, `tests/test_demo.py`): write failing checks for immutable recording import, invalid videos and capture command construction; implement unique match storage, bounded game-window FFmpeg capture and explicit stop/finalize. Verify using generated actual video.
2. Visual search (`smartreplay/demo/search.py`): write failing checks for timestamp grounding, absent/uncertain answers and selected clip bounds; implement timestamped frame sampling, structured model calls using the existing adapter, and dense verification. Fake only the external model in tests.
3. Session and HTTP (`smartreplay/demo/server.py`): write failing integration checks for request sequencing, failed queries, origin checks, media byte ranges and isolated match identity; implement threaded jobs and safe local routes. Video playback must remain available after a model error.
4. Demo page and launcher (`smartreplay/demo/static/`, `smartreplay/demo/__main__.py`, `Run-Demo.ps1`, `requirements-demo.txt`, `docs/STICK_FIGHT_DEMO.md`): implement the three-step UI and setup/run instructions; verify JS syntax and the browser flow, then exercise actual game-window capture when available.
5. Run full Python suite, review the change, fix important defects with regression checks and report actual live-model/capture evidence separately from simulated checks.

Review focus: media lifetime, cross-match stale responses, malformed model timestamps, no-match behavior, pending capture shutdown, local-server exposure, real range requests and browser seeking. Model accuracy and latency require real-model footage tests rather than injected answers.
