# Native full-session recording and concurrent questions

The user explicitly selected the existing native recorder and requested implementation: save full sessions to disk, retain a bounded RAM cache for quick replay, and ask questions while recording continues. This supersedes the first demo's bounded software capture. All changes to the GPL GameHQ host remain in its separate checkout; no native source is copied into the Python package. Existing unrelated changes remain intact.

## Recording and playback

The native WGC/Media Foundation recorder is the only encoder. Full-session mode writes finalized compressed segments and an atomic manifest in a unique persistent session directory. Ring cleanup must never remove these session segments. The original ring-only mode remains available. Full-session ownership pins capture to its original game while the browser or replay has focus, releasing it on explicit stop or target loss.

Native replay uses a bounded compressed-data cache (default 128 MiB, configurable); disk remains the durable source and large replays fall back to disk. Cache bytes, cap, hits and misses are visible. Eviction never deletes recordings. Player shutdown releases decoder, audio and cache leases, including deferred teardown. No second capture/encode pipeline is started for queries or replay.

The Python page starts/stops the native host, reads session history, builds immutable replay files by copying compressed streams from completed native segments, asks the existing model connection, and sends selected intervals to the native player. Questions default to recent footage; a session timeline makes older windows available, and a progressive visual index provides candidates for longer-session questions. The index is separate from capture, bounded in queued work, and reports coverage/errors honestly.

## Local bridge contract, version 1

Native flags: `--full-sessions --session-root <absolute sessions directory> --control-dir <absolute bridge directory> --media-root <absolute replay-export directory> --replay-cache-mb 128`.

Atomic `<control-dir>/status.json`: `version:1`, `instance_id:string`, `pid:number`, `recording:bool`, `enabled:bool`, `session_id:string|null`, `error:string`, `cache:{bytes,budget_bytes,hits,misses}`. A UTC ISO8601 `updated_at` heartbeat is published once per second, with immediate updates for state/command/cache changes; commands are polled every 100ms. `active_session` is a bounded summary `{id,game,created_at,state,duration_s}`, retained after stop. Full segment lists live only in session.json, avoiding ever-growing status writes. Stale status cannot pretend a recorder is alive.

Atomic `<session-root>/<id>/session.json`: `{id,game,created_at,state,duration_s,segments:[{path,start_s,end_s}]}`. `state` is recording/closed/error. Times are seconds on one session timeline, based on actual finalized segment durations. Paths are absolute and stay inside this session directory. `duration_s` is the finalized duration; incomplete media is never listed. Closed manifests remain discoverable after native exit.

Python publishes `<control-dir>/commands/<32-hex-id>.json`, native replies atomically to `<control-dir>/responses/<same-id>.json`. Every request includes `id`, `type`, and `instance_id`. Every reply echoes `id`, includes `ok` and `error` on failure. Reject stale instance IDs and malformed/oversized requests. Commands:

- `flush`, with `session_id`: finalize the current segment, immediately continue recording, publish the manifest, then acknowledge. A stale/wrong session is rejected.
- `play`, with `path,start_s,end_s`: path must be a finalized replay under `media-root`; times are relative to that replay file. Open in the native player, using RAM when within budget, with explicit fixed playback bounds. Validate finite bounds and refuse unrelated files.
- `stop`: finish session recording and publish the final manifest. The player/bridge may remain running, but auto-recording must stay disabled until restarted.
- `start`: re-enable full-session auto recording after stop, waiting for an eligible foreground game without changing focus. A new recording uses a new session ID.

An uncompleted command times out after eight seconds and receives a failed reply. Timed-out playback is hidden and unloaded before bridge serialization is released so Stop remains usable; late completions from timed-out commands cannot affect newer commands.

## Python boundaries

`NativeRecorder` owns launching the host (without visible console), status freshness, generation-tagged requests, persistent manifest discovery, validated segment selection and lossless replay assembly under MatchStore. A `NativeSnapshot` exposes `match` (existing Match), `path`, `duration`, `origin_s`, and `session_id`; `origin_s` maps the exported file's zero to the session timeline.

`SessionSearch` consumes plain manifest dictionaries and a snapshot callback `(session,start_s,end_s)->NativeSnapshot`. It can progressively index closed footage without stopping capture. `ask(session,question,start_s=None,end_s=None)` returns `{snapshot,result,session_id,scope,indexed_until_s}` where result is the existing find_moments result, and snapshot is the actual footage used. Search candidates must be visually checked. No-match/model-error behavior cannot replay unrelated footage or interrupt recording.

## Verification

Native generated-texture encode/decode tests must prove full history survives normal ring rotation, older intervals remain selectable, flush/replay permits continued encoding, cache stays within cap, and close releases media. Python tests use real generated media/FFmpeg and file IPC, with only model/OS boundaries doubled. Test stale generations, invalid paths/ranges, missing segments, a query concurrent with recorder progress, long-session history and timestamp offsets. Run existing native and Python suites. Do not move desktop focus, launch games or start live capture without a new explicit request; the user stopped earlier desktop automation. Report live-game/performance evidence separately from automated tests.
