# Native session demo

Run `./Run-Demo.ps1` from the repository folder. The page opens at <http://127.0.0.1:8766>. Keep the launcher running. Run it from your normal Windows terminal so it can see game windows and reach the model service.

The native host lives in the separate GameHQ checkout. Build it with `./experiments/gamehq-passive/spike/passive-replay/build.ps1` when missing or after native source changes. The page reports a missing build rather than falling back to another capture method. Python dependencies: `python -m pip install -r requirements-demo.txt`.

## Record and ask

1. Check **Model connected**. **Codex · fast** uses the existing Codex sign-in and GPT-6-Luna at low reasoning. Claude remains selectable; its standalone account can differ from the desktop app's account. **DeepSeek · fastest** (about half Codex's latency in testing) uses `DEEPSEEK_API_KEY` from `.env`; some networks, campus Wi-Fi especially, block it. The provider is also a setting in deja (**AI model**): the demo follows a save within a second and re-checks the connection, so switch there if one is blocked. The demo never changes any sign-in.
2. Open Stick Fight and join online with keyboard/mouse, or join locally with the borrowed controller. Keep the fullscreen mode you want to rehearse.
3. Click **Start native recording**, then switch to the game. The recorder starts when it detects an eligible foreground game. It pins that target when you return to the browser, so asking and replay do not relinquish recording ownership. Minimizing/hiding also retains the session, although Windows may pause frame delivery from a minimized game. Closing the game ends that target's recording.
4. Return to the page whenever you want to ask. Recording has its own status and remains independent of question processing. Stop stays available during a question.
5. Choose the recording session and search scope:
   - **Session · indexed moments:** uses the sampled background index to propose an older moment, then visually checks its footage. The page shows index coverage. This is not an exhaustive event inventory.
   - **Latest 25 seconds:** searches recent footage directly, including a flushed final segment.
   - **An earlier timestamp:** searches a 25-second window beginning at the session second you enter.
6. Describe something visible, preferably by player colour: “Show the last time yellow died” or “Show a player falling off the map.” Supported results select a fixed replay interval in the browser and send the same interval to the native player. Uncertain candidates require a click; absent events do not select unrelated footage.
7. **Full session** prepares a replay of the recorded history up to a fixed cutoff. Recording continues into later segments. **Watch this window** plays the selected snapshot in the browser. **Stop recording** finalizes the native session; its files remain available. Start again creates a new session.

Existing native defaults remain **Ctrl+Alt+R** show/hide replay, **Ctrl+Alt+F** switch replay/game controls, and **Ctrl+Alt+Shift+R** quit the native host. Laptop F8 toggles airplane mode, so it is not a default. Player settings remain in the existing settings app.

This machine's saved shortcuts are now **7** show/hide and **8** take/release replay controls. Release held movement/fire controls before entering. Control mode diverts Windows keyboard/mouse messages while the game remains foreground. The player stays nonactivating, displays a virtual pointer, and accepts its configured playback keys and buttons. Click outside the player or press **8** to return input; the first outside click is consumed so it cannot also fire a shot. Escape closes the player while replay controls are active; passive-player Escape belongs to the game. Held game-control releases pass through, and releases belonging to captured replay presses drain briefly on exit.

**7 now takes replay controls automatically by default.** Uncheck **Take controls when a quick replay opens** in Controls & playback for the passive opening behavior. **8 also summons the player when hidden and takes controls regardless of that preference.** If the replay is still loading, 8 toggles whether controls are wanted when ready. Automatic input routing waits for playable media and still requires the game to own foreground; hiding, errors or releasing controls cancel that intent.

**7 starts five seconds before the latest captured moment**, with the recent clip's earlier frames still seekable. Adjust **Start behind live** (1–300 seconds, default 5) under **Controls & playback** at <http://127.0.0.1:8765/#focus>, then Save; the next opening uses that value. This changes the initial cursor, not recording retention or the seek step. Imported videos and question-selected replay intervals keep their existing starting bounds.

For the entire recorded session, take controls with **8**, then choose **Full history** in the header or press **H**. The full timeline loads on demand and keeps the same session moment selected. Recording continues. Closing history and reopening a quick replay gives the latest quick request priority; interruption is cooperative between media samples, so an in-progress Media Foundation call must return first. Reaching the cutoff while controls are active holds the last moment; release controls to return to the game.

The input gate releases on foreground/target loss or a stale GUI heartbeat. It does not install hooks until you explicitly take controls. Automated checks cover the routing policy and inactive Qt action/mouse delivery, without live hooks. Full isolation from a game's Raw Input/DirectInput backend, cursor clipping/recentering and native fullscreen composition still need the real-game rehearsal. A system-navigation chord whose modifier was already captured may require a second press after the gate releases; the Windows key also releases it. Keep Computer Use inactive during play so Escape does not conflict with its emergency stop.

## Voice questions

With native recording running and the game foreground, **hold 9**, ask a short question such as “Show the last time yellow fell off the map,” then release. A passive status indicator shows listening and processing. The demo transcribes locally, searches the recorded session using the selected vision model, and opens the first visually supported match. Uncertain candidates remain available on the demo page for review. Silence and microphone failures produce an error instead of a question. Recording continues throughout.

Change or disable **Hold to ask about gameplay** under Hotkeys in the settings page. The default Windows microphone is used only while the shortcut is held; requests stop after 15 seconds and cancel if foreground or recording identity changes. Microphone WAVs are temporary and removed after processing. The transcript and result remain visible on the demo page.

Install `requirements-demo.txt`. The speech engine uses a locally cached faster-whisper `large-v3-turbo` model by default; `SMARTREPLAY_VOICE_MODEL` overrides its model name or directory. It never downloads a model during a voice question. This computer already has the default model cached. The model warms in the background when the native recorder starts. Initial loading measured about 16 seconds here; the repeated synthesized test question transcribed in 0.38 seconds. These timings exclude vision search and replay preparation. Rehearse live microphone pickup and end-to-end response time before the demo.

## Storage and memory

- `out/demo/sessions/<id>/`: full native session, finalized compressed segments and atomic `session.json`. Ring cleanup and RAM-cache eviction cannot delete these archives.
- `out/demo/matches/<id>/match.mp4`: immutable replay assembled by copying compressed streams, with no second video encoder. Existing snapshots are reused when their source segments are unchanged.
- `out/demo/indexes/`: sampled event evidence and coverage; indexing reads finalized segments without controlling capture.
- `out/demo/native/`: generation-tagged bridge commands, replies, status, and host logs.

The native player has a **128 MiB compressed replay cache** by default. Cached playback uses a memory-backed media device; files larger than the budget play from disk. Eviction considers active and retiring players' leases, and it never removes disk recordings. This budget covers encoded replay bytes, not the application's total RAM: Qt, capture surfaces and decoded frames also consume memory. Decoder/audio objects retire when playback closes. A new replay interval may need its first lossless disk assembly before it can be cached; repeat playback can reuse that export.

Full sessions have no automatic deletion policy in this implementation. Disk usage grows with recording length and bitrate; remove archived sessions only when you no longer need them and recording has stopped. Model processing has a separate bounded queue, interactive priority and error backoff. It can miss short or ambiguous events and cannot promise fixed response latency.

## Configuration and checks

### Question and speed rehearsal

Start with visible event questions: “What happened to blue before they died?”, “Which color won that round?”, “What weapon did green pick up?”, or “How did red recover from that fall?” Follow with “Did yellow get knocked off or launch themselves by shooting?” only when the before/contact/after footage is visible. Blocking and recovery questions appear in actual [blocking discussions](https://www.reddit.com/r/Stickfight/comments/l3lcf4/) and a [community skill map](https://www.reddit.com/r/Stickfight/comments/1buxdjo/stickfight_skill_map/). The [weapons wiki](https://stickfightgame.fandom.com/wiki/Weapons) supplies vocabulary for recoil, snake weapons and special effects; it is background knowledge, never evidence that a particular match contained that event. Coarse samples may miss a short block or round-ending screen, so do not promise exact counters or causal attribution without the supporting frames.

Query results include a `timings` breakdown. Frame sampling decodes forward using actual recorded timestamps, and an indexed candidate goes directly to dense verification. Indexed replay exports request ten seconds around the candidate, retaining the same verification frames and three-second lead-in/two-second tail; full session history remains on disk. A bounded sparse-segment fallback prevents healthy near-EOF seeks from stalling the index. Spoken questions reserve upcoming model access during listening/transcription; a model call already running may still finish first.

Codex now reuses one local app-server connection at Standard speed with low reasoning. Each call gets a fresh ephemeral thread, discarded afterward; answers still require actual visual verification. Apps, plugins, tools and external connections are disabled for these requests. Failed connections are retired and recreated on the next request, without automatically repeating inference. `SMARTREPLAY_CODEX_TRANSPORT=exec` restores the earlier one-process-per-request connection before restarting the demo. No Fast usage multiplier is enabled. Controlled timings are saved under `out/demo/transport-bench/`.

`SMARTREPLAY_NATIVE_EXE` overrides the native executable. `SMARTREPLAY_CODEX_MODEL` and `SMARTREPLAY_CODEX` select the Codex model/executable. For Claude, use `ANTHROPIC_API_KEY` locally or the signed-in CLI; `SMARTREPLAY_MODEL` and `SMARTREPLAY_CLAUDE` select its model/executable. Do not paste credentials into chat. For DeepSeek, `SMARTREPLAY_DEEPSEEK_MODEL` selects the model (default `deepseek-flash`). `SMARTREPLAY_DEMO_BACKEND=claude` overrides the saved provider at startup.

For an imported short-video check, use `./Run-Demo.ps1 -Video 'C:\path\clip.mp4'`. Imported videos are labelled and normalized to a browser-compatible MP4; that fallback keeps the previous 35-second limit. Native sessions do not have that recording limit. The local server binds only to loopback and serves only registered replay exports.

Automated verification uses generated native WARP textures, actual compressed media, the real local bridge, RAM-backed decoding, and model-boundary doubles. No game inputs or live desktop capture are required. Actual Stick Fight capture quality, foreground behavior and comparative performance still need a real-game rehearsal. Before presenting, try three fresh online/local sessions and include a request for an event that never happened.
