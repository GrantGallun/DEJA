# Smart Replay — HUD event discovery

This is the adaptability spike for Smart Replay. It answers one question
before any Windows plumbing gets written:

> Can a game's event vocabulary — death, kill, round start — be discovered
> from recorded footage alone, with no per-game configuration?

If yes, the rest of the product is engineering. If no, the "works on any game"
premise needs rethinking, and it is far cheaper to learn that now.

## The idea

Event indicators are HUD, and HUD is distinguishable from world geometry by
*temporal* statistics, not spatial ones:

| | behaviour over time |
|---|---|
| world geometry | broad spread of values, changing continuously |
| always-on HUD | barely moves |
| **event HUD** | **snaps to one tight value, holds for seconds, recurs** |

So the pipeline never asks "what does a death screen look like". It asks which
pixels behave like event HUD, groups them by *when* they fire, and hands the
result to a labelling step.

## Pipeline

1. **sample** — decode at ~2 fps into a thumbnail stack. `cv2.grab()` advances
   the decoder without paying for colour conversion; `retrieve()` runs only on
   kept frames. Letterboxing is detected and cropped.
2. **classify** (`discovery.py`) — per pixel, find the tight value it snaps to
   by histogram peak **prominence** (not argmax: the HUD value is a narrow
   spike on top of the broad base world content contributes). Then require
   **spatial consensus** — a neighbourhood must agree — and classify by run
   length, recurrence and prominence.
3. **group** (`grouping.py`) — cluster event pixels by the similarity of their
   activation time series. Spatial adjacency is the wrong primitive: a kill
   feed and a round timer sit side by side, and one element can span disjoint
   regions. The group's activation *is* the event timeline — rising edges are
   onsets, run lengths are durations.
4. **variants** (`variants.py`) — cluster each element's occurrences by
   difference hash, so a detector can later score against several exemplars
   instead of one blurred average. Exemplar imagery is re-read from the source
   at full resolution.
5. **report** (`report.py`) — a HUD map and one contact sheet per element, so
   a wrong answer is obvious by eye.
6. **profile** (`profile.py`) — a draft Game Profile: detector geometry in
   normalised coordinates, exemplar templates, the regions the interface
   occupies, and a suggested mini-player rectangle that occludes nothing.

Labels are left `null`. Naming the discovered elements is a separate step and
is the only place a model is needed — one call per game, ever.

## Usage

```bash
pip install -r requirements.txt

python -m smartreplay.cli discover footage/clip.mp4        # one clip
python -m smartreplay.cli batch   footage/                 # a folder
```

Useful flags: `--width` (analysis resolution; raise it for small HUD
elements), `--fps`, `--min-occurrences`, `--jaccard`, `--reuse-samples`.

Output per clip, under `out/<name>/` (`<name>` is the file stem, or
`<stem>-<ext>` when another video in the same folder shares the stem).
`--reuse-samples` only reuses a stack whose recorded source fingerprint and
sampling settings match the requested run; otherwise it resamples.
Files: `hud_map.png`, `group_NN.png`,
`events.json`, `profile_draft.json`, `templates.npz`, plus the sampled stacks.
`out/index.md` summarises a batch.

## Replay player settings

```bash
python -m smartreplay.frontend      # settings app in the browser (standard library only)
```

A local page for the replay player's hotkeys and in-player keys, its input
mode (keyboard only, mouse only, or both), size and position, theme and
opacity, and which buttons it shows, with a live preview. It writes
`%APPDATA%\SmartReplay\player-settings.json`; a running player applies each
save immediately and reports back, including any hotkey Windows refused.
[docs/PLAYER_SETTINGS.md](docs/PLAYER_SETTINGS.md) describes the file.

## Test substrates

Ground truth without hand labelling, in order of realism:

**Minecraft — the best realistic substrate.** `logs/latest.log` timestamps
every death, so truth is free and exact, and the death screen is a textbook
event overlay.

```bash
python tools/minecraft_truth.py footage/mc.mp4 --list   # check alignment first
python tools/minecraft_truth.py footage/mc.mp4
python -m smartreplay.cli discover footage/mc.mp4
python tools/verify.py footage/mc.truth.json out/mc
```

Log time is wall-clock, so it has to be aligned to the video: an explicit
`--start`, a timestamp in the filename (OBS and ShadowPlay write one), or file
mtime minus duration. Always sanity-check with `--list`. On a server the log
records *everyone's* deaths but only yours shows a death screen, so lines are
filtered to the local player (auto-detected from `Setting user:`).

**ViZDoom — real 3D footage, generated unattended.**

```bash
python tools/vizdoom_substrate.py footage/doom.mp4 --seconds 240
python -m smartreplay.cli discover footage/doom.mp4 --width 640
```

Caveat, found the hard way: ViZDoom's offscreen renderer draws the status bar
but **not** messages or the automap, so it cannot produce an event overlay —
`set_render_messages(True)` and `togglemap` both have no visible effect. It is
therefore useful as a **negative** test (real footage, static HUD only, correct
answer is "no events") rather than a positive one.

**Synthetic — the regression test.** A generated clip with planted events and
a truth file, for checking the mechanism end to end.

```bash
python tools/make_test_clip.py footage/synthetic.mp4
python -m smartreplay.cli discover footage/synthetic.mp4
python tools/verify.py footage/synthetic.truth.json out/synthetic
```

## Results so far

**Synthetic** (3 planted events, 120 s, 240 samples):

| event | truth | found | precision | recall | F1 | ROI IoU | onset error |
|---|---|---|---|---|---|---|---|
| death | 5 | 5 | 1.00 | 1.00 | **1.00** | 1.00 | 0.00 s |
| kill | 8 | 8 | 1.00 | 1.00 | **1.00** | 0.99 | 0.00 s |
| round | 3 | 4 | 0.75 | 1.00 | **0.86** | 0.07 | 0.00 s |

One false-positive group. Onsets are exact at the sampling resolution.

**Doom** (240 s of real gameplay, static HUD only): **0 confirmed elements** —
the correct answer — while static-HUD classification landed precisely on the
status bar (46,357 px of a 52,480 px bar). No false positives on real footage.

## Known limitations

- **Low-contrast elements lose their extent.** The round bar's *timing* is
  recovered exactly, but its ROI IoU is 0.07: when the world frequently passes
  through the element's own brightness band, only part of it is recovered.
  Timing survives because grouping needs only enough pixels to agree.
- **The per-pixel stage is deliberately permissive** — 37% of pixels on the
  synthetic clip, 62% on Doom. That is by design; group-level filters (size,
  fill, coherence, duration consistency) do the real rejecting. The HUD map
  distinguishes candidate from confirmed for exactly this reason.
- **Small elements need resolution.** At 384 px wide, a 40×8 native badge is
  ~10×2 px and will be lost to the 5×5 spatial consensus. Raise `--width`.
- **One polarity only.** The tight value is assumed to be the element's *on*
  state. Where an element covers an unusually static patch of world, the mode
  can land on the off state instead; those pixels are lost, and the element is
  recovered from its other pixels or not at all.
- **Variants under-split on large elements.** The death panel's dhash is
  dominated by its banner text, so three different killer names cluster as one
  variant. Fine for detection — the invariant part is what a detector should
  match — but it means the variant count is not a reliable count of looks.
- **Footage should be continuous in-game play.** Menus and edited cuts
  introduce their own stable UI and can register as events.
- **`min_coherence` trades recall against precision.** At 0.85 the synthetic
  clip yields zero false positives but loses the low-contrast event entirely;
  the default of 0.70 keeps all three. Recall is favoured deliberately: a
  missed event is invisible to the user, an extra group is just an unlabelled
  row the naming step discards.

## Tests

```bash
pip install -r requirements.txt pytest
python -m pytest tests                       # fast unit + regression suite
RUN_SLOW=1 python -m pytest tests            # adds the synthetic end-to-end run
```

`tests/test_regressions.py` pins every defect from `audit/findings.json`
(negative-control scoring, optimal onset matching, same-stem cache collisions,
the group cap being applied before validation, final-occurrence duration,
Minecraft player-name and midnight handling, letterbox crop in the profile,
stale template archives). All are fixed.

`tools/verify.py` accepts a negative control: a truth file with
`"events": {}` passes only when no groups are discovered.

```bash
echo '{"events": {}}' > footage/doom.negative.json
python tools/verify.py footage/doom.negative.json out/doom
```

## Layout

```
smartreplay/     sampling, discovery, grouping, variants, report, profile, cli
  player_settings.py  replay player settings: schema, validation, file I/O
  frontend/           settings app (local server + static page)
tools/           make_test_clip, vizdoom_substrate, minecraft_truth, verify, lookup_eval
tests/           pytest suite (regressions from the audit, synthetic end-to-end)
footage/         input clips
out/             results, one directory per clip
```
