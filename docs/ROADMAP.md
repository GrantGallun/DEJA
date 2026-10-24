# Smart Replay - status and roadmap

Updated 2026-10-24.

## Where things stand

| Area | State |
|---|---|
| HUD discovery (`smartreplay/`) | Works on synthetic (death/kill F1 1.00, round 0.86) and passes the Doom negative control. **No real-game positive result yet.** |
| Audit findings | All fixed, each pinned by a test in `tests/test_regressions.py`. |
| Web-image lookup | Not usable as an automatic step yet (see below). |
| Replay app | Passive-replay spike in progress in `experiments/gamehq-passive` (GameHQ fork, GPL, kept separate). |

## Licensing rule

GameHQ is GPL-3. Study it freely. SmartReplay's own code is written
independently, not copied or translated from GameHQ files, so it can stay MIT.
If we decide to reuse substantial GameHQ code, we switch the product to GPL on
purpose.

## Findings from this pass

**Web-image lookup: classical selection doesn't separate.** Pairwise
multi-scale edge-structure scores across the 29 cached Minecraft images (14
hand-labelled standard death screens): non-death images score as high as real
ones (e.g. a modded RPG screen has a top-5 mean of 0.365, while the positives
range from 0.26 to 0.51). Whole-frame structure is dominated by background. The
CLI also still uses the averaged consensus template, even though `build_bank`'s
own note says the bank does far better. Next step: filter candidates with a
vision model (CLIP or Claude), and treat lookup output as a label plus rough
geometry only.

**The semantic layer is cheap for text-bearing events.** Off-the-shelf
Tesseract on the death-screen text band read the cause of death correctly on 7
of 9 random web screenshots ("Efe was blown up by Creeper", "tried to swim in
lava"). Fuzzy-matching against Minecraft's fixed death-message templates and
mob list should close most of the gap. That is how "what mob did I die to"
gets answered without a model call.

## Roadmap

1. **Real-footage benchmark** (blocked on network access to video sites).
   Valorant, League, Deadlock and Minecraft: unedited VODs, labelled event
   intervals plus no-event footage, thresholds frozen before scoring.
2. **Event index.** Runtime detectors from confirmed profiles write
   (time, label, confidence, provenance, details) rows. Queries ("my last
   death", "where I found diamonds") are lookups over this index.
3. **Semantic details.** OCR on detected event frames (death messages,
   kill-feed names, advancement toasts like "Diamonds!"), with a VLM fallback
   for events without text.
4. **Labelling.** One VLM call per game names the discovered groups; users
   confirm or reject; the lookup only proposes.
5. **Replay surface.** Passive PiP with separate show and interact hotkeys,
   frozen snapshots while recording continues (see the prior-art report).
