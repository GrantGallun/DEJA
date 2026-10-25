# Contributing

Deja (Smart Replay) has one maintainer and changes daily. Issues and small pull requests are welcome.
For anything larger, open an issue first so the approach is agreed before you spend time on it.

## Setup

Windows 10 or 11 and Python 3.11 or newer. Clone the repository and install the dependencies:

```bash
git clone https://github.com/GrantGallun/DEJA.git
cd DEJA
python -m pip install -r requirements-demo.txt pytest ruff
```

`requirements.txt` alone is enough for the discovery pipeline. `requirements-demo.txt` adds the
recording demo's packages. The native player lives in a separate GameHQ checkout under
`experiments/` (see [docs/STICK_FIGHT_DEMO.md](docs/STICK_FIGHT_DEMO.md)). Tests that need it skip.

## Running the tests

```bash
python -m pytest -q -rs                 # unit and regression suite
node --test tests/demo_ui.test.cjs      # demo page script
ruff check                              # errors that break code, not style
```

Set `RUN_SLOW=1` to add the synthetic end-to-end run. It generates its own clip, so it needs no
footage:

```bash
RUN_SLOW=1 python -m pytest -q -rs
```

CI runs all three on Windows for every push and pull request.

## Rules that are not negotiable

- **No per-game configuration.** Smart Replay has to work on a game it has never seen. Fix a failure
  with a structural signal or invariant that holds across games, not a threshold tuned to one clip or
  a special case for one title.
- **No footage or credentials.** Never commit recordings, `.env`, API keys or sign-in tokens. Media
  under `footage/` and results under `out/` are gitignored for this reason.
- **Numbers are measured.** A claimed improvement in precision, recall or latency says what was
  measured, on which footage, and how to reproduce it.
- **Fixes come with a regression test.** `tests/test_regressions.py` pins every defect from
  `audit/findings.json`. Add to it, or to the closest test file, with a test that fails before the fix.

## Submitting a pull request

```bash
git checkout -b my-change
# edit, then run the tests above
git commit -am "Short summary of the change"
git push -u origin my-change
```

Then open a pull request against `main` and fill in the template. Keep it small and focused, and
update the docs that describe what you changed (`README.md`, `docs/PLAYER_SETTINGS.md` for the
settings file format).

Contributions are accepted under the project's [MIT license](LICENSE). By taking part you agree to the
[code of conduct](CODE_OF_CONDUCT.md).
