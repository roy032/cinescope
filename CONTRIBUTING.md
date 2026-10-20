# Contributing to CineScope

CineScope is a learning project, but it is built to run on anyone's machine with any video.
Issues and pull requests are welcome.

## Set up

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu   # CPU wheels are enough
pip install -r requirements.txt                    # core + dev
pip install -r requirements-full.txt               # optional: the whole pipeline and the web app
python tools/fetch_assets.py                       # optional: public test clips and library weights
```

Or let `python tools/run_everything.py` (Windows: `RUN_EVERYTHING.bat`, Unix: `make all`) do
all of it. Libraries used only as baselines are listed, commented out, at the bottom of
`requirements.txt`. Install the ones for the phase you are working on; tests that need a
missing library are skipped, not failed.

The web app front end lives in `app/frontend` (Node 22):

```bash
cd app/frontend && npm ci && npm test && npm run build
```

## Before you open a pull request

```bash
ruff check .            # lint (CI pins ruff 0.16.8; requirements.txt installs the same)
python -m pytest -q     # the test suite; no GPU, network or footage needed
```

CI runs both, plus the front end's tests and build and a Docker smoke test, on every pull
request.

## House rules

- **From scratch first, then the library.** Each phase implements the core idea itself and
  then compares it with the production library. Keep both paths, and report the comparison
  in the phase README.
- **Any clip.** Tools take video paths as arguments. No hard-coded file names, drives or
  personal footage in code, docs, tests or examples; use `your_clip.mp4` in examples.
- **No footage or weights in git.** `data/`, `trailers/`, `outputs/`, videos and model files
  are git-ignored. Results that are committed are small: tables, JSON scores and preview images.
- **Test clips are a test set.** Do not tune thresholds or hyperparameters on the public
  clips used for scoring.
- **Tests are synthetic and fast.** Generate the input in the test. A test that needs an
  optional library uses `pytest.importorskip`.
- Each roadmap week has a `tests/test_weekN.py`; add to the matching file. Tests that are not
  about one week (the command line, packaging) have their own file.

## Commit messages

Short imperative subject line (for example `Fix scrambled frame times on B-frame AVIs`),
then a body that says why when it is not obvious.
