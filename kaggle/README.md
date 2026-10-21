# Training on Kaggle (free GPU)

## Automated (recommended)

One command runs every remaining job on Kaggle, one after another, downloads the
results, fills in the README tables, runs the tests and commits:

```powershell
cd $HOME\Desktop\cinescope
.venv\Scripts\pip install kaggle              # once
.venv\Scripts\python kaggle\run_all.py
```

Needs your Kaggle API token once: kaggle.com → Settings → API → **Create New Token**,
then move the downloaded `kaggle.json` to `C:\Users\<you>\.kaggle\kaggle.json`.
Jobs: `p2_diagnose` (~15 min), `p2_train_log` (~4 h), `p1_own` (~1–2 h; needs the key frames from
`p1_cnn/build_own_shots.py`, which `RUN_OWN_SHOTS.bat` makes), `p9_tensorrt_v2` (~40 min: the
TensorRT pipeline re-timed after the variable-length fix) and `p2_train_v3` (~4 h: CenterNet with
whole large faces in crops and low-light augmentation). A job whose results are already on disk is
skipped, so on a machine that has run the earlier ones only the two new jobs run.
`p1_movieshots` runs only on request (`--only p1_movieshots`) once MovieShots frames are attached.

A job file whose first line is `# extends: <job>` sets a few constants and reuses that job's body
(`p2_train_v3` extends `p2_train_log`, `p9_tensorrt_v2` extends `p9_tensorrt`);
`tests/test_kaggle_jobs.py` checks that every job assembles into valid Python.

## What only your account and hardware can run

Everything else is built and tested in CI (lint, tests, front end, the Docker image with a
smoke test of the web app, the pipeline and the editorial export). These need you:

1. **Kaggle token, once.** kaggle.com → Settings → API → **Create New Token**; move the
   downloaded `kaggle.json` to `C:\Users\<you>\.kaggle\kaggle.json`. Phone-verify the account
   (Settings → Phone verification) or Kaggle gives no GPU or internet.
2. **Run the new GPU jobs.** Double-click `RUN_KAGGLE_JOBS.bat`, or
   `.venv\Scripts\python kaggle\run_all.py --only p9_tensorrt_v2 p2_train_v3`. About 5 h of
   Kaggle GPU (of the free 30 a week); the laptop may sleep: run it again and it picks up
   where Kaggle is. It downloads the results, fills in the README tables, runs the tests and
   commits on your machine.
3. **Score the new detector on movie frames.** `python tools/run_everything.py p2` (CPU,
   a few minutes): adds a `CenterNet-R18 (ours, v3)` row to the week-8 table.
4. **Push.** `git push` the commit run_all made. The TensorRT v2 numbers land in
   `p9_production/results/tensorrt_v2/`; the README text around the first run's table is
   written by hand, so ask Claude to write them up once they are pushed.
5. **Docker on your machine (optional).** With Docker Desktop running: `docker build -t
   cinescope .` then the `docker run` line at the top of the `Dockerfile`, and open
   http://localhost:8000. CI already builds and smoke-tests this exact image on every push.
Stopping the script is safe: running it again waits for, or fetches, what Kaggle
already ran. Job sources are in `kaggle/jobs/`; the log is `outputs/run_all.log`.

## By hand, in the Kaggle website

Training runs on Kaggle's free GPUs (about 30 GPU-hours a week); nothing heavy
runs on the laptop. Two notebooks, one per phase:

| Notebook | Trains | Data (downloaded inside Kaggle) |
|---|---|---|
| `p2_wider_face.ipynb` | CenterNet face detector | WIDER FACE from Hugging Face + the official eval toolkit |
| `p1_movieshots.ipynb` | ResNet-18 shot-scale classifier (ImageNet init and from scratch) | MovieShots frames (released by the authors on request) |

Start with Phase 2: its data download is the more reliable of the two.

## One-time setup

1. Kaggle account → Settings → **Phone verification** (needed for GPU and internet access).
2. On the laptop, pack the code:
   ```powershell
   cd $HOME\Desktop\cinescope
   .venv\Scripts\python kaggle\make_code_zip.py      # -> kaggle_upload\cinescope-code.zip
   ```
3. kaggle.com → **Datasets → New Dataset** → upload `cinescope-code.zip`, title
   `cinescope-code`, visibility **Private** → Create.

## Each run

1. kaggle.com → **Code → New Notebook** → File → **Import Notebook** → pick the `.ipynb` from this folder.
2. Right-hand panel: **Accelerator → GPU T4 x2** (or P100), **Internet → On**,
   **Add Input → Your Datasets → cinescope-code**.
3. **Save Version → Save & Run All (Commit)**. It runs in the background; you can close the browser.
4. When it finishes, open the version → **Output** → download `phase2_results.zip`
   (or `phase1_results.zip`) and unzip it into the matching phase folder
   (`p2_detection/`, `p1_cnn/`) on the laptop, then run
   `python tools/update_results.py` to fill in the READMEs' results tables.

If a run hits the 12-hour limit: open the notebook, **Add Input → Notebook
Output** → the previous version, and Save & Run All again — training resumes from
the last finished epoch.

After changing code on the laptop: re-run `make_code_zip.py` and upload it as a
**New Version** of the same `cinescope-code` dataset.
