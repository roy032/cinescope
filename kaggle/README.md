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
Jobs: `p2_diagnose` (~15 min), `p2_train_log` (~4 h), `p1_movieshots` (~4–7 h).
Stopping the script is safe: running it again waits for, or fetches, what Kaggle
already ran. Job sources are in `kaggle/jobs/`; the log is `outputs/run_all.log`.

## By hand, in the Kaggle website

Training runs on Kaggle's free GPUs (about 30 GPU-hours a week); nothing heavy
runs on the laptop. Two notebooks, one per phase:

| Notebook | Trains | Data (downloaded inside Kaggle) |
|---|---|---|
| `p2_wider_face.ipynb` | CenterNet face detector | WIDER FACE from Hugging Face + the official eval toolkit |
| `p1_movieshots.ipynb` | ResNet-18 shot-scale classifier (ImageNet init and from scratch) | MovieShots from the authors' Google Drive folder |

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
   (`p2_detection/`, `p1_cnn/`) on the laptop. Tell me and I'll fill in the READMEs' results tables.

If a run hits the 12-hour limit: open the notebook, **Add Input → Notebook
Output** → the previous version, and Save & Run All again — training resumes from
the last finished epoch.

After changing code on the laptop: re-run `make_code_zip.py` and upload it as a
**New Version** of the same `cinescope-code` dataset.
