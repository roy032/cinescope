#!/usr/bin/env python
"""Run every remaining CineScope training/evaluation job on Kaggle, unattended.

    .venv\\Scripts\\python kaggle\\run_all.py                  # all jobs not done yet
    .venv\\Scripts\\python kaggle\\run_all.py --only p2_diagnose

One-time setup: pip install kaggle, and your API token in C:\\Users\\<you>\\.kaggle\\kaggle.json
(kaggle.com -> Settings -> API -> Create New Token).

What it does, in order (a job is skipped when its results are already on disk):
  1. uploads the current code as a new version of the private "cinescope-code" dataset
     and the trained face-detector checkpoint as "cinescope-ckpt" (first time only)
  2. p2_diagnose    WIDER val, top-1000 peaks + per-face-size diagnosis     (~15 min GPU)
  3. p2_train_log   retrain CenterNet with log-size regression + evaluate   (~4 h GPU)
  4. p1_movieshots  MovieShots download, ImageNet-init and scratch ResNet-18 (~4-7 h GPU)
  5. downloads each job's results into p1_cnn/results or p2_detection/results,
     fills in the README tables (tools/update_results.py), runs the tests and commits.
Kaggle runs one job at a time here; the laptop only waits (it can sleep between jobs —
re-running this script picks up where it left off). Log: outputs/run_all.log
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "kaggle"))
import kagglerun as kr  # noqa: E402

JOBS = ROOT / "kaggle" / "jobs"
WORK = ROOT / "kaggle_upload"
LOG = ROOT / "outputs" / "run_all.log"

# job -> (results folder on disk, file that proves it is done, needs checkpoint dataset)
PLAN = {
    "p2_diagnose": (ROOT / "p2_detection", "results/centernet_r18/diagnosis.md", True),
    "p2_train_log": (ROOT / "p2_detection", "results/centernet_r18_log/wider_val.json", False),
    "p1_movieshots": (ROOT / "p1_cnn", "results/scratch/metrics.json", False),
}


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def upload_code(user: str) -> None:
    subprocess.run([sys.executable, str(ROOT / "kaggle" / "make_code_zip.py")], check=True)
    folder = WORK / "dataset-code"
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    shutil.copy(WORK / "cinescope-code.zip", folder / "cinescope-code.zip")
    kr.publish_dataset(folder, f"{user}/cinescope-code", "cinescope-code", "code update from run_all.py")


def upload_checkpoint(user: str) -> None:
    ref = f"{user}/cinescope-ckpt"
    if kr.dataset_exists(ref):
        return
    ckpt = ROOT / "p2_detection" / "checkpoints" / "centernet_r18.pt"
    if not ckpt.exists():
        sys.exit(f"{ckpt} is missing — unzip phase2_results.zip into p2_detection first")
    folder = WORK / "dataset-ckpt"
    folder.mkdir(parents=True, exist_ok=True)
    shutil.copy(ckpt, folder / ckpt.name)
    kr.publish_dataset(folder, ref, "cinescope-ckpt", "trained CenterNet-R18 (linear size)")


def run_job(name: str, user: str) -> bool:
    dest_root, done_file, needs_ckpt = PLAN[name]
    if (dest_root / done_file).exists():
        log(f"{name}: already done ({done_file})")
        return True
    folder = WORK / "jobs" / name
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    script = folder / f"{name}.py"
    script.write_text((JOBS / "_preamble.py").read_text(encoding="utf-8") + "\n"
                      + (JOBS / f"{name}.py").read_text(encoding="utf-8"), encoding="utf-8")
    datasets = [f"{user}/cinescope-code"] + ([f"{user}/cinescope-ckpt"] if needs_ckpt else [])
    ref = f"{user}/cinescope-{name.replace('_', '-')}"
    state = kr.status(ref)
    if state == "complete" and fetch(name, ref, dest_root, done_file):
        log(f"{name}: results fetched from the finished run {ref}")
        return True
    if state not in ("running", "queued"):
        log(f"{name}: starting on Kaggle as {ref}")
        ref = kr.push_kernel(folder, f"cinescope-{name.replace('_', '-')}", script.name, datasets)
    state = kr.wait_kernel(ref)
    log(f"{name}: {state}")
    ok = fetch(name, ref, dest_root, done_file)
    if not ok:
        log(f"{name}: no results — read the log at kaggle.com/code/{ref}")
    return ok


def fetch(name: str, ref: str, dest_root: Path, done_file: str) -> bool:
    out = WORK / "outputs" / name
    shutil.rmtree(out, ignore_errors=True)
    kr.download_output(ref, out)
    for sub in ("results", "checkpoints"):
        src = out / sub
        if src.exists():
            shutil.copytree(src, dest_root / sub, dirs_exist_ok=True)
    for extra in out.glob("*.log"):
        (dest_root / "results").mkdir(exist_ok=True)
        shutil.copy(extra, dest_root / "results" / f"{name}.log")
    return (dest_root / done_file).exists()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", choices=list(PLAN), help="run just these jobs")
    ap.add_argument("--no-upload", action="store_true", help="reuse the code already on Kaggle")
    args = ap.parse_args()
    user = kr.username()
    log("=== run_all.py started ===")
    if not args.no_upload:
        upload_code(user)
    jobs = args.only or list(PLAN)
    if "p2_diagnose" in jobs:
        upload_checkpoint(user)
    results = {j: run_job(j, user) for j in jobs}
    subprocess.run([sys.executable, str(ROOT / "tools" / "update_results.py")], check=False)
    tests = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT)
    if tests.returncode == 0:
        subprocess.run(["git", "add", "-A"], cwd=ROOT)
        subprocess.run(["git", "commit", "-m", "Kaggle results: " + ", ".join(j for j, ok in results.items() if ok)],
                       cwd=ROOT)
    log(f"=== done: {results} ===")


if __name__ == "__main__":
    main()
