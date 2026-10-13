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
  4. p1_own         auto-label your own videos' shots (p1_cnn/build_own_shots.py makes them),
                    ImageNet-init and scratch ResNet-18, test-split predictions (~1-2 h GPU)
     (p1_movieshots   the same on MovieShots, once its frames are attached: --only p1_movieshots)
  5. downloads each job's results into p1_cnn/results or p2_detection/results,
     fills in the README tables (tools/update_results.py), runs the tests and commits.
Kaggle runs one job at a time here; the laptop only waits (it can sleep between jobs —
re-running this script picks up where it left off). Log: outputs/run_all.log
"""
from __future__ import annotations

import argparse
import os
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

# job -> (results folder on disk, file that proves it is done, datasets besides the code)
PLAN = {
    "p2_diagnose": (ROOT / "p2_detection", "results/centernet_r18/diagnosis.md", ["cinescope-ckpt"]),
    "p2_train_log": (ROOT / "p2_detection", "results/centernet_r18_log/wider_val.json", []),
    "p1_own": (ROOT / "p1_cnn", "results/own/student_scratch.csv", ["cinescope-ckpt", "cinescope-own-shots"]),
    # MovieShots frames are released by the authors on request; run with --only p1_movieshots once attached
    "p1_movieshots": (ROOT / "p1_cnn", "results/scratch/metrics.json", []),
}
DEFAULT = ["p2_diagnose", "p2_train_log", "p1_own"]
OWN_ZIP = WORK / "dataset-own" / "own_shots.zip"


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


def lock_or_exit():
    """One run_all at a time: a second copy (a double-clicked .bat) would push the same kernels."""
    path = ROOT / "outputs" / "run_all.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(path, "a+")  # noqa: SIM115 - held open for the life of the process
    fh.seek(0)
    try:
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print("another run_all.py is already running; this copy exits and leaves the work to it")
        sys.exit(0)
    return fh


def builds_running() -> int:
    """How many p1_cnn/build_own_shots.py processes are running (Windows; 0 elsewhere)."""
    if os.name != "nt":
        return 0
    cmd = ("(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
           "Where-Object { $_.CommandLine -like '*build_own_shots*' } | Measure-Object).Count")
    out = subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True, text=True).stdout
    try:
        return int(out.strip() or 0)
    except ValueError:
        return 0


def upload_own_shots(user: str) -> None:
    """The frames from p1_cnn/build_own_shots.py; re-uploaded only when the zip changed.
    Waits for any episode cutting still running, then repacks from every cut episode."""
    while (n := builds_running()) > 0:
        log(f"waiting for {n} build_own_shots.py run(s) to finish cutting episodes")
        time.sleep(120)
    subprocess.run([sys.executable, str(ROOT / "p1_cnn" / "build_own_shots.py"), "--pack-only"], check=True)
    if not OWN_ZIP.exists():
        sys.exit(f"{OWN_ZIP} is missing: run p1_cnn/build_own_shots.py first")
    stamp = str(OWN_ZIP.stat().st_size)
    marker = OWN_ZIP.parent / "published.txt"
    ref = f"{user}/cinescope-own-shots"
    if marker.exists() and marker.read_text().strip() == stamp and kr.dataset_exists(ref):
        return
    log(f"uploading {OWN_ZIP.stat().st_size / 1e6:.0f} MB of key frames as {ref}")
    kr.publish_dataset(OWN_ZIP.parent, ref, "cinescope-own-shots", "key frames from build_own_shots.py")
    marker.write_text(stamp)


def run_job(name: str, user: str) -> bool:
    dest_root, done_file, extra = PLAN[name]
    if (dest_root / done_file).exists():
        log(f"{name}: already done ({done_file})")
        return True
    folder = WORK / "jobs" / name
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    script = folder / f"{name}.py"
    script.write_text((JOBS / "_preamble.py").read_text(encoding="utf-8") + "\n"
                      + (JOBS / f"{name}.py").read_text(encoding="utf-8"), encoding="utf-8")
    datasets = [f"{user}/cinescope-code"] + [f"{user}/{d}" for d in extra]
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
    _lock = lock_or_exit()  # noqa: F841 - released when the process exits
    log("=== run_all.py started ===")
    if not args.no_upload:
        upload_code(user)
    jobs = args.only or DEFAULT
    if any("cinescope-ckpt" in PLAN[j][2] for j in jobs):
        upload_checkpoint(user)
    if "p1_own" in jobs:
        upload_own_shots(user)
    results = {j: run_job(j, user) for j in jobs}
    if (ROOT / "p1_cnn" / "results" / "own" / "student_scratch.csv").exists():
        subprocess.run([sys.executable, str(ROOT / "p1_cnn" / "eval_own.py")], check=False)
    subprocess.run([sys.executable, str(ROOT / "tools" / "update_results.py")], check=False)
    tests = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT)
    if tests.returncode == 0:
        subprocess.run(["git", "add", "-A"], cwd=ROOT)
        subprocess.run(["git", "commit", "-m", "Kaggle results: " + ", ".join(j for j, ok in results.items() if ok)],
                       cwd=ROOT)
    log(f"=== done: {results} ===")


if __name__ == "__main__":
    main()
