"""Run Kaggle jobs from the command line: update a dataset, push a script kernel,
wait for it, download its output. Wraps the official `kaggle` CLI.

One-time setup (the token is yours; this module reads only the username from it):
  1. pip install kaggle
  2. kaggle.com -> Settings -> API -> "Create New Token" -> downloads kaggle.json
  3. move it to  C:\\Users\\<you>\\.kaggle\\kaggle.json
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path


def kaggle_exe() -> str:
    exe = shutil.which("kaggle")
    if exe:
        return exe
    here = Path(sys.executable).parent / ("kaggle.exe" if os.name == "nt" else "kaggle")
    if here.exists():
        return str(here)
    sys.exit("The kaggle CLI is not installed: run  pip install kaggle  in this environment")


def username() -> str:
    if os.environ.get("KAGGLE_USERNAME"):
        return os.environ["KAGGLE_USERNAME"]
    cfg = Path(os.environ.get("KAGGLE_CONFIG_DIR", Path.home() / ".kaggle")) / "kaggle.json"
    if not cfg.exists():
        sys.exit(f"No Kaggle API token at {cfg}. kaggle.com -> Settings -> API -> Create New Token, "
                 f"then move kaggle.json there.")
    return json.loads(cfg.read_text(encoding="utf-8"))["username"]


def run(*args: str, check: bool = True) -> str:
    print("$ kaggle", " ".join(args), flush=True)
    p = subprocess.run([kaggle_exe(), *args], capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    out = (p.stdout or "") + (p.stderr or "")
    if out.strip():
        print(out.strip()[-2000:], flush=True)
    if check and p.returncode != 0:
        sys.exit(f"kaggle {' '.join(args[:2])} failed (exit {p.returncode})")
    return out


def dataset_exists(ref: str) -> bool:
    out = run("datasets", "files", ref, check=False)
    return not re.search(r"404|not found|forbidden|403", out, re.I)


def publish_dataset(folder: Path, ref: str, title: str, message: str) -> None:
    """Create the dataset if it does not exist, otherwise add a new version.
    `folder` holds the files; dataset-metadata.json is written into it."""
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "dataset-metadata.json").write_text(json.dumps(
        {"title": title, "id": ref, "licenses": [{"name": "other"}]}), encoding="utf-8")
    if dataset_exists(ref):
        run("datasets", "version", "-p", str(folder), "-m", message, "-r", "zip")
    else:
        run("datasets", "create", "-p", str(folder), "-r", "zip")
    wait_dataset(ref)


def wait_dataset(ref: str, timeout: int = 900) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if "ready" in run("datasets", "status", ref, check=False).lower():
            return
        time.sleep(15)
    print(f"warning: dataset {ref} not ready after {timeout}s; continuing", flush=True)


def push_kernel(folder: Path, slug: str, code_file: str, datasets: list[str], gpu: bool = True,
                kernels: list[str] | None = None) -> str:
    """Push `folder/code_file` as a private script kernel and start it. Returns user/slug."""
    ref = f"{username()}/{slug}"
    (folder / "kernel-metadata.json").write_text(json.dumps({
        "id": ref, "title": slug, "code_file": code_file, "language": "python",
        "kernel_type": "script", "is_private": True, "enable_gpu": gpu, "enable_tpu": False,
        "enable_internet": True, "dataset_sources": datasets, "kernel_sources": kernels or [],
        "competition_sources": [], "model_sources": []}, indent=1), encoding="utf-8")
    run("kernels", "push", "-p", str(folder))
    return ref


def status(ref: str) -> str:
    out = run("kernels", "status", ref, check=False).lower()
    for s in ("complete", "error", "cancel", "running", "queued"):
        if s in out:
            return s
    return "unknown"


def wait_kernel(ref: str, poll: int = 120, timeout: int = 13 * 3600) -> str:
    t0 = time.time()
    time.sleep(30)
    while time.time() - t0 < timeout:
        s = status(ref)
        print(f"[{time.strftime('%H:%M')}] {ref}: {s} ({(time.time() - t0) / 60:.0f} min)", flush=True)
        if s in ("complete", "error", "cancel"):
            return s
        time.sleep(poll)
    return "timeout"


def download_output(ref: str, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    run("kernels", "output", ref, "-p", str(dest), "--force", check=False)
    return dest


def unzip(archive: Path, dest: Path) -> None:
    with zipfile.ZipFile(archive) as z:
        z.extractall(dest)
