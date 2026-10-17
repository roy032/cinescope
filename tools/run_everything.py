"""One command for the whole project: set up, fetch, check, test, score, run.

    python tools/run_everything.py                    # every step below except kaggle
    python tools/run_everything.py test pipeline      # just these steps, in this order
    python tools/run_everything.py --clip trailers/any_clip.mp4 pipeline
    python tools/run_everything.py --list             # the steps
    RUN_EVERYTHING.bat  /  make all                   # the same, from Windows Explorer / a Unix shell

Steps (default order):
  setup     create .venv if missing (then re-run inside it); install CPU PyTorch,
            requirements.txt and requirements-full.txt
  fetch     public test clips -> trailers/, library weights -> data/weights (tools/fetch_assets.py)
  lint      ruff check .
  test      python -m pytest -q
  frontend  npm ci && npm run build in app/frontend (skipped without npm)
  p1        score the episode-scale Phase 1 run (teacher + both ResNet-18s, trained on Kaggle by the
            kaggle step) against your hand labels of its test shots; skipped until you have
            labelled them in outputs/label_scales/index.html (RUN_OWN_SHOTS.bat makes that page)
  p2        regenerate the hand-labelled Phase 2 frames, score our CenterNet checkpoints (any in
            p2_detection/checkpoints) and YuNet on them, refresh the README tables
  pipeline  python -m cinescope.pipeline on --clip (default trailers/goldeneye.mp4), ONNX backend
  accel     ONNX export + agreement + timing (p9_production/export_onnx.py); with an NVIDIA GPU and
            onnxruntime-gpu, also the pipeline with --backend tensorrt (skipped otherwise)
  docker    docker build -t cinescope . (skipped without a Docker daemon; CI always builds it)
  kaggle    the GPU training jobs on Kaggle (kaggle/run_all.py; needs ~/.kaggle/kaggle.json).
            Not in the default set: it takes hours. Ask for it by name.

Every step is safe to re-run. The run stops at the first failing step and says which;
--keep-going runs the rest anyway. A summary with each step's time is printed at the end.
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
VENV = ROOT / ".venv"
VENV_PY = VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
DEFAULT = ["setup", "fetch", "lint", "test", "frontend", "p1", "p2", "pipeline", "accel", "docker"]
ALL = [*DEFAULT, "kaggle"]

# Phase 2 week 8: the frames each label file was drawn on (p2_detection/labels/README.md).
P2_FRAMES = {
    "goldeneye_dark": ["trailers/goldeneye.mp4", "--start", "43", "--seconds", "39.5", "--fps", "0.75",
                       "--tag", "dark", "--classes", "face,person,tv,car"],
    "head-pose-face-detection-female-and-male_profile": [
        "trailers/head-pose-face-detection-female-and-male.mp4", "--tag", "profile", "--classes", "face,person",
        "--times", "5,6.5,8,9.5,11,18,19.5,21,22.5,58,60,62,123,124,125,126,127,128,129,130,131,132,133,134,"
                   "135,136,137,138,139,140"],
    "classroom_crowd": ["trailers/classroom.mp4", "--start", "0.5", "--seconds", "32", "--fps", "0.94",
                        "--tag", "crowd", "--classes", "face,person"],
}


class Skip(Exception):
    """A step that cannot run on this machine (no npm, no Docker, no GPU): reported, not a failure."""


def sh(*cmd: str | Path, cwd: Path = ROOT) -> None:
    cmd = [str(c) for c in cmd]
    print("  $", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def py(*args: str | Path) -> None:
    sh(sys.executable, *args)


def in_venv() -> bool:
    return Path(sys.prefix).resolve() == VENV.resolve()


# ------------------------------------------------------------------ steps
def step_setup(args) -> None:
    if not in_venv():
        if not VENV_PY.exists():
            sh(sys.executable, "-m", "venv", VENV)
        # everything after this runs inside the venv
        raise SystemExit(subprocess.call([str(VENV_PY), __file__, *sys.argv[1:]], cwd=ROOT))
    py("-m", "pip", "install", "-q", "--upgrade", "pip")
    py("-m", "pip", "install", "-q", "torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cpu")
    py("-m", "pip", "install", "-q", "-r", "requirements.txt", "-r", "requirements-full.txt")


def step_fetch(args) -> None:
    py("tools/fetch_assets.py")


def step_lint(args) -> None:
    py("-m", "ruff", "check", ".")


def step_test(args) -> None:
    py("-m", "pytest", "-q")


def step_frontend(args) -> None:
    npm = shutil.which("npm")
    if not npm:
        raise Skip("npm not found (install Node.js 22 to build the web app's front end)")
    sh(npm, "ci", "--no-audit", "--no-fund", cwd=ROOT / "app" / "frontend")
    sh(npm, "run", "build", cwd=ROOT / "app" / "frontend")


def step_p1(args) -> None:
    own = ROOT / "p1_cnn" / "results" / "own"
    kaggle = ROOT / "kaggle_upload" / "outputs" / "p1_own" / "results" / "own"
    if not (own / "teacher.csv").exists() and (kaggle / "teacher.csv").exists():
        own.mkdir(parents=True, exist_ok=True)        # the Kaggle run's predictions (git-ignored: they name the videos)
        for f in kaggle.glob("*.csv"):
            shutil.copy2(f, own / f.name)
    if not (own / "teacher.csv").exists():
        raise Skip("no episode-scale predictions yet: run RUN_OWN_SHOTS.bat (or the kaggle step with p1_own)")
    labels = ROOT / "p1_cnn" / "labels" / "own_test.labels.json"
    if not labels.exists() and not list((Path.home() / "Downloads").glob("own_test.labels*.json")):
        raise Skip("the test shots are not hand-labelled yet: open outputs/label_scales/index.html, label, "
                   "press 'Download labels', then run this step again")
    py("p1_cnn/eval_own.py")
    py("tools/update_results.py")


def step_p2(args) -> None:
    labels = sorted((ROOT / "p2_detection" / "labels").glob("*.boxes.json"))
    for lab in labels:
        name = lab.name.removesuffix(".boxes.json")
        if not (ROOT / "outputs" / f"boxes_{name}" / "frames").is_dir():
            if name not in P2_FRAMES:
                raise RuntimeError(f"no frame recipe for {lab.name}: add it to P2_FRAMES")
            py("p2_detection/label_boxes.py", *P2_FRAMES[name])
    ckpts = sorted((ROOT / "p2_detection" / "checkpoints").glob("centernet_r18*.pt"))
    yunet = Path(os.environ.get("CINESCOPE_WEIGHTS", ROOT / "data" / "weights")) / "yunet.onnx"
    det = [a for c in ckpts for a in ("--ckpt", c)] + (["--yunet", yunet] if yunet.exists() else [])
    if not det:
        raise Skip("no CenterNet checkpoint in p2_detection/checkpoints and no yunet.onnx")
    if not ckpts:
        print("  (no CenterNet checkpoint in p2_detection/checkpoints: scoring YuNet only)")
    py("p2_detection/compare_faces.py", *labels, *det)
    py("tools/update_results.py")


def step_pipeline(args) -> None:
    clip = Path(args.clip)
    if not clip.is_absolute():
        clip = ROOT / clip
    if not clip.exists():
        raise RuntimeError(f"{clip} not found (the fetch step downloads the public clips)")
    py("-m", "cinescope.pipeline", clip, "--backend", "onnx", "--workers", str(args.workers), "--out", args.out)


def has_tensorrt() -> bool:
    try:
        import onnxruntime as ort
    except ImportError:
        return False
    return "TensorrtExecutionProvider" in ort.get_available_providers()


def step_accel(args) -> None:
    py("p9_production/export_onnx.py", "--out", Path(args.out) / "onnx.json")
    if not has_tensorrt():
        raise Skip("ONNX export checked; TensorRT needs an NVIDIA GPU and onnxruntime-gpu built with "
                   "TensorRT (onnxruntime here offers no TensorrtExecutionProvider)")
    clip = ROOT / args.clip if not Path(args.clip).is_absolute() else Path(args.clip)
    py("-m", "cinescope.pipeline", clip, "--backend", "tensorrt", "--out", Path(args.out) / "tensorrt")


def step_docker(args) -> None:
    docker = shutil.which("docker")
    if not docker or subprocess.run([docker, "info"], capture_output=True).returncode != 0:
        raise Skip("no Docker daemon here (CI builds and smoke-tests the image on every push)")
    sh(docker, "build", "-t", "cinescope", ".")


def step_kaggle(args) -> None:
    if not (Path.home() / ".kaggle" / "kaggle.json").exists() and not os.environ.get("KAGGLE_USERNAME"):
        raise Skip("no Kaggle API token (~/.kaggle/kaggle.json): kaggle.com -> Settings -> API -> Create New Token")
    py("-m", "pip", "install", "-q", "kaggle")
    py("kaggle/run_all.py", *(["--only", *args.kaggle_jobs] if args.kaggle_jobs else []))


STEPS = {n: globals()[f"step_{n}"] for n in ALL}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("steps", nargs="*", metavar="STEP", help=f"any of {' '.join(ALL)} (default: {' '.join(DEFAULT)})")
    ap.add_argument("--clip", default="trailers/goldeneye.mp4", help="clip for the pipeline step")
    ap.add_argument("--out", default="outputs", help="where the pipeline and accel steps write")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--kaggle-jobs", nargs="*", help="kaggle step: only these jobs (default: every remaining one)")
    ap.add_argument("--keep-going", action="store_true", help="run the remaining steps after a failure")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args(argv)
    if args.list:
        print(__doc__.split("Steps (default order):")[1].split("Every step")[0].rstrip())
        return 0
    steps = args.steps or DEFAULT
    if bad := [s for s in steps if s not in STEPS]:
        ap.error(f"unknown step(s) {bad}; choose from {ALL}")
    os.chdir(ROOT)
    os.environ.setdefault("PYTHONUTF8", "1")       # Windows consoles: the tables use non-ASCII dashes
    if "setup" not in steps and not in_venv() and VENV_PY.exists():
        return subprocess.call([str(VENV_PY), __file__, *sys.argv[1:]], cwd=ROOT)
    summary, failed = [], False
    for name in steps:
        print(f"\n=== {name} ===", flush=True)
        t0 = time.time()
        try:
            STEPS[name](args)
            status = "ok"
        except Skip as e:
            status = f"skipped: {e}"
        except (subprocess.CalledProcessError, RuntimeError) as e:
            status, failed = f"FAILED: {e}", True
        summary.append((name, status, time.time() - t0))
        print(f"--- {name}: {status}", flush=True)
        if failed and not args.keep_going:
            break
    print("\n=== summary ===")
    for name, status, dt in summary:
        print(f"{name:9s} {dt:7.1f} s  {status}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
