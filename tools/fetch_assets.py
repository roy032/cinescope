"""Download the public test clips and the library weights every phase reads.

    python tools/fetch_assets.py                    # clips -> trailers/, weights -> data/weights/
    python tools/fetch_assets.py --only clips       # or: weights
    python tools/fetch_assets.py --list             # what would be fetched, and from where

Everything comes from the projects' own public releases (the same URLs the phase
READMEs cite). Files already present are skipped, so it is safe to re-run. Nothing
here is committed: trailers/ and data/ are git-ignored (README: Footage and copyright).
"""
from __future__ import annotations

import argparse
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

_SCENEDETECT = "https://raw.githubusercontent.com/Breakthrough/PySceneDetect/resources/tests/resources"
_INTEL = "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master"
_ZOO = "https://github.com/opencv/opencv_zoo/raw/main/models"

# Public clips the phases are scored on: Phase 0 cuts, Phase 2 week 8, Phases 3-9.
CLIPS = {
    "goldeneye.mp4": f"{_SCENEDETECT}/goldeneye.mp4",
    "testvideo.mp4": f"{_SCENEDETECT}/testvideo.mp4",
    "head-pose-face-detection-female-and-male.mp4": f"{_INTEL}/head-pose-face-detection-female-and-male.mp4",
    "classroom.mp4": f"{_INTEL}/classroom.mp4",
}

# Library weights, under the names the code looks for in data/weights (or $CINESCOPE_WEIGHTS).
WEIGHTS = {
    "yunet.onnx": f"{_ZOO}/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "sface.onnx": f"{_ZOO}/face_recognition_sface/face_recognition_sface_2021dec.onnx",
    "light_asd_ava.model": "https://github.com/Junhua-Liao/Light-ASD/raw/main/weight/pretrain_AVA_CVPR.model",
    "vit_b_32-quickgelu-laion400m_e32.pt":
        "https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_b_32-quickgelu-laion400m_e32-46683a32.pt",
    "yolov8n.pt": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt",
    "yolov8n-pose.pt": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n-pose.pt",
    "resnet18_a1_0-d63eafa0.pth":
        "https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet18_a1_0-d63eafa0.pth",
}


def weights_dir() -> Path:
    return Path(os.environ.get("CINESCOPE_WEIGHTS", ROOT / "data" / "weights"))


def fetch(url: str, dest: Path) -> bool:
    """Download url -> dest unless it exists. A partial download never lands under the final name."""
    if dest.exists() and dest.stat().st_size > 0:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "cinescope-fetch"})
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    if tmp.stat().st_size < 1024:                 # an HTML error page or an LFS pointer, not the file
        tmp.unlink()
        raise RuntimeError(f"{url}: suspiciously small download")
    tmp.replace(dest)
    return True


def plan(only: str | None = None) -> list[tuple[str, Path]]:
    jobs = []
    if only in (None, "clips"):
        jobs += [(u, ROOT / "trailers" / n) for n, u in CLIPS.items()]
    if only in (None, "weights"):
        jobs += [(u, weights_dir() / n) for n, u in WEIGHTS.items()]
    return jobs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=["clips", "weights"])
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args(argv)
    failed = 0
    for url, dest in plan(args.only):
        rel = dest.relative_to(ROOT) if dest.is_relative_to(ROOT) else dest
        if args.list:
            print(f"{rel}  <-  {url}")
            continue
        try:
            print(f"{'fetched' if fetch(url, dest) else 'present'}  {rel}")
        except Exception as e:                    # keep going: one dead mirror should not block the rest
            failed += 1
            print(f"FAILED   {rel}: {e}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
