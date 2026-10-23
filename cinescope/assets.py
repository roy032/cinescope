"""The model weights CineScope reads, where to get them, and how to check them.

One table for both sides: tools/fetch_assets.py downloads from it, and
cinescope.doctor checks the files against it. Each weight lists its mirrors
in the order they are tried (GitHub's LFS host refuses some networks, so
files that live there have a second source) and its SHA-256 where it is
known, so a truncated download, an HTML error page or a different file under
the right name is caught instead of failing deep inside a stage.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

_ZOO = "https://github.com/opencv/opencv_zoo/raw/main/models"
_ZOO_LFS = "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models"
_ULTRA = "https://github.com/ultralytics/assets/releases/download/v8.2.0"


@dataclass(frozen=True)
class Weight:
    name: str                       # file name in the weights folder (what the code opens)
    urls: tuple[str, ...]           # mirrors, tried in order
    sha256: str | None              # None: not pinned (only size / not-a-pointer checks)
    used_by: tuple[str, ...]        # pipeline stages (or tools) that read it
    note: str = ""


WEIGHTS: dict[str, Weight] = {w.name: w for w in [
    Weight("yunet.onnx",
           (f"{_ZOO}/face_detection_yunet/face_detection_yunet_2023mar.onnx",
            f"{_ZOO_LFS}/face_detection_yunet/face_detection_yunet_2023mar.onnx",
            "https://raw.githubusercontent.com/ShiqiYu/libfacedetection.train/master/onnx/yunet_s_640_640.onnx"),
           "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4", ("faces",), "YuNet face detector"),
    Weight("sface.onnx",
           (f"{_ZOO}/face_recognition_sface/face_recognition_sface_2021dec.onnx",
            f"{_ZOO_LFS}/face_recognition_sface/face_recognition_sface_2021dec.onnx"),
           None, ("characters",), "SFace face embeddings (who is who)"),
    Weight("light_asd_ava.model",
           ("https://raw.githubusercontent.com/Junhua-Liao/Light-ASD/main/weight/pretrain_AVA_CVPR.model",
            "https://github.com/Junhua-Liao/Light-ASD/raw/main/weight/pretrain_AVA_CVPR.model"),
           "d44bc3ea7baa8e0946fa3921311714a630ed8b90a1928fab0dbe30d918909317", ("speaking",),
           "Light-ASD active speaker (optional: a built-in fallback runs without it)"),
    Weight("vit_b_32-quickgelu-laion400m_e32.pt",
           ("https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/"
            "vit_b_32-quickgelu-laion400m_e32-46683a32.pt",),
           "46683a32721d5c68911153698992361285d20ca690bb4f317c11e45c03d798fa", ("search", "scenes"),
           "OpenCLIP ViT-B/32 (scene search, scene types)"),
    Weight("yolov8n.pt", (f"{_ULTRA}/yolov8n.pt",),
           "f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36", ("p2_detection",),
           "YOLOv8n detector (Phase 2 baseline)"),
    Weight("yolov8n-pose.pt", (f"{_ULTRA}/yolov8n-pose.pt",),
           "c6fa93dd1ee4a2c18c900a45c1d864a1c6f7aba75d84f91648a30b7fb641d212", ("pose",), "YOLOv8n-pose"),
    Weight("yolov8n-seg.pt", (f"{_ULTRA}/yolov8n-seg.pt",),
           "a7cd8f929e1903d78a12a48efecab430209f18dc46cb96c3599a5980c63c423c", ("cutouts",),
           "YOLOv8n-seg person boxes for cut-outs"),
    Weight("resnet18_a1_0-d63eafa0.pth",
           ("https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/"
            "resnet18_a1_0-d63eafa0.pth",),
           "d63eafa07a6e32a39d328e364f8c9f89d671444ecc7f02aa0f7eb8882af3dd29", ("p6_pose_video",),
           "ImageNet ResNet-18 backbone (pose training)"),
]}


def weights_dir() -> Path:
    return Path(os.environ.get("CINESCOPE_WEIGHTS", REPO / "data" / "weights"))


def sha256(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def looks_wrong(path: Path) -> str | None:
    """Why a downloaded file is clearly not the weight (None when it looks fine)."""
    size = path.stat().st_size
    if size < 1024:
        with open(path, "rb") as f:
            head = f.read(200)
        if head.startswith(b"version https://git-lfs"):
            return "a Git LFS pointer, not the file (download it from a mirror)"
        if head.lstrip().lower().startswith((b"<!doctype", b"<html")):
            return "an HTML page, not the file"
        return f"only {size} bytes"
    return None


def check(w: Weight, folder: Path | None = None, hash_it: bool = True) -> dict:
    """{"name", "path", "status": ok | missing | bad | unverified, "detail"} for one weight."""
    p = (folder or weights_dir()) / w.name
    row = {"name": w.name, "path": str(p), "used_by": list(w.used_by), "note": w.note}
    if not p.exists():
        return {**row, "status": "missing", "detail": "not downloaded"}
    bad = looks_wrong(p)
    if bad:
        return {**row, "status": "bad", "detail": bad}
    if w.sha256 is None or not hash_it:
        return {**row, "status": "unverified" if w.sha256 is None else "ok",
                "detail": f"{p.stat().st_size / 2**20:.1f} MB"}
    got = sha256(p)
    if got != w.sha256:
        return {**row, "status": "bad", "detail": f"SHA-256 {got[:12]}… is not the expected {w.sha256[:12]}…"}
    return {**row, "status": "ok", "detail": f"{p.stat().st_size / 2**20:.1f} MB, SHA-256 verified"}
