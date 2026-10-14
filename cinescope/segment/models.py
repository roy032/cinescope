"""Production segmenters behind one interface: SAM 2 tiny, MobileSAM and
YOLOv8n-seg through ultralytics, and the person detector the scripts use to
find boxes.

Weights are local files, never auto-downloaded: `weights_dir()` is
`$CINESCOPE_WEIGHTS` or `<repo root>/data/weights`.

CPU threads: on a shared CPU, PyTorch with more than one OpenMP thread can be
~10x *slower* (threads spin waiting for each other while other processes hold
the cores), and ultralytics resets the thread count on its first call. Every
wrapper here re-applies `threads` after each ultralytics call.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]


def weights_dir() -> Path:
    return Path(os.environ.get("CINESCOPE_WEIGHTS", REPO_ROOT / "data" / "weights"))


def weight_path(name: str) -> str:
    p = Path(name)
    if p.exists():
        return str(p)
    q = weights_dir() / name
    if not q.exists():
        raise FileNotFoundError(f"{name}: not found (looked in {weights_dir()}; set CINESCOPE_WEIGHTS)")
    return str(q)


def set_threads(n: int) -> None:
    import torch

    torch.set_num_threads(max(1, n))


class PersonDetector:
    """YOLOv8n-seg people: boxes (N, 4) xyxy, scores (N,), full-res masks (N, H, W)."""

    def __init__(self, weights: str = "yolov8n-seg.pt", conf: float = 0.4, threads: int = 1,
                 imgsz: int = 640) -> None:
        from ultralytics import YOLO

        self.model = YOLO(weight_path(weights))
        self.conf, self.threads, self.imgsz = conf, threads, imgsz

    def __call__(self, rgb: np.ndarray):
        r = self.model(np.ascontiguousarray(rgb[:, :, ::-1]), classes=[0], conf=self.conf,
                       imgsz=self.imgsz, verbose=False, retina_masks=True)[0]
        set_threads(self.threads)
        h, w = rgb.shape[:2]
        if r.boxes is None or len(r.boxes) == 0:
            return np.zeros((0, 4)), np.zeros(0), np.zeros((0, h, w), bool)
        boxes = r.boxes.xyxy.cpu().numpy().astype(np.float64)
        scores = r.boxes.conf.cpu().numpy().astype(np.float64)
        masks = r.masks.data.cpu().numpy() > 0.5 if r.masks is not None else np.zeros((len(boxes), h, w), bool)
        if masks.shape[1:] != (h, w):
            import cv2

            masks = np.stack([cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                              for m in masks]).astype(bool)
        return boxes, scores, masks


class YoloBoxSegmenter:
    """Make YOLOv8n-seg answer a box prompt: run it on the whole frame and
    return the person mask whose box overlaps the prompt best (empty if none
    overlaps by IoU >= 0.3). It is a detector, not a promptable model, so this
    is the fairest way to put it in the same table."""

    def __init__(self, detector: PersonDetector | None = None) -> None:
        self.det = detector or PersonDetector()
        self._cache: tuple[int, tuple] | None = None

    def __call__(self, rgb: np.ndarray, boxes) -> list[np.ndarray]:
        key = id(rgb)
        if self._cache is None or self._cache[0] != key:
            self._cache = (key, self.det(rgb))
        dboxes, _, masks = self._cache[1]
        out = []
        for b in np.asarray(boxes, np.float64).reshape(-1, 4):
            if len(dboxes) == 0:
                out.append(np.zeros(rgb.shape[:2], bool))
                continue
            from ..detect.boxes import iou

            ov = iou(b[None], dboxes)[0]
            k = int(ov.argmax())
            out.append(masks[k] if ov[k] >= 0.3 else np.zeros(rgb.shape[:2], bool))
        return out


class SamSegmenter:
    """SAM 2 tiny ('sam2_t.pt') or MobileSAM ('mobile_sam.pt') with box prompts.
    All boxes of a frame go in one call, so the image encoder runs once."""

    def __init__(self, weights: str = "sam2_t.pt", threads: int = 1) -> None:
        from ultralytics import SAM

        self.model = SAM(weight_path(weights))
        self.threads = threads

    def __call__(self, rgb: np.ndarray, boxes) -> list[np.ndarray]:
        boxes = np.asarray(boxes, np.float64).reshape(-1, 4)
        if len(boxes) == 0:
            return []
        r = self.model(np.ascontiguousarray(rgb[:, :, ::-1]), bboxes=boxes.tolist(), verbose=False)[0]
        set_threads(self.threads)
        m = r.masks.data.cpu().numpy() > 0.5
        return [m[i] for i in range(len(boxes))]
