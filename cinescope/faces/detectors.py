"""Pluggable face detectors for tracking.

Everything downstream (tracker, embeddings, active speaker) needs, per frame,
face boxes, confidence scores and -- when the detector provides them -- five
landmarks (eyes, nose tip, mouth corners). Two detectors share that interface:

  yunet      OpenCV's YuNet (opencv_zoo, `cv2.FaceDetectorYN`): 0.2 MB, fast on
             CPU, returns landmarks. The default.
  centernet  The Phase 2 CenterNet-R18 face detector trained here
             (`cinescope/detect/faces.py`). Boxes only; landmarks are estimated
             from the box with average face proportions.

`make_detector("yunet")` / `make_detector("centernet", ckpt=...)`. Weights
default to `<repo>/data/weights` (env `CINESCOPE_WEIGHTS` overrides).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]


def weights_dir() -> Path:
    return Path(os.environ.get("CINESCOPE_WEIGHTS", REPO / "data" / "weights"))


def weight_path(name: str) -> Path:
    p = Path(name)
    return p if p.exists() else weights_dir() / name


# Mean landmark positions as fractions of a face box (x, y); from YuNet's own
# output averaged over frontal faces. Used when a detector gives no landmarks.
MEAN_LANDMARKS = np.array([[0.31, 0.40], [0.69, 0.40], [0.50, 0.58], [0.35, 0.77], [0.65, 0.77]])


@dataclass
class FaceDets:
    boxes: np.ndarray       # (N, 4) xyxy in pixels of the frame given
    scores: np.ndarray      # (N,)
    landmarks: np.ndarray   # (N, 5, 2): right eye, left eye, nose, right mouth, left mouth (image coords)

    def __len__(self) -> int:
        return len(self.scores)

    @staticmethod
    def empty() -> FaceDets:
        return FaceDets(np.zeros((0, 4)), np.zeros(0), np.zeros((0, 5, 2)))


def landmarks_from_boxes(boxes: np.ndarray) -> np.ndarray:
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    wh = b[:, 2:] - b[:, :2]
    return b[:, None, :2] + MEAN_LANDMARKS[None] * wh[:, None, :]


class YuNet:
    name = "yunet"

    def __init__(self, model: str | Path = "yunet.onnx", score_threshold: float = 0.3,
                 nms_threshold: float = 0.3, max_side: int = 960):
        import cv2

        self.cv2 = cv2
        self.path = str(weight_path(str(model)))
        if not Path(self.path).exists():
            raise FileNotFoundError(f"{self.path} (download face_detection_yunet_2023mar.onnx from opencv_zoo)")
        self.score_threshold, self.nms_threshold, self.max_side = score_threshold, nms_threshold, max_side
        self._det = None
        self._size = None

    def __call__(self, rgb: np.ndarray) -> FaceDets:
        cv2 = self.cv2
        h, w = rgb.shape[:2]
        s = min(1.0, self.max_side / max(h, w))
        img = rgb if s == 1.0 else cv2.resize(rgb, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
        size = (img.shape[1], img.shape[0])
        if self._det is None:
            self._det = cv2.FaceDetectorYN.create(self.path, "", size, self.score_threshold,
                                                  self.nms_threshold, 5000)
        if size != self._size:
            self._det.setInputSize(size)
            self._size = size
        _, f = self._det.detect(np.ascontiguousarray(img[:, :, ::-1]))
        if f is None or not len(f):
            return FaceDets.empty()
        f = f.astype(np.float64)
        scores, f = f[:, 14], f[:, :14] / s
        boxes = np.c_[f[:, :2], f[:, :2] + f[:, 2:4]]
        return FaceDets(boxes, scores, f[:, 4:14].reshape(-1, 5, 2))


class CenterNetFaces:
    name = "centernet"

    def __init__(self, ckpt: str | Path, threshold: float = 0.3, device: str | None = None):
        from ..detect.faces import FaceDetector

        self.det = FaceDetector(ckpt, device=device, threshold=threshold)

    def __call__(self, rgb: np.ndarray) -> FaceDets:
        r = self.det.detect([rgb])[0]
        if not len(r["scores"]):
            return FaceDets.empty()
        b = np.asarray(r["boxes"], dtype=np.float64)
        return FaceDets(b, np.asarray(r["scores"], dtype=np.float64), landmarks_from_boxes(b))


def make_detector(kind: str = "yunet", ckpt: str | Path | None = None, **kw):
    if kind == "yunet":
        return YuNet(**kw)
    if kind == "centernet":
        if ckpt is None:
            raise ValueError("centernet needs --ckpt (a Phase 2 checkpoint)")
        return CenterNetFaces(ckpt, **kw)
    raise ValueError(f"unknown detector {kind!r}")
