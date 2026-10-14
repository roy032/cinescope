"""OpenCV's YuNet face detector (Wu et al., 2023) behind a small, uniform interface.

YuNet is the production baseline the week-8 comparison scores our CenterNet
against, and the fallback face cue of the Phase 1 teacher when no CenterNet
checkpoint is at hand. The model file comes from the OpenCV model zoo
(github.com/opencv/opencv_zoo, models/face_detection_yunet); the default path
is `$CINESCOPE_WEIGHTS/yunet.onnx` or `<repo>/data/weights/yunet.onnx`.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]


def default_model() -> Path:
    return Path(os.environ.get("CINESCOPE_WEIGHTS", REPO / "data" / "weights")) / "yunet.onnx"


class YuNet:
    """`det(rgb)` -> (boxes (N, 4) xyxy in pixels, scores (N,)).

    `score` is the lowest confidence returned: keep it low for AP (the
    precision-recall curve needs the low-scoring tail) and threshold afterwards.
    """

    name = "OpenCV YuNet"

    def __init__(self, model: str | Path | None = None, score: float = 0.05, nms_iou: float = 0.3,
                 top_k: int = 5000):
        import cv2

        self.model = str(model or default_model())
        if not Path(self.model).exists():
            raise FileNotFoundError(f"{self.model} not found: download face_detection_yunet_*.onnx from the "
                                    "OpenCV model zoo or set CINESCOPE_WEIGHTS")
        self.cv2, self.score, self.nms_iou, self.top_k = cv2, score, nms_iou, top_k
        self.det = None

    def __call__(self, rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h, w = rgb.shape[:2]
        if self.det is None:
            self.det = self.cv2.FaceDetectorYN.create(self.model, "", (w, h), self.score, self.nms_iou, self.top_k)
        self.det.setInputSize((w, h))
        _, faces = self.det.detect(self.cv2.cvtColor(np.ascontiguousarray(rgb), self.cv2.COLOR_RGB2BGR))
        if faces is None:
            return np.zeros((0, 4)), np.zeros(0)
        xywh = faces[:, :4].astype(np.float64)
        return np.c_[xywh[:, :2], xywh[:, :2] + xywh[:, 2:]], faces[:, 14].astype(np.float64)

    def detect(self, frames: list[np.ndarray]) -> list[dict]:
        """Same output as `FaceDetector.detect`: per frame {"boxes": (N, 4) xyxy, "scores": (N,)}."""
        out = []
        for f in frames:
            b, s = self(f)
            out.append({"boxes": b, "scores": s})
        return out
