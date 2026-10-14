"""YOLOv8n-pose (Ultralytics): the library baseline, and the teacher whose
outputs the from-scratch network is distilled from.

YOLOv8-pose is bottom-up-ish: one network predicts person boxes *and* their 17
keypoints in one pass (a keypoint head on each detection anchor), so its cost
does not grow with the number of people. The top-down SimpleBaseline needs a
person detector first and runs once per person.

Weights are always loaded from an explicit local path (Ultralytics otherwise
tries to download them).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .simplebaseline import weights_dir


@dataclass
class People:
    boxes: np.ndarray       # (P, 4) xyxy, frame pixels
    scores: np.ndarray      # (P,)
    kpts: np.ndarray        # (P, 17, 3): x, y, confidence

    def __len__(self) -> int:
        return len(self.boxes)

    def to_dict(self, nd: int = 1) -> dict:
        return {"boxes": np.round(self.boxes, nd).tolist(), "scores": np.round(self.scores, 3).tolist(),
                "kpts": np.round(self.kpts, nd + 1).tolist()}


class YoloPose:
    def __init__(self, weights: str | Path | None = None, conf: float = 0.25, imgsz: int = 640,
                 pose: bool = True):
        from ultralytics import YOLO

        name = "yolov8n-pose.pt" if pose else "yolov8n.pt"
        self.model = YOLO(str(weights or weights_dir() / name))
        self.conf, self.imgsz, self.pose = conf, imgsz, pose

    def __call__(self, frames: list[np.ndarray]) -> list[People]:
        """RGB frames -> people per frame (persons only, class 0)."""
        if not frames:
            return []
        bgr = [np.ascontiguousarray(f[:, :, ::-1]) for f in frames]
        res = self.model.predict(bgr, conf=self.conf, imgsz=self.imgsz, verbose=False, classes=[0])
        out = []
        for r in res:
            b = r.boxes.xyxy.cpu().numpy().astype(np.float64)
            s = r.boxes.conf.cpu().numpy().astype(np.float64)
            if self.pose and r.keypoints is not None and len(b):
                k = r.keypoints.data.cpu().numpy().astype(np.float64)
            else:
                k = np.zeros((len(b), 17, 3))
            out.append(People(b.reshape(-1, 4), s.reshape(-1), k.reshape(-1, 17, 3)))
        return out
