"""Run a trained CenterNet face detector on images or video frames."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .wider import MEAN, STD


class FaceDetector:
    def __init__(self, checkpoint: str | Path, device: str | None = None, threshold: float = 0.3,
                 max_side: int = 1024):
        import torch

        from .centernet import CenterNet

        self.torch = torch
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        state = torch.load(checkpoint, map_location=self.device, weights_only=False)
        self.model = CenterNet(n_classes=state.get("n_classes", 1))
        self.model.load_state_dict(state["model"])
        self.model.to(self.device).eval()
        self.threshold, self.max_side = threshold, max_side

    def _prep(self, rgb: np.ndarray) -> tuple[np.ndarray, float]:
        from PIL import Image

        h, w = rgb.shape[:2]
        s = min(1.0, self.max_side / max(h, w))
        if s < 1.0:
            rgb = np.asarray(Image.fromarray(rgb).resize((round(w * s), round(h * s)), Image.BILINEAR))
        a = (rgb.astype(np.float32) / 255.0 - MEAN) / STD
        H, W = -(-a.shape[0] // 32) * 32, -(-a.shape[1] // 32) * 32
        out = np.zeros((H, W, 3), np.float32)
        out[:a.shape[0], :a.shape[1]] = a
        return out.transpose(2, 0, 1), s

    def detect(self, frames: list[np.ndarray], nms_iou: float | None = 0.5) -> list[dict]:
        """RGB uint8 frames of one size -> per frame {"boxes": (N,4) xyxy, "scores": (N,)}.
        Peak extraction already removes most duplicates; a light NMS catches the
        rest (two adjacent peaks on one large face)."""
        from ..detect.nms import nms
        from .centernet import decode

        batch, scale = zip(*(self._prep(f) for f in frames), strict=True)
        x = self.torch.from_numpy(np.stack(batch)).to(self.device)
        with self.torch.no_grad():
            dets = decode(self.model(x), threshold=self.threshold)
        out = []
        for (b, s, _), sc in zip(dets, scale, strict=True):
            b, s = b.cpu().numpy() / sc, s.cpu().numpy()
            if nms_iou is not None and len(b):
                keep = nms(b, s, nms_iou)
                b, s = b[keep], s[keep]
            out.append({"boxes": b, "scores": s})
        return out
