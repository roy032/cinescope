"""Label every shot of an analysed clip with its framing (Phase 1 in the pipeline).

    clf = FramingClassifier("p1_cnn/checkpoints/pretrained.pt")
    labels = clf.label_shots("clip.mkv", analysis.shots)

Like MovieShots, each shot is represented by three key frames (at 25%, 50% and
75% of its length); the class probabilities of the three are averaged.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..io.video import VideoReader
from . import SCALE_NAMES


def keyframe_indices(shots, positions: tuple[float, ...] = (0.25, 0.5, 0.75)) -> dict[int, list[int]]:
    """Decoded-frame indices of the key frames of each shot, keyed by shot index."""
    out = {}
    for s in shots:
        span = s.end - s.start
        out[s.index] = sorted({s.start + int(round(p * span)) for p in positions})
    return out


def read_frames(video: str | Path, wanted: set[int], width: int = 480) -> dict[int, np.ndarray]:
    frames: dict[int, np.ndarray] = {}
    last = max(wanted) if wanted else -1
    with VideoReader(video, width=width) as vr:
        for i, f in enumerate(vr.frames()):
            if i in wanted:
                frames[i] = f.rgb
            if i >= last:
                break
    return frames


class FramingClassifier:
    def __init__(self, checkpoint: str | Path, device: str | None = None):
        import torch

        from .data import build_transforms
        from .resnet import resnet18

        self.torch = torch
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        state = torch.load(checkpoint, map_location=self.device, weights_only=False)
        self.classes = tuple(state["classes"])
        self.model = resnet18(len(self.classes))
        self.model.load_state_dict(state["model"])
        self.model.to(self.device).eval()
        self.tf = build_transforms(False, tuple(state.get("size", (224, 384))))

    def predict(self, frames: list[np.ndarray]) -> np.ndarray:
        """Class probabilities (N, n_classes) for RGB uint8 frames."""
        from PIL import Image

        x = self.torch.stack([self.tf(Image.fromarray(f)) for f in frames]).to(self.device)
        with self.torch.no_grad():
            return self.model(x).softmax(1).cpu().numpy()

    def label_shots(self, video: str | Path, shots, batch: int = 32) -> list[dict]:
        keys = keyframe_indices(shots)
        frames = read_frames(video, {i for v in keys.values() for i in v})
        order = [(s, i) for s, idx in keys.items() for i in idx if i in frames]
        probs: dict[int, list[np.ndarray]] = {}
        for b in range(0, len(order), batch):
            chunk = order[b:b + batch]
            p = self.predict([frames[i] for _, i in chunk])
            for (s, _), row in zip(chunk, p, strict=True):
                probs.setdefault(s, []).append(row)
        out = []
        for s in shots:
            if s.index not in probs:
                out.append({"label": None, "name": None, "confidence": 0.0})
                continue
            p = np.mean(probs[s.index], 0)
            k = int(p.argmax())
            out.append({"label": self.classes[k], "name": SCALE_NAMES.get(self.classes[k], self.classes[k]),
                        "confidence": round(float(p[k]), 3),
                        "probs": {c: round(float(v), 3) for c, v in zip(self.classes, p, strict=True)}})
        return out
