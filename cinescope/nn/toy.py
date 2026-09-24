"""A synthetic "shot scale" task for the NumPy CNN.

Each image is a coloured disk (the subject) on a noisy background; its class is
how much of the frame the subject fills — the same cue a close-up / medium /
wide label depends on. The subject is lit and the set is darker, as in
most shots; position, size within the class, colours and noise vary. Small
enough to train on a CPU in seconds.
"""
from __future__ import annotations

import numpy as np

from .layers import (
    BatchNorm2d,
    Conv2d,
    GlobalAvgPool,
    Linear,
    MaxPool2d,
    ReLU,
    Residual,
    Sequential,
)

CLASSES = ("close", "medium", "wide")
_RADIUS = {0: (0.38, 0.50), 1: (0.20, 0.28), 2: (0.07, 0.12)}     # fraction of image side


def make_framing_toy(n: int, size: int = 24, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 3, n)
    yy, xx = np.mgrid[:size, :size] + 0.5
    x = np.empty((n, 3, size, size))
    for i, c in enumerate(y):
        r = rng.uniform(*_RADIUS[int(c)]) * size
        cy, cx = rng.uniform(r * 0.6, size - r * 0.6, 2) if r < size / 2 else rng.uniform(size * 0.4, size * 0.6, 2)
        disk = ((yy - cy) ** 2 + (xx - cx) ** 2) < r * r
        bg, fg = rng.uniform(0, 0.4, 3), rng.uniform(0.6, 1, 3)   # lit subject, darker set
        img = np.where(disk[None], fg[:, None, None], bg[:, None, None])
        x[i] = img + rng.normal(0, 0.08, img.shape)
    return (x - 0.5) / 0.25, y


def tiny_resnet(n_classes: int = 3, width: int = 8, seed: int = 0) -> Sequential:
    w = width
    return Sequential(
        Conv2d(3, w, 3, padding=1, bias=False, seed=seed), BatchNorm2d(w), ReLU(), MaxPool2d(2),
        Residual(Sequential(Conv2d(w, w, 3, padding=1, bias=False, seed=seed + 1), BatchNorm2d(w), ReLU(),
                            Conv2d(w, w, 3, padding=1, bias=False, seed=seed + 2), BatchNorm2d(w))),
        Conv2d(w, 2 * w, 3, stride=2, padding=1, bias=False, seed=seed + 3), BatchNorm2d(2 * w), ReLU(),
        GlobalAvgPool(), Linear(2 * w, n_classes, seed=seed + 4),
    )
