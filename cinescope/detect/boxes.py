"""Box geometry in NumPy.

Two formats, as in COCO and most detectors:
  xyxy  [x1, y1, x2, y2]  corners — what IoU and NMS compute on
  xywh  [x, y, w, h]      top-left + size — what COCO JSON stores

Coordinates are continuous (no "+1" pixel convention): a box from x=10 to
x=20 has width 10. PASCAL VOC's old code added 1; COCO does not, and mixing
the two shifts IoU for small boxes by several points — faces in WIDER FACE are
often under 20 px, where that matters.
"""
from __future__ import annotations

import numpy as np


def xywh_to_xyxy(b: np.ndarray) -> np.ndarray:
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    return np.c_[b[:, :2], b[:, :2] + b[:, 2:]]


def xyxy_to_xywh(b: np.ndarray) -> np.ndarray:
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    return np.c_[b[:, :2], b[:, 2:] - b[:, :2]]


def area(b: np.ndarray) -> np.ndarray:
    """Area of xyxy boxes (0 for degenerate ones)."""
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    return np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)


def intersection(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(N, M) intersection areas of xyxy boxes a (N) and b (M)."""
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    return wh[..., 0] * wh[..., 1]


def iou(a: np.ndarray, b: np.ndarray, crowd: np.ndarray | None = None) -> np.ndarray:
    """(N, M) intersection over union of xyxy boxes.

    `crowd` (length M, bool) follows COCO: for a crowd region (a group of people
    labelled as one box) the denominator is the *detection's* area, so a
    detection lying inside the crowd counts as fully overlapping it instead of
    being punished for covering only part of the region.
    """
    inter = intersection(a, b)
    aa, ab = area(a)[:, None], area(b)[None, :]
    union = aa + ab - inter
    if crowd is not None:
        union = np.where(np.asarray(crowd, bool)[None, :], aa, union)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(union > 0, inter / union, 0.0)


def giou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Generalised IoU (Rezatofighi et al., 2019): IoU minus the fraction of the
    smallest enclosing box not covered by the union. Unlike IoU it still has a
    gradient when boxes do not overlap, which is why it is used as a loss."""
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    inter = intersection(a, b)
    union = area(a)[:, None] + area(b)[None, :] - inter
    lt = np.minimum(a[:, None, :2], b[None, :, :2])
    rb = np.maximum(a[:, None, 2:], b[None, :, 2:])
    hull = np.prod(np.clip(rb - lt, 0, None), axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        i = np.where(union > 0, inter / union, 0.0)
        return i - np.where(hull > 0, (hull - union) / hull, 0.0)
