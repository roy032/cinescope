"""Non-maximum suppression.

A detector fires many overlapping boxes on one object. Greedy NMS keeps the
highest-scoring box, deletes every remaining box that overlaps it by more than
a threshold, and repeats. Its failure mode is crowds: two real faces that
overlap by more than the threshold (a hug, a crowd shot, a face in front of
another) are treated as duplicates and one disappears. Soft-NMS (Bodla et al.,
2017) decays the scores of overlapping boxes instead of deleting them, so a
strongly-scored second face survives with a lower score.
"""
from __future__ import annotations

import numpy as np

from .boxes import iou


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float = 0.5) -> np.ndarray:
    """Indices of the kept xyxy boxes, highest score first (like torchvision.ops.nms).
    Boxes overlapping a kept box by IoU > iou_thresh are removed."""
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    order = np.argsort(-np.asarray(scores, dtype=np.float64), kind="stable")
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        overlap = iou(boxes[i:i + 1], boxes[order[1:]])[0]
        order = order[1:][overlap <= iou_thresh]
    return np.asarray(keep, dtype=np.int64)


def batched_nms(boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray,
                iou_thresh: float = 0.5) -> np.ndarray:
    """NMS within each class only: a phone held to a face must not suppress the face.
    Implemented by shifting each class's boxes far apart, then one NMS pass."""
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    if not len(boxes):
        return np.zeros(0, dtype=np.int64)
    offset = np.asarray(classes, dtype=np.float64)[:, None] * (boxes.max() + 1)
    return nms(boxes + offset, scores, iou_thresh)


def soft_nms(boxes: np.ndarray, scores: np.ndarray, sigma: float = 0.5, iou_thresh: float = 0.3,
             score_thresh: float = 0.001, method: str = "gaussian") -> tuple[np.ndarray, np.ndarray]:
    """Soft-NMS. Returns (kept indices in selection order, their decayed scores).

    gaussian: s *= exp(-iou² / sigma)
    linear:   s *= (1 - iou) where iou > iou_thresh
    Boxes whose score falls under score_thresh are dropped.
    """
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    s = np.asarray(scores, dtype=np.float64).copy()
    idx = np.arange(len(s))
    keep, kept_scores = [], []
    while idx.size:
        j = int(np.argmax(s[idx]))
        i = idx[j]
        keep.append(i)
        kept_scores.append(s[i])
        idx = np.delete(idx, j)
        if not idx.size:
            break
        ov = iou(boxes[i:i + 1], boxes[idx])[0]
        if method == "gaussian":
            s[idx] *= np.exp(-(ov ** 2) / sigma)
        elif method == "linear":
            s[idx] *= np.where(ov > iou_thresh, 1 - ov, 1.0)
        else:
            raise ValueError(f"unknown soft-NMS method {method!r}")
        idx = idx[s[idx] >= score_thresh]
    return np.asarray(keep, dtype=np.int64), np.asarray(kept_scores)
