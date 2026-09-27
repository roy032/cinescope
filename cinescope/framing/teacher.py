"""Automatic shot-scale labels for footage that has none (Phase 1 on our own episodes).

Two independent cues, fused per shot:

1. Zero-shot image-text matching (SigLIP). Each class is described by a few
   prompts; a frame's class score is the mean logit over that class's prompts.
2. Face size from the Phase 2 detector. Shot scale is, by definition, how much
   of a person the frame shows, and the height of the largest face relative to
   the frame height tracks that closely when a face is visible:
       ECS  face taller than the frame (cropped)     CS  ~0.35-0.6
       MS   ~0.15-0.3                                FS  ~0.06-0.12   LS  smaller
   The ratio becomes a soft prior: a Gaussian in log space around each class's
   typical ratio, so a ratio between two classes splits the mass between them.

The fused label is argmax(log p_text + w * log p_face) when a face was found,
and the text prediction alone otherwise. Neither cue is trusted as ground
truth: the test set is labelled by hand (p1_cnn/label_scales.py) and every cue
is scored against it separately (p1_cnn/eval_own.py).
"""
from __future__ import annotations

import numpy as np

from . import SCALES

PROMPTS: dict[str, list[str]] = {
    "ECS": ["an extreme close-up shot showing only part of a face, such as the eyes or mouth",
            "an extreme close-up of a small object or detail filling the whole frame",
            "a macro shot of a hand or an object, very close to the camera"],
    "CS": ["a close-up shot of a person's face",
           "a close-up shot where a head and shoulders fill the frame",
           "a close-up portrait shot from a film"],
    "MS": ["a medium shot of a person from the waist up",
           "a medium shot of two people talking, shown from the chest up",
           "a medium shot from a TV series, upper body visible"],
    "FS": ["a full shot showing a person's whole body from head to feet",
           "a full-length shot of people standing in a room",
           "a full shot where the entire figure is visible"],
    "LS": ["a long shot of a wide landscape or street with small distant people",
           "a wide establishing shot of a building or a city",
           "a long shot where people are tiny in a large space"],
}

# typical largest-face-height / frame-height per class (geometric centres of the ranges above)
FACE_CENTRES = {"ECS": 0.9, "CS": 0.45, "MS": 0.2, "FS": 0.085, "LS": 0.03}
FACE_SIGMA = 0.45          # log units: ~1.6x either side of a centre before the prior halves twice


def class_scores(logits: np.ndarray, prompt_class: list[str], classes: tuple[str, ...] = SCALES) -> np.ndarray:
    """(N, n_prompts) image-text logits -> (N, n_classes) mean logit per class."""
    logits = np.asarray(logits, dtype=np.float64)
    out = np.empty((logits.shape[0], len(classes)))
    for j, c in enumerate(classes):
        cols = [i for i, pc in enumerate(prompt_class) if pc == c]
        out[:, j] = logits[:, cols].mean(1)
    return out


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def face_ratio(boxes: np.ndarray, scores: np.ndarray, frame_h: int, min_score: float = 0.5) -> float | None:
    """Height of the largest confidently detected face / frame height (None: no face)."""
    boxes, scores = np.asarray(boxes).reshape(-1, 4), np.asarray(scores).reshape(-1)
    keep = scores >= min_score
    if not keep.any():
        return None
    return float((boxes[keep, 3] - boxes[keep, 1]).max() / frame_h)


def face_prior(ratio: float | None, classes: tuple[str, ...] = SCALES,
               sigma: float = FACE_SIGMA) -> np.ndarray | None:
    """Soft class distribution from a face-height ratio; None when there is no face."""
    if ratio is None or not np.isfinite(ratio) or ratio <= 0:
        return None
    d = np.array([np.log(ratio) - np.log(FACE_CENTRES[c]) for c in classes])
    return softmax(-0.5 * (d / sigma) ** 2)


def fuse(p_text: np.ndarray, p_face: np.ndarray | None, w_face: float = 1.0) -> np.ndarray:
    """Product of experts: p ∝ p_text * p_face^w (text alone when there is no face)."""
    if p_face is None:
        return np.asarray(p_text, dtype=np.float64)
    logp = np.log(np.clip(p_text, 1e-9, 1)) + w_face * np.log(np.clip(p_face, 1e-9, 1))
    return softmax(logp)
