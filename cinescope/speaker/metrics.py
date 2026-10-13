"""Frame-level scores for active-speaker detection.

AVA-ActiveSpeaker reports mean average precision over face-frames; here the
same: every (face, frame) pair is one example, the score ranks them, and AP
is the area under the precision-recall curve computed as the mean precision
at each true positive (the non-interpolated form, as in
sklearn.metrics.average_precision_score). F1 needs a threshold and is
reported at a threshold fixed in advance.
"""
from __future__ import annotations

import numpy as np


def average_precision(y: np.ndarray, s: np.ndarray) -> float:
    y = np.asarray(y, bool)
    s = np.asarray(s, np.float64)
    if not y.any():
        return float("nan")
    order = np.argsort(-s, kind="stable")
    ys, ss = y[order], s[order]
    # Ties: a block of equal scores enters the ranking at once.
    last = np.r_[np.flatnonzero(np.diff(ss)), len(ss) - 1]
    tp = np.cumsum(ys)[last]
    n = last + 1
    prec = tp / n
    rec = tp / y.sum()
    dr = np.diff(np.r_[0.0, rec])
    return float((dr * prec).sum())


def f1_at(y: np.ndarray, s: np.ndarray, threshold: float) -> dict:
    y = np.asarray(y, bool)
    p = np.asarray(s) >= threshold
    tp = int((p & y).sum())
    fp = int((p & ~y).sum())
    fn = int((~p & y).sum())
    pr = tp / max(tp + fp, 1)
    rc = tp / max(tp + fn, 1)
    return {"precision": pr, "recall": rc, "f1": 2 * pr * rc / max(pr + rc, 1e-12), "tp": tp, "fp": fp, "fn": fn}


def roc_auc(y: np.ndarray, s: np.ndarray) -> float:
    """Probability a random positive outranks a random negative (ties count half)."""
    y = np.asarray(y, bool)
    s = np.asarray(s, np.float64)
    if y.all() or not y.any():
        return float("nan")
    from scipy.stats import rankdata

    r = rankdata(s)
    n1, n0 = y.sum(), (~y).sum()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
