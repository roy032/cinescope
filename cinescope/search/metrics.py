"""Retrieval metrics for scene search, with relevance defined by time overlap.

Ground truth is a list of time intervals (seconds) per query, not segment ids, so
the labels stay valid if the segmentation changes (another shot detector, another
max segment length). A retrieved segment is relevant when it overlaps a
ground-truth interval by at least half of the shorter of the two: a 6 s segment
that grazes a 5 s event by 0.4 s does not count; a 1 s shot inside a 20 s event does.

Recall@k here is the text-to-video retrieval convention ("success@k"): 1 if at
least one relevant segment is in the top k, averaged over queries. MRR is the
mean of 1 / (rank of the first relevant segment); a query whose relevant segments
are never returned (possible with an approximate index) scores 0.
"""
from __future__ import annotations

import numpy as np


def overlaps(t0: float, t1: float, gt: list[list[float]], frac: float = 0.5) -> bool:
    for a, b in gt:
        ov = min(t1, b) - max(t0, a)
        if ov > 0 and ov >= frac * min(t1 - t0, b - a):
            return True
    return False


def first_relevant_rank(ranked: list[dict], gt: list[list[float]], clip: str | None = None) -> int | None:
    """1-based rank of the first relevant document; None if none was returned.
    With `clip`, documents from other clips (library search) are never relevant."""
    for r, d in enumerate(ranked, 1):
        if (clip is None or d.get("clip") == clip) and overlaps(d["t0"], d["t1"], gt):
            return r
    return None


def summarise(ranks: list[int | None], ks: tuple[int, ...] = (1, 5)) -> dict:
    n = len(ranks)
    out = {f"R@{k}": round(sum(1 for r in ranks if r is not None and r <= k) / max(n, 1), 4) for k in ks}
    out["MRR"] = round(float(np.mean([1.0 / r if r else 0.0 for r in ranks])) if n else 0.0, 4)
    out["n"] = n
    return out
