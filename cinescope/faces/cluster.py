"""Tracks -> characters: agglomerative clustering from scratch, and its metrics.

Each track gets one embedding (the normalised mean of its frames'
embeddings). Average-linkage agglomerative clustering starts with every
track as its own cluster and repeatedly merges the two closest clusters,
where the distance between clusters is the mean pairwise cosine distance
of their members, until the closest pair is farther apart than a threshold.

Why agglomerative and not k-means: the number of characters is unknown, and
a distance threshold is a more natural knob than k ("how different must two
faces be to be different people?"). It also takes **cannot-link
constraints** for free: two tracks visible in the same frame are two
different people, so the distance between their clusters is set to
infinity and they never merge. That is free supervision a film always has.

Lance-Williams update for average linkage, after merging a and b into m:
    d(m, k) = (n_a d(a, k) + n_b d(b, k)) / (n_a + n_b)
O(n^2) memory, O(n^2) per merge -- fine for the hundreds of tracks in a clip.
"""
from __future__ import annotations

from collections import Counter
from itertools import combinations

import numpy as np


def cosine_distances(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    return np.clip(1.0 - x @ x.T, 0.0, 2.0)


def agglomerative(dist: np.ndarray, threshold: float, linkage: str = "average",
                  cannot_link: list[tuple[int, int]] | None = None,
                  n_clusters: int | None = None) -> np.ndarray:
    """Cluster labels 0..k-1 for an (n, n) distance matrix.

    Merging stops when the closest pair of clusters is farther than
    `threshold` (or, if n_clusters is given, when that many remain).
    linkage: "average", "single" or "complete".
    """
    d = np.array(dist, dtype=np.float64)
    n = len(d)
    if n == 0:
        return np.zeros(0, np.int64)
    np.fill_diagonal(d, np.inf)
    for a, b in cannot_link or ():
        d[a, b] = d[b, a] = np.inf
    size = np.ones(n)
    alive = np.ones(n, bool)
    members = [[i] for i in range(n)]
    while alive.sum() > (n_clusters or 1):
        sub = np.where(alive[:, None] & alive[None, :], d, np.inf)
        flat = int(np.argmin(sub))
        a, b = divmod(flat, n)
        if not np.isfinite(sub[a, b]) or (n_clusters is None and sub[a, b] > threshold):
            break
        if linkage == "average":
            new = (size[a] * d[a] + size[b] * d[b]) / (size[a] + size[b])
        elif linkage == "single":
            new = np.minimum(d[a], d[b])
        elif linkage == "complete":
            new = np.maximum(d[a], d[b])
        else:
            raise ValueError(linkage)
        # A cannot-link with either side forbids the merged cluster (inf propagates
        # through max and average; for single linkage it must be forced).
        new[np.isinf(d[a]) | np.isinf(d[b])] = np.inf
        d[a], d[:, a] = new, new
        d[a, a] = np.inf
        d[b], d[:, b] = np.inf, np.inf
        alive[b] = False
        size[a] += size[b]
        members[a] += members[b]
    labels = np.empty(n, np.int64)
    for k, i in enumerate(np.flatnonzero(alive)):
        labels[members[i]] = k
    return labels


# ------------------------------------------------------------------ metrics
def purity(pred: np.ndarray, truth: list, weights: np.ndarray | None = None) -> float:
    """Fraction of items whose cluster's majority identity is their own.
    weights (e.g. track lengths in frames) make it 'fraction of screen time'.
    One cluster per item scores 1.0, so always read it next to the cluster count
    or inverse purity."""
    w = np.ones(len(pred)) if weights is None else np.asarray(weights, dtype=np.float64)
    tot = 0.0
    for c in set(pred.tolist()):
        idx = np.flatnonzero(pred == c)
        cnt = Counter()
        for i in idx:
            cnt[truth[i]] += w[i]
        tot += max(cnt.values())
    return tot / max(w.sum(), 1e-12)


def inverse_purity(pred: np.ndarray, truth: list, weights: np.ndarray | None = None) -> float:
    """Purity with the roles swapped: how much of each true identity sits in its
    largest cluster (penalises splitting one character into many clusters)."""
    t = np.asarray([hash(x) for x in truth])
    tl = {h: k for k, h in enumerate(dict.fromkeys(t.tolist()))}
    return purity(np.array([tl[h] for h in t.tolist()]), list(pred.tolist()), weights)


def pairwise_f1(pred: np.ndarray, truth: list) -> dict:
    """Over all pairs of items: precision = same-cluster pairs that are the same
    identity; recall = same-identity pairs put in the same cluster."""
    tp = fp = fn = 0
    for i, j in combinations(range(len(pred)), 2):
        same_c, same_t = pred[i] == pred[j], truth[i] == truth[j]
        tp += same_c and same_t
        fp += same_c and not same_t
        fn += same_t and not same_c
    # No pair claimed / no pair to find counts as perfect (every item its own identity).
    p = tp / (tp + fp) if tp + fp else 1.0
    r = tp / (tp + fn) if tp + fn else 1.0
    return {"precision": p, "recall": r, "f1": 2 * p * r / max(p + r, 1e-12)}


def cooccurring_pairs(frames_per_track: list) -> list[tuple[int, int]]:
    """Pairs of tracks that share at least one frame (= different people)."""
    sets = [set(f) for f in frames_per_track]
    return [(i, j) for i, j in combinations(range(len(sets)), 2) if sets[i] & sets[j]]


def auto_threshold(same: np.ndarray, diff: np.ndarray) -> float:
    """The distance that best separates known-same pairs (two halves of one
    track) from known-different pairs (co-occurring tracks): minimise
    (miss rate + false-alarm rate) over candidate thresholds. Needs no labels,
    so it can run on any clip. Within-track pairs are easier than same-person
    pairs across shots, so this errs towards too many clusters."""
    same, diff = np.asarray(same, float), np.asarray(diff, float)
    if not len(same) or not len(diff):
        return float("nan")
    cand = np.unique(np.r_[same, diff])
    err = [(same > t).mean() + (diff <= t).mean() for t in cand]
    k = int(np.argmin(err))
    hi = cand[k + 1] if k + 1 < len(cand) else cand[k]
    return float((cand[k] + hi) / 2)
