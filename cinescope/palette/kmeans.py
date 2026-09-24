"""k-means from scratch, and colour palettes / movie barcodes built on it.

k-means (Lloyd's algorithm)
  1. pick k initial centres            (k-means++: spread them out, see below)
  2. assign every point to its nearest centre
  3. move each centre to the mean of its points
  4. repeat 2-3 until assignments stop changing
Each step can only lower the within-cluster sum of squares, so it converges —
to a *local* minimum, which is why the initialisation matters and why several
restarts (`n_init`) keep the best run.

k-means++ (Arthur & Vassilvitskii, 2007): the first centre is a random point;
each next centre is drawn with probability proportional to D(x)², the squared
distance to the nearest centre chosen so far. It is O(log k)-competitive with
the optimum in expectation, and in practice removes most bad restarts.

Palettes are clustered in Lab (see imgproc/color.py): a cluster there is a set
of colours a viewer sees as "the same", which is what a palette should show.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..imgproc.color import lab_to_rgb, rgb_to_lab


def _sq_dists(x: np.ndarray, c: np.ndarray) -> np.ndarray:
    """(N, K) squared Euclidean distances, via |x|² - 2x·c + |c|² (no (N, K, D) tensor)."""
    return np.maximum((x * x).sum(1)[:, None] - 2 * x @ c.T + (c * c).sum(1)[None, :], 0.0)


def kmeans_pp_init(x: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    centres = [x[rng.integers(len(x))]]
    d2 = _sq_dists(x, np.asarray(centres))[:, 0]
    for _ in range(1, k):
        total = d2.sum()
        idx = rng.integers(len(x)) if total == 0 else rng.choice(len(x), p=d2 / total)
        centres.append(x[idx])
        d2 = np.minimum(d2, _sq_dists(x, x[idx][None, :])[:, 0])
    return np.asarray(centres, dtype=np.float64)


@dataclass
class KMeansResult:
    centres: np.ndarray      # (k, D)
    labels: np.ndarray       # (N,)
    inertia: float           # within-cluster sum of squares
    n_iter: int


def kmeans(x: np.ndarray, k: int, n_init: int = 4, max_iter: int = 100, tol: float = 1e-4,
           seed: int = 0) -> KMeansResult:
    x = np.asarray(x, dtype=np.float64)
    if len(x) < k:
        raise ValueError(f"need at least k={k} points, got {len(x)}")
    rng = np.random.default_rng(seed)
    best: KMeansResult | None = None
    for _ in range(n_init):
        c = kmeans_pp_init(x, k, rng)
        labels = np.full(len(x), -1)
        for it in range(1, max_iter + 1):
            d = _sq_dists(x, c)
            new = d.argmin(1)
            moved = c.copy()
            for j in range(k):
                members = x[new == j]
                if len(members):
                    moved[j] = members.mean(0)
                else:                       # empty cluster: re-seed at the worst-served point
                    moved[j] = x[d.min(1).argmax()]
            shift = np.abs(moved - c).max()
            c = moved
            if np.array_equal(new, labels) or shift < tol:
                labels = new
                break
            labels = new
        inertia = float(_sq_dists(x, c)[np.arange(len(x)), labels].sum())
        if best is None or inertia < best.inertia:
            best = KMeansResult(c, labels, inertia, it)
    assert best is not None
    return best


@dataclass
class Palette:
    rgb: np.ndarray          # (k, 3) uint8, most common first
    weights: np.ndarray      # (k,) fractions summing to 1
    lab: np.ndarray          # (k, 3)


def palette(frames: np.ndarray, k: int = 5, max_pixels: int = 6000, seed: int = 0,
            ignore_black_bars: bool = True, n_init: int = 2) -> Palette:
    """Dominant colours of one or more RGB frames (uint8, (..., H, W, 3))."""
    px = np.asarray(frames).reshape(-1, 3)
    if ignore_black_bars:
        # Letterbox bars and pure-black frames would otherwise win a cluster in
        # every widescreen film.
        keep = px.max(1) > 12
        px = px[keep] if keep.sum() >= k else px
    rng = np.random.default_rng(seed)
    if len(px) > max_pixels:
        px = px[rng.choice(len(px), max_pixels, replace=False)]
    lab = rgb_to_lab(px)
    res = kmeans(lab, k, seed=seed, n_init=n_init)
    counts = np.bincount(res.labels, minlength=k).astype(float)
    order = np.argsort(-counts)
    centres = res.centres[order]
    rgb = np.round(lab_to_rgb(centres) * 255).astype(np.uint8)
    return Palette(rgb, counts[order] / counts.sum(), centres)


def palette_strip(p: Palette, width: int = 400, height: int = 60) -> np.ndarray:
    """Horizontal bar, each colour's width proportional to its share."""
    out = np.zeros((height, width, 3), np.uint8)
    edges = np.round(np.concatenate([[0], np.cumsum(p.weights)]) * width).astype(int)
    for c, a, b in zip(p.rgb, edges[:-1], edges[1:], strict=True):
        out[:, a:b] = c
    return out


def barcode(frames_or_colours: np.ndarray, height: int = 200, mode: str = "mean") -> np.ndarray:
    """The classic "movie barcode": one vertical line per sampled frame.

    mode="mean"     the frame's average colour (computed in linear-light Lab, so a
                    half-red half-green frame averages to a perceptual mix)
    mode="dominant" the largest k-means cluster of the frame (k=3) — more vivid,
                    because averaging a whole frame drifts every colour to brown
    """
    arr = np.asarray(frames_or_colours)
    if arr.ndim == 2:                     # already one colour per column
        cols = arr.astype(np.uint8)
    else:
        cols = []
        for f in arr:
            if mode == "dominant":
                cols.append(palette(f, k=3, max_pixels=3000).rgb[0])
            else:
                px = f.reshape(-1, 3)
                px = px[px.max(1) > 12] if (px.max(1) > 12).sum() > 10 else px
                cols.append(np.round(lab_to_rgb(rgb_to_lab(px).mean(0)) * 255).astype(np.uint8))
        cols = np.asarray(cols, np.uint8)
    return np.repeat(cols[None, :, :], height, axis=0)
