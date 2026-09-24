"""Homography estimation: normalised DLT + RANSAC, in NumPy.

A homography H (3x3, 8 degrees of freedom) maps points of one image plane to
another: [x', y', 1]ᵀ ~ H [x, y, 1]ᵀ. Between two frames of a shot it describes
the camera's motion exactly when the camera only rotates/zooms (pan, tilt, zoom
— the moves Phase 0 classifies), or when the scene is far away / planar.

DLT (direct linear transform)
  Each correspondence gives two linear equations in the 9 entries of H; stack
  them into A (2N x 9) and take the right singular vector with the smallest
  singular value (the least-squares solution of A h = 0 with |h| = 1).
  Hartley normalisation first — translate points to zero mean and scale them to
  mean distance √2 — because raw pixel coordinates (x ~ 500, x² ~ 250000) make A
  terribly conditioned. Hartley (1997) is titled "In defense of the eight-point
  algorithm" for exactly this reason.

RANSAC (Fischler & Bolles, 1981)
  Optical-flow tracks include outliers: points on moving actors, bad tracks. Fit
  H to a random minimal sample (4 points), count inliers (reprojection error <
  threshold), keep the best, refit on its inliers. RANSAC assumes (1) the model
  explains a large enough fraction w of the data and (2) a minimal sample of
  all-inliers gives a good model. To see one all-inlier sample with probability
  p you need N = log(1 - p) / log(1 - w^s) iterations (s = 4 here), recomputed as
  the best inlier ratio improves. At p = 0.99: w = 0.5 needs 72 iterations,
  w = 0.2 needs 2876 (the default here is p = 0.995: 83 and 3309).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _normalise(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    c = pts.mean(0)
    d = np.sqrt(((pts - c) ** 2).sum(1)).mean()
    s = np.sqrt(2) / d if d > 0 else 1.0
    t = np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1]])
    return (pts - c) * s, t


def dlt(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Least-squares homography from >= 4 correspondences (normalised DLT)."""
    src, dst = np.asarray(src, float), np.asarray(dst, float)
    if len(src) < 4:
        raise ValueError("need at least 4 correspondences")
    ns, ts = _normalise(src)
    nd, td = _normalise(dst)
    x, y = ns[:, 0], ns[:, 1]
    u, v = nd[:, 0], nd[:, 1]
    z, o = np.zeros_like(x), np.ones_like(x)
    a = np.concatenate([
        np.stack([-x, -y, -o, z, z, z, u * x, u * y, u], 1),
        np.stack([z, z, z, -x, -y, -o, v * x, v * y, v], 1),
    ])
    _, _, vt = np.linalg.svd(a)
    hn = vt[-1].reshape(3, 3)
    h = np.linalg.inv(td) @ hn @ ts                       # undo the normalisation
    return h / h[2, 2] if abs(h[2, 2]) > 1e-12 else h


def apply(h: np.ndarray, pts: np.ndarray) -> np.ndarray:
    p = np.c_[pts, np.ones(len(pts))] @ h.T
    return p[:, :2] / p[:, 2:3]


def reprojection_error(h: np.ndarray, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        e = np.sqrt(((apply(h, src) - dst) ** 2).sum(1))
    return np.where(np.isfinite(e), e, np.inf)


@dataclass
class RansacResult:
    h: np.ndarray | None
    inliers: np.ndarray            # bool mask
    iterations: int

    @property
    def inlier_ratio(self) -> float:
        return float(self.inliers.mean()) if len(self.inliers) else 0.0


def ransac_homography(src: np.ndarray, dst: np.ndarray, thresh: float = 2.0, p: float = 0.995,
                      max_iter: int = 2000, seed: int = 0) -> RansacResult:
    src, dst = np.asarray(src, float), np.asarray(dst, float)
    n = len(src)
    if n < 4:
        return RansacResult(None, np.zeros(n, bool), 0)
    rng = np.random.default_rng(seed)
    best = np.zeros(n, bool)
    needed, it = max_iter, 0
    while it < min(needed, max_iter):
        it += 1
        idx = rng.choice(n, 4, replace=False)
        try:
            h = dlt(src[idx], dst[idx])
        except np.linalg.LinAlgError:
            continue
        inl = reprojection_error(h, src, dst) < thresh
        if inl.sum() > best.sum():
            best = inl
            w = inl.mean()
            if w >= 1:
                needed = it
            elif w > 0:
                needed = int(np.ceil(np.log(1 - p) / np.log(1 - w ** 4)))
    if best.sum() < 4:
        return RansacResult(None, best, it)
    h = dlt(src[best], dst[best])                      # refit on all inliers
    inliers = reprojection_error(h, src, dst) < thresh
    if inliers.sum() >= 4:
        h = dlt(src[inliers], dst[inliers])
    return RansacResult(h, inliers, it)
