"""Harris corner detector (Harris & Stephens, 1988), in NumPy.

Shift a small window by (u, v). The change in intensity is, to first order,

    E(u, v) ≈ [u v] M [u v]ᵀ,    M = Σ_window w(x, y) [[Ix², IxIy], [IxIy, Iy²]]

M's eigenvalues λ1, λ2 say how the patch changes:
  both small  -> flat region (moving the window changes nothing)
  one large   -> edge (changes across the edge, not along it: the aperture problem)
  both large  -> corner (changes in every direction) — a point you can track

Harris avoids the eigen-decomposition with R = det(M) - k·trace(M)², which is
large and positive only when both eigenvalues are large. (Shi-Tomasi's
min(λ1, λ2) is the other common score; `score="shi-tomasi"` computes it.)

The same matrix M is the one Lucas-Kanade inverts, so "good corners" are
exactly the points where optical flow is well-conditioned.
"""
from __future__ import annotations

import numpy as np

from ..imgproc.filters import gaussian_blur, sobel


def structure_tensor(gray: np.ndarray, sigma: float = 1.5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gx, gy = sobel(gray)
    return (gaussian_blur(gx * gx, sigma), gaussian_blur(gx * gy, sigma),
            gaussian_blur(gy * gy, sigma))


def corner_response(gray: np.ndarray, sigma: float = 1.5, k: float = 0.04,
                    score: str = "harris") -> np.ndarray:
    sxx, sxy, syy = structure_tensor(gray, sigma)
    if score == "harris":
        return sxx * syy - sxy ** 2 - k * (sxx + syy) ** 2
    if score == "shi-tomasi":
        half_trace = (sxx + syy) / 2
        return half_trace - np.sqrt(((sxx - syy) / 2) ** 2 + sxy ** 2)   # smaller eigenvalue
    raise ValueError(score)


def _local_max(r: np.ndarray, radius: int) -> np.ndarray:
    """True where r equals the max of its (2·radius+1)² neighbourhood."""
    pad = np.pad(r, radius, mode="constant", constant_values=-np.inf)
    h, w = r.shape
    m = np.full_like(r, -np.inf)
    for dy in range(2 * radius + 1):
        for dx in range(2 * radius + 1):
            m = np.maximum(m, pad[dy:dy + h, dx:dx + w])
    return r >= m


def detect_corners(gray: np.ndarray, max_corners: int = 200, quality: float = 0.01,
                   min_distance: int = 5, sigma: float = 1.5, border: int = 4,
                   score: str = "harris") -> np.ndarray:
    """(N, 2) array of (x, y) corner positions, strongest first.

    quality: keep responses above quality * max response (OpenCV's qualityLevel).
    min_distance: non-maximum suppression radius, so corners spread over the frame
                  instead of clustering on one textured patch.
    """
    gray = np.asarray(gray, dtype=np.float64)
    r = corner_response(gray, sigma=sigma, score=score)
    if border:
        r[:border] = r[-border:] = -np.inf
        r[:, :border] = r[:, -border:] = -np.inf
    rmax = r[np.isfinite(r)].max(initial=0.0)
    if rmax <= 0:
        return np.zeros((0, 2))
    mask = (r > quality * rmax) & _local_max(r, min_distance)
    ys, xs = np.nonzero(mask)
    order = np.argsort(-r[ys, xs])[:max_corners]
    return np.stack([xs[order], ys[order]], axis=1).astype(np.float64)
