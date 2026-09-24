"""Pyramidal Lucas-Kanade optical flow (Lucas & Kanade 1981; Bouguet 2000), in NumPy.

Brightness constancy says a point keeps its intensity as it moves:
I(x + u, y + v, t + 1) = I(x, y, t). Linearising gives one equation per pixel,

    Ix·u + Iy·v = -It

— two unknowns, one equation (the aperture problem). Lucas-Kanade assumes all
pixels in a small window move together and solves the least-squares system

    [ΣIx²  ΣIxIy] [u]     [ΣIx·It]
    [ΣIxIy ΣIy² ] [v] = - [ΣIy·It]          (the matrix is Harris's M)

It is only valid for motion of about a pixel, so it runs coarse-to-fine on a
Gaussian pyramid: a 16-pixel pan is a 2-pixel pan three octaves down. At each
level the estimate is refined iteratively (Newton-Raphson on the warped
window), then doubled and handed to the next finer level.

Points are dropped (status False) when the window's M is near-singular (flat
or edge-like: min eigenvalue below `min_eig`) or the point leaves the image.
"""
from __future__ import annotations

import numpy as np

from ..imgproc.filters import correlate2d, gaussian_blur


def pyramid(img: np.ndarray, levels: int) -> list[np.ndarray]:
    """Gaussian pyramid, finest first. Blur (σ=1) before 2x decimation to avoid aliasing."""
    out = [np.asarray(img, dtype=np.float64)]
    for _ in range(levels - 1):
        prev = out[-1]
        if min(prev.shape) < 16:
            break
        out.append(gaussian_blur(prev, 1.0)[::2, ::2])
    return out


def bilinear(img: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Sample img at float coordinates (any shape), clamping at the border."""
    h, w = img.shape
    x = np.clip(x, 0, w - 1.001)
    y = np.clip(y, 0, h - 1.001)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    ax, ay = x - x0, y - y0
    return ((1 - ax) * (1 - ay) * img[y0, x0] + ax * (1 - ay) * img[y0, x0 + 1]
            + (1 - ax) * ay * img[y0 + 1, x0] + ax * ay * img[y0 + 1, x0 + 1])


_DX = np.array([[-0.5, 0.0, 0.5]])            # central difference


def track(prev: np.ndarray, nxt: np.ndarray, pts: np.ndarray, win: int = 7, levels: int = 3,
          iters: int = 20, eps: float = 0.01, min_eig: float = 1e-3) -> tuple[np.ndarray, np.ndarray]:
    """Track (N, 2) (x, y) points from `prev` to `nxt`. Returns (new_pts, status).

    win: half-size of the square window (15x15 for win=7, OpenCV's default).
    """
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    n = len(pts)
    if n == 0:
        return pts.copy(), np.zeros(0, bool)
    pp, pn = pyramid(prev, levels), pyramid(nxt, levels)
    levels = len(pp)
    offs = np.arange(-win, win + 1, dtype=np.float64)
    wx, wy = np.meshgrid(offs, offs)                       # (W, W) window offsets
    guess = np.zeros((n, 2))
    status = np.ones(n, bool)

    for lvl in range(levels - 1, -1, -1):
        scale = 2.0 ** lvl
        a, b = pp[lvl], pn[lvl]
        ix = correlate2d(a, _DX, border="edge")
        iy = correlate2d(a, _DX.T, border="edge")
        p = pts / scale                                     # (N, 2) at this level
        xs = p[:, 0, None, None] + wx                       # (N, W, W)
        ys = p[:, 1, None, None] + wy
        tpl = bilinear(a, xs, ys)
        gx, gy = bilinear(ix, xs, ys), bilinear(iy, xs, ys)
        gxx, gxy, gyy = (gx * gx).sum((1, 2)), (gx * gy).sum((1, 2)), (gy * gy).sum((1, 2))
        det = gxx * gyy - gxy ** 2
        tr = gxx + gyy
        lam_min = (tr - np.sqrt(np.maximum(tr ** 2 - 4 * det, 0))) / 2 / (2 * win + 1) ** 2
        ok = (lam_min > min_eig) & (det > 1e-12)
        if lvl == 0:
            status &= ok
        inv = np.zeros((n, 2, 2))
        safe = np.where(ok, det, 1.0)
        inv[:, 0, 0], inv[:, 0, 1] = gyy / safe, -gxy / safe
        inv[:, 1, 0], inv[:, 1, 1] = -gxy / safe, gxx / safe
        v = guess.copy()
        for _ in range(iters):
            warped = bilinear(b, xs + v[:, 0, None, None], ys + v[:, 1, None, None])
            it = warped - tpl                               # temporal difference
            bvec = np.stack([(it * gx).sum((1, 2)), (it * gy).sum((1, 2))], axis=1)
            step = -np.einsum("nij,nj->ni", inv, bvec)
            step[~ok] = 0
            v += step
            if np.all(np.abs(step[ok]) < eps) if ok.any() else True:
                break
        guess = v * 2.0 if lvl > 0 else v
    new = pts + guess
    h, w = prev.shape
    inside = (new[:, 0] >= 0) & (new[:, 0] <= w - 1) & (new[:, 1] >= 0) & (new[:, 1] <= h - 1)
    return new, status & inside & np.isfinite(new).all(1)
