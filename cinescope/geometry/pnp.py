"""Camera resection (Perspective-n-Point): pose of a calibrated camera from
known 3D points and their 2D projections. This is how incremental SfM adds
each new view to the reconstruction.

1. Linear DLT on normalised coordinates: x × ([R|t] X) = 0 is linear in the 12
   entries of [R|t]; six points determine it. The 3x3 part is then snapped to
   the nearest rotation, and the scale fixed by its singular values.
2. RANSAC over 6-point samples, inliers by reprojection error in pixels.
3. Gauss-Newton refinement of (axis-angle, t) on the inliers, minimising the
   reprojection error — the linear solution minimises an algebraic error only.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .camera import log_rotation, nearest_rotation, rodrigues
from .lm import levenberg_marquardt
from .twoview import ransac_iterations


def pnp_dlt(x3: np.ndarray, xn: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(R, t) from >= 6 points; xn are normalised image coordinates."""
    n = len(x3)
    if n < 6:
        raise ValueError("need at least 6 points")
    c = x3.mean(0)
    s = np.sqrt(3) / (np.sqrt(((x3 - c) ** 2).sum(1)).mean() + 1e-12)
    xs = (x3 - c) * s                                        # condition the 3D points too
    xh = np.c_[xs, np.ones(n)]
    z = np.zeros((n, 4))
    a = np.concatenate([np.concatenate([xh, z, -xn[:, 0:1] * xh], 1),
                        np.concatenate([z, xh, -xn[:, 1:2] * xh], 1)])
    _, _, vt = np.linalg.svd(a)
    pm = vt[-1].reshape(3, 4)
    u, sv, wt = np.linalg.svd(pm[:, :3])
    scale = sv.mean()
    rot = u @ wt
    sign = 1.0
    if np.linalg.det(rot) < 0:
        rot, sign = -rot, -1.0
    t = sign * pm[:, 3] / scale
    rot = nearest_rotation(rot)
    # undo the 3D normalisation: X_s = s (X - c)  =>  R X_s + t = s R X + (t - s R c), divide by s
    t = t / s - rot @ c
    return rot, t


def reprojection_errors(rot, t, x3, xn, focal: float) -> np.ndarray:
    xc = x3 @ rot.T + t
    z = xc[:, 2]
    proj = xc[:, :2] / np.where(np.abs(z) < 1e-9, 1e-9, z)[:, None]
    err = np.linalg.norm(proj - xn, axis=1) * focal
    return np.where(z > 0, err, np.inf)


def refine_pose(rot, t, x3, xn, iters: int = 20) -> tuple[np.ndarray, np.ndarray]:
    def res(p):
        r = rodrigues(p[:3])
        xc = x3 @ r.T + p[3:]
        return (xc[:, :2] / xc[:, 2:3] - xn).ravel()

    out = levenberg_marquardt(res, np.r_[log_rotation(rot), t], max_iter=iters)
    return rodrigues(out.x[:3]), out.x[3:]


@dataclass
class PnPResult:
    rot: np.ndarray | None
    t: np.ndarray | None
    inliers: np.ndarray


def ransac_pnp(x3: np.ndarray, xn: np.ndarray, focal: float, thresh_px: float = 2.0,
               p: float = 0.999, max_iter: int = 1000, seed: int = 0) -> PnPResult:
    n = len(x3)
    if n < 6:
        return PnPResult(None, None, np.zeros(n, bool))
    rng = np.random.default_rng(seed)
    best = np.zeros(n, bool)
    needed, it = max_iter, 0
    while it < min(needed, max_iter):
        it += 1
        idx = rng.choice(n, 6, replace=False)
        try:
            rot, t = pnp_dlt(x3[idx], xn[idx])
        except np.linalg.LinAlgError:
            continue
        inl = reprojection_errors(rot, t, x3, xn, focal) < thresh_px
        if inl.sum() > best.sum():
            best = inl
            w = inl.mean()
            needed = ransac_iterations(w, 6, p, it, max_iter)
    if best.sum() < 6:
        return PnPResult(None, None, best)
    rot, t = pnp_dlt(x3[best], xn[best])
    rot, t = refine_pose(rot, t, x3[best], xn[best])
    inl = reprojection_errors(rot, t, x3, xn, focal) < thresh_px
    if inl.sum() >= 6:
        rot, t = refine_pose(rot, t, x3[inl], xn[inl])
        inl = reprojection_errors(rot, t, x3, xn, focal) < thresh_px
    return PnPResult(rot, t, inl)
