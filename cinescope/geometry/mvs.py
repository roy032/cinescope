"""Plane-sweep multi-view stereo (Collins, 1996), in NumPy.

SfM gives depth only at a few hundred tracked corners. To get a depth for
every pixel of a reference keyframe, sweep a family of planes through the
scene — here fronto-parallel to the reference camera, z = d, for depths
d_1 … d_D spaced uniformly in *inverse* depth (equal steps in disparity, so
near depths are sampled as finely as the pixels can resolve) — and ask, for
each pixel, at which depth the other views agree with the reference.

A world plane nᵀX = d induces a homography between two views. With
X_src = R X_ref + t:   H = K_src (R + t nᵀ / d) K_ref⁻¹.
For each plane, warp every source image into the reference with its H,
compare a small window around each pixel (zero-mean normalised
cross-correlation — insensitive to the exposure changes common between film
frames), and keep the depth with the best score (winner-takes-all), refined to
sub-plane precision with a parabola through the best score and its two
neighbours.

Occlusion: a pixel visible in the reference may be hidden in one of the
sources, so the per-pixel cost is the mean of the best half of the sources'
costs, not of all of them.

Confidence: textureless regions (sky, a blank wall) match equally well at every
depth. Pixels are kept only where the reference window has texture and the cost
curve has a clear minimum.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..motion.lk import bilinear


def box_mean(img: np.ndarray, r: int) -> np.ndarray:
    """Mean over a (2r+1)² window via an integral image (O(1) per pixel; edges
    use the part of the window inside the image)."""
    h, w = img.shape[-2:]
    pad = np.zeros(img.shape[:-2] + (h + 1, w + 1))
    pad[..., 1:, 1:] = img.cumsum(-2).cumsum(-1)
    ys, xs = np.arange(h), np.arange(w)
    y0, y1 = np.clip(ys - r, 0, h), np.clip(ys + r + 1, 0, h)
    x0, x1 = np.clip(xs - r, 0, w), np.clip(xs + r + 1, 0, w)
    s = (pad[..., y1[:, None], x1[None, :]] - pad[..., y0[:, None], x1[None, :]]
         - pad[..., y1[:, None], x0[None, :]] + pad[..., y0[:, None], x0[None, :]])
    area = (y1 - y0)[:, None] * (x1 - x0)[None, :]
    return s / area


def zncc(a: np.ndarray, b: np.ndarray, r: int, eps: float = 1e-3) -> np.ndarray:
    """Windowed zero-mean normalised cross-correlation in [-1, 1]."""
    ma, mb = box_mean(a, r), box_mean(b, r)
    va = box_mean(a * a, r) - ma ** 2
    vb = box_mean(b * b, r) - mb ** 2
    cov = box_mean(a * b, r) - ma * mb
    return cov / np.sqrt(np.maximum(va, eps) * np.maximum(vb, eps))


def plane_homography(k_ref, k_src, rot_rel, t_rel, depth: float) -> np.ndarray:
    n = np.array([0, 0, 1.0])
    return k_src @ (rot_rel + np.outer(t_rel, n) / depth) @ np.linalg.inv(k_ref)


@dataclass
class DepthResult:
    depth: np.ndarray           # (H, W), nan where not confident
    raw_depth: np.ndarray       # (H, W) winner-takes-all everywhere
    cost: np.ndarray            # (H, W) best cost (1 - zncc)
    confident: np.ndarray       # bool mask
    depths: np.ndarray          # the swept planes


def plane_sweep(ref: np.ndarray, srcs: list[np.ndarray], k: np.ndarray, pose_ref, poses_src,
                near: float, far: float, n_planes: int = 64, window: int = 3,
                min_texture: float = 4.0, max_cost: float = 0.5, best_frac: float = 0.5) -> DepthResult:
    """ref, srcs: gray float images (same size); poses are (R, t) world->camera."""
    ref = np.asarray(ref, np.float64)
    h, w = ref.shape
    r_ref, t_ref = pose_ref
    inv = np.linspace(1.0 / far, 1.0 / near, n_planes)
    depths = 1.0 / inv
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
    pix = np.stack([xs.ravel(), ys.ravel(), np.ones(h * w)])
    costs = np.empty((n_planes, h, w))
    rels = []
    for r_s, t_s in poses_src:
        rot_rel = r_s @ r_ref.T                 # X_src = R_s R_refᵀ (X_ref - t_ref) + t_s
        rels.append((rot_rel, t_s - rot_rel @ t_ref))
    keep = max(1, int(round(best_frac * len(srcs))))
    for i, d in enumerate(depths):
        per_src = []
        for src, (rot_rel, t_rel) in zip(srcs, rels, strict=True):
            hm = plane_homography(k, k, rot_rel, t_rel, d)
            q = hm @ pix
            u, v = q[0] / q[2], q[1] / q[2]
            warped = bilinear(np.asarray(src, np.float64), u.reshape(h, w), v.reshape(h, w))
            outside = (u < 0) | (u > w - 1) | (v < 0) | (v > h - 1) | (q[2] <= 0)
            c = 1.0 - zncc(ref, warped, window)
            c[outside.reshape(h, w)] = 2.0
            per_src.append(c)
        per_src = np.sort(np.stack(per_src), 0)
        costs[i] = per_src[:keep].mean(0)
    best = np.argmin(costs, 0)
    c0 = np.take_along_axis(costs, best[None], 0)[0]
    # sub-plane refinement in inverse depth (parabola through three costs)
    lo, hi = np.clip(best - 1, 0, n_planes - 1), np.clip(best + 1, 0, n_planes - 1)
    cl = np.take_along_axis(costs, lo[None], 0)[0]
    ch = np.take_along_axis(costs, hi[None], 0)[0]
    den = cl - 2 * c0 + ch
    off = np.where((den > 1e-9) & (best > 0) & (best < n_planes - 1), 0.5 * (cl - ch) / np.maximum(den, 1e-9), 0.0)
    step = inv[1] - inv[0] if n_planes > 1 else 0.0
    inv_best = inv[best] + np.clip(off, -0.5, 0.5) * step
    raw = 1.0 / inv_best
    texture = np.sqrt(np.maximum(box_mean(ref * ref, window) - box_mean(ref, window) ** 2, 0))
    confident = (texture > min_texture) & (c0 < max_cost) & (best > 0) & (best < n_planes - 1)
    return DepthResult(np.where(confident, raw, np.nan), raw, c0, confident, depths)
