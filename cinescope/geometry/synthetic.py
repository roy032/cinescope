"""A synthetic film set with exact ground truth: a textured room, ray-cast in NumPy.

Real footage never says where the camera really was, so every algorithm in
this phase is first scored here, where it does: camera poses, focal length
and per-pixel depth are known exactly.

The set is a box room (floor, ceiling, three walls; y points down, z forward)
with two textured boxes standing on the floor — something in the foreground
to create parallax, the thing a homography cannot explain and SfM needs.
Each surface carries its own random texture (blurred noise plus a coarse
grid), sampled bilinearly in the surface's own 2D coordinates, and a little
shading by surface orientation so the boxes' faces differ.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .camera import intrinsics, look_at


def _texture(seed: int, size: int = 256) -> np.ndarray:
    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(seed)
    base = sum(gaussian_filter(rng.random((size, size)), s, mode="wrap") * w
               for s, w in ((1.0, 0.6), (3.0, 1.0), (8.0, 1.5)))
    base = (base - base.min()) / np.ptp(base)
    yy, xx = np.mgrid[0:size, 0:size]
    grid = ((xx // 32 + yy // 32) % 2) * 0.25
    tint = rng.uniform(0.4, 1.0, 3)
    return np.clip((0.25 + 0.6 * base + grid)[..., None] * tint, 0, 1)


def _sample(tex: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Bilinear, wrapping; (u, v) in texture pixels."""
    s = tex.shape[0]
    u, v = np.mod(u, s), np.mod(v, s)
    u0, v0 = np.floor(u).astype(int), np.floor(v).astype(int)
    a, b = (u - u0)[..., None], (v - v0)[..., None]
    u1, v1 = (u0 + 1) % s, (v0 + 1) % s
    return ((1 - a) * (1 - b) * tex[v0, u0] + a * (1 - b) * tex[v0, u1]
            + (1 - a) * b * tex[v1, u0] + a * b * tex[v1, u1])


@dataclass
class Room:
    half_width: float = 3.0       # walls at x = ±half_width
    floor: float = 1.5            # floor at y = +floor (y down)
    ceiling: float = -2.0
    back: float = 9.0             # back wall at z = back
    front: float = -4.0           # wall behind the cameras
    boxes: tuple = (((-1.2, 0.3, 4.0), (0.6, 1.2, 0.6)), ((1.0, 0.7, 6.0), (0.8, 0.8, 0.8)))   # (centre, half size)
    texels_per_unit: float = 48.0
    seed: int = 0

    def __post_init__(self):
        self.tex = [_texture(self.seed * 100 + i) for i in range(6 + 3 * len(self.boxes))]


def render(room: Room, k: np.ndarray, rot: np.ndarray, t: np.ndarray, width: int, height: int
           ) -> tuple[np.ndarray, np.ndarray]:
    """(rgb uint8 (H, W, 3), z-depth (H, W)) of the room seen by camera (K, R, t)."""
    vs, us = np.mgrid[0:height, 0:width].astype(float) + 0.5
    rays_c = np.stack([(us - k[0, 2]) / k[0, 0], (vs - k[1, 2]) / k[1, 1], np.ones_like(us)], -1)
    d = rays_c @ rot                                  # world directions (Rᵀ r, row-vector form)
    o = -rot.T @ t
    best = np.full(us.shape, np.inf)
    col = np.zeros(us.shape + (3,))
    s = room.texels_per_unit

    def hit_plane(axis, value, tex_id, uv_axes, shade):
        nonlocal best
        with np.errstate(divide="ignore", invalid="ignore"):
            lam = (value - o[axis]) / d[..., axis]
        ok = (lam > 1e-6) & (lam < best)
        if not ok.any():
            return
        p = o + lam[..., None] * d
        c = _sample(room.tex[tex_id], p[..., uv_axes[0]] * s, p[..., uv_axes[1]] * s) * shade
        best = np.where(ok, lam, best)
        col[ok] = c[ok]

    hit_plane(1, room.floor, 0, (0, 2), 0.9)
    hit_plane(1, room.ceiling, 1, (0, 2), 0.7)
    hit_plane(0, -room.half_width, 2, (2, 1), 0.8)
    hit_plane(0, room.half_width, 3, (2, 1), 0.85)
    hit_plane(2, room.back, 4, (0, 1), 1.0)
    hit_plane(2, room.front, 5, (0, 1), 1.0)
    for b, (centre, half) in enumerate(room.boxes):
        c, hsz = np.asarray(centre, float), np.asarray(half, float)
        with np.errstate(divide="ignore", invalid="ignore"):
            t1 = (c - hsz - o) / d
            t2 = (c + hsz - o) / d
        tmin = np.nanmax(np.minimum(t1, t2), -1)
        tmax = np.nanmin(np.maximum(t1, t2), -1)
        ok = (tmax >= tmin) & (tmin > 1e-6) & (tmin < best)
        if not ok.any():
            continue
        p = o + tmin[..., None] * d
        axis = np.argmax(np.abs((p - c) / hsz), -1)            # which face was hit
        for ax in range(3):
            m = ok & (axis == ax)
            if not m.any():
                continue
            uv = [a for a in range(3) if a != ax]
            cc = _sample(room.tex[6 + 3 * b + ax], p[..., uv[0]] * s, p[..., uv[1]] * s) * (0.6 + 0.15 * ax)
            col[m] = cc[m]
        best = np.where(ok, tmin, best)
    z = best * rays_c[..., 2]                          # ray parameter along a z=1 ray is the z-depth
    return (np.clip(col, 0, 1) * 255).astype(np.uint8), z


@dataclass
class SyntheticShot:
    frames: np.ndarray            # (N, H, W, 3) uint8
    depth: np.ndarray             # (N, H, W)
    k: np.ndarray
    rots: np.ndarray              # (N, 3, 3) world -> camera
    ts: np.ndarray                # (N, 3)


def dolly_shot(n: int = 24, width: int = 160, height: int = 120, focal: float | None = None,
               move: str = "dolly", seed: int = 0, room: Room | None = None) -> SyntheticShot:
    """A camera move through the room. move: "dolly" (sideways track while
    looking at the set), "push" (forward), "pan" (rotation only — SfM must fail
    to triangulate this one: there is no baseline), "orbit"."""
    room = room or Room(seed=seed)
    f = focal or 0.9 * width
    k = intrinsics(f, width / 2, height / 2)
    frames, depth, rots, ts = [], [], [], []
    for i in range(n):
        a = i / max(n - 1, 1)
        if move == "dolly":
            eye, target = np.array([-1.0 + 2.0 * a, -0.2, -0.5]), np.array([0.2 * a, 0.2, 6.0])
        elif move == "push":
            eye, target = np.array([0.1, -0.2, -1.0 + 2.5 * a]), np.array([0.0, 0.1, 7.0])
        elif move == "pan":
            eye, target = np.array([0.0, -0.2, -0.5]), np.array([-2.0 + 4.0 * a, 0.2, 6.0])
        elif move == "orbit":
            ang = -0.35 + 0.7 * a
            eye = np.array([4.0 * np.sin(ang), -0.3, 5.0 - 5.5 * np.cos(ang)])
            target = np.array([0.0, 0.3, 5.0])
        else:
            raise ValueError(move)
        rot, t = look_at(eye, target)
        img, z = render(room, k, rot, t, width, height)
        frames.append(img)
        depth.append(z)
        rots.append(rot)
        ts.append(t)
    return SyntheticShot(np.stack(frames), np.stack(depth), k, np.stack(rots), np.stack(ts))
