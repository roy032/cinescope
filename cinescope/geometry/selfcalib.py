"""Self-calibration fallbacks for film footage, which comes with no chessboard.

Two estimates of the focal length (in pixels), principal point assumed at the
image centre and pixels assumed square — true for nearly all digital video:

1. A prior from the image size. f = 1.2 · max(w, h) is COLMAP's default guess
   and corresponds to a horizontal field of view of about 45° — a "normal"
   lens. Typical film lenses (25-85 mm on Super 35) span roughly 60° to 17°,
   so the prior is off by up to ~2x; bundle adjustment can refine it when the
   camera translates.

2. Vanishing points. Parallel 3D lines meet in the image at a vanishing point v,
   and the ray K⁻¹v points along their 3D direction. Two orthogonal directions
   (a floor edge and a wall edge) give (K⁻¹v1)·(K⁻¹v2) = 0, which with
   K = [[f, 0, cx], [0, f, cy], [0, 0, 1]] becomes

       f² = -(v1 - p)·(v2 - p),       p = (cx, cy).

   VPs are found by RANSAC over line segments: two segments propose an
   intersection, segments pointing at it vote. Man-made sets (corridors,
   rooms, buildings) work; nature shots and close-ups have no straight lines,
   which the result reports as None rather than guessing.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def focal_prior(width: int, height: int, factor: float = 1.2) -> float:
    return factor * max(width, height)


def _lines(segs: np.ndarray) -> np.ndarray:
    """(N, 4) segments x1 y1 x2 y2 -> (N, 3) homogeneous lines (normalised)."""
    p1 = np.c_[segs[:, :2], np.ones(len(segs))]
    p2 = np.c_[segs[:, 2:], np.ones(len(segs))]
    ln = np.cross(p1, p2)
    return ln / np.linalg.norm(ln[:, :2], axis=1, keepdims=True)


def _angle_to_vp(segs: np.ndarray, vp: np.ndarray) -> np.ndarray:
    """Angle (deg) between each segment and the direction from its midpoint to vp.
    vp is homogeneous; a VP at infinity (vp[2] = 0) is a direction."""
    mid = (segs[:, :2] + segs[:, 2:]) / 2
    d = segs[:, 2:] - segs[:, :2]
    to = vp[:2] / vp[2] - mid if abs(vp[2]) > 1e-09 else np.broadcast_to(vp[:2], mid.shape)
    cos = np.abs((d * to).sum(1)) / (np.linalg.norm(d, axis=1) * np.linalg.norm(to, axis=1) + 1e-12)
    return np.degrees(np.arccos(np.clip(cos, 0, 1)))


def _refine_vp(segs: np.ndarray) -> np.ndarray:
    """Least-squares intersection of lines: the smallest singular vector of the
    stacked line equations, each weighted by segment length."""
    ln = _lines(segs) * np.linalg.norm(segs[:, 2:] - segs[:, :2], axis=1, keepdims=True)
    _, _, vt = np.linalg.svd(ln)
    return vt[-1]


def ransac_vanishing_points(segs: np.ndarray, n_vps: int = 3, thresh_deg: float = 2.0,
                            iters: int = 500, min_support: int = 6, seed: int = 0) -> list[tuple[np.ndarray, int]]:
    """Greedy sequential RANSAC: best VP, remove its segments, repeat.
    Returns [(homogeneous vp, n_supporting_segments)]."""
    rng = np.random.default_rng(seed)
    segs = np.asarray(segs, float)
    segs = segs[np.linalg.norm(segs[:, 2:] - segs[:, :2], axis=1) > 1e-6]
    weight = np.linalg.norm(segs[:, 2:] - segs[:, :2], axis=1)
    out = []
    remaining = np.ones(len(segs), bool)
    for _ in range(n_vps):
        idx = np.flatnonzero(remaining)
        if len(idx) < max(2, min_support):
            break
        ln = _lines(segs[idx])
        best, best_score = None, 0.0
        for _ in range(iters):
            a, b = rng.choice(len(idx), 2, replace=False)
            vp = np.cross(ln[a], ln[b])
            if np.linalg.norm(vp) < 1e-12:
                continue
            inl = _angle_to_vp(segs[idx], vp) < thresh_deg
            score = weight[idx][inl].sum()
            if score > best_score:
                best, best_score = inl, score
        if best is None or best.sum() < min_support:
            break
        members = idx[best]
        vp = _refine_vp(segs[members])
        members = idx[_angle_to_vp(segs[idx], vp) < thresh_deg]
        out.append((vp / (np.linalg.norm(vp) + 1e-12), int(len(members))))
        remaining[members] = False
    return out


def focal_from_vps(v1: np.ndarray, v2: np.ndarray, centre: tuple[float, float]) -> float | None:
    """f from two orthogonal finite vanishing points (homogeneous), or None."""
    if abs(v1[2]) < 1e-9 or abs(v2[2]) < 1e-9:
        return None
    p = np.asarray(centre, float)
    f2 = -((v1[:2] / v1[2] - p) @ (v2[:2] / v2[2] - p))
    return float(np.sqrt(f2)) if f2 > 0 else None


@dataclass
class FocalEstimate:
    f: float
    method: str                 # "vanishing-points" or "prior"
    prior: float
    vps: list                   # [(x, y) or None per VP, support]

    def to_dict(self) -> dict:
        return {"f": round(self.f, 2), "method": self.method, "prior": round(self.prior, 2),
                "vps": [[None if v is None else [round(c, 1) for c in v], s] for v, s in self.vps]}


def estimate_focal(segs: np.ndarray, width: int, height: int, f_range: tuple[float, float] = (0.3, 5.0),
                   seed: int = 0) -> FocalEstimate:
    """Focal length from line segments, falling back to the prior.

    Accept a VP pair only if f lands within f_range x max(w, h) (roughly 11° to
    118° horizontal field of view); among acceptable pairs take the one with the
    most supporting segments."""
    prior = focal_prior(width, height)
    vps = ransac_vanishing_points(segs, seed=seed) if len(segs) >= 4 else []
    listed = [(None if abs(v[2]) < 1e-9 else (v[:2] / v[2]).tolist(), s) for v, s in vps]
    best = None
    for i in range(len(vps)):
        for j in range(i + 1, len(vps)):
            f = focal_from_vps(vps[i][0], vps[j][0], (width / 2, height / 2))
            if f is None or not (f_range[0] * max(width, height) <= f <= f_range[1] * max(width, height)):
                continue
            support = min(vps[i][1], vps[j][1])
            if best is None or support > best[1]:
                best = (f, support)
    if best is None:
        return FocalEstimate(prior, "prior", prior, listed)
    return FocalEstimate(best[0], "vanishing-points", prior, listed)


def detect_segments(gray: np.ndarray, min_length: float = 0.05) -> np.ndarray:
    """Straight segments with OpenCV (Canny + probabilistic Hough). Line detection is
    the library part here; the vanishing points and the focal length are ours."""
    import cv2

    g = np.asarray(gray, np.float32)
    g = (255 * (g - g.min()) / (np.ptp(g) + 1e-6)).astype(np.uint8)
    edges = cv2.Canny(g, 60, 160)
    min_len = int(min_length * max(g.shape))
    segs = cv2.HoughLinesP(edges, 1, np.pi / 360, threshold=40, minLineLength=min_len, maxLineGap=4)
    return np.zeros((0, 4)) if segs is None else segs.reshape(-1, 4).astype(float)
