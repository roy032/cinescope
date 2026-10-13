"""Two-view geometry: fundamental / essential matrix, relative pose, triangulation.

Epipolar constraint. A point seen at x1 in view 1 and x2 in view 2 (homogeneous
pixels) satisfies x2ᵀ F x1 = 0; in normalised coordinates (K⁻¹x) the same
constraint uses the essential matrix E = [t]ₓ R = K2ᵀ F K1.

Normalised 8-point algorithm (Longuet-Higgins 1981, Hartley 1997). Each match
gives one linear equation in F's nine entries; eight matches (or more, least
squares) give F as the null vector of the stacked system, after Hartley's
normalisation (zero mean, mean distance √2) without which the system is
hopelessly ill-conditioned. Then enforce the constraints: rank 2 for F, and
singular values (σ, σ, 0) for E.

RANSAC around it, scoring every match by its Sampson distance — the first-order
approximation of the geometric (reprojection) distance to the epipolar
constraint, in pixels. The minimal solver for E is Nistér's five-point
algorithm (what OpenCV's findEssentialMat uses); this module samples eight
points instead — simpler, but RANSAC needs more iterations (w⁸ instead of w⁵).

Pose from E. E = U diag(1,1,0) Vᵀ gives four (R, t) candidates: R = U W Vᵀ or
U Wᵀ Vᵀ, t = ±u3. Only one puts the triangulated points in front of both
cameras (the cheirality check); count points with positive depth in both.

Triangulation: linear (DLT). x × (P X) = 0 gives two equations per view;
stack and take the SVD null vector.
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


def eight_point(x1: np.ndarray, x2: np.ndarray, essential: bool = False) -> np.ndarray:
    """F (or E, for normalised coordinates with essential=True) from >= 8 matches."""
    if len(x1) < 8:
        raise ValueError("need at least 8 correspondences")
    n1, t1 = _normalise(np.asarray(x1, float))
    n2, t2 = _normalise(np.asarray(x2, float))
    a = np.stack([n2[:, 0] * n1[:, 0], n2[:, 0] * n1[:, 1], n2[:, 0],
                  n2[:, 1] * n1[:, 0], n2[:, 1] * n1[:, 1], n2[:, 1],
                  n1[:, 0], n1[:, 1], np.ones(len(n1))], 1)
    _, _, vt = np.linalg.svd(a)
    f = vt[-1].reshape(3, 3)
    f = t2.T @ f @ t1
    u, s, vt = np.linalg.svd(f)
    s = np.array([1.0, 1.0, 0.0]) if essential else np.array([s[0], s[1], 0.0])
    f = u @ np.diag(s) @ vt
    return f / np.linalg.norm(f)


def sampson_distance(f: np.ndarray, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """First-order geometric error of x2ᵀ F x1 = 0 (same units as x1, x2)."""
    h1 = np.c_[x1, np.ones(len(x1))]
    h2 = np.c_[x2, np.ones(len(x2))]
    fx1 = h1 @ f.T
    ftx2 = h2 @ f
    num = (h2 * fx1).sum(1) ** 2
    den = fx1[:, 0] ** 2 + fx1[:, 1] ** 2 + ftx2[:, 0] ** 2 + ftx2[:, 1] ** 2
    return np.sqrt(num / np.maximum(den, 1e-30))


def ransac_iterations(w: float, s: int, p: float, it: int, max_iter: int) -> int:
    """N = log(1 - p) / log(1 - wˢ): samples needed to draw one all-inlier
    minimal set with probability p (capped; log1p keeps tiny wˢ accurate)."""
    if w >= 1:
        return it
    q = np.log1p(-(w ** s))
    if q >= 0 or not np.isfinite(q):
        return max_iter
    return int(min(max_iter, np.ceil(np.log(1 - p) / q)))


@dataclass
class EssentialResult:
    e: np.ndarray | None
    inliers: np.ndarray
    iterations: int


def ransac_essential(xn1: np.ndarray, xn2: np.ndarray, focal: float, thresh_px: float = 1.0,
                     p: float = 0.999, max_iter: int = 3000, seed: int = 0) -> EssentialResult:
    """E from normalised coordinates; `focal` converts the pixel threshold."""
    n = len(xn1)
    if n < 8:
        return EssentialResult(None, np.zeros(n, bool), 0)
    rng = np.random.default_rng(seed)
    thr = thresh_px / focal
    best, best_e = np.zeros(n, bool), None
    needed, it = max_iter, 0
    while it < min(needed, max_iter):
        it += 1
        idx = rng.choice(n, 8, replace=False)
        try:
            e = eight_point(xn1[idx], xn2[idx], essential=True)
        except np.linalg.LinAlgError:
            continue
        inl = sampson_distance(e, xn1, xn2) < thr
        if inl.sum() > best.sum():
            # Local optimisation (LO-RANSAC, Chum et al. 2003): an 8-point model
            # from noisy points under-counts its own inliers; refitting on them
            # (a few times, with a looser threshold first) recovers the rest, and
            # the better inlier ratio also ends the sampling much sooner.
            for k_thr in (2.0, 1.0, 1.0):
                if inl.sum() < 8:
                    break
                e_lo = eight_point(xn1[inl], xn2[inl], essential=True)
                inl_lo = sampson_distance(e_lo, xn1, xn2) < k_thr * thr
                if k_thr == 1.0 and inl_lo.sum() < inl.sum():
                    break
                e, inl = e_lo, inl_lo
            inl = sampson_distance(e, xn1, xn2) < thr
            if inl.sum() > best.sum():
                best, best_e = inl, e
                needed = ransac_iterations(inl.mean(), 8, p, it, max_iter)
    if best.sum() < 8:
        return EssentialResult(None, best, it)
    return EssentialResult(best_e, best, it)



def triangulate(p_list: list[np.ndarray], x_list: list[np.ndarray]) -> np.ndarray:
    """Linear triangulation. p_list: 3x4 projection matrices (in the same units as
    x_list: normalised coordinates with P = [R|t], or pixels with P = K[R|t]);
    x_list: matching (N, 2) points per view. Returns (N, 3)."""
    n = len(x_list[0])
    a = np.zeros((n, 2 * len(p_list), 4))
    for v, (pm, x) in enumerate(zip(p_list, x_list, strict=True)):
        a[:, 2 * v] = x[:, 0:1] * pm[2] - pm[0]
        a[:, 2 * v + 1] = x[:, 1:2] * pm[2] - pm[1]
    # row scaling does not change the null vector but helps the conditioning
    a /= np.linalg.norm(a, axis=2, keepdims=True) + 1e-12
    _, _, vt = np.linalg.svd(a)
    xh = vt[:, -1]
    w = xh[:, 3:4]
    return xh[:, :3] / np.where(np.abs(w) < 1e-12, 1e-12, w)


def decompose_essential(e: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    u, _, vt = np.linalg.svd(e)
    if np.linalg.det(u) < 0:
        u = -u
    if np.linalg.det(vt) < 0:
        vt = -vt
    w = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]])
    r1, r2 = u @ w @ vt, u @ w.T @ vt
    t = u[:, 2]
    return [(r1, t), (r1, -t), (r2, t), (r2, -t)]


def recover_pose(e: np.ndarray, xn1: np.ndarray, xn2: np.ndarray, max_depth: float = 100.0
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(R, t, mask of points in front of both cameras) — the cheirality check.
    Camera 1 is [I|0]; t has unit length (scale is unobservable from two views)."""
    p1 = np.c_[np.eye(3), np.zeros(3)]
    best = None
    for rot, t in decompose_essential(e):
        x = triangulate([p1, np.c_[rot, t]], [xn1, xn2])
        z1 = x[:, 2]
        z2 = x @ rot[2] + t[2]
        ok = (z1 > 0) & (z2 > 0) & (z1 < max_depth) & (z2 < max_depth)
        if best is None or ok.sum() > best[2].sum():
            best = (rot, t, ok)
    return best


def triangulation_angle(x: np.ndarray, c1: np.ndarray, c2: np.ndarray) -> np.ndarray:
    """Angle (deg) at each 3D point between the rays to the two camera centres."""
    a, b = c1 - x, c2 - x
    cos = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def refine_relative_pose(rot: np.ndarray, t: np.ndarray, xn1: np.ndarray, xn2: np.ndarray,
                         iters: int = 30) -> tuple[np.ndarray, np.ndarray]:
    """Minimise the (signed) Sampson error over the 5 degrees of freedom of
    (R, t/|t|) with Levenberg-Marquardt. The 8-point solution minimises an
    algebraic error and ignores that E has only five degrees of freedom; this
    step is what makes the translation direction accurate."""
    from .camera import log_rotation, rodrigues, skew
    from .lm import levenberg_marquardt

    h1 = np.c_[xn1, np.ones(len(xn1))]
    h2 = np.c_[xn2, np.ones(len(xn2))]

    def res(p):
        r = rodrigues(p[:3])
        tt = p[3:]
        e = skew(tt) @ r
        fx1 = h1 @ e.T
        ftx2 = h2 @ e
        num = (h2 * fx1).sum(1)
        den = np.sqrt(fx1[:, 0] ** 2 + fx1[:, 1] ** 2 + ftx2[:, 0] ** 2 + ftx2[:, 1] ** 2 + 1e-30)
        return np.r_[num / den, 10.0 * (tt @ tt - 1.0)]       # keep |t| = 1 (scale gauge)

    out = levenberg_marquardt(res, np.r_[log_rotation(rot), t / np.linalg.norm(t)], max_iter=iters)
    return rodrigues(out.x[:3]), out.x[3:] / np.linalg.norm(out.x[3:])
