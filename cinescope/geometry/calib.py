"""Camera calibration with Zhang's method (Zhang, 2000), in NumPy.

A flat calibration target (a chessboard) at Z = 0 maps to the image by a
homography,  s [u v 1]ᵀ = K [r1 r2 t] [X Y 1]ᵀ,  so H = λ K [r1 r2 t].

1. One homography per view (normalised DLT, reused from Phase 0).
2. r1 and r2 are orthonormal, which gives two linear constraints per view on
   the symmetric matrix B = K⁻ᵀ K⁻¹ (the image of the absolute conic):
       h1ᵀ B h2 = 0,       h1ᵀ B h1 = h2ᵀ B h2.
   Stack them (plus "zero skew", B12 = 0) and solve V b = 0 by SVD; read K off
   B in closed form. Three views are enough; more average out noise.
3. Each view's R, t from K⁻¹H (then snap R onto SO(3)).
4. Radial distortion (k1, k2) by linear least squares, given everything else.
5. Levenberg-Marquardt over all of it — fx, fy, cx, cy, k1, k2 and six pose
   numbers per view — minimising the reprojection error in pixels. The closed
   form minimises an algebraic error; only this step minimises the error that
   means something (the maximum-likelihood estimate under Gaussian pixel noise).

Pixel coordinates are first normalised to about [-1, 1]; otherwise B's entries
span ten orders of magnitude and the SVD in step 2 loses the small ones.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..motion.homography import dlt
from .camera import log_rotation, nearest_rotation, project, rodrigues
from .lm import levenberg_marquardt


@dataclass
class Calibration:
    k: np.ndarray                       # 3x3 intrinsics
    dist: np.ndarray                    # (k1, k2)
    rvecs: list[np.ndarray]             # per view, axis-angle
    tvecs: list[np.ndarray]
    rms: float                          # sqrt(mean squared point error), as cv2.calibrateCamera
    per_view_rms: list[float] = field(default_factory=list)
    closed_form_k: np.ndarray | None = None
    iterations: int = 0

    def to_dict(self) -> dict:
        return {"fx": float(self.k[0, 0]), "fy": float(self.k[1, 1]), "cx": float(self.k[0, 2]),
                "cy": float(self.k[1, 2]), "k1": float(self.dist[0]), "k2": float(self.dist[1]),
                "rms_px": float(self.rms), "per_view_rms_px": [round(v, 4) for v in self.per_view_rms]}


def chessboard_points(cols: int, rows: int, square: float = 1.0) -> np.ndarray:
    """(cols*rows, 2) planar target coordinates, in the row-major order
    cv2.findChessboardCorners returns."""
    xs, ys = np.meshgrid(np.arange(cols), np.arange(rows))
    return np.stack([xs.ravel(), ys.ravel()], 1).astype(float) * square


def _v(h: np.ndarray, i: int, j: int) -> np.ndarray:
    a, b = h[:, i], h[:, j]
    return np.array([a[0] * b[0], a[0] * b[1] + a[1] * b[0], a[1] * b[1],
                     a[2] * b[0] + a[0] * b[2], a[2] * b[1] + a[1] * b[2], a[2] * b[2]])


def intrinsics_from_homographies(hs: list[np.ndarray], zero_skew: bool = True) -> np.ndarray:
    """Closed-form K from >= 3 plane homographies (Zhang's Appendix B)."""
    rows = []
    for h in hs:
        rows.append(_v(h, 0, 1))
        rows.append(_v(h, 0, 0) - _v(h, 1, 1))
    if zero_skew:
        rows.append(np.array([0, 1.0, 0, 0, 0, 0]))
    _, _, vt = np.linalg.svd(np.asarray(rows))
    b11, b12, b22, b13, b23, b33 = vt[-1]
    if b11 < 0:                         # B is defined up to sign; it must be positive definite
        b11, b12, b22, b13, b23, b33 = -vt[-1]
    v0 = (b12 * b13 - b11 * b23) / (b11 * b22 - b12 ** 2)
    lam = b33 - (b13 ** 2 + v0 * (b12 * b13 - b11 * b23)) / b11
    alpha = np.sqrt(lam / b11)
    beta = np.sqrt(lam * b11 / (b11 * b22 - b12 ** 2))
    gamma = -b12 * alpha ** 2 * beta / lam
    u0 = gamma * v0 / beta - b13 * alpha ** 2 / lam
    return np.array([[alpha, gamma, u0], [0, beta, v0], [0, 0, 1.0]])


def extrinsics_from_homography(k: np.ndarray, h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    m = np.linalg.inv(k) @ h
    lam = 1.0 / np.linalg.norm(m[:, 0])
    if m[2, 2] * lam < 0:               # the board must be in front of the camera
        lam = -lam
    r1, r2, t = lam * m[:, 0], lam * m[:, 1], lam * m[:, 2]
    rot = nearest_rotation(np.stack([r1, r2, np.cross(r1, r2)], 1))
    return rot, t


def _obj3(obj: np.ndarray) -> np.ndarray:
    return np.c_[obj, np.zeros(len(obj))]


def _unpack(x: np.ndarray, n_views: int):
    fx, fy, cx, cy, k1, k2 = x[:6]
    k = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
    poses = x[6:].reshape(n_views, 6)
    return k, np.array([k1, k2]), poses


def _residuals(x: np.ndarray, obj: list[np.ndarray], img: list[np.ndarray]) -> np.ndarray:
    k, dist, poses = _unpack(x, len(obj))
    out = []
    for o, i, p in zip(obj, img, poses, strict=True):
        out.append((project(_obj3(o), k, rodrigues(p[:3]), p[3:], dist) - i).ravel())
    return np.concatenate(out)


def _jacobian(x: np.ndarray, obj: list[np.ndarray], img: list[np.ndarray], eps: float = 1e-6) -> np.ndarray:
    """Forward differences, exploiting the block structure: the six intrinsic
    columns touch every residual, but a view's six pose columns touch only that
    view's residuals — so ~12 residual evaluations instead of 6 + 6·views."""
    r0 = _residuals(x, obj, img)
    jac = np.zeros((len(r0), len(x)))
    for i in range(6):
        h = eps * max(1.0, abs(x[i]))
        xp = x.copy()
        xp[i] += h
        jac[:, i] = (_residuals(xp, obj, img) - r0) / h
    k, dist, poses = _unpack(x, len(obj))
    row = 0
    for v, (o, im, p) in enumerate(zip(obj, img, poses, strict=True)):
        n = 2 * len(o)
        base = (project(_obj3(o), k, rodrigues(p[:3]), p[3:], dist) - im).ravel()
        for j in range(6):
            h = eps * max(1.0, abs(p[j]))
            pp = p.copy()
            pp[j] += h
            jac[row:row + n, 6 + 6 * v + j] = ((project(_obj3(o), k, rodrigues(pp[:3]), pp[3:], dist) - im).ravel()
                                               - base) / h
        row += n
    return jac


def calibrate(obj_pts: list[np.ndarray], img_pts: list[np.ndarray], image_size: tuple[int, int],
              refine: bool = True) -> Calibration:
    """obj_pts: per view (N, 2) planar target coordinates; img_pts: per view (N, 2) pixels;
    image_size: (width, height)."""
    w, h = image_size
    s = 2.0 / max(w, h)
    norm = np.array([[s, 0, -s * w / 2], [0, s, -s * h / 2], [0, 0, 1.0]])
    hs_n = [dlt(o, (i - [w / 2, h / 2]) * s) for o, i in zip(obj_pts, img_pts, strict=True)]
    k = np.linalg.inv(norm) @ intrinsics_from_homographies(hs_n)
    k /= k[2, 2]
    k[0, 1] = 0.0
    closed = k.copy()
    hs = [np.linalg.inv(norm) @ hn for hn in hs_n]
    poses = [extrinsics_from_homography(k, hh) for hh in hs]

    # Linear radial distortion given K and the poses (Zhang, section 3.3).
    a_rows, b_rows = [], []
    for o, i, (rot, t) in zip(obj_pts, img_pts, poses, strict=True):
        xc = _obj3(o) @ rot.T + t
        xy = xc[:, :2] / xc[:, 2:3]
        r2 = (xy ** 2).sum(1)
        uv = xy @ k[:2, :2].T + k[:2, 2]
        d = uv - k[:2, 2]
        a_rows.append(np.stack([d[:, 0] * r2, d[:, 0] * r2 ** 2], 1))
        a_rows.append(np.stack([d[:, 1] * r2, d[:, 1] * r2 ** 2], 1))
        b_rows.append(i[:, 0] - uv[:, 0])
        b_rows.append(i[:, 1] - uv[:, 1])
    dist = np.linalg.lstsq(np.concatenate(a_rows), np.concatenate(b_rows), rcond=None)[0]

    x0 = np.concatenate([[k[0, 0], k[1, 1], k[0, 2], k[1, 2]], dist]
                        + [np.r_[log_rotation(r), t] for r, t in poses])
    iters = 0
    if refine:
        res = levenberg_marquardt(lambda x: _residuals(x, obj_pts, img_pts), x0,
                                  jac=lambda x: _jacobian(x, obj_pts, img_pts), max_iter=200)
        x0, iters = res.x, res.iterations
    k, dist, pv = _unpack(x0, len(obj_pts))
    r = _residuals(x0, obj_pts, img_pts).reshape(-1, 2)
    n = np.cumsum([0] + [len(o) for o in obj_pts])
    per_view = [float(np.sqrt((r[a:b] ** 2).sum(1).mean())) for a, b in zip(n[:-1], n[1:], strict=True)]
    rms = float(np.sqrt((r ** 2).sum(1).mean()))
    return Calibration(k, dist, [p[:3] for p in pv], [p[3:] for p in pv], rms, per_view, closed, iters)
