"""Rotations and the pinhole camera with radial distortion, in NumPy.

Axis-angle (Rodrigues) vectors are how every optimiser here parameterises a
rotation: three numbers, no constraint to maintain (a 3x3 matrix has nine
numbers and six constraints). exp/log convert between the two.

Pinhole projection with radial distortion (Brown's model, two terms, as in
Zhang 2000 and OpenCV with the tangential terms off):

    x_c = R X + t                       camera coordinates
    (x, y) = (x_c / z_c, y_c / z_c)     normalised image plane
    r² = x² + y²
    (x_d, y_d) = (x, y) (1 + k1 r² + k2 r⁴)
    u = fx x_d + s y_d + cx,  v = fy y_d + cy
"""
from __future__ import annotations

import numpy as np


def skew(v: np.ndarray) -> np.ndarray:
    """[v]_x, the matrix with [v]_x w = v x w."""
    x, y, z = v
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=float)


def rodrigues(r: np.ndarray) -> np.ndarray:
    """Axis-angle vector -> rotation matrix (the exponential map of SO(3))."""
    r = np.asarray(r, float)
    th = np.linalg.norm(r)
    if th < 1e-12:
        return np.eye(3) + skew(r)
    k = skew(r / th)
    return np.eye(3) + np.sin(th) * k + (1 - np.cos(th)) * k @ k


def rodrigues_batch(r: np.ndarray) -> np.ndarray:
    """(N, 3) axis-angle -> (N, 3, 3) rotations."""
    r = np.asarray(r, float).reshape(-1, 3)
    th = np.linalg.norm(r, axis=1)
    safe = np.where(th < 1e-12, 1.0, th)
    k = r / safe[:, None]
    kx = np.zeros((len(r), 3, 3))
    kx[:, 0, 1], kx[:, 0, 2], kx[:, 1, 2] = -k[:, 2], k[:, 1], -k[:, 0]
    kx[:, 1, 0], kx[:, 2, 0], kx[:, 2, 1] = k[:, 2], -k[:, 1], k[:, 0]
    s, c = np.sin(th)[:, None, None], (1 - np.cos(th))[:, None, None]
    out = np.eye(3)[None] + s * kx + c * kx @ kx
    small = th < 1e-12
    if small.any():
        out[small] = np.stack([np.eye(3) + skew(v) for v in r[small]])
    return out


def log_rotation(rot: np.ndarray) -> np.ndarray:
    """Rotation matrix -> axis-angle vector (the inverse of `rodrigues`)."""
    rot = np.asarray(rot, float)
    c = np.clip((np.trace(rot) - 1) / 2, -1.0, 1.0)
    th = np.arccos(c)
    w = np.array([rot[2, 1] - rot[1, 2], rot[0, 2] - rot[2, 0], rot[1, 0] - rot[0, 1]])
    if th < 1e-7:
        return w / 2
    if np.pi - th < 1e-4:
        # sin(th) ~ 0: read the axis off the symmetric part, R = 2 a aᵀ - I
        a = np.sqrt(np.maximum((np.diag(rot) + 1) / 2, 0))
        i = int(np.argmax(a))
        a[np.arange(3) != i] = (rot[i] + rot[:, i])[np.arange(3) != i] / (4 * a[i])
        return a / np.linalg.norm(a) * th
    return w * th / (2 * np.sin(th))


def right_jacobian(r: np.ndarray) -> np.ndarray:
    """J_r(r): exp(r + δ) ≈ exp(r) exp(J_r δ). Used for analytic BA Jacobians."""
    r = np.asarray(r, float)
    th = np.linalg.norm(r)
    k = skew(r)
    if th < 1e-6:
        return np.eye(3) - 0.5 * k
    return np.eye(3) - (1 - np.cos(th)) / th ** 2 * k + (th - np.sin(th)) / th ** 3 * k @ k


def nearest_rotation(m: np.ndarray) -> np.ndarray:
    """Closest rotation (Frobenius norm) to a 3x3 matrix, via SVD."""
    u, _, vt = np.linalg.svd(m)
    r = u @ vt
    if np.linalg.det(r) < 0:
        r = u @ np.diag([1, 1, -1]) @ vt
    return r


def rotation_angle_deg(ra: np.ndarray, rb: np.ndarray) -> float:
    """Angle of the relative rotation ra rbᵀ, in degrees."""
    c = (np.trace(ra @ rb.T) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def intrinsics(f: float, cx: float, cy: float, fy: float | None = None) -> np.ndarray:
    return np.array([[f, 0, cx], [0, f if fy is None else fy, cy], [0, 0, 1.0]])


def distort(xy: np.ndarray, dist) -> np.ndarray:
    """Apply radial distortion (k1, k2) to normalised image coordinates."""
    if dist is None:
        return xy
    k1, k2 = (list(dist) + [0, 0])[:2]
    r2 = (xy ** 2).sum(-1, keepdims=True)
    return xy * (1 + k1 * r2 + k2 * r2 ** 2)


def undistort(xy: np.ndarray, dist, iters: int = 20) -> np.ndarray:
    """Invert `distort` by fixed-point iteration (what cv2.undistortPoints does)."""
    if dist is None:
        return xy
    k1, k2 = (list(dist) + [0, 0])[:2]
    und = xy.copy()
    for _ in range(iters):
        r2 = (und ** 2).sum(-1, keepdims=True)
        und = xy / (1 + k1 * r2 + k2 * r2 ** 2)
    return und


def project(x_world: np.ndarray, k: np.ndarray, rot: np.ndarray, t: np.ndarray,
            dist=None) -> np.ndarray:
    """(N, 3) world points -> (N, 2) pixels. Points behind the camera give nan-free garbage;
    check depth separately with `depths`."""
    xc = x_world @ rot.T + t
    z = xc[:, 2:3]
    xy = xc[:, :2] / np.where(np.abs(z) < 1e-12, 1e-12, z)
    xy = distort(xy, dist)
    return xy @ k[:2, :2].T + k[:2, 2]


def depths(x_world: np.ndarray, rot: np.ndarray, t: np.ndarray) -> np.ndarray:
    return x_world @ rot[2] + t[2]


def pixels_to_normalised(uv: np.ndarray, k: np.ndarray, dist=None) -> np.ndarray:
    """Pixels -> undistorted normalised coordinates (x/z, y/z)."""
    kinv = np.linalg.inv(k)
    xy = uv @ kinv[:2, :2].T + kinv[:2, 2]
    return undistort(xy, dist)


def camera_centre(rot: np.ndarray, t: np.ndarray) -> np.ndarray:
    return -rot.T @ t


def look_at(eye: np.ndarray, target: np.ndarray, up=(0, -1, 0)) -> tuple[np.ndarray, np.ndarray]:
    """(R, t) of a camera at `eye` looking at `target` (y-down image convention)."""
    eye, target = np.asarray(eye, float), np.asarray(target, float)
    z = target - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, np.asarray(up, float))
    if np.linalg.norm(x) < 1e-9:
        x = np.cross(z, [0, 0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    rot = np.stack([x, y, z])
    return rot, -rot @ eye
