"""Scores for reconstructions and depth maps against ground truth.

Monocular SfM recovers the scene only up to a similarity transform (rotation,
translation, *scale*): nothing in the images says whether the set is a model
or a real room. So trajectories are compared after the best similarity
alignment (Umeyama, 1991), and depth maps after the best scale (and, for
networks that predict inverse depth up to an affine map, shift).
"""
from __future__ import annotations

import numpy as np

from .camera import nearest_rotation, rotation_angle_deg


def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool = True) -> tuple[float, np.ndarray, np.ndarray]:
    """s, R, t minimising Σ‖dst - (s R src + t)‖² (closed form, via SVD)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    a, b = src - mu_s, dst - mu_d
    cov = b.T @ a / len(src)
    u, d, vt = np.linalg.svd(cov)
    s_fix = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        s_fix[2, 2] = -1
    rot = u @ s_fix @ vt
    var = (a ** 2).sum() / len(src)
    scale = float(np.trace(np.diag(d) @ s_fix) / var) if with_scale and var > 0 else 1.0
    return scale, rot, mu_d - scale * rot @ mu_s


def trajectory_errors(est_centres: np.ndarray, gt_centres: np.ndarray, est_rots: np.ndarray,
                      gt_rots: np.ndarray) -> dict:
    """Pose errors after aligning the reconstruction's world frame to the ground truth.

    The aligning rotation comes from the camera orientations (R_est = R_gt A for
    every camera, so A = nearest rotation to Σ R_gtᵀ R_est) — Umeyama on the
    centres alone is ill-posed for a dolly, whose centres lie on a line and
    leave the rotation about that line free. Scale and offset then come from
    the centres. ATE is reported relative to the GT path length (scale-free)."""
    a = nearest_rotation(sum(rg.T @ re for re, rg in zip(est_rots, gt_rots, strict=True)))
    src = est_centres @ a.T                     # est centres in GT orientation
    mu_s, mu_d = src.mean(0), gt_centres.mean(0)
    den = ((src - mu_s) ** 2).sum()
    scale = float(((src - mu_s) * (gt_centres - mu_d)).sum() / den) if den > 0 else 1.0
    aligned = scale * (src - mu_s) + mu_d
    path = np.linalg.norm(np.diff(gt_centres, axis=0), axis=1).sum()
    ate = np.linalg.norm(aligned - gt_centres, axis=1)
    rot_err = [rotation_angle_deg(re @ a.T, rg) for re, rg in zip(est_rots, gt_rots, strict=True)]
    return {"ate_rmse_rel_path": float(np.sqrt((ate ** 2).mean()) / max(path, 1e-12)),
            "rot_err_deg_mean": float(np.mean(rot_err)), "rot_err_deg_max": float(np.max(rot_err)),
            "scale": scale}


def translation_direction_error_deg(t_est: np.ndarray, t_gt: np.ndarray) -> float:
    a, b = t_est / np.linalg.norm(t_est), t_gt / np.linalg.norm(t_gt)
    return float(np.degrees(np.arccos(np.clip(a @ b, -1, 1))))


def scale_invariant_log_rmse(pred: np.ndarray, gt: np.ndarray) -> float:
    """Eigen et al. (2014): RMSE of log depth after removing the mean log ratio —
    invariant to a global scale of either map."""
    d = np.log(pred) - np.log(gt)
    return float(np.sqrt(np.mean(d ** 2) - np.mean(d) ** 2))


def median_scale(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.median(gt) / np.median(pred))


def abs_rel(pred: np.ndarray, gt: np.ndarray) -> float:
    """Mean |pred - gt| / gt after median scaling of pred."""
    p = pred * median_scale(pred, gt)
    return float(np.mean(np.abs(p - gt) / gt))


def delta_accuracy(pred: np.ndarray, gt: np.ndarray, thresh: float = 1.25) -> float:
    """Share of points with max(p/g, g/p) < 1.25 after median scaling."""
    p = pred * median_scale(pred, gt)
    return float(np.mean(np.maximum(p / gt, gt / p) < thresh))


def fit_inverse_depth_affine(inv_pred: np.ndarray, depth: np.ndarray) -> tuple[float, float]:
    """a, b with a · inv_pred + b ≈ 1 / depth (least squares). MiDaS predicts
    inverse depth up to an unknown scale and shift; this undoes both."""
    a = np.stack([inv_pred, np.ones_like(inv_pred)], 1)
    sol, *_ = np.linalg.lstsq(a, 1.0 / depth, rcond=None)
    return float(sol[0]), float(sol[1])


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation: does the ordering of depths agree, whatever the scale?"""
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    return float((ra * rb).sum() / np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))
