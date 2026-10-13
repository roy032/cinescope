"""Camera path of one shot: where the camera was and where it pointed, frame by frame.

Two cases, decided by the data:

* the camera translates (dolly, crane, Steadicam): incremental SfM gives the
  full 6-DoF path (up to scale) and a sparse point cloud;
* it only rotates or zooms (pan, tilt, zoom, a locked-off tripod): there is no
  parallax and SfM rightly refuses. Then frame-to-frame homographies give the
  rotation: for a rotating camera H = K R K⁻¹, so R ≈ K⁻¹ H K (snapped onto
  SO(3); a zoom shows up as scale the rotation cannot absorb, reported apart).
  The centre stays at the origin.

Angles are reported as pan (yaw, about the camera's vertical axis), tilt
(pitch, about its horizontal axis) and roll, relative to the first frame —
positive pan = turning right, positive tilt = looking up.
"""
from __future__ import annotations

import numpy as np

from ..motion.homography import ransac_homography
from .camera import camera_centre, nearest_rotation
from .sfm import Reconstruction
from .tracks import Tracks


def pan_tilt_roll(rot_rel: np.ndarray) -> tuple[float, float, float]:
    """Degrees, from R_rel = R_i R_0ᵀ (world->camera, y down, z forward).
    The camera's viewing direction in first-frame coordinates is R_relᵀ e_z."""
    d = rot_rel.T @ np.array([0, 0, 1.0])
    pan = np.degrees(np.arctan2(d[0], d[2]))
    tilt = np.degrees(np.arctan2(-d[1], np.hypot(d[0], d[2])))
    right = rot_rel.T @ np.array([1.0, 0, 0])
    roll = np.degrees(np.arctan2(right[1], right[0]))
    return float(pan), float(tilt), float(roll)


def rotation_only_path(tracks: Tracks, k: np.ndarray, max_misfit: float = 0.05) -> dict[int, dict]:
    """Chain frame-to-frame homographies into per-frame rotations (and zoom).
    Steps whose homography is not a rotation + zoom (misfit > max_misfit) are held."""
    kinv = np.linalg.inv(k)
    frames = tracks.frames
    rot = np.eye(3)
    log_zoom = 0.0
    out = {frames[0]: {"R": rot, "zoom": 1.0, "inliers": 1.0}}
    for a, b in zip(frames[:-1], frames[1:], strict=True):
        ids, pa, pb = tracks.common(a, b)
        if len(ids) >= 8:
            res = ransac_homography(pa, pb, thresh=1.5)
            if res.h is not None:
                m = kinv @ res.h @ k
                if np.linalg.det(m) < 0:
                    m = -m
                # With a zoom by s, m ~ diag(s, s, 1) R. The polar decomposition of
                # D R (D symmetric positive definite) has rotation factor exactly R,
                # so the nearest rotation is the camera rotation, and D = m Rᵀ.
                r_step = nearest_rotation(m)
                dm = m @ r_step.T
                d = np.diag(dm)
                # How far is H from "rotation + zoom"? A translating camera, a
                # moving screen graphic or a bad fit leaves off-diagonal terms
                # (or unequal x/y scale). Such steps are not trusted: hold the pose.
                misfit = float(np.linalg.norm(dm - np.diag(d)) / np.linalg.norm(dm)
                               + abs(d[0] - d[1]) / (abs(d[0]) + abs(d[1])))
                if misfit < max_misfit:
                    log_zoom += float(np.log(max((d[0] + d[1]) / (2 * d[2]), 1e-6)))
                    rot = r_step @ rot
                    out[b] = {"R": rot, "zoom": float(np.exp(log_zoom)), "inliers": res.inlier_ratio,
                              "misfit": misfit}
                    continue
                out[b] = {"R": rot, "zoom": float(np.exp(log_zoom)), "inliers": res.inlier_ratio,
                          "misfit": misfit, "held": True}
                continue
        out[b] = {"R": rot, "zoom": float(np.exp(log_zoom)), "inliers": 0.0, "held": True}
    return out


def path_from_reconstruction(rec: Reconstruction) -> list[dict]:
    frames = sorted(rec.poses)
    r0 = rec.poses[frames[0]][0]
    c0 = camera_centre(*rec.poses[frames[0]])
    out = []
    for f in frames:
        rot, t = rec.poses[f]
        c = camera_centre(rot, t)
        pan, tilt, roll = pan_tilt_roll(rot @ r0.T)
        out.append({"frame": int(f), "keyframe": f in rec.keyframes, "R": np.round(rot, 6).tolist(),
                    "t": np.round(t, 6).tolist(), "centre": np.round(c, 6).tolist(),
                    "pan_deg": round(pan, 3), "tilt_deg": round(tilt, 3), "roll_deg": round(roll, 3),
                    "distance_from_start": round(float(np.linalg.norm(c - c0)), 6)})
    return out


def path_from_rotations(rots: dict[int, dict]) -> list[dict]:
    out = []
    for f in sorted(rots):
        r = rots[f]["R"]
        pan, tilt, roll = pan_tilt_roll(r)
        out.append({"frame": int(f), "keyframe": False, "R": np.round(r, 6).tolist(), "t": [0.0, 0.0, 0.0],
                    "centre": [0.0, 0.0, 0.0], "pan_deg": round(pan, 3), "tilt_deg": round(tilt, 3),
                    "roll_deg": round(roll, 3), "zoom": round(rots[f]["zoom"], 4),
                    "held": bool(rots[f].get("held", False)), "distance_from_start": 0.0})
    return out
