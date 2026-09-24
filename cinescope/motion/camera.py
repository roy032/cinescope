"""Classify a shot's camera move: static, pan, tilt, zoom or handheld.

Per shot:
  1. every `step` frames, detect Harris corners on frame a, track them to
     frame b with pyramidal Lucas-Kanade, fit a homography with RANSAC
  2. read the frame-to-frame motion off H at the image centre:
       translation (tx, ty)  where the centre moves, in % of frame width per second
       zoom                  log of the local scale, sqrt(|det J|) of H's Jacobian
  3. aggregate over the shot and apply simple, explainable rules:

  static    median speed and zoom rate both tiny, little jitter
  zoom      scale change explains more of the motion than translation
  pan/tilt  steady horizontal / vertical motion (the image moves opposite to
            the camera: content sliding left means the camera pans right)
  handheld  motion that keeps changing direction (high jitter relative to
            the steady component) — the shake of a hand-held or Steadicam rig

A dolly or tracking shot of a nearby scene violates the homography model
(parallax), which shows up as a low RANSAC inlier ratio; it is reported
rather than hidden, and Phase 5's structure-from-motion handles it properly.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .harris import detect_corners
from .homography import ransac_homography
from .lk import track


@dataclass
class CameraMove:
    label: str               # static | pan | tilt | zoom | handheld | unknown
    direction: str           # left/right, up/down, in/out, or ""
    speed: float             # |median translation|, % of width per second
    zoom_rate: float         # median log-scale change per second
    jitter: float            # std of per-step translation, % width per second
    inlier_ratio: float      # median RANSAC inlier share (low = parallax / moving subject)
    steps: int

    def to_dict(self) -> dict:
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in asdict(self).items()}


@dataclass
class CameraRules:
    static_speed: float = 2.0        # %width/s below this counts as not moving
    static_zoom: float = 0.03        # log-scale/s
    # Tracking noise matters here: at 96 px, a 0.5 px error every 1/6 s is
    # already ~3 %width/s of apparent jitter, so the handheld threshold sits
    # well above that floor.
    handheld_jitter: float = 6.0     # %width/s
    handheld_ratio: float = 1.2      # jitter / speed above this (and above handheld_jitter) = handheld


def frame_motion(a: np.ndarray, b: np.ndarray, max_corners: int = 150) -> tuple[np.ndarray, float, float] | None:
    """(t_xy in pixels, log-scale, inlier ratio) between two gray frames, or None."""
    # quality 0.001: in a dark frame one bright practical light (a lamp) dominates
    # the response map, and a 1% cut-off would leave only the lamp's corners.
    pts = detect_corners(a, max_corners=max_corners, min_distance=4, quality=0.001)
    if len(pts) < 8:
        return None
    new, ok = track(a, b, pts, win=5, levels=3)
    if ok.sum() < 8:
        return None
    res = ransac_homography(pts[ok], new[ok], thresh=1.5)
    if res.h is None:
        return None
    h = res.h
    cx, cy = a.shape[1] / 2, a.shape[0] / 2
    c = h @ np.array([cx, cy, 1.0])
    c = c[:2] / c[2]
    # Jacobian of the homography at the centre -> local scale
    w = h[2] @ np.array([cx, cy, 1.0])
    jac = (h[:2, :2] - np.outer(c, h[2, :2])) / w
    scale = np.sqrt(abs(np.linalg.det(jac)))
    return c - np.array([cx, cy]), float(np.log(max(scale, 1e-6))), res.inlier_ratio


def _normalise(g: np.ndarray) -> np.ndarray:
    """Zero-mean, fixed-contrast frame: a night scene then has gradients as
    strong as a daylight one, so corner and eigenvalue thresholds still apply."""
    g = np.asarray(g, dtype=np.float64)
    return (g - g.mean()) / (g.std() + 1e-6) * 50.0 + 128.0


def classify_shot(gray: np.ndarray, fps: float, step: int = 4,
                  rules: CameraRules | None = None) -> CameraMove:
    """gray: (N, H, W) frames of ONE shot."""
    rules = rules or CameraRules()
    n = len(gray)
    if n < step + 1:
        return CameraMove("unknown", "", 0.0, 0.0, 0.0, 0.0, 0)
    width = gray.shape[2]
    per_s = fps / step
    trans, zooms, inl = [], [], []
    for i in range(0, n - step, step):
        m = frame_motion(_normalise(gray[i]), _normalise(gray[i + step]))
        if m is None:
            continue
        t, z, r = m
        trans.append(t / width * 100 * per_s)          # % of width per second
        zooms.append(z * per_s)
        inl.append(r)
    if len(trans) < 2:
        return CameraMove("unknown", "", 0.0, 0.0, 0.0, 0.0, len(trans))
    tr = np.asarray(trans)
    med = np.median(tr, axis=0)
    speed = float(np.hypot(*med))
    zoom = float(np.median(zooms))
    jitter = float(np.sqrt(((tr - med) ** 2).sum(1).mean()))
    inlier = float(np.median(inl))

    if jitter > rules.handheld_jitter and jitter > rules.handheld_ratio * max(speed, 1e-6):
        label, direction = "handheld", ""
    elif speed < rules.static_speed and abs(zoom) < rules.static_zoom:
        label, direction = "static", ""
    elif abs(zoom) * 100 >= 0.5 * speed:
        # Scale change explains more of the motion than translation does. (A
        # zoom about the frame centre moves the centre very little, so speed
        # alone would call a slow zoom "static" or a tiny pan.)
        label, direction = "zoom", "in" if zoom > 0 else "out"
    elif abs(med[0]) >= abs(med[1]):
        # content moves left (tx < 0) when the camera pans right
        label, direction = "pan", "right" if med[0] < 0 else "left"
    else:
        label, direction = "tilt", "down" if med[1] < 0 else "up"
    return CameraMove(label, direction, speed, zoom, jitter, inlier, len(trans))
