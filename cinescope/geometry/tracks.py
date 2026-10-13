"""Feature tracks through a shot: Phase 0's Harris corners + pyramidal LK.

Structure from motion needs the same 3D point identified in many frames. A
film shot is a continuous video, so instead of matching descriptors between
distant keyframes (what COLMAP does with SIFT), points are *tracked* frame to
frame, where motion is small and LK is reliable, and chained into tracks.

  * forward-backward check (Kalal et al., 2010): track a → b, then b → a; a
    good track returns to where it started. Errors above `fb_thresh` pixels
    (occlusion, drift onto a different texture) end the track.
  * replenishing: when tracks die (points leave the frame or get occluded),
    new corners are detected away from the surviving ones, so coverage stays
    even across the frame.

Result: for every track id, the frames it was seen in and its (x, y) there.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..motion.harris import detect_corners
from ..motion.lk import track


@dataclass
class Tracks:
    frames: list[int]                                   # frame numbers the tracks were run on
    obs: dict[int, dict[int, np.ndarray]] = field(default_factory=dict)   # track -> {frame: xy}

    def in_frame(self, f: int) -> tuple[np.ndarray, np.ndarray]:
        """(track ids, (N, 2) points) seen in frame f."""
        ids = [t for t, o in self.obs.items() if f in o]
        if not ids:
            return np.zeros(0, int), np.zeros((0, 2))
        return np.asarray(ids), np.stack([self.obs[t][f] for t in ids])

    def common(self, fa: int, fb: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ids = [t for t, o in self.obs.items() if fa in o and fb in o]
        if not ids:
            return np.zeros(0, int), np.zeros((0, 2)), np.zeros((0, 2))
        return (np.asarray(ids), np.stack([self.obs[t][fa] for t in ids]),
                np.stack([self.obs[t][fb] for t in ids]))

    def lengths(self) -> np.ndarray:
        return np.array([len(o) for o in self.obs.values()])


def _far_from(pts: np.ndarray, existing: np.ndarray, radius: float) -> np.ndarray:
    if len(existing) == 0 or len(pts) == 0:
        return np.ones(len(pts), bool)
    d2 = ((pts[:, None, :] - existing[None, :, :]) ** 2).sum(-1)
    return d2.min(1) > radius ** 2


def track_shot(gray: np.ndarray, frame_ids: list[int] | None = None, max_corners: int = 600,
               min_distance: int = 8, fb_thresh: float = 1.0, win: int = 7, levels: int = 4,
               quality: float = 0.002, border: int = 8) -> Tracks:
    """gray: (N, H, W) consecutive frames of one shot (float or uint8)."""
    gray = np.asarray(gray, np.float64)
    n = len(gray)
    frame_ids = list(range(n)) if frame_ids is None else list(frame_ids)
    out = Tracks(frame_ids)
    pts = detect_corners(gray[0], max_corners=max_corners, min_distance=min_distance,
                         quality=quality, border=border)
    ids = np.arange(len(pts))
    next_id = len(pts)
    for t, p in zip(ids, pts, strict=True):
        out.obs[int(t)] = {frame_ids[0]: p}
    h, w = gray.shape[1:]
    for k in range(1, n):
        if len(pts):
            fwd, ok = track(gray[k - 1], gray[k], pts, win=win, levels=levels)
            back, ok2 = track(gray[k], gray[k - 1], fwd, win=win, levels=levels)
            fb = np.linalg.norm(back - pts, axis=1)
            keep = ok & ok2 & (fb < fb_thresh)
            keep &= (fwd[:, 0] > border) & (fwd[:, 0] < w - border) & (fwd[:, 1] > border) & (fwd[:, 1] < h - border)
            pts, ids = fwd[keep], ids[keep]
            for t, p in zip(ids, pts, strict=True):
                out.obs[int(t)][frame_ids[k]] = p
        if len(pts) < 0.8 * max_corners:
            new = detect_corners(gray[k], max_corners=max_corners, min_distance=min_distance,
                                 quality=quality, border=border)
            new = new[_far_from(new, pts, min_distance)][: max_corners - len(pts)]
            if len(new):
                new_ids = np.arange(next_id, next_id + len(new))
                next_id += len(new)
                for t, p in zip(new_ids, new, strict=True):
                    out.obs[int(t)] = {frame_ids[k]: p}
                pts = np.concatenate([pts, new])
                ids = np.concatenate([ids, new_ids])
    out.obs = {t: o for t, o in out.obs.items() if len(o) >= 2}
    return out
