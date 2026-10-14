"""Per-person pose tracks: link detections frame to frame by box IoU.

The simplest tracker that works within a shot: each frame, solve the
assignment between live tracks (their boxes moved on at constant velocity)
and new detections that maximises total IoU
(Hungarian algorithm), accept pairs with IoU >= `iou_thresh`, start new tracks
for the rest, and end tracks unseen for `max_age` frames. There is no motion
filter and no appearance: two people crossing can swap identities. Phase 3
builds the proper tracker (Kalman motion, appearance embeddings); a shot-level
action breakdown only needs tracks that hold for a second or two.

Tracking happens inside a shot: a cut ends every track.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..detect.boxes import iou


@dataclass
class Track:
    id: int
    times: list[float] = field(default_factory=list)
    boxes: list[np.ndarray] = field(default_factory=list)
    kpts: list[np.ndarray] = field(default_factory=list)
    misses: int = 0

    def arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return np.array(self.times), np.array(self.boxes).reshape(-1, 4), np.array(self.kpts).reshape(-1, 17, 3)

    def __len__(self) -> int:
        return len(self.times)


def _predict(tr: Track) -> np.ndarray:
    """Where the box should be now: its centre moved on at the velocity of the
    last two observations, its size kept. Without it a runner (a quarter of a
    body height per frame at 8 fps) overlaps their previous box too little and
    the track breaks. Only the centre is extrapolated: box edges jump with
    every arm and leg swing."""
    b = tr.boxes[-1]
    if len(tr.boxes) < 2:
        return b
    c = lambda x: np.array([(x[0] + x[2]) / 2, (x[1] + x[3]) / 2])  # noqa: E731
    v = (c(b) - c(tr.boxes[-2])) / max(tr.times[-1] - tr.times[-2], 1e-6)
    dt = (tr.times[-1] - tr.times[-2]) * (1 + tr.misses)
    return b + np.tile(v * dt, 2)


def track_people(times: list[float] | np.ndarray, boxes: list[np.ndarray], kpts: list[np.ndarray],
                 iou_thresh: float = 0.3, max_age: int = 3, min_len: int = 3) -> list[Track]:
    """Per frame (P, 4) xyxy boxes and (P, 17, 3) keypoints -> tracks of >= min_len frames."""
    live: list[Track] = []
    done: list[Track] = []
    next_id = 0
    for t, b, k in zip(times, boxes, kpts, strict=True):
        b = np.asarray(b, np.float64).reshape(-1, 4)
        k = np.asarray(k, np.float64).reshape(-1, 17, 3)
        matched_d: set[int] = set()
        if live and len(b):
            ov = iou(np.array([_predict(tr) for tr in live]), b)
            rows, cols = linear_sum_assignment(-ov)
            for r, c in zip(rows, cols, strict=True):
                if ov[r, c] >= iou_thresh:
                    tr = live[r]
                    tr.times.append(float(t))
                    tr.boxes.append(b[c])
                    tr.kpts.append(k[c])
                    tr.misses = -1                      # reset below
                    matched_d.add(int(c))
        for tr in live:
            tr.misses += 1
        for c in range(len(b)):
            if c not in matched_d:
                live.append(Track(next_id, [float(t)], [b[c]], [k[c]]))
                next_id += 1
        still = []
        for tr in live:
            (done if tr.misses > max_age else still).append(tr)
        live = still
    return [tr for tr in done + live if len(tr) >= min_len]
