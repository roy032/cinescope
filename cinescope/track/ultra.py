"""Ultralytics' ByteTrack and BoT-SORT behind CineScope's tracker interface.

The library comparison for Week 9: same detections in, same Track objects out,
so the from-scratch tracker and the production ones are scored identically.
Ultralytics normally runs its trackers inside `model.track()`; here they are
fed our detections directly (the trackers only need `conf`, `cls`, `xywh`).
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from .tracker import Track


class _Dets:
    """The minimum of ultralytics' `Boxes` the trackers read."""

    def __init__(self, xyxy: np.ndarray, conf: np.ndarray):
        self.xyxy = np.asarray(xyxy, dtype=np.float32).reshape(-1, 4)
        self.conf = np.asarray(conf, dtype=np.float32).reshape(-1)
        self.cls = np.zeros(len(self.conf), np.float32)
        wh = self.xyxy[:, 2:] - self.xyxy[:, :2]
        self.xywh = np.c_[self.xyxy[:, :2] + wh / 2, wh]

    def __len__(self) -> int:
        return len(self.conf)

    def __getitem__(self, m) -> _Dets:
        return _Dets(self.xyxy[m], self.conf[m])


def _args(kind: str, **over) -> SimpleNamespace:
    a = {"tracker_type": kind, "track_high_thresh": 0.25, "track_low_thresh": 0.1, "new_track_thresh": 0.25,
             "track_buffer": 30, "match_thresh": 0.8, "fuse_score": True}
    if kind == "botsort":
        a.update(gmc_method="sparseOptFlow", proximity_thresh=0.5, appearance_thresh=0.8,
                 with_reid=False, model="auto")
    a.update(over)
    return SimpleNamespace(**a)


class UltralyticsTracker:
    """kind: "bytetrack" or "botsort". Defaults are ultralytics' own YAML files
    (`cfg/trackers/*.yaml`); pass keyword overrides to align thresholds."""

    def __init__(self, kind: str = "bytetrack", **over):
        from ultralytics.trackers.bot_sort import BOTSORT
        from ultralytics.trackers.byte_tracker import BYTETracker

        self.kind, self.over = kind, over
        self._cls = BYTETracker if kind == "bytetrack" else BOTSORT
        self._tracks: dict[int, Track] = {}
        self._shot = 0
        self._new()

    def _new(self) -> None:
        args = _args(self.kind, **self.over)
        if self.kind == "botsort" and args.gmc_method in (None, "none"):
            args.gmc_method = None
        self._trk = self._cls(args)

    def reset(self) -> None:
        self._shot += 1
        self._new()

    def update(self, frame: int, boxes: np.ndarray, scores: np.ndarray, img: np.ndarray | None = None):
        out = self._trk.update(_Dets(boxes, scores), img)
        res = []
        for row in np.asarray(out).reshape(-1, 8):
            tid = int(row[4]) + 1_000_000 * self._shot
            d = int(row[7])
            t = self._tracks.setdefault(tid, Track(tid, shot=self._shot))
            t.frames.append(frame)
            t.boxes.append(np.asarray(boxes[d], dtype=np.float64))
            t.scores.append(float(scores[d]))
            t.det_index.append(d)
            res.append((tid, d))
        return res

    def finish(self) -> list[Track]:
        return sorted(self._tracks.values(), key=lambda t: t.id)
