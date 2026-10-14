"""A ByteTrack-style face tracker, from scratch.

Tracking by detection: a detector runs on every frame; the tracker links the
boxes into tracks ("this face in frame 10 is the same face as in frame 9").

  1. Every track predicts its box in the new frame (Kalman filter).
  2. High-confidence detections are matched to all tracks by IoU with the
     Hungarian algorithm (SORT, Bewley et al. 2016).
  3. ByteTrack's idea (Zhang et al. 2022): *low*-confidence detections are not
     thrown away. They are matched, in a second round, to the tracks still
     unmatched. A face that turns to profile or blurs drops to score 0.3 for a
     few frames; a threshold-then-track pipeline loses it and starts a new
     identity afterwards, ByteTrack keeps the track alive through the dip. Low
     detections never *start* a track, so they add little noise.
  4. Unmatched tracks are kept "lost" for `max_lost` frames (predicted only) so
     a short occlusion does not split the track; new tracks start from
     unmatched high detections and count only after `min_hits` matches.

Films add one rule that surveillance trackers do not have: **a shot cut ends
every track.** Across a cut the camera is somewhere else, and IoU overlap
between a face before and a face after the cut is a coincidence, not
evidence. `Tracker.reset()` is called at every cut from Phase 0; linking the
same character across shots is the job of Week 10's face embeddings.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..detect.boxes import iou
from .hungarian import match
from .kalman import BoxKalman


@dataclass
class TrackerParams:
    high: float = 0.6        # detections at or above this start/extend tracks in round 1
    low: float = 0.3         # detections in [low, high) are only used in round 2
    new_track: float = 0.7   # a new track needs a detection this confident
    match_iou: float = 0.2   # round 1: IoU at least this (cost 1 - IoU <= 0.8, ByteTrack's default)
    low_iou: float = 0.5     # round 2 is stricter: low detections are noisier
    max_lost: int = 30       # frames a track survives without a detection (ByteTrack: 30 at 30 fps)
    min_hits: int = 2        # matches before a new track counts (filters one-frame false positives)
    fuse_score: bool = True  # ByteTrack: weight IoU by detection score in round 1


@dataclass
class Track:
    id: int
    frames: list[int] = field(default_factory=list)
    boxes: list[np.ndarray] = field(default_factory=list)     # detection boxes, xyxy
    scores: list[float] = field(default_factory=list)
    det_index: list[int] = field(default_factory=list)        # index into that frame's detections
    shot: int = 0

    def __len__(self) -> int:
        return len(self.frames)

    def to_dict(self) -> dict:
        return {"id": self.id, "shot": self.shot, "frames": self.frames,
                "boxes": np.round(np.asarray(self.boxes), 1).tolist(),
                "scores": np.round(self.scores, 3).tolist()}


class _Live:
    __slots__ = ("track", "kf", "hits", "lost", "confirmed")

    def __init__(self, track: Track, box: np.ndarray):
        self.track, self.kf = track, BoxKalman(box)
        self.hits, self.lost, self.confirmed = 1, 0, False


class Tracker:
    """    trk = Tracker()
        for frame_idx, (boxes, scores) in enumerate(detections):
            if frame_idx in cuts:
                trk.reset()
            active = trk.update(frame_idx, boxes, scores)   # [(track_id, det_index), ...]
        tracks = trk.finish()                               # every confirmed Track
    """

    def __init__(self, params: TrackerParams | None = None):
        self.p = params or TrackerParams()
        self._live: list[_Live] = []
        self._done: list[Track] = []
        self._next_id = 0
        self._shot = 0
        self._first_frame = True

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """End every track (call at a shot cut)."""
        self._retire(self._live)
        self._live = []
        self._shot += 1
        self._first_frame = True

    def _retire(self, lives: list[_Live]) -> None:
        for lv in lives:
            if lv.confirmed:
                self._done.append(lv.track)

    def _cost(self, lives: list[_Live], pred: np.ndarray, boxes: np.ndarray, scores: np.ndarray,
              fuse: bool) -> np.ndarray:
        if not lives or not len(boxes):
            return np.zeros((len(lives), len(boxes)))
        sim = iou(pred, boxes)
        if fuse:
            sim = sim * scores[None, :]
        return 1.0 - sim

    def update(self, frame: int, boxes: np.ndarray, scores: np.ndarray) -> list[tuple[int, int]]:
        """Feed one frame's detections (xyxy, scores). Returns (track id, detection
        index) for every confirmed track that took a detection in this frame."""
        p = self.p
        boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        hi = np.flatnonzero(scores >= p.high)
        lo = np.flatnonzero((scores >= p.low) & (scores < p.high))

        pred = np.array([lv.kf.predict() for lv in self._live]).reshape(-1, 4)
        assigned: dict[int, int] = {}      # live index -> detection index

        # Round 1: confirmed + lost tracks vs high detections.
        pool = [i for i, lv in enumerate(self._live) if lv.confirmed]
        c = self._cost([self._live[i] for i in pool], pred[pool], boxes[hi], scores[hi], p.fuse_score)
        pairs, un_t, un_d = match(c, 1.0 - p.match_iou)
        for a, b in pairs:
            assigned[pool[a]] = int(hi[b])
        rest_hi = hi[un_d]

        # Round 2: still-unmatched tracks that were seen last frame vs low detections.
        pool2 = [pool[a] for a in un_t if self._live[pool[a]].lost == 0]
        c = self._cost([self._live[i] for i in pool2], pred[pool2], boxes[lo], scores[lo], False)
        pairs, _, _ = match(c, 1.0 - p.low_iou)
        for a, b in pairs:
            assigned[pool2[a]] = int(lo[b])

        # Round 3: tentative tracks vs the high detections nobody took.
        pool3 = [i for i, lv in enumerate(self._live) if not lv.confirmed]
        c = self._cost([self._live[i] for i in pool3], pred[pool3], boxes[rest_hi], scores[rest_hi],
                       p.fuse_score)
        pairs, _, un_d3 = match(c, 1.0 - p.match_iou)
        for a, b in pairs:
            assigned[pool3[a]] = int(rest_hi[b])
        rest_hi = rest_hi[un_d3]

        out: list[tuple[int, int]] = []
        keep: list[_Live] = []
        for i, lv in enumerate(self._live):
            if i in assigned:
                d = assigned[i]
                lv.kf.update(boxes[d])
                lv.hits += 1
                lv.lost = 0
                t = lv.track
                t.frames.append(frame)
                t.boxes.append(boxes[d].copy())
                t.scores.append(float(scores[d]))
                t.det_index.append(d)
                if not lv.confirmed and lv.hits >= p.min_hits:
                    lv.confirmed = True
                if lv.confirmed:
                    out.append((t.id, d))
                keep.append(lv)
            else:
                lv.lost += 1
                if lv.confirmed and lv.lost <= p.max_lost:
                    keep.append(lv)
                else:
                    self._retire([lv])           # tentative tracks die on their first miss
        self._live = keep

        # New tracks. On the first frame of a shot they are confirmed at once
        # (ByteTrack does the same), so a face visible from the cut is not lost.
        for d in rest_hi:
            if scores[d] < p.new_track:
                continue
            t = Track(self._next_id, [frame], [boxes[d].copy()], [float(scores[d])], [int(d)], self._shot)
            self._next_id += 1
            lv = _Live(t, boxes[d])
            if self._first_frame or p.min_hits <= 1:
                lv.confirmed = True
                out.append((t.id, int(d)))
            self._live.append(lv)
        self._first_frame = False
        return out

    def finish(self) -> list[Track]:
        """End the clip: every confirmed track, ordered by id."""
        self._retire(self._live)
        self._live = []
        return sorted(self._done, key=lambda t: t.id)


def track_detections(dets: list[tuple[np.ndarray, np.ndarray]], cuts: set[int] | None = None,
                     params: TrackerParams | None = None) -> list[Track]:
    """Track a whole sequence: dets[k] = (boxes xyxy, scores) of frame k; `cuts`
    holds the frame numbers that start a new shot."""
    trk = Tracker(params)
    cuts = cuts or set()
    for k, (b, s) in enumerate(dets):
        if k in cuts and k > 0:
            trk.reset()
        trk.update(k, b, s)
    return trk.finish()
