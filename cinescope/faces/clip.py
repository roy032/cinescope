"""Clip -> face detections -> shot-aware tracks -> per-track face crops.

Two decoding passes, because what to crop depends on the finished tracks:

  pass 1  detect faces in every frame (the expensive part; cached to .npz),
          cut the clip into shots (Phase 0), track faces within each shot;
  pass 2  decode again and cut out, per track frame,
            - a 112x112 aligned face (grayscale every frame, for mouth motion;
              colour on up to `max_rgb` evenly spaced frames, for identity),
            - Light-ASD's 112x112 grayscale crop from median-smoothed boxes.

Storing grayscale for every frame and colour only for a sample keeps a
2-minute clip at ~100 MB; a feature film would stream pass 2 per shot.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..io.video import VideoReader
from ..track.tracker import Track, Tracker, TrackerParams
from .align import align, asd_crop
from .detectors import FaceDets


@dataclass
class Detections:
    video: str
    fps: float
    width: int
    height: int
    times: np.ndarray                 # (F,) seconds
    frames: list[FaceDets]

    def save(self, path: str | Path) -> None:
        n = np.array([len(d) for d in self.frames], np.int64)
        def cat(k, shape):
            return (np.concatenate([getattr(d, k) for d in self.frames]).reshape(shape)
                       if n.sum() else np.zeros(shape))
        np.savez_compressed(path, video=self.video, fps=self.fps, size=[self.width, self.height],
                            times=self.times, n=n, boxes=cat("boxes", (-1, 4)),
                            scores=cat("scores", (-1,)), landmarks=cat("landmarks", (-1, 5, 2)))

    @staticmethod
    def load(path: str | Path) -> Detections:
        z = np.load(path, allow_pickle=False)
        off = np.r_[0, np.cumsum(z["n"])]
        frames = [FaceDets(z["boxes"][a:b], z["scores"][a:b], z["landmarks"][a:b])
                  for a, b in zip(off[:-1], off[1:], strict=True)]
        w, h = z["size"]
        return Detections(str(z["video"]), float(z["fps"]), int(w), int(h), z["times"], frames)


def mmap_large_buffers() -> None:
    """Keep whole-frame buffers out of the malloc heap (glibc only).

    glibc serves big allocations with mmap at first, but after the first few
    are freed it raises its mmap threshold to their size, and from then on
    every decoded frame (2-6 MB) lands on the heap. Small arrays that we keep
    (detections, crops) get placed between them, the heap can never shrink,
    and memory grows by about one frame per frame: 4 GB after a few thousand
    1080p frames. Pinning the threshold at 128 KB keeps frames in mmap, which
    is returned to the system on free.
    """
    import contextlib
    import ctypes
    import sys

    if sys.platform.startswith("linux"):
        with contextlib.suppress(OSError, AttributeError):
            ctypes.CDLL("libc.so.6").mallopt(-3, 128 * 1024)       # M_MMAP_THRESHOLD


def detect_clip(video: str | Path, detector, cache: str | Path | None = None, log=print) -> Detections:
    if cache is not None and Path(cache).exists():
        return Detections.load(cache)
    mmap_large_buffers()
    t0 = time.perf_counter()
    frames, times = [], []
    with VideoReader(video) as vr:
        info = vr.info
        for f in vr.frames():
            frames.append(detector(f.rgb))
            times.append(f.t)
            if log and len(frames) % 500 == 0:
                log(f"  detected {len(frames)} frames ({time.perf_counter() - t0:.0f} s)")
    d = Detections(str(video), info.fps, info.width, info.height, np.asarray(times), frames)
    if cache is not None:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        d.save(cache)
    return d


def shot_starts(video: str | Path, n_frames: int | None = None) -> list[int]:
    """First frame of every shot (Phase 0's detector)."""
    from ..shots.pipeline import analyse

    a = analyse(video, with_palettes=False)
    return [s.start for s in a.shots]


def track_clip(dets: Detections, cuts: list[int] | set[int], params: TrackerParams | None = None,
               tracker=None) -> list[Track]:
    trk = tracker or Tracker(params)
    cuts = set(cuts)
    for k, fd in enumerate(dets.frames):
        if k in cuts and k > 0:
            trk.reset()
        trk.update(k, fd.boxes, fd.scores)
    return trk.finish()


@dataclass
class TrackCrops:
    track: Track
    gray: np.ndarray                  # (T, 112, 112) aligned grayscale, every track frame
    asd: np.ndarray                   # (T, 112, 112) Light-ASD crop, every track frame
    rgb: np.ndarray                   # (K, 112, 112, 3) aligned colour, K <= max_rgb
    rgb_frames: list[int] = field(default_factory=list)


def _medfilt(x: np.ndarray, k: int = 13) -> np.ndarray:
    from scipy.signal import medfilt

    k = min(k, len(x) if len(x) % 2 else len(x) - 1)
    return medfilt(x, k) if k >= 3 else x


def extract_crops(video: str | Path, dets: Detections, tracks: list[Track], max_rgb: int = 24,
                  min_len: int = 1) -> list[TrackCrops]:
    tracks = [t for t in tracks if len(t) >= min_len]
    need: dict[int, list[tuple[int, int]]] = {}       # frame -> [(track idx, position in track)]
    smooth = []
    for ti, t in enumerate(tracks):
        b = np.asarray(t.boxes)
        cx, cy = (b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2
        s = np.maximum(b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]) / 2
        cx, cy, s = _medfilt(cx), _medfilt(cy), _medfilt(s)
        smooth.append(np.c_[cx - s, cy - s, cx + s, cy + s])
        for k, f in enumerate(t.frames):
            need.setdefault(f, []).append((ti, k))
    rgb_pick = [set(np.linspace(0, len(t) - 1, min(max_rgb, len(t))).round().astype(int)) for t in tracks]
    gray = [np.zeros((len(t), 112, 112), np.uint8) for t in tracks]
    asd = [np.zeros((len(t), 112, 112), np.uint8) for t in tracks]
    rgb: list[list] = [[] for _ in tracks]
    rgb_frames: list[list] = [[] for _ in tracks]
    import cv2

    mmap_large_buffers()
    with VideoReader(video) as vr:
        for k, f in enumerate(vr.frames()):
            for ti, pos in need.get(k, ()):
                t = tracks[ti]
                lm = dets.frames[k].landmarks[t.det_index[pos]]
                a = align(f.rgb, lm)
                gray[ti][pos] = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
                asd[ti][pos] = asd_crop(f.rgb, smooth[ti][pos])
                if pos in rgb_pick[ti]:
                    rgb[ti].append(a)
                    rgb_frames[ti].append(k)
    return [TrackCrops(t, g, a, np.asarray(r, np.uint8).reshape(-1, 112, 112, 3), rf)
            for t, g, a, r, rf in zip(tracks, gray, asd, rgb, rgb_frames, strict=True)]


def save_crops(path: str | Path, crops: list[TrackCrops]) -> None:
    import json

    lens = np.array([len(c.gray) for c in crops], np.int64)
    nrgb = np.array([len(c.rgb) for c in crops], np.int64)

    def cat(xs, shape):
        return np.concatenate(xs) if xs else np.zeros(shape, np.uint8)

    np.savez(path, lens=lens, nrgb=nrgb,
             gray=cat([c.gray for c in crops], (0, 112, 112)), asd=cat([c.asd for c in crops], (0, 112, 112)),
             rgb=cat([c.rgb for c in crops], (0, 112, 112, 3)),
             rgb_frames=np.array([f for c in crops for f in c.rgb_frames], np.int64),
             tracks=json.dumps([{**c.track.to_dict(), "det_index": c.track.det_index} for c in crops]))


def load_crops(path: str | Path) -> list[TrackCrops]:
    import json

    z = np.load(path)
    out = []
    a = np.r_[0, np.cumsum(z["lens"])]
    r = np.r_[0, np.cumsum(z["nrgb"])]
    for i, d in enumerate(json.loads(str(z["tracks"]))):
        t = Track(d["id"], d["frames"], [np.asarray(b) for b in d["boxes"]], d["scores"], d["det_index"],
                  d["shot"])
        out.append(TrackCrops(t, z["gray"][a[i]:a[i + 1]], z["asd"][a[i]:a[i + 1]], z["rgb"][r[i]:r[i + 1]],
                              z["rgb_frames"][r[i]:r[i + 1]].tolist()))
    return out
