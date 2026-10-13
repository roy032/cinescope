"""Glue between a video file and the geometry code: load one shot's frames,
track them, reconstruct. Works on any clip — shots come from Phase 0's
detector (cached in outputs/<clip>/shots.json when available)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..io.video import VideoReader
from .camera import intrinsics
from .selfcalib import focal_prior
from .sfm import Reconstruction, SfMParams, run_sfm
from .tracks import Tracks, track_shot


@dataclass
class ShotFrames:
    video: str
    shot: int
    start: int                  # first frame index of the shot in the clip
    end: int                    # last (inclusive)
    fps: float
    frames: np.ndarray          # (N, H, W, 3) uint8, every `stride`-th frame of the shot
    ids: list[int]              # their frame indices in the clip


def find_shots(video: str | Path, cache_dir: str | Path = "outputs") -> list[dict]:
    """Shots of a clip as [{"index", "start", "end", "t0", "t1", "camera"?}]."""
    cache = Path(cache_dir) / Path(video).stem / "shots.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))["shots"]
    from ..shots.pipeline import analyse

    res = analyse(video, with_camera=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    res.save(cache)
    return res.to_json()["shots"]


def load_shot(video: str | Path, shot: int, width: int = 480, stride: int = 1,
              max_frames: int | None = None, cache_dir: str | Path = "outputs",
              start: int | None = None, end: int | None = None) -> ShotFrames:
    """Frames of shot `shot` (or of the frame range start..end if given)."""
    if start is None or end is None:
        s = find_shots(video, cache_dir)[shot]
        start, end = int(s["start"]), int(s["end"])
    # skip a couple of frames at each end: dissolves and motion blur at cuts
    a, b = start + 2, end - 2
    if b <= a:
        a, b = start, end
    wanted = list(range(a, b + 1, stride))
    if max_frames and len(wanted) > max_frames:
        wanted = wanted[:max_frames]
    want = set(wanted)
    frames, ids = [], []
    with VideoReader(video, width=width) as vr:
        fps = vr.info.fps
        for f in vr.frames(start_s=max(0.0, (a - 1) / fps - 0.5)):
            if f.index in want:
                frames.append(f.rgb)
                ids.append(f.index)
            if f.index >= wanted[-1]:
                break
    return ShotFrames(str(video), shot, start, end, fps, np.stack(frames), ids)


def to_gray(frames: np.ndarray) -> np.ndarray:
    return frames.astype(np.float64) @ np.array([0.299, 0.587, 0.114])


def normalise_contrast(gray: np.ndarray) -> np.ndarray:
    """Same idea as Phase 0's camera classifier: a dark scene gets the same
    contrast as a bright one, so corner thresholds behave the same."""
    m, s = gray.mean(), gray.std() + 1e-6
    return (gray - m) / s * 50.0 + 128.0


def default_k(width: int, height: int, focal: float | None = None) -> np.ndarray:
    return intrinsics(focal or focal_prior(width, height), width / 2, height / 2)


def reconstruct(sf: ShotFrames, focal: float | None = None, params: SfMParams | None = None,
                max_corners: int = 800) -> tuple[Reconstruction, Tracks]:
    gray = normalise_contrast(to_gray(sf.frames))
    tracks = track_shot(gray, sf.ids, max_corners=max_corners)
    h, w = sf.frames.shape[1:3]
    k = default_k(w, h, focal)
    p = params or SfMParams(refine_focal=focal is None)
    import copy

    rec = run_sfm(copy.deepcopy(tracks), k, w, p)
    return rec, tracks
