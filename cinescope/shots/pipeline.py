"""Phase 0 end to end: video -> shots (+ palettes, barcode, camera moves).

    from cinescope.shots.pipeline import analyse
    result = analyse("trailers/clip.mkv")

Everything is computed from one decoding pass at `width` pixels (160 by
default): shot detection needs every frame, but not the pixels of a full HD one.
Small thumbnails (96 px) of every frame are kept for palettes and camera
motion — about 20 KB a frame, fine for clips; a full film would use a second
decoding pass instead (Phase 9 work).
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..io.video import VideoReader
from ..palette.kmeans import barcode, palette
from . import cuts, features


@dataclass
class Analysis:
    video: str
    fps: float
    n_frames: int
    transitions: list[cuts.Transition]
    shots: list[cuts.Shot]
    palettes: list[dict] = field(default_factory=list)      # per shot
    barcode: np.ndarray | None = None                      # (1, n_cols, 3) uint8
    camera: list[dict] = field(default_factory=list)       # per shot (Week 3)
    framing: list[dict] = field(default_factory=list)      # per shot (Phase 1, needs a checkpoint)
    seconds: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return {
            "video": self.video, "fps": self.fps, "n_frames": self.n_frames,
            "transitions": [t.to_dict() for t in self.transitions],
            "shots": [
                {**s.to_dict(),
                 **({"palette": self.palettes[i]} if i < len(self.palettes) else {}),
                 **({"camera": self.camera[i]} if i < len(self.camera) else {}),
                 **({"framing": self.framing[i]} if i < len(self.framing) else {})}
                for i, s in enumerate(self.shots)
            ],
            "timing_s": self.seconds,
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_json(), indent=1), encoding="utf-8")


def analyse(video: str | Path, width: int = 160, thumb_width: int = 96, params: cuts.CutParams | None = None,
            with_palettes: bool = True, with_camera: bool = False, k: int = 5,
            barcode_fps: float = 1.0) -> Analysis:
    t0 = time.perf_counter()
    with VideoReader(video, width=width) as vr:
        fps = vr.info.fps
        feat = features.extract(vr.frames(), keep_thumbs=True, thumb_width=thumb_width)
    t1 = time.perf_counter()
    transitions = cuts.detect_transitions(feat, params)
    shots = cuts.shots_from_transitions(transitions, len(feat), feat.times)
    t2 = time.perf_counter()
    result = Analysis(str(video), fps, len(feat), transitions, shots,
                      seconds={"decode+features": round(t1 - t0, 2), "boundaries": round(t2 - t1, 2)})
    if with_palettes and feat.thumbs is not None:
        step = max(1, int(round(fps / 2)))                # 2 frames per second of shot
        for s in shots:
            frames = feat.thumbs[s.start:s.end + 1:step]
            p = palette(frames, k=min(k, max(1, frames.shape[0] * 4)))
            result.palettes.append({"rgb": p.rgb.tolist(), "weights": np.round(p.weights, 4).tolist()})
        cols = feat.thumbs[:: max(1, int(round(fps / barcode_fps)))]
        result.barcode = barcode(cols, height=1)
        result.seconds["palettes"] = round(time.perf_counter() - t2, 2)
    if with_camera and feat.thumbs is not None:
        from ..motion.camera import classify_shot

        t3 = time.perf_counter()
        gray = feat.thumbs.astype(np.float64).mean(-1)
        for s in shots:
            result.camera.append(classify_shot(gray[s.start:s.end + 1], fps).to_dict())
        result.seconds["camera"] = round(time.perf_counter() - t3, 2)
    return result
