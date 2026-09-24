"""Video decoding and frame sampling.

Everything downstream sees frames as `uint8` RGB arrays of shape (H, W, 3) plus
a timestamp in seconds, so the rest of CineScope never cares which decoder ran.

Two backends behind one interface:

  PyAV    (default) - FFmpeg bindings. Frame-accurate timestamps from the
          container, seeking by time, any codec FFmpeg knows (the .mkv/H.264
          episodes in trailers/ included).
  OpenCV  (fallback) - `cv2.VideoCapture`. Available almost everywhere, but its
          timestamps are derived from a frame counter and the nominal fps,
          which drifts on variable-frame-rate files. Good enough for tests.

Why sample at all: a 2-hour film at 24 fps is ~170k frames. Shot detection
needs every frame (cuts happen between two adjacent frames); palettes and
thumbnails are fine at 2-4 fps. `sample_fps` makes that a parameter instead of
a copy-pasted `if i % n` in every script.
"""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class VideoInfo:
    path: str
    width: int
    height: int
    fps: float
    duration_s: float
    backend: str

    @property
    def n_frames_estimate(self) -> int:
        return int(round(self.fps * self.duration_s))


@dataclass
class Frame:
    index: int          # index among *decoded* frames (0-based, before sampling)
    t: float            # presentation time in seconds
    rgb: np.ndarray     # (H, W, 3) uint8


def _resize(rgb: np.ndarray, width: int | None) -> np.ndarray:
    if not width or rgb.shape[1] == width:
        return rgb
    from PIL import Image

    h = max(1, round(rgb.shape[0] * width / rgb.shape[1]))
    return np.asarray(Image.fromarray(rgb).resize((width, h), Image.BILINEAR))


def _have_av() -> bool:
    try:
        import av  # noqa: F401
    except ImportError:
        return False
    return True


class VideoReader:
    """Iterate frames of a video file.

        with VideoReader("data/clips/trailer.mp4", width=320) as vr:
            print(vr.info)
            for f in vr.frames(sample_fps=2, start_s=10, end_s=40):
                ...

    width: resize every frame to this width (aspect kept). Shot detection and
           palettes work on ~320 px; nothing in Phase 0 needs full resolution.
    """

    def __init__(self, path: str | Path, width: int | None = None, backend: str = "auto") -> None:
        self.path = str(path)
        if not Path(self.path).exists():
            raise FileNotFoundError(self.path)
        self.width = width
        if backend == "auto":
            backend = "av" if _have_av() else "opencv"
        if backend not in ("av", "opencv"):
            raise ValueError(f"unknown backend {backend!r}")
        self.backend = backend
        self._container = None
        self._cap = None
        self.info = self._probe()

    # ------------------------------------------------------------ probing
    def _probe(self) -> VideoInfo:
        if self.backend == "av":
            import av

            self._container = av.open(self.path)
            stream = self._container.streams.video[0]
            fps = float(stream.average_rate or stream.guessed_rate or 25)
            if stream.duration is not None and stream.time_base is not None:
                duration = float(stream.duration * stream.time_base)
            elif self._container.duration is not None:
                duration = self._container.duration / 1_000_000   # microseconds
            else:
                duration = 0.0
            return VideoInfo(self.path, stream.codec_context.width, stream.codec_context.height,
                             fps, duration, "av")
        import cv2

        self._cap = cv2.VideoCapture(self.path)
        if not self._cap.isOpened():
            raise OSError(f"OpenCV could not open {self.path}")
        fps = self._cap.get(cv2.CAP_PROP_FPS) or 25.0
        n = self._cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return VideoInfo(self.path, w, h, float(fps), float(n / fps) if n else 0.0, "opencv")

    # ------------------------------------------------------------ decoding
    def _raw(self, start_s: float) -> Iterator[tuple[int, float, np.ndarray]]:
        if self.backend == "av":
            assert self._container is not None
            stream = self._container.streams.video[0]
            stream.thread_type = "AUTO"            # multi-threaded decode
            if start_s > 0:
                # Seek lands on the keyframe *before* start_s; frames are then
                # decoded and skipped up to start_s, so the result is exact.
                self._container.seek(int(start_s / stream.time_base), stream=stream,
                                     backward=True, any_frame=False)
            fps = self.info.fps
            for frame in self._container.decode(stream):
                t = float(frame.pts * stream.time_base) if frame.pts is not None else 0.0
                yield int(round(t * fps)), t, frame.to_ndarray(format="rgb24")
            return
        import cv2

        assert self._cap is not None
        fps = self.info.fps
        if start_s > 0:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_s * fps))
        i = int(self._cap.get(cv2.CAP_PROP_POS_FRAMES))
        while True:
            ok, bgr = self._cap.read()
            if not ok:
                return
            yield i, i / fps, bgr[:, :, ::-1].copy()
            i += 1

    def frames(self, sample_fps: float | None = None, start_s: float = 0.0,
               end_s: float | None = None) -> Iterator[Frame]:
        """Yield frames in [start_s, end_s). sample_fps=None yields every frame;
        otherwise the first frame at or after each 1/sample_fps tick."""
        step = None if not sample_fps else 1.0 / sample_fps
        next_t = start_s
        for idx, t, rgb in self._raw(start_s):
            if t < start_s - 1e-6:
                continue
            if end_s is not None and t >= end_s:
                break
            if step is not None:
                if t + 1e-6 < next_t:
                    continue
                # Advance to the first tick after t (skips ticks if frames are sparse).
                next_t += step * (int((t - next_t) / step) + 1)
            yield Frame(idx, t, _resize(rgb, self.width))

    def read_all(self, **kw) -> tuple[np.ndarray, np.ndarray]:
        """(frames[N,H,W,3] uint8, times[N]) — convenient for short clips only."""
        frames, times = [], []
        for f in self.frames(**kw):
            frames.append(f.rgb)
            times.append(f.t)
        if not frames:
            return np.zeros((0, 0, 0, 3), np.uint8), np.zeros(0)
        return np.stack(frames), np.asarray(times)

    # ------------------------------------------------------------ lifecycle
    def close(self) -> None:
        if self._container is not None:
            self._container.close()
            self._container = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> VideoReader:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def write_video(path: str | Path, frames: np.ndarray, fps: float = 24.0) -> None:
    """Write RGB uint8 frames to a video file (used to build synthetic test clips).
    PyAV when available, OpenCV otherwise."""
    path = str(path)
    frames = np.asarray(frames, dtype=np.uint8)
    h, w = frames.shape[1:3]
    if _have_av():
        import av
        from fractions import Fraction

        with av.open(path, "w") as out:
            s = out.add_stream("mpeg4", rate=Fraction(fps).limit_denominator(1001))
            s.width, s.height, s.pix_fmt = w, h, "yuv420p"
            s.bit_rate = 4_000_000
            for rgb in frames:
                for packet in s.encode(av.VideoFrame.from_ndarray(rgb, format="rgb24")):
                    out.mux(packet)
            for packet in s.encode():
                out.mux(packet)
        return
    import cv2

    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for rgb in frames:
        vw.write(np.ascontiguousarray(rgb[:, :, ::-1]))
    vw.release()
