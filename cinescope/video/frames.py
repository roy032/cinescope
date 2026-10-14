"""Grab frames at given times in one decoding pass."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..io.video import VideoReader


def frames_at(path: str | Path, times: list[float], width: int | None = None) -> list[np.ndarray]:
    """RGB frames nearest to each requested time (any order; duplicates allowed).

    Decodes the clip once from the earliest time and keeps, for each request,
    the first frame at or after it (the last frame if past the end)."""
    order = np.argsort(times, kind="stable")
    out: list[np.ndarray | None] = [None] * len(times)
    k = 0
    last = None
    with VideoReader(path, width=width) as vr:
        start = max(0.0, float(times[order[0]]) - 0.5) if len(times) else 0.0
        for f in vr.frames(start_s=start):
            last = f.rgb
            while k < len(order) and f.t + 1e-3 >= times[order[k]]:
                out[order[k]] = f.rgb
                k += 1
            if k == len(order):
                break
    for i in range(len(out)):
        if out[i] is None:
            out[i] = last
    return out  # type: ignore[return-value]


def clip_frames(path: str | Path, t0: float, t1: float, n: int, width: int | None = None) -> list[np.ndarray]:
    """n frames evenly spaced over [t0, t1)."""
    ts = list(t0 + (np.arange(n) + 0.5) * (t1 - t0) / n)
    return frames_at(path, ts, width)
