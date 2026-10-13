"""Where is text in a frame? A from-scratch text map and line finder.

Burned-in subtitles and credits share three properties that ordinary image
content rarely has at once, and the map tests for each:

  1. thin strokes much brighter (or darker) than their surroundings — the
     morphological top-hat, Y - opening(Y), keeps bright structures narrower
     than the structuring element and removes everything wider (sky, walls,
     faces);
  2. many short on/off alternations along a row — a word is a comb of
     strokes; a lamp post or a horizon is one long edge;
  3. a horizontal layout — candidate pixels are merged along rows, and lines
     are found by row and then column projection profiles.

Everything here is NumPy (min/max filters as sliding windows); no OpenCV.
Temporal stability (subtitles stay still while the picture moves) is used
one level up, in cinescope.ocr.subtitles.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def _filt1d(a: np.ndarray, k: int, axis: int, fn) -> np.ndarray:
    """Running min/max over a window of k along `axis`, edge-padded."""
    if k <= 1:
        return a
    pad = [(0, 0)] * a.ndim
    pad[axis] = (k // 2, k - 1 - k // 2)
    win = sliding_window_view(np.pad(a, pad, mode="edge"), k, axis=axis)
    return fn(win, axis=-1)


def erode(a, kh, kw):
    return _filt1d(_filt1d(a, kh, 0, np.min), kw, 1, np.min)


def dilate(a, kh, kw):
    return _filt1d(_filt1d(a, kh, 0, np.max), kw, 1, np.max)


def box_mean_rows(a: np.ndarray, k: int) -> np.ndarray:
    """Mean over a horizontal window of k pixels (cumulative sums)."""
    pad = np.pad(a.astype(np.float32), ((0, 0), (k // 2 + 1, k - 1 - k // 2)), mode="edge")
    c = np.cumsum(pad, axis=1)
    return (c[:, k:] - c[:, :-k]) / k


def luma(rgb: np.ndarray) -> np.ndarray:
    a = np.asarray(rgb, dtype=np.float32)
    return a if a.ndim == 2 else a[..., 0] * 0.299 + a[..., 1] * 0.587 + a[..., 2] * 0.114


@dataclass
class TextParams:
    """Thresholds of the text map. Defaults were chosen on the tuning clips
    (people-detection, classroom with burned Pride and Prejudice lines)."""
    stroke_k: float = 0.025      # top-hat element size, as a fraction of the frame height
    tophat_thr: float = 28.0     # grey levels above the local opening
    near: float = 45.0           # a stroke is within this many grey levels of its neighbourhood's peak
    win: float = 0.05            # horizontal window for density/transitions, fraction of frame height
    min_density: float = 0.12    # fraction of stroke pixels in the window
    max_density: float = 0.85
    min_trans: float = 0.06      # on/off transitions per pixel in the window
    min_h: float = 0.022         # line height range, fraction of frame height
    max_h: float = 0.11
    min_aspect: float = 1.8      # line width / height
    polarity: str = "bright"     # "bright", "dark" or "both"


def stroke_map(y: np.ndarray, frame_h: int, p: TextParams) -> np.ndarray:
    """Binary map of thin bright (and/or dark) strokes.

    Top-hat alone also fires on background *next to* an outlined letter: the
    opening dips where it meets the dark outline, so a light wall between two
    letters looks like a bright stroke and whole words become solid blobs. A
    stroke pixel must therefore also be nearly as bright as the brightest
    pixel around it (within `near` grey levels) — true of the letter itself,
    not of the wall beside it."""
    k = max(3, int(round(p.stroke_k * frame_h)) | 1)
    out = np.zeros(y.shape, bool)
    if p.polarity in ("bright", "both"):
        top = (y - dilate(erode(y, k, k), k, k)) > p.tophat_thr            # white top-hat
        out |= top & (y >= dilate(y, k, k) - p.near)
    if p.polarity in ("dark", "both"):
        top = (erode(dilate(y, k, k), k, k) - y) > p.tophat_thr            # black top-hat
        out |= top & (y <= erode(y, k, k) + p.near)
    return out


def candidate_map(strokes: np.ndarray, frame_h: int, p: TextParams) -> np.ndarray:
    """Pixels inside a comb of strokes: stroke density in a moderate range and
    many on/off transitions along the row."""
    k = max(5, int(round(p.win * frame_h)))
    s = strokes.astype(np.float32)
    density = box_mean_rows(s, k)
    trans = np.zeros_like(s)
    trans[:, 1:] = np.abs(np.diff(s, axis=1))
    trans = box_mean_rows(trans, k)
    cand = (density > p.min_density) & (density < p.max_density) & (trans > p.min_trans)
    return dilate(cand.astype(np.uint8), 1, k // 2 | 1).astype(bool)


def _runs(flags: np.ndarray, max_gap: int = 0) -> list[tuple[int, int]]:
    """[start, end) runs of True, bridging gaps of up to max_gap False values."""
    idx = np.flatnonzero(flags)
    if not idx.size:
        return []
    runs, s, prev = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - prev - 1 > max_gap:
            runs.append((int(s), int(prev) + 1))
            s = i
        prev = i
    runs.append((int(s), int(prev) + 1))
    return runs


def split_rows(prof: np.ndarray, min_h: float, core: float = 0.4) -> list[tuple[int, int]]:
    """Split a band of rows into text lines using its stroke row profile.

    Two subtitle lines often sit so close that their candidate rows touch. Each
    line still has a dense *core* — the x-height zone, where every letter has
    strokes — while ascenders, descenders and the gap between lines are thin.
    Rows above `core` × the peak form cores; with two or more cores (each at
    least a third of a line tall), the band is cut at the emptiest row between
    neighbouring cores. One line has one core, however its profile steps down
    to the ascender zone."""
    h = len(prof)
    if h < 1.6 * min_h or not prof.max(initial=0) > 0:
        return [(0, h)]
    cores = [(a, b) for a, b in _runs(prof >= core * prof.max(), max_gap=1) if b - a >= min_h / 3]
    if len(cores) < 2:
        return [(0, h)]
    cuts = [c0[1] + int(np.argmin(prof[c0[1]:c1[0]])) if c1[0] > c0[1] else c0[1]
            for c0, c1 in zip(cores, cores[1:], strict=False)]
    edges = [0, *cuts, h]
    return [(a, b) for a, b in zip(edges, edges[1:], strict=False) if b > a]


def find_lines(cand: np.ndarray, frame_h: int, p: TextParams,
               strokes: np.ndarray | None = None) -> list[tuple[int, int, int, int]]:
    """Text line boxes (x0, y0, x1, y1) from a candidate map, by projections:
    rows with enough candidates form bands, bands are split into lines at
    stroke-profile valleys; inside a line, columns with candidates form
    segments (gaps shorter than ~0.8 line height — a word space — are bridged)."""
    hmin, hmax = p.min_h * frame_h, p.max_h * frame_h
    rows = cand.sum(1)
    boxes = []
    for y0, y1 in _runs(rows >= max(3, p.min_aspect * hmin), max_gap=1):
        if y1 - y0 < hmin * 0.6:
            continue
        src = strokes if strokes is not None else cand
        cols_any = cand[y0:y1].any(0)
        prof = src[y0:y1][:, cols_any].sum(1).astype(float)
        for a, b in split_rows(prof, hmin):
            piece = prof[a:b]
            if b - a > hmax and piece.max() > 0:
                # texture touching the line (a collar, a sweater) made the band
                # too tall: keep the line's core plus room for ascenders and descenders
                c0, c1 = max(_runs(piece >= 0.4 * piece.max(), max_gap=1), key=lambda r: r[1] - r[0])
                ext = int(round(0.45 * (c1 - c0)))
                a, b = a + max(0, c0 - ext), a + min(b - a, c1 + ext)
            b0, b1 = y0 + a, y0 + b
            bh = b1 - b0
            if not (hmin * 0.6 <= bh <= hmax):
                continue
            cols = cand[b0:b1].sum(0) >= max(1, 0.2 * bh)
            for x0, x1 in _runs(cols, max_gap=int(0.8 * bh)):
                if x1 - x0 >= p.min_aspect * bh:
                    boxes.append((x0, b0, x1, b1))
    return boxes


def detect_lines(rgb: np.ndarray, p: TextParams | None = None, frame_h: int | None = None):
    """(boxes, stroke map) for one image. frame_h: the height of the full frame
    this image (e.g. a band of it) comes from, which sets the expected scale."""
    p = p or TextParams()
    y = luma(rgb)
    fh = frame_h or y.shape[0]
    if p.polarity != "both":
        strokes = stroke_map(y, fh, p)
        cand = candidate_map(strokes, fh, p)
        return find_lines(cand, fh, p, strokes), strokes
    # each polarity separately: on light-on-dark text the *dark* top-hat marks
    # the gaps between letters, and the union of both would be a solid block
    out_s, out_c = np.zeros(y.shape, bool), np.zeros(y.shape, bool)
    for pol in ("bright", "dark"):
        q = replace(p, polarity=pol)
        s = stroke_map(y, fh, q)
        c = candidate_map(s, fh, q)
        out_c |= c
        out_s |= s & c
    return find_lines(out_c, fh, p, out_s), out_s


def pad_box(box, h_img: int, w_img: int, fy: float = 0.18, fx: float = 0.35):
    x0, y0, x1, y1 = box
    h = y1 - y0
    return (max(0, int(x0 - fx * h)), max(0, int(y0 - fy * h)),
            min(w_img, int(x1 + fx * h)), min(h_img, int(y1 + fy * h)))
