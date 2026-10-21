"""Editing grammar: how the cuts are put together, in an editor's words.

    pacing        cuts per minute over time (a sliding window over the cut times),
                  and where the cutting is fastest
    setups        shots filmed from the same camera set-up: the editor cuts back to
                  the same angle, so the two shots look alike (same layout of light
                  and dark, same colours) even when far apart in time
    shot/reverse-shot
                  a run of at least four shots alternating between two set-ups,
                  A B A B: the classic coverage of a conversation
    montage       a run of at least five consecutive shots much shorter than the
                  clip's own median shot
    long take     a shot at least four times the median shot and 15 s or longer

A shot's descriptor is a coarse 8 x 6 luma layout (z-normalised, so exposure
changes do not matter) and a hue / saturation / value histogram of a few of its
thumbnails. Two shots are the same set-up when both the layouts correlate and
the histograms overlap strongly. Thresholds are round numbers fixed up front, not
fitted to any clip.
"""
from __future__ import annotations

import numpy as np

from ..imgproc.color import rgb_to_hsv

LAYOUT = (6, 8)          # rows, columns of the luma layout
LAYOUT_CORR = 0.8        # same set-up: layouts correlate at least this much ...
HIST_OVERLAP = 0.6       # ... and colour histograms intersect at least this much
WINDOW = 12              # only compare shots this many apart (set-ups recur within a scene)
MIN_EXCHANGE = 4         # shots in a shot/reverse-shot run
MIN_MONTAGE = 5          # shots in a montage run
LONG_TAKE = (4.0, 15.0)  # x median shot, and seconds


def shot_descriptor(frames: np.ndarray) -> dict:
    """frames: (N, H, W, 3) uint8 thumbnails of one shot."""
    f = np.asarray(frames)
    f = f.astype(np.float64) / 255.0 if f.dtype == np.uint8 else f.astype(np.float64)
    if f.ndim == 3:
        f = f[None]
    mean = f.mean(0)
    y = mean @ np.array([0.2126, 0.7152, 0.0722])
    rows, cols = LAYOUT
    h, w = y.shape
    ys = np.linspace(0, h, rows + 1).astype(int)
    xs = np.linspace(0, w, cols + 1).astype(int)
    lay = np.array([[y[ys[i]:max(ys[i + 1], ys[i] + 1), xs[j]:max(xs[j + 1], xs[j] + 1)].mean()
                     for j in range(cols)] for i in range(rows)]).ravel()
    lay = lay - lay.mean()
    n = np.linalg.norm(lay)
    lay = lay / n if n > 1e-6 else np.zeros_like(lay)
    hsv = rgb_to_hsv(f).reshape(-1, 3)
    hs, _, _ = np.histogram2d(hsv[:, 0], hsv[:, 1], bins=(12, 4), range=((0, 360), (0, 1)))
    hv, _ = np.histogram(hsv[:, 2], bins=8, range=(0, 1))
    hist = np.concatenate([hs.ravel() / max(hs.sum(), 1), hv / max(hv.sum(), 1)]) / 2
    return {"layout": lay, "hist": hist, "flat": n <= 1e-6}


def same_setup(a: dict, b: dict) -> bool:
    if a["flat"] or b["flat"]:
        return False                       # a black frame or a flat card says nothing about the camera
    corr = float(a["layout"] @ b["layout"])
    overlap = float(np.minimum(a["hist"], b["hist"]).sum())
    return corr >= LAYOUT_CORR and overlap >= HIST_OVERLAP


def find_setups(descs: list[dict], window: int = WINDOW) -> list[int]:
    """Set-up id per shot (0, 1, 2 ... in order of first appearance); shots linked by
    same_setup within `window` shots of each other share an id (union-find)."""
    n = len(descs)
    parent = list(range(n))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, min(n, i + window + 1)):
            if same_setup(descs[i], descs[j]):
                parent[root(j)] = root(i)
    ids: dict[int, int] = {}
    return [ids.setdefault(root(i), len(ids)) for i in range(n)]


def exchanges(setups: list[int], min_len: int = MIN_EXCHANGE) -> list[tuple[int, int]]:
    """Maximal runs [i, j] (shot positions, inclusive) that alternate A B A B ..."""
    out = []
    i, n = 0, len(setups)
    while i < n - 1:
        if setups[i] == setups[i + 1]:
            i += 1
            continue
        j = i + 1
        while j + 1 < n and setups[j + 1] == setups[j - 1] and setups[j + 1] != setups[j]:
            j += 1
        if j - i + 1 >= min_len:
            out.append((i, j))
            i = j
        else:
            i += 1
    return out


def montages(durations: list[float], min_len: int = MIN_MONTAGE) -> list[tuple[int, int]]:
    """Runs [i, j] of consecutive shots shorter than 0.6 x the median shot (kept
    between 0.75 s and 2 s, so a clip of only fast cuts is not one long montage)."""
    if not durations:
        return []
    thr = float(np.clip(0.6 * np.median(durations), 0.75, 2.0))
    out, start = [], None
    for k, d in enumerate([*durations, np.inf]):
        if d < thr:
            start = k if start is None else start
        else:
            if start is not None and k - start >= min_len:
                out.append((start, k - 1))
            start = None
    return out


def long_takes(durations: list[float]) -> list[int]:
    if not durations:
        return []
    thr = max(LONG_TAKE[0] * float(np.median(durations)), LONG_TAKE[1])
    return [k for k, d in enumerate(durations) if d >= thr]


def pacing_curve(cut_times: list[float], duration: float, window: float | None = None,
                 points: int = 200) -> dict:
    """Cuts per minute in a centred window, sampled at `points` times. The window is
    30 s, or a third of the clip when that is shorter (but at least 2 s)."""
    if duration <= 0:
        return {"t": [], "cpm": [], "window_s": 0.0}
    win = window or max(2.0, min(30.0, duration / 3))
    t = np.linspace(0, duration, max(2, points))
    c = np.sort(np.asarray(cut_times, np.float64))
    lo = np.clip(t - win / 2, 0, None)
    hi = np.clip(t + win / 2, None, duration)
    count = np.searchsorted(c, hi, "right") - np.searchsorted(c, lo, "left")
    span = np.maximum(hi - lo, 1e-6)
    cpm = 60.0 * count / span
    return {"t": np.round(t, 3).tolist(), "cpm": np.round(cpm, 2).tolist(), "window_s": round(win, 2)}


def analyse(shots: list[dict], descs: list[dict], duration: float) -> dict:
    """Everything above for one clip. shots: the pipeline's shot dicts (t0, t1, duration_s)."""
    setups = find_setups(descs)
    size: dict[int, int] = {}
    for s in setups:
        size[s] = size.get(s, 0) + 1
    durs = [float(s["duration_s"]) for s in shots]
    ex, mo, lt = exchanges(setups), montages(durs), long_takes(durs)
    timeline = ([{"t0": shots[i]["t0"], "t1": shots[j]["t1"], "label": "shot/reverse-shot", "shots": j - i + 1}
                 for i, j in ex]
                + [{"t0": shots[i]["t0"], "t1": shots[j]["t1"], "label": "montage", "shots": j - i + 1} for i, j in mo]
                + [{"t0": shots[k]["t0"], "t1": shots[k]["t1"], "label": "long take", "shots": 1} for k in lt])
    timeline.sort(key=lambda e: e["t0"])
    pace = pacing_curve([s["t0"] for s in shots[1:]], duration)
    peak = int(np.argmax(pace["cpm"])) if pace["cpm"] else None
    per_shot = [{"setup": s, "recurs": size[s] > 1} for s in setups]
    return {
        "per_shot": per_shot,
        "timeline": timeline,
        "pacing": pace,
        "summary": {
            "setups": len(size),
            "recurring_setups": sum(1 for v in size.values() if v > 1),
            "shots_in_recurring_setups": sum(v for v in size.values() if v > 1),
            "exchanges": len(ex),
            "montages": len(mo),
            "long_takes": len(lt),
            "fastest_t": pace["t"][peak] if peak is not None else None,
            "fastest_cpm": pace["cpm"][peak] if peak is not None else None,
        },
    }
