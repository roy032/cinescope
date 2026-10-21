"""Story structure: scenes, the three acts, turning points and the climax.

    scenes        runs of shots in one place and time. A cut starts a new scene
                  when no camera set-up is shared across it (cinescope.cinema.editing)
                  and the colours of the shots either side of it do not overlap,
                  or when the soundtrack drops to silence across it and the colours
                  differ less strongly
    intensity     a 0..1 curve over the clip: the mean of three rank-normalised
                  signals, the cutting rate, the loudness and the picture's own
                  motion (mean frame difference), smoothed over a twentieth of the
                  clip. Rank normalisation needs no units and no tuning
    turning points
                  the classic beats of a three-act structure, each searched in its
                  conventional window of the running time and placed at the scene
                  change (else cut) where the intensity changes most:
                      plot point 1   15-35 %    end of act 1, the story turns
                      midpoint       40-60 %
                      plot point 2   65-90 %    end of act 2
                  and the climax is the intensity peak in the last 40 %

The windows are the screenwriting conventions (Syd Field's paradigm), not
fitted to any clip; on a trailer or a short the "acts" describe its shape, not
a plot, and the summary says so (structure: "short").
"""
from __future__ import annotations

import numpy as np

CONTEXT = 3                # shots each side of a cut that are compared
SCENE_SIM = 0.45           # colour overlap below this across a cut: a new scene
SILENT_SIM = 0.6           # ... or below this when the cut is in a silence
MIN_SCENE_SHOTS = 2
MIN_SCENE_S = 4.0
WINDOWS = {"plot point 1": (0.15, 0.35), "midpoint": (0.40, 0.60), "plot point 2": (0.65, 0.90)}
CLIMAX_FROM = 0.6
SHORT_S = 300.0            # under five minutes the acts describe a shape, not a plot


def colour_overlap(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.minimum(a, b).sum())


def cut_similarity(hists: list[np.ndarray], i: int, k: int = CONTEXT) -> float:
    """Best colour overlap between the k shots before cut i (between shots i-1 and i) and the k after."""
    before = hists[max(0, i - k):i]
    after = hists[i:i + k]
    return max((colour_overlap(a, b) for a in before for b in after), default=1.0)


def _in_silence(t: float, silences: list[tuple[float, float]], pad: float = 0.25) -> bool:
    return any(a - pad <= t <= b + pad for a, b in silences)


def scene_starts(shots: list[dict], hists: list[np.ndarray], setups: list[int] | None = None,
                 silences: list[tuple[float, float]] = ()) -> list[int]:
    """Shot positions where a scene begins (always starting with 0)."""
    n = len(shots)
    cand = []
    for i in range(1, n):
        if setups is not None:
            before = set(setups[max(0, i - CONTEXT):i])
            if before & set(setups[i:i + CONTEXT]):
                continue                        # the editor cuts back to an angle used just before
        sim = cut_similarity(hists, i)
        thr = SILENT_SIM if _in_silence(float(shots[i]["t0"]), silences) else SCENE_SIM
        if sim < thr:
            cand.append((sim - thr, i))         # most negative: the clearest break
    starts = [0]
    for _, i in sorted(cand):
        t = float(shots[i]["t0"])
        ok = True
        for j in [*starts, n]:
            tj = float(shots[j]["t0"]) if j < n else float(shots[-1]["t1"])
            if abs(i - j) < MIN_SCENE_SHOTS or abs(t - tj) < MIN_SCENE_S:
                ok = False
                break
        if ok:
            starts.append(i)
    return sorted(starts)


def _ranks(v: np.ndarray) -> np.ndarray:
    """Values -> 0..1 by rank (ties share their mean rank); a constant signal gives 0.5."""
    v = np.asarray(v, float)
    if len(v) < 2 or np.ptp(v) < 1e-12:
        return np.full(len(v), 0.5)
    order = np.argsort(v, kind="stable")
    r = np.empty(len(v))
    r[order] = np.arange(len(v))
    for u in np.unique(v):
        m = v == u
        if m.sum() > 1:
            r[m] = r[m].mean()
    return r / (len(v) - 1)


def _smooth(v: np.ndarray, width: int) -> np.ndarray:
    if width <= 1 or len(v) < 3:
        return v
    k = np.hanning(width + 2)[1:-1]
    k /= k.sum()
    pad = width // 2
    return np.convolve(np.pad(v, pad, mode="edge"), k, mode="same")[pad:pad + len(v)]


def intensity(duration: float, cut_t: list[float], loud: tuple[np.ndarray, np.ndarray] | None = None,
              motion: tuple[np.ndarray, np.ndarray] | None = None, points: int = 400) -> dict:
    """The intensity curve on a regular grid. loud / motion: (times, values) samples."""
    dur = max(float(duration), 1e-3)
    n = int(max(8, min(points, round(dur * 2))))
    dt = dur / n
    t = (np.arange(n) + 0.5) * dt
    window = float(np.clip(dur / 20, 2.0, 90.0))
    cuts = np.sort(np.asarray(cut_t, float))
    lo, hi = t - window / 2, t + window / 2
    rate = (np.searchsorted(cuts, hi) - np.searchsorted(cuts, lo)) / window
    parts = {"cutting": rate}
    for name, sig in (("loudness", loud), ("motion", motion)):
        if sig is not None and len(sig[0]) >= 2:
            parts[name] = np.interp(t, np.asarray(sig[0], float), np.asarray(sig[1], float))
    w = max(1, int(round(window / dt)))
    comp = {k: _smooth(_ranks(v), w) for k, v in parts.items()}
    curve = np.mean(list(comp.values()), 0)
    curve = _smooth(curve, w)
    lo_, hi_ = float(curve.min()), float(curve.max())
    curve = (curve - lo_) / (hi_ - lo_) if hi_ - lo_ > 1e-9 else np.full(n, 0.5)
    return {"t": t, "value": curve, "dt": dt, "window_s": window, "signals": sorted(parts)}


def turning_points(curve: dict, duration: float, scene_t: list[float], cut_t: list[float] = ()) -> list[dict]:
    """The three structural beats and the climax (see the module docstring). Each beat
    is placed on a scene change in its window, else on a cut, else anywhere."""
    t, v = curve["t"], curve["value"]
    h = curve["window_s"]
    dur = max(float(duration), 1e-3)
    out = []
    for label, (a, b) in WINDOWS.items():
        lo, hi = a * dur, b * dur
        cand = [c for c in scene_t if lo <= c <= hi]
        snapped = "scene change" if cand else "cut"
        cand = cand or [c for c in cut_t if lo <= c <= hi]
        if not cand:
            snapped = None
            cand = list(t[(t >= lo) & (t <= hi)]) or [(lo + hi) / 2]
        change = [float(np.interp(c + h / 2, t, v) - np.interp(c - h / 2, t, v)) for c in cand]
        k = int(np.argmax(np.abs(change)))
        out.append({"label": label, "t": round(float(cand[k]), 2), "change": round(change[k], 3),
                    "rises": bool(change[k] >= 0), "on": snapped})
    tail = t >= CLIMAX_FROM * dur
    if tail.any():
        k = int(np.flatnonzero(tail)[np.argmax(v[tail])])
        out.append({"label": "climax", "t": round(float(t[k]), 2), "intensity": round(float(v[k]), 3)})
    return out


def analyse(shots: list[dict], hists: list[np.ndarray], duration: float, setups: list[int] | None = None,
            silences: list[tuple[float, float]] = (), loud: tuple | None = None,
            motion: tuple | None = None) -> dict:
    """Scenes, intensity, acts and beats for one clip. shots: the pipeline's shot dicts."""
    if not shots:
        return {"per_shot": [], "timeline": [], "scenes": [], "beats": [], "acts": [], "summary": {}}
    dur = float(duration) or float(shots[-1]["t1"])
    starts = scene_starts(shots, hists, setups, silences)
    bounds = [*starts, len(shots)]
    scenes = []
    for k, (a, b) in enumerate(zip(bounds[:-1], bounds[1:], strict=True)):
        scenes.append({"t0": shots[a]["t0"], "t1": shots[b - 1]["t1"], "label": "scene", "scene": k,
                       "first_shot": a, "shots": b - a})
    curve = intensity(dur, [s["t0"] for s in shots[1:]], loud, motion)
    beats = turning_points(curve, dur, [s["t0"] for s in scenes[1:]], [s["t0"] for s in shots[1:]])
    pp = {b["label"]: b["t"] for b in beats}
    edges = [0.0, pp["plot point 1"], pp["plot point 2"], dur]
    acts = [{"act": i + 1, "t0": round(edges[i], 2), "t1": round(edges[i + 1], 2),
             "share": round((edges[i + 1] - edges[i]) / dur, 3)} for i in range(3)]
    scene_of = np.repeat(np.arange(len(scenes)), [s["shots"] for s in scenes])
    per_shot = []
    for i, s in enumerate(shots):
        mid = (s["t0"] + s["t1"]) / 2
        act = 1 + int(mid >= edges[1]) + int(mid >= edges[2])
        per_shot.append({"scene": int(scene_of[i]), "act": act})
    sl = [s["t1"] - s["t0"] for s in scenes]
    v = curve["value"]
    k = max(1, len(v) // 10)
    summary = {
        "scenes": len(scenes),
        "mean_scene_s": round(float(np.mean(sl)), 2),
        "longest_scene_s": round(float(np.max(sl)), 2),
        "structure": "short" if dur < SHORT_S else "feature",
        "act_shares": [a["share"] for a in acts],
        "climax_t": next((b["t"] for b in beats if b["label"] == "climax"), None),
        # rising action: does the last tenth run hotter than the first tenth?
        "arc": "rising" if v[-k:].mean() > v[:k].mean() + 0.1 else
               "falling" if v[-k:].mean() < v[:k].mean() - 0.1 else "level",
        "signals": curve["signals"],
    }
    step = max(1, len(v) // 300)
    return {"per_shot": per_shot, "timeline": scenes, "scenes": scenes, "acts": acts, "beats": beats,
            "intensity": {"t0": round(float(curve["t"][0]), 3), "dt": round(curve["dt"] * step, 4),
                          "window_s": round(curve["window_s"], 2),
                          "value": np.round(v[::step], 3).tolist()},
            "summary": summary}
