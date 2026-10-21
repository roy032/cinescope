"""Editing pace: the numbers film scholars and editors quote about a cut.

ASL (average shot length) and the median shot length are the standard
measures (Barry Salt's statistical style analysis, the Cinemetrics database);
the median is the steadier of the two, since a few long takes pull the mean up.
The cutting-rate curve is cuts per minute in a sliding window, so a chase
shows up as a peak and a dialogue scene played in one take as a trough.
"""
from __future__ import annotations

import numpy as np


def pacing(report: dict, n_points: int = 200) -> dict | None:
    """None when the report has fewer than two shots (nothing to measure)."""
    shots = report.get("shots") or []
    d = np.asarray([s["t1"] - s["t0"] for s in shots], float)
    d = d[d > 0]
    if len(shots) < 2 or not len(d):
        return None
    dur = float(report["info"]["duration_s"]) or float(d.sum())
    cuts = np.asarray([s["t0"] for s in shots[1:]], float)
    window = float(np.clip(dur / 12, 5.0, 120.0))
    ts = np.linspace(0.0, dur, n_points)
    window = min(window, dur)
    lo = np.clip(ts - window / 2, 0.0, max(dur - window, 0.0))   # a full window, slid inside the clip at its ends
    hi = lo + window
    counts = np.searchsorted(cuts, hi, "right") - np.searchsorted(cuts, lo, "left")
    rate = 60.0 * counts / np.maximum(hi - lo, 1e-6)
    fast = int(np.argmax(rate))
    slow = int(np.argmin(rate))
    edges = [0, 1, 2, 4, 8, 16, 32, float("inf")]
    hist = np.histogram(d, bins=edges)[0]
    return {
        "n_shots": len(shots),
        "asl_s": round(float(d.mean()), 3),
        "median_s": round(float(np.median(d)), 3),
        "std_s": round(float(d.std()), 3),
        "longest_s": round(float(d.max()), 3),
        "shortest_s": round(float(d.min()), 3),
        "cuts_per_min": round(60.0 * (len(shots) - 1) / dur, 3) if dur > 0 else 0.0,
        "window_s": round(window, 2),
        "curve": {"t": [round(float(t), 2) for t in ts], "cuts_per_min": [round(float(r), 2) for r in rate]},
        "fastest": {"t": round(float(ts[fast]), 2), "cuts_per_min": round(float(rate[fast]), 2)},
        "slowest": {"t": round(float(ts[slow]), 2), "cuts_per_min": round(float(rate[slow]), 2)},
        "length_histogram": [{"lo_s": a, "hi_s": None if b == float("inf") else b, "n": int(n)}
                             for a, b, n in zip(edges[:-1], edges[1:], hist, strict=True)],
    }
