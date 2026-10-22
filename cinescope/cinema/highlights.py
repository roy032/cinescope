"""Highlights: the strongest moments of a clip, cut down to a short reel.

    score       each shot gets the mean of the rank-normalised signals that are
                available: the story's intensity curve over the shot (cutting
                rate + loudness + motion), the shot's loudest moment (LUFS
                envelope) and how fast the cutting is around it (1 / shot
                length). Rank normalisation, as in cinescope.cinema.story,
                needs no units and no tuning
    moments     the best shot not yet used seeds a moment; the moment grows
                into the better-scoring neighbour shot while it is shorter
                than `max_moment_s`. A shot longer than that is trimmed to the
                window around its loudest point. Moments are taken until the
                reel is `target_s` long (at most `max_share` of the clip, so a
                short clip still gets a summary, not itself back). A first pass
                keeps seeds a twentieth of the running time away from the
                moments already taken, so the reel samples the whole clip
                rather than one busy passage; a second pass fills what is left.
                Moments that end up touching are joined into one
    climax      the story's climax is always in the reel, as its own moment
    beats       a trimmed edge (one not on a cut already) moves to the nearest
                beat within a third of a beat, so the reel's own cuts land on
                the music where there is a beat to land on
    order       chronological, the way a trailer keeps the story's order

The reel itself is rendered by cinescope.editorial.reel (MP4) and listed as a
CMX 3600 EDL by cinescope.editorial.exports.highlights_edl.
"""
from __future__ import annotations

import numpy as np

TARGET_S = 30.0
MAX_SHARE = 0.25           # the reel is at most this share of the clip
MAX_MOMENT_S = 6.0
MIN_MOMENT_S = 1.5         # shorter moments flash by; they grow into a neighbour or are trimmed around
EDGE_S = 0.5               # the first and last half second (logos, fades) never seed a moment
SPREAD = 0.05              # first pass: seeds at least this share of the running time from a moment


def ranks(x: np.ndarray) -> np.ndarray:
    """0..1 ranks (ties share the mean rank); NaNs rank 0.5."""
    x = np.asarray(x, float)
    out = np.full(len(x), 0.5)
    ok = ~np.isnan(x)
    if ok.sum() > 1:
        v = x[ok]
        order = np.argsort(v, kind="stable")
        r = np.empty(len(v))
        r[order] = np.arange(len(v))
        for u in np.unique(v):                      # ties
            m = v == u
            r[m] = r[m].mean()
        out[ok] = r / (len(v) - 1)
    return out


def _curve_mean(t0: float, t1: float, start: float, dt: float, values: np.ndarray) -> float:
    if not len(values) or dt <= 0:
        return float("nan")
    a = int(np.floor((t0 - start) / dt))
    b = int(np.ceil((t1 - start) / dt))
    a, b = max(a, 0), min(max(b, a + 1), len(values))
    return float(np.mean(values[a:b])) if b > a else float(values[min(max(a, 0), len(values) - 1)])


def _peak_time(t0: float, t1: float, env: dict | None) -> float | None:
    if not env or not env.get("lufs"):
        return None
    v = np.asarray(env["lufs"], float)
    t = env["t0"] + env["dt"] * np.arange(len(v))
    m = (t >= t0) & (t <= t1)
    return float(t[m][np.argmax(v[m])]) if m.any() else None


def shot_scores(shots: list[dict], intensity: dict | None = None, envelope: dict | None = None) -> np.ndarray:
    """0..1 per shot, the mean of the available rank-normalised signals."""
    n = len(shots)
    signals = []
    if intensity and intensity.get("value"):
        v = np.asarray(intensity["value"], float)
        signals.append(ranks(np.array([_curve_mean(s["t0"], s["t1"], intensity["t0"], intensity["dt"], v)
                                       for s in shots])))
    if envelope and envelope.get("lufs"):
        v = np.maximum(np.asarray(envelope["lufs"], float), -60.0)
        t = envelope["t0"] + envelope["dt"] * np.arange(len(v))
        peak = [float(v[(t >= s["t0"]) & (t <= s["t1"])].max()) if ((t >= s["t0"]) & (t <= s["t1"])).any()
                else float("nan") for s in shots]
        signals.append(ranks(np.array(peak)))
    # cutting rate around the shot: shots in a fast passage are short, and so are their neighbours
    lengths = np.array([max(s["t1"] - s["t0"], 1e-3) for s in shots])
    local = np.array([lengths[max(0, i - 2):i + 3].mean() for i in range(n)])
    signals.append(ranks(1.0 / local))
    return np.mean(signals, axis=0) if n else np.zeros(0)


def _snap(t: float, beats: np.ndarray, lo: float, hi: float, period: float | None) -> float:
    if not len(beats) or not period:
        return t
    j = int(np.argmin(np.abs(beats - t)))
    b = float(beats[j])
    return b if abs(b - t) <= period / 3 and lo <= b <= hi else t


def pick(shots: list[dict], duration: float, scores: np.ndarray, *, envelope: dict | None = None,
         beats: list[float] = (), climax_t: float | None = None, target_s: float = TARGET_S,
         max_share: float = MAX_SHARE, max_moment_s: float = MAX_MOMENT_S,
         min_moment_s: float = MIN_MOMENT_S) -> list[dict]:
    """Chronological moments {t0, t1, shots: [first, last], score, climax}."""
    n = len(shots)
    dur = float(duration) or (float(shots[-1]["t1"]) if n else 0.0)
    if not n or dur <= 0:
        return []
    budget = min(float(target_s), max_share * dur)
    budget = max(budget, min(min_moment_s, dur))
    beats = np.asarray(sorted(beats), float)
    period = float(np.median(np.diff(beats))) if len(beats) > 2 else None
    used = np.zeros(n, bool)
    moments: list[dict] = []
    total = 0.0

    def window(a: int, b: int) -> tuple[float, float]:
        """The moment's times: whole shots a..b, or a trimmed window of a single long shot."""
        t0, t1 = float(shots[a]["t0"]), float(shots[b]["t1"])
        room = min(max_moment_s, budget - total)
        if t1 - t0 <= room + 1e-6:
            return t0, t1
        centre = _peak_time(t0, t1, envelope)
        if centre is None:
            centre = (t0 + t1) / 2
        w = max(min(room, max_moment_s), min(min_moment_s, t1 - t0))
        s0 = min(max(centre - w / 2, t0), t1 - w)
        s0 = _snap(s0, beats, t0, t1 - min_moment_s, period)
        s1 = _snap(s0 + w, beats, s0 + min_moment_s, t1, period)
        return round(s0, 3), round(s1, 3)

    def grow(i: int) -> tuple[int, int]:
        a = b = i
        while float(shots[b]["t1"]) - float(shots[a]["t0"]) < min(max_moment_s, budget - total):
            left = scores[a - 1] if a > 0 and not used[a - 1] else -1.0
            right = scores[b + 1] if b + 1 < n and not used[b + 1] else -1.0
            if max(left, right) < 0:
                break
            nb = (a - 1, b) if left > right else (a, b + 1)
            if float(shots[nb[1]]["t1"]) - float(shots[nb[0]]["t0"]) > max_moment_s and \
                    float(shots[b]["t1"]) - float(shots[a]["t0"]) >= min_moment_s:
                break                                   # the neighbour would overrun; long enough already
            a, b = nb
        return a, b

    def take(i: int, climax: bool = False) -> None:
        nonlocal total
        a, b = grow(i)
        t0, t1 = window(a, b)
        used[a:b + 1] = True
        t1 = min(t1, t0 + max(budget - total, min(min_moment_s, t1 - t0)))   # the last one fits the budget
        if t1 - t0 < 0.2:
            return
        moments.append({"t0": t0, "t1": t1, "shots": [int(a), int(b)],
                        "score": round(float(scores[a:b + 1].mean()), 3), "climax": climax})
        total += t1 - t0

    if climax_t is not None:
        ci = next((i for i, s in enumerate(shots) if s["t0"] <= climax_t < s["t1"]), n - 1)
        take(ci, climax=True)
    edge = min(EDGE_S, dur / 10)
    gap = SPREAD * dur
    for spread in (True, False):
        for i in np.argsort(-scores, kind="stable"):
            if budget - total < min(min_moment_s, budget) - 1e-6 or total >= budget - 0.25:
                break
            if used[i] or shots[i]["t1"] <= edge or shots[i]["t0"] >= dur - edge:
                continue
            if spread and any(shots[i]["t0"] < m["t1"] + gap and shots[i]["t1"] > m["t0"] - gap for m in moments):
                continue
            take(int(i))
    return _join(sorted(moments, key=lambda m: m["t0"]), scores)


def _join(moments: list[dict], scores: np.ndarray, tol: float = 0.1) -> list[dict]:
    """Moments that touch (one ends where the next starts) become one."""
    out: list[dict] = []
    for m in moments:
        if out and m["t0"] - out[-1]["t1"] <= tol:
            p = out[-1]
            a, b = p["shots"][0], max(p["shots"][1], m["shots"][1])
            out[-1] = {"t0": p["t0"], "t1": max(p["t1"], m["t1"]), "shots": [a, b],
                       "score": round(float(scores[a:b + 1].mean()), 3), "climax": p["climax"] or m["climax"]}
        else:
            out.append(m)
    return out


def analyse(shots: list[dict], duration: float, story: dict | None = None, sound: dict | None = None,
            rhythm: dict | None = None, target_s: float = TARGET_S, max_share: float = MAX_SHARE,
            max_moment_s: float = MAX_MOMENT_S, min_moment_s: float = MIN_MOMENT_S) -> dict:
    """The highlights stage's result; every input but the shots is optional."""
    if not shots:
        return {"moments": [], "timeline": [], "summary": {}}
    story = story or {}
    envelope = (sound or {}).get("envelope")
    scores = shot_scores(shots, story.get("intensity"), envelope)
    climax = next((b["t"] for b in story.get("beats", []) if b.get("label") == "climax"), None)
    moments = pick(shots, duration, scores, envelope=envelope, beats=(rhythm or {}).get("beats") or [],
                   climax_t=climax, target_s=target_s, max_share=max_share, max_moment_s=max_moment_s,
                   min_moment_s=min_moment_s)
    total = sum(m["t1"] - m["t0"] for m in moments)
    signals = ["cutting rate"] + (["intensity"] if story.get("intensity") else []) + \
        (["loudness"] if envelope else [])
    return {
        "moments": moments,
        "timeline": [{"t0": m["t0"], "t1": m["t1"], "label": "climax" if m["climax"] else "highlight"}
                     for m in moments],
        "per_shot": [round(float(s), 3) for s in scores],
        "summary": {"moments": len(moments), "duration_s": round(total, 2),
                    "share": round(total / max(float(duration), 1e-6), 3), "signals": signals,
                    "on_beat": bool((rhythm or {}).get("beats"))},
    }
