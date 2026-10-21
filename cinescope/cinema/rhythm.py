"""Rhythm: the soundtrack's beat, and whether the picture is cut to it.

    onset envelope   half-wave rectified log-spectral flux at 100 frames a second,
                     minus its own 1 s moving mean (how much new sound starts)
    tempo            the autocorrelation of the envelope between 40 and 240 BPM,
                     weighted by a log-normal prior around 120 BPM one octave wide
                     (Ellis 2007), so the tempo an audience would tap wins over its
                     double and half
    pulse clarity    the autocorrelation at that tempo over the autocorrelation at
                     zero lag: the share of the envelope's variance the beat
                     explains. Sections of the clip where an 8 s window's clarity
                     reaches PULSE are "rhythmic" (music with a beat, not speech or
                     ambience)
    beats            dynamic-programming beat tracking (Ellis 2007): the beat times
                     that best sit on onsets while keeping the tempo
    cuts on the beat the share of cuts inside rhythmic sections within TOL of a
                     beat, against the share a cut at a random time would get

PULSE and TOL are round numbers fixed up front (TOL is two frames at 24 fps),
not fitted to any clip.
"""
from __future__ import annotations

import numpy as np

SR = 16000
FPS = 100.0               # envelope frames per second
BPM_RANGE = (40.0, 240.0)
PRIOR_BPM, PRIOR_OCTAVES = 120.0, 1.0
PULSE = 0.3
TOL = 0.083
WINDOW_S, HOP_S = 8.0, 2.0
TIGHTNESS = 100.0         # beat tracker: how strongly the tempo is kept


def onset_envelope(x: np.ndarray, sr: int = SR, fps: float = FPS) -> np.ndarray:
    n, h = 1024, int(round(sr / fps))
    if x is None or len(x) < n + h:
        return np.zeros(0)
    x = np.pad(np.asarray(x, np.float64), (n // 2, 0))      # frame i is centred on i / fps
    count = 1 + (len(x) - n) // h
    win = np.hanning(n)
    fr = np.fft.rfftfreq(n, 1 / sr)
    keep = (fr >= 30) & (fr <= 8000)
    prev = None
    env = np.empty(count)
    for s in range(0, count, 4096):
        idx = np.arange(s, min(count, s + 4096))[:, None] * h + np.arange(n)[None, :]
        S = np.log1p(1000.0 * np.abs(np.fft.rfft(x[idx] * win, axis=1))[:, keep])
        first = S[:1] if prev is None else prev
        d = np.maximum(np.diff(np.concatenate([first, S]), axis=0), 0).sum(1)
        env[s:s + len(idx)] = d
        prev = S[-1:]
    k = int(fps) | 1
    mean = np.convolve(np.pad(env, k // 2, mode="edge"), np.ones(k) / k, mode="valid")
    env = np.maximum(env - mean, 0)
    sd = env.std()
    return env / sd if sd > 1e-12 else env


def _autocorr(e: np.ndarray, max_lag: int) -> np.ndarray:
    e = e - e.mean()
    n = len(e)
    f = np.fft.rfft(e, 2 * n)
    ac = np.fft.irfft(f * np.conj(f))[:max_lag + 1]
    return ac / ac[0] if ac[0] > 1e-12 else np.zeros(max_lag + 1)


def tempo(env: np.ndarray, fps: float = FPS) -> dict:
    """{"bpm", "period_s", "clarity"} of an onset envelope (bpm None when too short)."""
    lo, hi = int(np.floor(fps * 60 / BPM_RANGE[1])), int(np.ceil(fps * 60 / BPM_RANGE[0]))
    if len(env) < 2 * hi:
        return {"bpm": None, "period_s": None, "clarity": 0.0}
    ac = _autocorr(env, hi)
    lags = np.arange(lo, hi + 1)
    bpm = 60.0 * fps / lags
    prior = np.exp(-0.5 * (np.log2(bpm / PRIOR_BPM) / PRIOR_OCTAVES) ** 2)
    k = int(np.argmax(np.maximum(ac[lags], 0) * prior))
    lag = int(lags[k])
    # parabolic interpolation for a sub-frame period
    if lo < lag < hi:
        a, b, c = ac[lag - 1], ac[lag], ac[lag + 1]
        den = a - 2 * b + c
        off = 0.5 * (a - c) / den if abs(den) > 1e-12 else 0.0
    else:
        off = 0.0
    period = (lag + float(np.clip(off, -0.5, 0.5))) / fps
    return {"bpm": round(60.0 / period, 1), "period_s": round(period, 4), "clarity": round(float(max(ac[lag], 0)), 3)}


def track_beats(env: np.ndarray, period_s: float, fps: float = FPS) -> np.ndarray:
    """Beat times (s) by dynamic programming over the onset envelope."""
    p = period_s * fps
    n = len(env)
    if n < 2 or p < 1:
        return np.zeros(0)
    score = env.astype(np.float64).copy()
    back = np.full(n, -1)
    lo_o, hi_o = int(round(2 * p)), max(1, int(round(p / 2)))
    offs = np.arange(hi_o, lo_o + 1)
    penalty = -TIGHTNESS * np.log(offs / p) ** 2
    for t in range(hi_o, n):
        cand = t - offs
        ok = cand >= 0
        if not ok.any():
            continue
        v = score[cand[ok]] + penalty[ok]
        j = int(np.argmax(v))
        if v[j] > 0:
            score[t] += v[j]
            back[t] = cand[ok][j]
    tail = max(0, n - int(np.ceil(p)))
    t = tail + int(np.argmax(score[tail:]))
    beats = []
    while t >= 0:
        beats.append(t)
        t = back[t]
    return np.asarray(beats[::-1], float) / fps


def local_clarity(env: np.ndarray, fps: float = FPS) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(window centre times, pulse clarity, period) over sliding 8 s windows."""
    w, hop = int(WINDOW_S * fps), int(HOP_S * fps)
    if len(env) < w:
        tp = tempo(env, fps)
        return np.array([len(env) / fps / 2]), np.array([tp["clarity"]]), np.array([tp["period_s"] or 0.0])
    ts, cs, ps = [], [], []
    for s in range(0, len(env) - w + 1, hop):
        tp = tempo(env[s:s + w], fps)
        ts.append((s + w / 2) / fps)
        cs.append(tp["clarity"])
        ps.append(tp["period_s"] or 0.0)
    return np.asarray(ts), np.asarray(cs), np.asarray(ps)


def rhythmic_sections(t: np.ndarray, clarity: np.ndarray, duration: float) -> list[tuple[float, float]]:
    """Merged spans where the windows' clarity reaches PULSE. Each window vouches for
    the middle half of itself (a window only partly over music can still pass), and
    the first and last windows for the clip's ends."""
    spans = []
    for k, (c, ci) in enumerate(zip(t, clarity, strict=True)):
        if ci < PULSE:
            continue
        a = 0.0 if k == 0 else max(0.0, c - WINDOW_S / 4)
        b = duration if k == len(t) - 1 else min(duration, c + WINDOW_S / 4)
        if spans and a <= spans[-1][1]:
            spans[-1] = (spans[-1][0], max(spans[-1][1], b))
        else:
            spans.append((a, b))
    return spans


def cuts_on_beat(cuts: list[float], beats: np.ndarray, sections: list[tuple[float, float]],
                 period_s: float | None, tol: float = TOL) -> dict:
    inside = [c for c in cuts if any(a <= c <= b for a, b in sections)]
    if not inside or not len(beats) or not period_s:
        return {"cuts": len(inside), "on_beat": None, "chance": None}
    b = np.asarray(beats)
    near = np.array([np.abs(b - c).min() for c in inside])
    return {"cuts": len(inside), "on_beat": round(float((near <= tol).mean()), 3),
            "chance": round(float(min(1.0, 2 * tol / period_s)), 3)}


def analyse(x: np.ndarray | None, shots: list[dict], duration: float, sr: int = SR) -> dict:
    """Tempo, beats, rhythmic sections and cuts on the beat for one clip."""
    env = onset_envelope(x, sr) if x is not None else np.zeros(0)
    if not len(env):
        return {"audio": x is not None, "per_shot": [None] * len(shots), "timeline": [], "beats": [], "summary": {}}
    tp = tempo(env)
    lt, lc, _ = local_clarity(env)
    sections = rhythmic_sections(lt, lc, duration)
    beats = track_beats(env, tp["period_s"]) if tp["period_s"] else np.zeros(0)
    # beats only count inside rhythmic sections: outside them the tracker is guessing
    in_music = np.array([any(a <= t <= b for a, b in sections) for t in beats], bool)
    beats = beats[in_music] if len(beats) else beats
    per_shot = []
    for s in shots:
        nb = int(((beats >= s["t0"]) & (beats < s["t1"])).sum())
        per_shot.append({"beats": nb, "length_beats": round((s["t1"] - s["t0"]) / tp["period_s"], 2)
                         if tp["period_s"] and nb else None})
    share = sum(b - a for a, b in sections) / max(duration, 1e-6)
    summary = {"bpm": tp["bpm"] if sections else None, "pulse_clarity": tp["clarity"],
               "rhythmic_share": round(min(share, 1.0), 3), "beats": int(len(beats)),
               "cuts_on_beat": cuts_on_beat([s["t0"] for s in shots[1:]], beats, sections, tp["period_s"])}
    step = max(1, len(lc) // 300)
    return {"audio": True, "per_shot": per_shot,
            "timeline": [{"t0": round(a, 2), "t1": round(b, 2), "label": "rhythmic"} for a, b in sections],
            "beats": np.round(beats, 3).tolist(), "summary": summary,
            "clarity": {"t": np.round(lt[::step], 2).tolist(), "value": np.round(lc[::step], 3).tolist()}}
