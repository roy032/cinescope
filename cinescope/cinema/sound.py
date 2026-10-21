"""The soundtrack, in a re-recording mixer's words: loudness, dynamics, peaks and
silences, how much of it is speech, and whether the picture cuts on the sound.

Loudness follows ITU-R BS.1770 / EBU R 128 (the measure broadcasters and
streaming services deliver to), computed in the frequency domain:

  K-weighting   the standard's two filters (a +4 dB high shelf above ~1.5 kHz and
                a 38 Hz high-pass) applied as a gain on each block's power spectrum.
                Mean-square energy is the same in either domain (Parseval), so this
                matches filtering the waveform up to block-edge effects.
  momentary     loudness of 400 ms blocks every 100 ms: -0.691 + 10 log10(mean square)
  short-term    3 s blocks, used for the loudness range
  integrated    mean energy of the momentary blocks above the absolute gate
                (-70 LUFS) and the relative gate (10 LU under the ungated mean)
  range (LRA)   10th to 95th percentile of short-term loudness after the
                -70 LUFS and -20 LU gates

The pipeline decodes audio as 16 kHz mono (as Phase 3 does), so these are close
to, not certified, LUFS: content above 8 kHz and channel weights are left out.

Events, with thresholds fixed from the definitions:
  peak          momentary loudness 8 LU or more above the clip's typical level (the
                median momentary loudness above -70 LUFS) for at least 0.3 s: an
                explosion, a sting, a scream. (Not the integrated loudness: its
                relative gate lets a loud minute set the level of a quiet film.)
  silence       at least 1 s quieter than max(typical - 25 LU, -60 LUFS)
  onsets        sudden rises of broadband energy (spectral flux peaks); a cut "on
                the sound" is within 0.1 s of one, compared against chance
"""
from __future__ import annotations

import numpy as np

SR = 16000
BLOCK, HOP = 0.4, 0.1
ABS_GATE = -70.0
PEAK_LU, PEAK_MIN_S = 8.0, 0.3
SILENCE_LU, SILENCE_FLOOR, SILENCE_MIN_S = 25.0, -60.0, 1.0
ONSET_TOL = 0.1

# BS.1770 K-weighting biquads at 48 kHz (shelf, then high-pass)
_KW = (([1.53512485958697, -2.69169618940638, 1.19839281085285], [1.0, -1.69065929318241, 0.73248077421585]),
       ([1.0, -2.0, 1.0], [1.0, -1.99004745483398, 0.99007225036621]))


def k_weight_gain(freqs: np.ndarray) -> np.ndarray:
    """|H(f)|^2 of the K-weighting filter at the given frequencies (Hz, < 24 kHz)."""
    z = np.exp(-1j * 2 * np.pi * np.asarray(freqs, np.float64) / 48000.0)
    g = np.ones_like(z)
    for b, a in _KW:
        g = g * (b[0] + b[1] * z + b[2] * z * z) / (a[0] + a[1] * z + a[2] * z * z)
    return np.abs(g) ** 2


def _block_power(x: np.ndarray, sr: int, block: float, hop: float) -> np.ndarray:
    """K-weighted mean square of each block (blocks start every `hop` s)."""
    n, h = int(round(block * sr)), int(round(hop * sr))
    if len(x) < n:
        x = np.pad(x, (0, n - len(x)))
    count = 1 + (len(x) - n) // h
    gain = k_weight_gain(np.fft.rfftfreq(n, 1 / sr))
    gain[1:-1 if n % 2 == 0 else None] *= 2          # one-sided spectrum: count the negative half too
    out = np.empty(count)
    step = max(1, 2_000_000 // n)                    # blocks per FFT batch: about 16 MB at a time
    for s in range(0, count, step):
        idx = np.arange(s, min(count, s + step))[:, None] * h + np.arange(n)[None, :]
        spec = np.abs(np.fft.rfft(x[idx].astype(np.float64), axis=1)) ** 2
        out[s:s + len(idx)] = (spec * gain).sum(1) / (n * n)
    return out


def lufs(power) -> np.ndarray | float:
    return -0.691 + 10 * np.log10(np.maximum(power, 1e-12))


def integrated(power: np.ndarray) -> float:
    """Gated integrated loudness of momentary block powers (-inf if all is silence)."""
    p = np.asarray(power)
    p = p[lufs(p) > ABS_GATE]
    if not len(p):
        return float("-inf")
    rel = lufs(p.mean()) - 10.0
    p = p[lufs(p) > rel]
    return float(lufs(p.mean())) if len(p) else float("-inf")


def loudness_range(short_power: np.ndarray) -> float:
    p = np.asarray(short_power)
    p = p[lufs(p) > ABS_GATE]
    if len(p) < 2:
        return 0.0
    rel = lufs(p.mean()) - 20.0
    lv = lufs(p[lufs(p) > rel])
    return float(np.percentile(lv, 95) - np.percentile(lv, 10)) if len(lv) > 1 else 0.0


def runs(mask: np.ndarray, min_len: int) -> list[tuple[int, int]]:
    """[start, end) index runs of True at least min_len long."""
    m = np.concatenate([[False], np.asarray(mask, bool), [False]])
    d = np.diff(m.astype(np.int8))
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    return [(int(a), int(b)) for a, b in zip(starts, ends, strict=True) if b - a >= min_len]


def onsets(x: np.ndarray, sr: int = SR, hop: float = 0.01) -> np.ndarray:
    """Onset times (s): peaks of half-wave rectified log-spectral flux over 6
    log-spaced bands (150 Hz up), above a 0.5 s moving median + 3 robust standard
    deviations of the clip's flux + 0.5 (a rise of about 3 dB summed over bands)."""
    n, h = int(0.064 * sr), int(hop * sr)
    if len(x) < n + h:
        return np.zeros(0)
    count = 1 + (len(x) - n) // h
    edges = np.geomspace(150, sr / 2, 7)          # from 150 Hz: lower bands hold too few bins to be steady
    fr = np.fft.rfftfreq(n, 1 / sr)
    band = np.clip(np.searchsorted(edges, fr) - 1, 0, 5)
    win = np.hanning(n)
    E = np.empty((count, 6))
    for s in range(0, count, 8192):
        idx = np.arange(s, min(count, s + 8192))[:, None] * h + np.arange(n)[None, :]
        spec = np.abs(np.fft.rfft(x[idx] * win, axis=1)) ** 2
        for b in range(6):
            E[s:s + len(idx), b] = spec[:, band == b].sum(1)
    logE = np.log10(E + 1e-10)
    flux = np.maximum(np.diff(logE, axis=0), 0).sum(1)
    flux = np.concatenate([[0.0], flux])
    k = int(0.5 / hop) | 1
    pad = np.pad(flux, (k // 2, k // 2), mode="edge")
    med = np.median(np.lib.stride_tricks.sliding_window_view(pad, k), axis=1)
    mad = np.median(np.abs(flux - np.median(flux))) + 1e-9
    thr = med + 3.0 * mad * 1.4826 + 0.5
    peak = (flux > thr) & (flux >= np.roll(flux, 1)) & (flux >= np.roll(flux, -1))
    t = np.flatnonzero(peak) * hop + n / (2 * sr)
    # at most one onset per 50 ms
    keep, last = [], -1.0
    for v in t:
        if v - last >= 0.05:
            keep.append(v)
            last = v
    return np.asarray(keep)


def cut_sync(cut_times: list[float], onset_t: np.ndarray, duration: float, tol: float = ONSET_TOL) -> dict:
    """Share of cuts within `tol` of an onset, and the share expected by chance
    (the fraction of the timeline that is within `tol` of some onset)."""
    c = np.asarray(cut_times, np.float64)
    if not len(c) or not len(onset_t) or duration <= 0:
        return {"cuts": int(len(c)), "on_sound": 0.0, "chance": 0.0}
    o = np.sort(onset_t)
    pos = np.clip(np.searchsorted(o, c), 1, len(o) - 1) if len(o) > 1 else np.zeros(len(c), int)
    near = np.minimum(np.abs(o[pos] - c), np.abs(o[np.maximum(pos - 1, 0)] - c))
    grid = np.arange(0, duration, 0.01)
    gp = np.clip(np.searchsorted(o, grid), 1, len(o) - 1) if len(o) > 1 else np.zeros(len(grid), int)
    gnear = np.minimum(np.abs(o[gp] - grid), np.abs(o[np.maximum(gp - 1, 0)] - grid))
    return {"cuts": int(len(c)), "on_sound": round(float((near <= tol).mean()), 3),
            "chance": round(float((gnear <= tol).mean()), 3)}


def analyse(x: np.ndarray | None, shots: list[dict], duration: float, sr: int = SR,
            speech: np.ndarray | None = None, speech_fps: float = 100.0, points: int = 600) -> dict:
    """Loudness, events and per-shot sound for one clip. x: mono float audio, or None.
    speech: optional per-frame speech score 0..1 (Phase 3's vad) at speech_fps."""
    if x is None or not len(x):
        return {"audio": False, "per_shot": [None] * len(shots), "timeline": [], "summary": {}}
    mom = _block_power(x, sr, BLOCK, HOP)
    short = _block_power(x, sr, 3.0, 1.0) if len(x) >= 3 * sr else mom
    mt = np.arange(len(mom)) * HOP + BLOCK / 2                   # block centre times
    il = integrated(mom)
    ml = lufs(mom)
    audible = ml[ml > ABS_GATE]
    ref = float(np.median(audible)) if len(audible) else ABS_GATE
    timeline = []
    for a, b in runs(ml >= ref + PEAK_LU, int(round(PEAK_MIN_S / HOP))):
        timeline.append({"t0": round(float(mt[a] - HOP / 2), 2), "t1": round(float(mt[b - 1] + HOP / 2), 2),
                         "label": "peak", "lufs": round(float(ml[a:b].max()), 1)})
    for a, b in runs(ml < max(ref - SILENCE_LU, SILENCE_FLOOR), int(round(SILENCE_MIN_S / HOP))):
        timeline.append({"t0": round(float(mt[a] - HOP / 2), 2), "t1": round(float(mt[b - 1] + HOP / 2), 2),
                         "label": "silence"})
    timeline.sort(key=lambda e: e["t0"])
    on = onsets(x, sr)
    per_shot = []
    for s in shots:
        sel = (mt >= s["t0"]) & (mt < s["t1"])
        if not sel.any():
            sel = np.zeros(len(mt), bool)
            sel[min(int(s["t0"] / HOP), len(mt) - 1)] = True
        li = integrated(mom[sel])
        row = {"lufs": round(li, 1) if np.isfinite(li) else None, "max_lufs": round(float(ml[sel].max()), 1),
               "onsets": int(((on >= s["t0"]) & (on < s["t1"])).sum())}
        if speech is not None and len(speech):
            a, b = int(s["t0"] * speech_fps), max(int(s["t1"] * speech_fps), int(s["t0"] * speech_fps) + 1)
            seg = speech[a:b]
            row["speech"] = round(float((seg > 0.5).mean()), 3) if len(seg) else 0.0
        per_shot.append(row)
    # the envelope for the app: momentary loudness, max-pooled to at most `points` values
    k = max(1, int(np.ceil(len(ml) / points)))
    env = np.pad(ml, (0, (-len(ml)) % k), constant_values=-120).reshape(-1, k).max(1)
    loud = int(np.argmax(ml))
    summary = {
        "integrated_lufs": round(il, 1) if np.isfinite(il) else None,
        "typical_lufs": round(ref, 1),
        "lra_lu": round(loudness_range(short), 1),
        "peak_dbfs": round(float(20 * np.log10(max(float(np.abs(x).max()), 1e-9))), 1),
        "loudest_t": round(float(mt[loud]), 2), "loudest_lufs": round(float(ml[loud]), 1),
        "peaks": sum(e["label"] == "peak" for e in timeline),
        "silences": sum(e["label"] == "silence" for e in timeline),
        "silence_s": round(sum(e["t1"] - e["t0"] for e in timeline if e["label"] == "silence"), 1),
        "onsets_per_min": round(60.0 * len(on) / max(duration, 1e-6), 1),
        "cut_sync": cut_sync([s["t0"] for s in shots[1:]], on, duration),
    }
    if speech is not None and len(speech):
        summary["speech_share"] = round(float((speech > 0.5).mean()), 3)
    return {"audio": True, "per_shot": per_shot, "timeline": timeline, "summary": summary,
            "envelope": {"dt": round(HOP * k, 3), "t0": round(BLOCK / 2, 3),
                         "lufs": np.round(np.maximum(env, -90), 1).tolist()}}
