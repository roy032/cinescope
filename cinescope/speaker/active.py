"""Who is talking: an audio-visual active-speaker score, from scratch.

For every face track and every frame: is *this* face speaking now? Three
signals, none learned:

  speech     the audio VAD score at that frame (is anyone speaking?);
  mouth      how much the mouth region moves, relative to the rest of the
             face. Faces are aligned on their landmarks first, so head motion
             is mostly removed; dividing by the upper face's motion cancels
             what is left (and blinks barely reach the ratio's numerator).
             The track's own "quiet" level comes from its frames where the
             VAD says nobody speaks -- a per-face, per-clip calibration that
             needs no labels: z = (log ratio - median_quiet) / spread_quiet;
  synchrony  the correlation, over a 1 s window, between the mouth-motion
             signal and the speech-band audio envelope's changes, maximised
             over +-2 frames of audio-video offset. A face that moves its
             mouth *with* the sound is the likelier speaker; dubbing, voice-
             over and a listener chewing gum are not in sync.

score = speech * sigmoid(z - 1) * (0.5 + 0.5 * max(synchrony, 0))

The constants (z offset 1, window lengths) were fixed before scoring the
labelled clip; Week 11's README reports the result once.
"""
from __future__ import annotations

import numpy as np

from .audio import at_times

# Regions in the 112x112 aligned face (TEMPLATE_112: mouth corners at y=92, x=42..71).
MOUTH = (slice(78, 108), slice(30, 82))
UPPER = (slice(28, 64), slice(22, 90))


def _roi_motion(gray: np.ndarray, roi) -> np.ndarray:
    """Mean absolute frame difference inside roi, after removing each frame's
    mean brightness there (lighting flicker is not motion)."""
    r = gray[:, roi[0], roi[1]].astype(np.float32)
    r = r - r.mean(axis=(1, 2), keepdims=True)
    d = np.abs(np.diff(r, axis=0)).mean(axis=(1, 2))
    return np.r_[d[:1] if len(d) else [0.0], d]


def mouth_activity(gray: np.ndarray) -> np.ndarray:
    """log(mouth motion / upper-face motion) per frame of an aligned track."""
    if len(gray) < 2:
        return np.zeros(len(gray))
    m = _roi_motion(gray, MOUTH)
    u = _roi_motion(gray, UPPER)
    return np.log((m + 0.5) / (u + 0.5))


def _smooth(x: np.ndarray, n: int) -> np.ndarray:
    if n <= 1 or len(x) < 2:
        return x
    k = np.ones(n) / n
    return np.convolve(np.pad(x, (n // 2, n - 1 - n // 2), mode="edge"), k, mode="valid")


def windowed_corr(a: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:
    """Pearson correlation of a and b over a centred window of n samples."""
    out = np.zeros(len(a))
    h = n // 2
    for i in range(len(a)):
        s = slice(max(0, i - h), min(len(a), i + h + 1))
        x, y = a[s] - a[s].mean(), b[s] - b[s].mean()
        den = np.sqrt((x * x).sum() * (y * y).sum())
        out[i] = (x * y).sum() / den if den > 1e-9 else 0.0
    return out


def score_track(gray: np.ndarray, frames: list[int], times: np.ndarray, vad: dict,
                fps: float, max_lag: int = 2) -> dict:
    """Per-frame active-speaker score for one track.

    gray:   (T, 112, 112) aligned grayscale faces of the track
    frames: the T frame numbers; times: video frame times (all frames)
    vad:    `audio.vad(...)` output for the clip
    """
    t = np.asarray(times)[frames]
    span = 1.0 / fps
    speech = at_times(vad["score"], vad["fps"], t, span)
    env = at_times(vad["energy_db"], vad["fps"], t, span)
    act = _smooth(mouth_activity(gray), max(1, int(round(0.2 * fps))))
    quiet = act[speech < 0.2]
    ref = quiet if len(quiet) >= max(5, 0.1 * len(act)) else act
    med = np.median(ref)
    spread = 1.4826 * np.median(np.abs(ref - med)) + 0.05
    z = (act - med) / spread
    p_mouth = 1 / (1 + np.exp(-(z - 1.0)))
    # Synchrony between the raw mouth motion and audio-envelope change.
    motion = _roi_motion(gray, MOUTH) if len(gray) > 1 else np.zeros(len(gray))
    denv = np.abs(np.r_[0, np.diff(env)])
    n = max(3, int(round(fps)))
    sync = np.full(len(t), -1.0)
    for lag in range(-max_lag, max_lag + 1):
        sync = np.maximum(sync, windowed_corr(motion, np.roll(denv, lag), n))
    score = speech * p_mouth * (0.5 + 0.5 * np.clip(sync, 0, None))
    return {"score": score, "speech": speech, "mouth_z": z, "sync": sync}
