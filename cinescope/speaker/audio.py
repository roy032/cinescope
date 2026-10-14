"""Audio for active-speaker detection: decoding, MFCCs, and a voice-activity
detector, from scratch in NumPy.

Who is talking needs two answers: *is anyone speaking now* (audio only) and
*whose mouth moves with it* (audio-visual). This module answers the first.

Voice activity, the classical way. Film soundtracks mix speech with music,
effects and silence, and loudness alone cannot tell them apart. Three cues
that are cheap and fairly specific to speech, each on 10 ms frames:

  1. speech-band energy -- 300-3400 Hz (the telephone band) holds most of
     speech's energy; measured relative to the clip's own noise floor
     (a low percentile), so the threshold adapts to each mix;
  2. syllabic modulation -- speech switches on and off 3-8 times a second
     (syllables), so the speech-band envelope has strong 3-8 Hz modulation;
     sustained music and rumble do not;
  3. spectral flatness -- voiced speech has harmonics (a peaky spectrum,
     low flatness); noise and explosions are flat.

They are combined into a 0-1 score with fixed, hand-set weights (no
training): the point is to see how far signal processing gets before the
learned model (Light-ASD) is brought in.
"""
from __future__ import annotations

import numpy as np

SR = 16000


def load_audio(path: str, sr: int = SR) -> np.ndarray | None:
    """Mono float32 in [-1, 1] at `sr` Hz, or None when the file has no
    (decodable) audio. Undecodable packets are skipped, not fatal."""
    import av

    try:
        c = av.open(str(path))
    except Exception:
        return None
    if not c.streams.audio:
        c.close()
        return None
    res = av.AudioResampler(format="s16", layout="mono", rate=sr)
    chunks = []
    stream = c.streams.audio[0]
    for packet in c.demux(stream):
        try:
            frames = packet.decode()
        except Exception:
            continue
        for fr in frames:
            for r in res.resample(fr):
                chunks.append(r.to_ndarray().ravel())
    c.close()
    if not chunks:
        return None
    return np.concatenate(chunks).astype(np.float32) / 32768.0


def frames_of(x: np.ndarray, win: int, hop: int) -> np.ndarray:
    """(n, win) overlapping frames; the last partial frame is zero-padded."""
    n = 1 + max(0, int(np.ceil((len(x) - win) / hop)))
    pad = np.zeros((n - 1) * hop + win, x.dtype)
    pad[:len(x)] = x
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    return pad[idx]


# ------------------------------------------------------------------ MFCC
def mel(f):
    return 2595 * np.log10(1 + np.asarray(f) / 700.0)


def mel_inv(m):
    return 700 * (10 ** (np.asarray(m) / 2595.0) - 1)


def mel_filterbank(n_filt: int = 26, nfft: int = 512, sr: int = SR, low: float = 0, high: float | None = None):
    """Triangular filters equally spaced on the mel scale (python_speech_features' layout)."""
    high = high or sr / 2
    pts = mel_inv(np.linspace(mel(low), mel(high), n_filt + 2))
    bins = np.floor((nfft + 1) * pts / sr).astype(int)
    fb = np.zeros((n_filt, nfft // 2 + 1))
    for j in range(n_filt):
        a, b, c = bins[j], bins[j + 1], bins[j + 2]
        for i in range(a, b):
            fb[j, i] = (i - a) / (b - a)
        for i in range(b, c):
            fb[j, i] = (c - i) / (c - b)
    return fb


def mfcc(x: np.ndarray, sr: int = SR, numcep: int = 13, winlen: float = 0.025, winstep: float = 0.010,
         nfilt: int = 26, nfft: int = 512, preemph: float = 0.97, ceplifter: int = 22) -> np.ndarray:
    """MFCCs exactly as python_speech_features.mfcc computes them (what Light-ASD
    was trained on): pre-emphasis, rectangular 25 ms windows every 10 ms, power
    spectrum, 26 mel filters, log, DCT-II (orthonormal), sinusoidal liftering,
    and c0 replaced by the log frame energy. 100 frames per second."""
    x = np.asarray(x, dtype=np.float64)
    x = np.r_[x[:1], x[1:] - preemph * x[:-1]]
    win, hop = int(round(winlen * sr)), int(round(winstep * sr))
    n = 1 if len(x) <= win else 1 + int(np.ceil((len(x) - win) / hop))
    pad = np.zeros((n - 1) * hop + win)
    pad[:len(x)] = x
    fr = pad[np.arange(win)[None, :] + hop * np.arange(n)[:, None]]
    pspec = np.abs(np.fft.rfft(fr, nfft)) ** 2 / nfft
    energy = pspec.sum(1)
    energy = np.where(energy == 0, np.finfo(float).eps, energy)
    feat = pspec @ mel_filterbank(nfilt, nfft, sr).T
    feat = np.log(np.where(feat == 0, np.finfo(float).eps, feat))
    # Orthonormal DCT-II, written out.
    k = np.arange(nfilt)
    basis = np.cos(np.pi / nfilt * (k[None, :] + 0.5) * np.arange(numcep)[:, None])
    scale = np.full(numcep, np.sqrt(2.0 / nfilt))
    scale[0] = np.sqrt(1.0 / nfilt)
    cep = (feat @ basis.T) * scale
    lift = 1 + (ceplifter / 2.0) * np.sin(np.pi * np.arange(numcep) / ceplifter)
    cep = cep * lift
    cep[:, 0] = np.log(energy)
    return cep


# ------------------------------------------------------------------ VAD
def band_envelope(x: np.ndarray, sr: int = SR, lo: float = 300, hi: float = 3400,
                  win: float = 0.025, hop: float = 0.010) -> tuple[np.ndarray, np.ndarray]:
    """(speech-band energy in dB, spectral flatness of that band), one value per hop."""
    w, h = int(win * sr), int(hop * sr)
    fr = frames_of(np.asarray(x, np.float64), w, h) * np.hanning(w)
    nfft = 1 << (w - 1).bit_length()
    spec = np.abs(np.fft.rfft(fr, nfft)) ** 2
    f = np.fft.rfftfreq(nfft, 1 / sr)
    band = spec[:, (f >= lo) & (f <= hi)] + 1e-12
    e_db = 10 * np.log10(band.sum(1))
    flat = np.exp(np.log(band).mean(1)) / band.mean(1)
    return e_db, flat


def _smooth(x: np.ndarray, n: int) -> np.ndarray:
    if n <= 1:
        return x
    k = np.ones(n) / n
    return np.convolve(np.pad(x, (n // 2, n - 1 - n // 2), mode="edge"), k, mode="valid")


def vad(x: np.ndarray, sr: int = SR, hop: float = 0.010) -> dict:
    """Frame-level speech score in [0, 1] at 1/hop frames per second, with its parts.

    energy      sigmoid((E - floor - 12 dB) / 3): 12 dB above the clip's 10th-percentile
                speech-band energy counts as "something is there".
    modulation  fraction of the band envelope's 1 s-window variance in 3-8 Hz.
    voicing     1 - spectral flatness (peaky = harmonic).
    score       energy * sigmoid(8 * (0.5 * modulation + 0.5 * voicing - 0.45)), smoothed 0.2 s.
    The constants were set by reasoning about the cues, before any labelled clip was scored.
    """
    e_db, flat = band_envelope(x, sr, hop=hop)
    fps = 1.0 / hop
    floor = np.percentile(e_db, 10)
    energy = 1 / (1 + np.exp(-(e_db - floor - 12.0) / 3.0))
    # Syllabic modulation: band-pass the dB envelope to 3-8 Hz by FFT over 1 s windows.
    n = int(fps)
    env = e_db - _smooth(e_db, n)
    win = frames_of(np.pad(env, (n // 2, n - n // 2)), n, 1)[:len(env)] * np.hanning(n)
    spec = np.abs(np.fft.rfft(win, axis=1)) ** 2
    fr = np.fft.rfftfreq(n, 1 / fps)
    sel = (fr >= 3) & (fr <= 8)
    modulation = spec[:, sel].sum(1) / (spec[:, fr >= 1].sum(1) + 1e-9)
    voicing = 1.0 - _smooth(flat, 5)
    cue = 0.5 * modulation + 0.5 * voicing
    score = _smooth(energy * (1 / (1 + np.exp(-8 * (cue - 0.45)))), int(0.2 * fps))
    return {"fps": fps, "score": score, "energy_db": e_db, "energy": energy,
            "modulation": modulation, "voicing": voicing}


def at_times(values: np.ndarray, fps: float, times: np.ndarray, span: float = 0.0) -> np.ndarray:
    """Sample a per-hop signal at video frame times (mean over [t, t + span) if span > 0)."""
    times = np.asarray(times, dtype=np.float64)
    if span <= 0:
        idx = np.clip(np.round(times * fps).astype(int), 0, len(values) - 1)
        return values[idx]
    c = np.r_[0, np.cumsum(values)]
    a = np.clip(np.floor(times * fps).astype(int), 0, len(values))
    b = np.clip(np.maximum(a + 1, np.floor((times + span) * fps).astype(int)), 0, len(values))
    return (c[b] - c[a]) / np.maximum(b - a, 1)


def silero_vad(x: np.ndarray, sr: int = SR, hop: float = 0.010) -> np.ndarray | None:
    """The library baseline: Silero VAD (pip install silero-vad), speech probability
    per 32 ms chunk, resampled to 1/hop fps. None if the package is missing."""
    try:
        import torch
        from silero_vad import load_silero_vad
    except ImportError:
        return None
    model = load_silero_vad()
    chunk = 512
    probs = []
    with torch.no_grad():
        for i in range(0, len(x) - chunk + 1, chunk):
            probs.append(float(model(torch.from_numpy(x[i:i + chunk].astype(np.float32)), sr)))
    probs = np.asarray(probs)
    t = np.arange(int(len(x) / sr / hop)) * hop
    return probs[np.clip((t * sr / chunk).astype(int), 0, len(probs) - 1)]
