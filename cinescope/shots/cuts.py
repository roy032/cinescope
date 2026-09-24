"""Shot-boundary detection: hard cuts, fades and dissolves, from histogram distances.

Hard cuts
  d[t] = chi2(hist[t-1], hist[t]). A cut is a *spike*: d[t] is far above the
  distances around it. A single global threshold fails in both directions:
  a cut between two shots of the same room (shot / reverse-shot dialogue)
  can score 0.15, while a handheld chase produces in-shot distances of 0.1.
  So the threshold is relative to the neighbourhood:

      d[t] > max(min_cut, ratio * P90(d[t-w .. t+w] without t-1, t, t+1))
      and d[t] is the local maximum within ±min_shot frames

  (A percentile, not mean/std: an earlier cut inside the window would inflate
  the std and hide the next one. On the first labelled clip every true cut had
  d[t] at least ~12x its neighbourhood P90 and every non-cut under ~2x, which
  is why ratio=4 sits comfortably in the gap.)

  Flash guard: a camera flash or muzzle flash makes frame t differ from t-1,
  but frame t+2 looks like t-1 again. A real cut does not come back.

Gradual transitions (why histograms struggle with them)
  In a dissolve every frame is a blend of shot A and shot B, so each adjacent
  pair differs only a little — no spike ever crosses the cut threshold. The
  twin-comparison method (Zhang, Kankanhalli & Smoliar, 1993) watches for a
  *run* of moderately elevated distances and compares the frame before the run
  with the frame after it: if they are as different as a cut, it was a gradual
  transition.

Fades
  A fade to/from black passes through frames that are dark AND flat (low mean,
  low std). The boundary is placed at the darkest frame.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .features import FrameFeatures


@dataclass
class Transition:
    kind: str            # "cut" | "dissolve" | "fade"
    frame: int           # first frame of the new shot (for gradual: centre of the transition)
    start: int           # first affected frame (== frame for cuts)
    end: int             # last affected frame (== frame for cuts)
    t: float             # seconds
    score: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Shot:
    index: int
    start: int           # first frame (inclusive)
    end: int             # last frame (inclusive)
    t0: float
    t1: float

    @property
    def n_frames(self) -> int:
        return self.end - self.start + 1

    def to_dict(self) -> dict:
        return {**asdict(self), "duration_s": round(self.t1 - self.t0, 3)}


@dataclass
class CutParams:
    window: int = 24              # frames each side for the local statistics (~1 s)
    ratio: float = 4.0            # spike must exceed ratio x the neighbourhood's 90th percentile
    min_cut: float = 0.08         # absolute floor on chi2 distance for a hard cut
    min_shot: int = 6             # frames; no two boundaries closer than this
    flash_ratio: float = 0.5      # d(t-1, t+2) < flash_ratio * d[t]  -> flash, not cut
    grad_low: float = 0.04        # twin comparison: start of a candidate gradual run
    grad_high: float = 0.35       # accumulated difference that makes it a transition
    grad_min_len: int = 6
    grad_max_len: int = 60
    fade_mean: float = 18.0       # luma mean below this (0..255) ...
    fade_std: float = 8.0         # ... and std below this = a "black" frame


def neighbourhood_p90(d: np.ndarray, w: int) -> np.ndarray:
    """90th percentile of d over [t-w, t+w], excluding t-1, t, t+1 (a cut's
    immediate neighbours can be slightly elevated by motion blur or encoding)."""
    n = len(d)
    out = np.empty(n)
    for t in range(n):
        win = np.concatenate([d[max(0, t - w):max(0, t - 1)], d[min(n, t + 2):min(n, t + w + 1)]])
        out[t] = np.percentile(win, 90) if win.size else 0.0
    return out


def _is_flash(feat: FrameFeatures, d: np.ndarray, t: int, p: CutParams, max_len: int = 3) -> bool:
    """A flash (camera flash, muzzle flash, lightning) changes a few frames and
    then the shot resumes. Spike at t is the flash's onset if a frame shortly
    after it resembles t-1 again, and its end if frame t resembles a frame just
    before the flash began. A real cut never "comes back"."""
    n = len(d)
    limit = p.flash_ratio * d[t]
    for j in range(t + 1, min(n, t + 1 + max_len)):          # onset: t-1 ~ t+j ?
        if feat.distance(t - 1, j) < limit:
            return True
    # end: does t resemble a frame from just before the flash began?
    return any(feat.distance(j, t) < limit for j in range(max(0, t - 1 - max_len), t - 1))


def detect_cuts(feat: FrameFeatures, p: CutParams | None = None) -> list[Transition]:
    p = p or CutParams()
    d = feat.pair_distance()
    n = len(d)
    if n < 3:
        return []
    thresh = np.maximum(p.min_cut, p.ratio * neighbourhood_p90(d, p.window))
    cuts: list[Transition] = []
    for t in range(1, n):
        if d[t] <= thresh[t]:
            continue
        lo, hi = max(1, t - p.min_shot), min(n, t + p.min_shot + 1)
        if d[t] < d[lo:hi].max():
            continue                                        # not the local peak
        if _is_flash(feat, d, t, p):
            continue
        if cuts and t - cuts[-1].frame < p.min_shot:
            continue
        cuts.append(Transition("cut", t, t, t, float(feat.times[t]), float(d[t])))
    return cuts


def detect_fades(feat: FrameFeatures, p: CutParams | None = None) -> list[Transition]:
    p = p or CutParams()
    black = (feat.luma_mean < p.fade_mean) & (feat.luma_std < p.fade_std)
    out: list[Transition] = []
    t = 0
    n = len(black)
    while t < n:
        if not black[t]:
            t += 1
            continue
        s = t
        while t < n and black[t]:
            t += 1
        e = t - 1
        if s == 0 or e == n - 1:
            continue                  # black at the very start/end of a clip is not a boundary
        mid = s + int(np.argmin(feat.luma_mean[s:e + 1]))
        out.append(Transition("fade", mid, s, e, float(feat.times[mid]),
                              float(255 - feat.luma_mean[mid])))
    return out


def detect_dissolves(feat: FrameFeatures, hard: list[Transition],
                     p: CutParams | None = None) -> list[Transition]:
    """Twin-comparison over the frames that no hard cut explains."""
    p = p or CutParams()
    d = feat.pair_distance()
    n = len(d)
    blocked = np.zeros(n, bool)
    for c in hard:
        blocked[max(0, c.start - 2):min(n, c.end + 3)] = True
    out: list[Transition] = []
    t = 1
    while t < n:
        if blocked[t] or d[t] < p.grad_low:
            t += 1
            continue
        s = t
        while t < n and not blocked[t] and d[t] >= p.grad_low * 0.5 and t - s < p.grad_max_len:
            t += 1
        e = t - 1
        if e - s + 1 >= p.grad_min_len:
            total = feat.distance(max(0, s - 1), min(n - 1, e))
            # A camera move or a person crossing frame also produces a run of
            # moderate distances; what separates a dissolve is that the frames
            # on either side are as different as a cut AND the run is smooth
            # (no single pair explains the change).
            if total >= p.grad_high and d[s:e + 1].max() < 0.75 * total:
                mid = (s + e) // 2
                out.append(Transition("dissolve", mid, s, e, float(feat.times[mid]), total))
        t = max(t, s + 1)
    return out


def detect_transitions(feat: FrameFeatures, p: CutParams | None = None,
                       gradual: bool = True) -> list[Transition]:
    p = p or CutParams()
    cuts = detect_cuts(feat, p)
    found = list(cuts)
    if gradual:
        fades = detect_fades(feat, p)
        found += [f for f in fades if all(abs(f.frame - c.frame) >= p.min_shot for c in cuts)]
        found += detect_dissolves(feat, found, p)
    found.sort(key=lambda x: x.frame)
    merged: list[Transition] = []
    for tr in found:                   # keep one boundary per min_shot neighbourhood
        if merged and tr.start - merged[-1].end < p.min_shot // 2:
            if tr.score > merged[-1].score:
                merged[-1] = tr
            continue
        merged.append(tr)
    return merged


def shots_from_transitions(transitions: list[Transition], n_frames: int,
                           times: np.ndarray) -> list[Shot]:
    bounds = [0] + [tr.frame for tr in transitions if 0 < tr.frame < n_frames] + [n_frames]
    shots = []
    for a, b in zip(bounds[:-1], bounds[1:], strict=True):
        if b <= a:
            continue
        shots.append(Shot(len(shots), a, b - 1, float(times[a]), float(times[b - 1])))
    return shots
