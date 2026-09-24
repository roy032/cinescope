"""Per-frame features for shot-boundary detection, computed once per video.

A cut is a discontinuity *between* two frames, so the detector needs one
number per adjacent frame pair: how different are frame t-1 and frame t?

Colour histograms answer that question robustly. They ignore *where* colours
are, so a person walking across a static shot barely changes the histogram,
while a cut to a different set changes it a lot. That same property is their
weakness: two different shots of the same room (a shot/reverse-shot dialogue)
can have similar histograms, which is why an edge-based signal is kept too.

Features per frame
  hist   HSV joint histogram (16 hue x 4 sat x 4 val = 256 bins), L1-normalised.
         Hue gets the most bins because it is the most shot-specific channel;
         value gets few so that lighting flicker moves little mass.
  luma   mean and standard deviation of brightness — fades go to black (low
         mean AND low std), which histograms alone describe poorly.
  edges  fraction of strong-gradient pixels, a cheap "structure" signal.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..imgproc.color import rgb_to_hsv
from ..imgproc.filters import to_gray

H_BINS, S_BINS, V_BINS = 16, 4, 4


def hsv_histogram(rgb: np.ndarray) -> np.ndarray:
    """256-bin joint HSV histogram, L1-normalised, via one np.bincount."""
    hsv = rgb_to_hsv(rgb).reshape(-1, 3)
    h = np.minimum((hsv[:, 0] / 360.0 * H_BINS).astype(int), H_BINS - 1)
    s = np.minimum((hsv[:, 1] * S_BINS).astype(int), S_BINS - 1)
    v = np.minimum((hsv[:, 2] * V_BINS).astype(int), V_BINS - 1)
    idx = (h * S_BINS + s) * V_BINS + v
    hist = np.bincount(idx, minlength=H_BINS * S_BINS * V_BINS).astype(np.float64)
    return hist / max(hist.sum(), 1.0)


def chi2_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Symmetric chi-square distance between histograms, in [0, 1] for L1-normalised input.
    Works on single histograms or row-wise on (N, bins) arrays."""
    num = (a - b) ** 2
    den = a + b
    return 0.5 * np.sum(np.where(den > 0, num / np.where(den > 0, den, 1), 0.0), axis=-1)


def edge_density(gray: np.ndarray, thresh: float = 60.0) -> float:
    """Share of pixels whose central-difference gradient magnitude exceeds `thresh`.
    (Central differences rather than the full Sobel: this runs on every frame.)"""
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
    gy[1:-1, :] = gray[2:, :] - gray[:-2, :]
    return float(np.mean(np.hypot(gx, gy) > thresh))


def _shrink(rgb: np.ndarray, width: int) -> np.ndarray:
    if rgb.shape[1] <= width:
        return rgb
    from PIL import Image

    h = max(1, round(rgb.shape[0] * width / rgb.shape[1]))
    return np.asarray(Image.fromarray(rgb).resize((width, h), Image.BILINEAR))


@dataclass
class FrameFeatures:
    times: np.ndarray        # (N,) seconds
    hist: np.ndarray         # (N, 256)
    luma_mean: np.ndarray    # (N,) 0..255
    luma_std: np.ndarray     # (N,)
    edges: np.ndarray        # (N,)
    thumbs: np.ndarray | None = None   # (N, h, w, 3) small RGB frames, for palettes/plots

    def __len__(self) -> int:
        return len(self.times)

    def pair_distance(self) -> np.ndarray:
        """d[t] = distance between frame t-1 and frame t (d[0] = 0)."""
        d = np.zeros(len(self))
        d[1:] = chi2_distance(self.hist[:-1], self.hist[1:])
        return d

    def distance(self, i: int, j: int) -> float:
        return float(chi2_distance(self.hist[i], self.hist[j]))


def extract(frames, keep_thumbs: bool = True, thumb_width: int = 64) -> FrameFeatures:
    """frames: iterable of cinescope.io.video.Frame (already resized for speed)."""
    times, hists, means, stds, edges, thumbs = [], [], [], [], [], []
    for f in frames:
        gray = to_gray(f.rgb)
        times.append(f.t)
        hists.append(hsv_histogram(f.rgb))
        means.append(float(gray.mean()))
        stds.append(float(gray.std()))
        edges.append(edge_density(gray))
        if keep_thumbs:
            thumbs.append(_shrink(f.rgb, thumb_width))
    return FrameFeatures(np.asarray(times), np.asarray(hists), np.asarray(means),
                         np.asarray(stds), np.asarray(edges),
                         np.stack(thumbs) if keep_thumbs and thumbs else None)
