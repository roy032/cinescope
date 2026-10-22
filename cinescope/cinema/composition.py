"""Composition: where the eye goes in the frame, and how the frame is balanced.

    saliency      what stands out, as the mean of two classic models: the spectral
                  residual of the log spectrum (Hou & Zhang 2007: whatever the
                  image's own smooth spectrum does not predict) and colour
                  contrast against the frame's mean Lab colour (Achanta et al.
                  2009). Letterbox bars are trimmed first
    centre of interest
                  the saliency-weighted centroid of the salient region: pixels at
                  least half as salient as the peak (and at most the top fifth of
                  the frame), in 0..1 picture coordinates
    placement     where it sits horizontally: "centre" (within 1/12 of the
                  middle), "thirds" (within 1/12 of a third line) or "edge"; and
                  "power point" when it is near a thirds intersection both ways
    balance       right minus left saliency mass over the total, -1..1
    headroom      the space above the salient region, a fraction of the height
    focus         the edge sharpness inside the salient region over that outside
                  it (strong-edge Laplacian over strong-edge gradient: how narrow
                  the edges are, whatever their contrast); "shallow" (a sharp subject on a soft background)
                  when it is at least SHALLOW

The 1/12 bands split the frame halfway between the centre and the third lines;
SHALLOW is a round number fixed up front, not fitted to any clip.
"""
from __future__ import annotations

import numpy as np

from ..imgproc.color import rgb_to_lab
from ..imgproc.filters import correlate2d, gaussian_blur

WORK_W = 128            # saliency is computed at this width
TOP_SHARE = 0.2         # the salient region is at most the top fifth of the pixels ...
HALF = 0.5              # ... at least half as salient as the peak
BAND = 1 / 12           # half the gap between the centre line and a third line
SHALLOW = 1.5           # focus ratio for "shallow focus"
BAR_LUMA = 0.06         # rows darker than this at the top / bottom are letterbox bars
LAPLACE = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], np.float64)


def _unit(rgb: np.ndarray) -> np.ndarray:
    a = np.asarray(rgb)
    return a.astype(np.float64) / 255.0 if a.dtype == np.uint8 else a.astype(np.float64)


def _resize(img: np.ndarray, width: int) -> np.ndarray:
    """Area-average resize to `width` (aspect kept); smaller images are returned as is."""
    h, w = img.shape[:2]
    if w <= width:
        return img
    nh = max(1, round(h * width / w))
    ys = np.linspace(0, h, nh + 1).astype(int)
    xs = np.linspace(0, w, width + 1).astype(int)
    extra = (1,) * (img.ndim - 2)
    rows = np.add.reduceat(img, ys[:-1], axis=0) / np.diff(ys).reshape(-1, 1, *extra)
    return np.add.reduceat(rows, xs[:-1], axis=1) / np.diff(xs).reshape(1, -1, *extra)


def trim_bars(f: np.ndarray) -> np.ndarray:
    """Cut dark letterbox rows off the top and bottom (at most a quarter each)."""
    y = f.mean(-1).mean(1)
    h = len(y)
    top = 0
    while top < h // 4 and y[top] < BAR_LUMA:
        top += 1
    bot = h
    while bot > h - h // 4 and y[bot - 1] < BAR_LUMA:
        bot -= 1
    return f[top:bot] if bot - top > h // 2 else f      # all dark: a black frame, not bars


def saliency(rgb: np.ndarray) -> np.ndarray:
    """0..1 saliency map of one frame at WORK_W width (after trimming letterbox bars)."""
    f = _resize(trim_bars(_unit(rgb)), WORK_W)
    g = f @ np.array([0.299, 0.587, 0.114])
    F = np.fft.fft2(g)
    amp = np.log(np.abs(F) + 1e-9)
    resid = amp - correlate2d(amp, np.full((3, 3), 1 / 9))
    sr = np.abs(np.fft.ifft2(np.exp(resid + 1j * np.angle(F)))) ** 2
    sigma = max(1.0, f.shape[1] / 40)
    sr = gaussian_blur(sr, sigma)
    lab = rgb_to_lab(f)
    cc = np.linalg.norm(gaussian_blur_rgb(lab, sigma / 2) - lab.reshape(-1, 3).mean(0), axis=-1)
    out = 0.5 * _norm(sr) + 0.5 * _norm(cc)
    return _norm(out)


def gaussian_blur_rgb(img: np.ndarray, sigma: float) -> np.ndarray:
    return np.stack([gaussian_blur(img[..., c], sigma) for c in range(img.shape[-1])], -1)


def _norm(m: np.ndarray) -> np.ndarray:
    lo, hi = float(m.min()), float(m.max())
    return (m - lo) / (hi - lo) if hi - lo > 1e-12 else np.zeros_like(m)


def placement(cx: float, cy: float) -> str:
    if abs(cx - 0.5) < BAND:
        return "centre"
    if min(abs(cx - 1 / 3), abs(cx - 2 / 3)) < BAND:
        if min(abs(cy - 1 / 3), abs(cy - 2 / 3)) < BAND:
            return "power point"
        return "thirds"
    return "edge"


def edge_sharpness(lap: np.ndarray, grad: np.ndarray, sel: np.ndarray) -> float | None:
    """Strong-edge Laplacian over strong-edge gradient in a region: for an edge blurred
    by a Gaussian of width s this goes as 1/s whatever the edge's contrast. None when
    the region has no edges at all (a flat or fully defocused background)."""
    if not sel.any():
        return None
    gq = float(np.quantile(grad[sel], 0.9))
    if gq < 0.02:
        return None
    return float(np.quantile(lap[sel], 0.9)) / gq


def focus_ratio(g: np.ndarray, m: np.ndarray) -> float:
    """Sharpness of the salient region over the rest (capped at 99)."""
    from ..imgproc.filters import gradient_magnitude

    lap = np.abs(correlate2d(g, LAPLACE))
    grad = gradient_magnitude(g)[0] / 8.0          # Sobel sums 8 weighted differences
    a, b = edge_sharpness(lap, grad, m), edge_sharpness(lap, grad, ~m)
    if a is None:
        return 1.0
    if b is None or b < a / 99:
        return 99.0
    return a / b


def frame_composition(rgb: np.ndarray) -> dict:
    """Composition of one frame (see the module docstring)."""
    f = _resize(trim_bars(_unit(rgb)), WORK_W)
    s = saliency(rgb)
    h, w = s.shape
    thr = max(float(np.quantile(s, 1 - TOP_SHARE)), HALF)
    m = s >= thr
    wt = np.where(m, s, 0.0)
    tot = wt.sum()
    yy, xx = np.mgrid[0:h, 0:w]
    if tot <= 1e-12:
        cx = cy = 0.5
    else:
        cx = float((wt * (xx + 0.5)).sum() / tot / w)
        cy = float((wt * (yy + 0.5)).sum() / tot / h)
    left, right = float(s[:, : w // 2].sum()), float(s[:, (w + 1) // 2:].sum())
    rows = np.flatnonzero(m.any(1))
    head = float(rows[0] / h) if len(rows) else 0.0
    g = f @ np.array([0.299, 0.587, 0.114])
    ratio = focus_ratio(g, m)
    return {"x": round(cx, 3), "y": round(cy, 3), "placement": placement(cx, cy),
            "balance": round((right - left) / max(right + left, 1e-9), 3), "headroom": round(head, 3),
            "focus_ratio": round(min(ratio, 99.0), 2), "shallow": bool(ratio >= SHALLOW)}


def shot_composition(frames: np.ndarray) -> dict | None:
    """The mean over a shot's frames; placement from the mean centre of interest."""
    rows = [frame_composition(fr) for fr in frames]
    if not rows:
        return None
    cx = float(np.mean([r["x"] for r in rows]))
    cy = float(np.mean([r["y"] for r in rows]))
    fr = float(np.median([r["focus_ratio"] for r in rows]))
    return {"x": round(cx, 3), "y": round(cy, 3), "placement": placement(cx, cy),
            "balance": round(float(np.mean([r["balance"] for r in rows])), 3),
            "headroom": round(float(np.mean([r["headroom"] for r in rows])), 3),
            "focus_ratio": round(fr, 2), "shallow": sum(r["shallow"] for r in rows) * 2 > len(rows)}


PLACES = ("centre", "thirds", "power point", "edge")


def summarise(per_shot: list[dict | None], durations: list[float], grid: int = 6) -> dict:
    """Screen-time weighted shares, the clip's habits in a sentence, and a heat map of
    where the centre of interest sits (grid x grid, rows top to bottom)."""
    rows = [(c, d) for c, d in zip(per_shot, durations, strict=False) if c]
    if not rows:
        return {}
    w = np.array([d for _, d in rows], float) + 1e-9
    w = w / w.sum()
    share = {p: round(float(sum(wi for (c, _), wi in zip(rows, w, strict=True) if c["placement"] == p)), 3)
             for p in PLACES}
    heat = np.zeros((grid, grid))
    for (c, _), wi in zip(rows, w, strict=True):
        heat[min(int(c["y"] * grid), grid - 1), min(int(c["x"] * grid), grid - 1)] += wi
    shallow = round(float(sum(wi for (c, _), wi in zip(rows, w, strict=True) if c["shallow"])), 3)
    bal = float(sum(wi * c["balance"] for (c, _), wi in zip(rows, w, strict=True)))
    thirds = share["thirds"] + share["power point"]
    if share["centre"] >= 0.5:
        style = "mostly centred, symmetrical framing"
    elif thirds >= 0.5:
        style = "mostly framed on the thirds"
    elif share["edge"] >= 0.35:
        style = "often off-centre, with the subject near the frame's edge"
    else:
        style = "a mix of centred and thirds framing"
    focus = "shallow focus" if shallow >= 0.5 else "deep focus" if shallow <= 0.15 else "mixed focus"
    return {"share": share, "thirds_share": round(thirds, 3), "shallow_share": shallow,
            "balance": round(bal, 3), "headroom": round(float(sum(wi * c["headroom"] for (c, _), wi in
                                                                    zip(rows, w, strict=True))), 3),
            "heatmap": np.round(heat, 3).tolist(), "description": f"{style.capitalize()}; {focus}."}
