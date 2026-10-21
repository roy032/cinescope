"""The look of a shot: how it is lit and graded, in a colourist's words.

Every measure works on a handful of small RGB frames of the shot (the
pipeline's 96 px thumbnails), so a whole film costs seconds:

    brightness    mean Rec. 709 luma, 0..1
    contrast      RMS contrast (standard deviation of luma), 0..~0.5
    range         5th to 95th percentile of luma: how much of the tonal scale is used
    key           low-key (dark, hard shadows), high-key (bright, flat) or normal
    saturation    mean HSV chroma (max - min of R, G, B), 0..1
    warmth        mean CIE b* minus a quarter of a*, in Lab units: > 0 warm (amber),
                  < 0 cool (blue / teal). Daylight-neutral footage sits near 0.
    temperature   warm / neutral / cool from warmth
    teal_orange   0..1, how much of the colour sits in the two complementary bands of
                  the "teal and orange" blockbuster grade (skin-orange 15-50°, teal
                  165-215° in HSV hue), high only when both bands are present
    symmetry      0..1, correlation of the frame with its mirror image (1 = a perfectly
                  centred, symmetrical composition, the Kubrick / Wes Anderson look)
    crushed       share of pixels in near-black (luma < 0.03)
    clipped       share of pixels in near-white (luma > 0.97)
    aspect        aspect ratio of the picture inside any letterbox / pillarbox bars,
                  and its common name (2.39:1 scope, 1.85:1 flat, 16:9 ...)

Letterbox bars are found first (rows and columns that stay black in every frame
of the shot, from the edges in) and left out of every other measure, so a
scope film shown in a 16:9 frame is not called dark because of its bars.

The thresholds below are round numbers fixed from the definitions, not fitted
to any clip.
"""
from __future__ import annotations

import numpy as np

from ..imgproc.color import rgb_to_hsv, rgb_to_lab

_LUMA = np.array([0.2126, 0.7152, 0.0722])
LOW_KEY = 0.25          # mean luma below this, with real shadows: low-key
HIGH_KEY = 0.62         # mean luma above this, with a gentle tonal range: high-key
WARM, COOL = 6.0, -6.0  # warmth (Lab units) beyond these: warm / cool
ORANGE = (15.0, 50.0)   # HSV hue bands, degrees
TEAL = (165.0, 215.0)


def _unit(frames: np.ndarray) -> np.ndarray:
    f = np.asarray(frames)
    f = f.astype(np.float64) / 255.0 if f.dtype == np.uint8 else f.astype(np.float64)
    return f[None] if f.ndim == 3 else f


# common delivery / projection ratios, widest first
RATIOS = ((2.76, "2.76:1 Ultra Panavision"), (2.39, "2.39:1 scope"), (2.2, "2.20:1 70 mm"),
          (2.0, "2:1 Univisium"), (1.85, "1.85:1 flat"), (1.78, "16:9"), (1.66, "1.66:1 European"),
          (1.43, "1.43:1 IMAX"), (1.37, "1.37:1 Academy"), (1.33, "4:3"), (1.0, "1:1 square"))


def ratio_name(r: float) -> str:
    return min(RATIOS, key=lambda kv: abs(kv[0] - r))[1]


def active_area(y: np.ndarray, black: float = 0.06) -> tuple[int, int, int, int]:
    """(top, bottom, left, right) of the picture inside black bars, from (N, H, W) luma.
    A bar is a run of rows (columns) from the edge that is black in every frame; the
    picture always keeps at least half the frame, so a dark shot is not cropped away."""
    h, w = y.shape[1:]
    rows = y.max(axis=(0, 2)) < black
    cols = y.max(axis=(0, 1)) < black

    def run(mask: np.ndarray) -> int:
        nz = np.flatnonzero(~mask)
        return int(nz[0]) if len(nz) else len(mask)

    top, bottom = run(rows), run(rows[::-1])
    left, right = run(cols), run(cols[::-1])
    if top + bottom > h // 2:
        top = bottom = 0
    if left + right > w // 2:
        left = right = 0
    return top, h - bottom, left, w - right


def key_label(brightness: float, contrast: float, rng: float) -> str:
    if brightness < LOW_KEY and contrast > 0.08:
        return "low-key"
    if brightness > HIGH_KEY and rng < 0.6:
        return "high-key"
    return "normal"


def temperature_label(warmth: float) -> str:
    return "warm" if warmth > WARM else "cool" if warmth < COOL else "neutral"


def teal_orange_score(hsv: np.ndarray) -> float:
    """sqrt(orange share x teal share) of the chroma-weighted colour, x2 so that an
    even split between the two bands scores 1 and either band alone scores 0."""
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    w = s * v * (s > 0.12) * (v > 0.08)
    tot = w.sum()
    if tot <= 1e-9:
        return 0.0
    orange = w[(h >= ORANGE[0]) & (h < ORANGE[1])].sum() / tot
    teal = w[(h >= TEAL[0]) & (h < TEAL[1])].sum() / tot
    return float(min(1.0, 2.0 * np.sqrt(orange * teal)))


def symmetry_score(gray: np.ndarray) -> float:
    """Correlation between the frame and its mirror image, clipped to 0..1 and averaged
    over frames: 1 for a frame equal to its mirror, about 0 when the two halves are
    unrelated. A flat frame has no composition to judge and scores 0."""
    g = gray if gray.ndim == 3 else gray[None]
    out = []
    for f in g:
        a = f - f.mean()
        den = float((a * a).sum())
        out.append(0.0 if den < 1e-9 else float((a * a[:, ::-1]).sum()) / den)
    return float(np.clip(np.mean(out), 0.0, 1.0))


def shot_look(frames: np.ndarray) -> dict:
    """frames: (N, H, W, 3) uint8 or float RGB of one shot (or one (H, W, 3) frame)."""
    f = _unit(frames)
    y = f @ _LUMA
    h, w = y.shape[1:]
    t, b, left, r = active_area(y)
    aspect = (r - left) / max(b - t, 1)
    f, y = f[:, t:b, left:r], y[:, t:b, left:r]
    hsv = rgb_to_hsv(f)
    lab = rgb_to_lab(f)
    brightness = float(y.mean())
    contrast = float(np.mean([fr.std() for fr in y]))
    lo, hi = np.percentile(y, [5, 95])
    warmth = float(lab[..., 2].mean() - 0.25 * lab[..., 1].mean())
    out = {
        "brightness": brightness,
        "contrast": contrast,
        "range": float(hi - lo),
        "saturation": float((f.max(-1) - f.min(-1)).mean()),
        "warmth": warmth,
        "teal_orange": teal_orange_score(hsv),
        "symmetry": symmetry_score(y),
        "crushed": float((y < 0.03).mean()),
        "clipped": float((y > 0.97).mean()),
    }
    out = {k: round(v, 4) for k, v in out.items()}
    out["key"] = key_label(brightness, contrast, out["range"])
    out["temperature"] = temperature_label(warmth)
    out["aspect"] = round(float(aspect), 3)
    out["bars"] = (t, b, left, r) != (0, h, 0, w)
    return out


_NUMERIC = ("brightness", "contrast", "range", "saturation", "warmth", "teal_orange", "symmetry",
            "crushed", "clipped")


def film_look(looks: list[dict | None], durations: list[float]) -> dict:
    """Screen-time weighted averages over the shots, the share of screen time in each
    key and temperature, and a one-line description of the overall look."""
    rows = [(lk, max(d, 0.0)) for lk, d in zip(looks, durations, strict=True) if lk]
    if not rows:
        return {}
    w = np.array([d for _, d in rows]) + 1e-9
    w = w / w.sum()
    out: dict = {k: round(float(sum(wi * lk[k] for (lk, _), wi in zip(rows, w, strict=True))), 4)
                 for k in _NUMERIC}
    for cat in ("key", "temperature"):
        share: dict[str, float] = {}
        for (lk, _), wi in zip(rows, w, strict=True):
            share[lk[cat]] = share.get(lk[cat], 0.0) + float(wi)
        out[f"{cat}_share"] = {k: round(v, 3) for k, v in sorted(share.items(), key=lambda kv: -kv[1])}
    out["key"] = key_label(out["brightness"], out["contrast"], out["range"])
    out["temperature"] = temperature_label(out["warmth"])
    # the ratio most of the screen time is shown in (a weighted median, so a few
    # very dark shots whose edges look like bars do not move it)
    asp = sorted((lk["aspect"], wi) for (lk, _), wi in zip(rows, w, strict=True))
    acc = np.cumsum([wi for _, wi in asp])
    out["aspect"] = asp[int(np.searchsorted(acc, 0.5))][0]
    out["aspect_name"] = ratio_name(out["aspect"])
    out["letterboxed"] = round(float(sum(wi for (lk, _), wi in zip(rows, w, strict=True) if lk["bars"])), 3)
    out["description"] = describe(out)
    return out


def describe(s: dict) -> str:
    """'Low-key, cool, desaturated, high contrast; teal-and-orange grade.'"""
    words = []
    words.append({"low-key": "Low-key", "high-key": "High-key"}.get(s["key"], "Naturally exposed"))
    words.append(s["temperature"])
    words.append("desaturated" if s["saturation"] < 0.12 else "vivid" if s["saturation"] > 0.35 else "moderately saturated")
    words.append("high contrast" if s["contrast"] > 0.24 else "soft contrast" if s["contrast"] < 0.12 else "medium contrast")
    text = ", ".join(words)
    extras = [f"{s['aspect_name']} frame"] if "aspect_name" in s else []
    if s["teal_orange"] > 0.5:
        extras.append("teal-and-orange grade")
    if s["symmetry"] > 0.75:
        extras.append("symmetrical framing")
    if s["crushed"] > 0.15:
        extras.append("crushed blacks")
    if s["clipped"] > 0.1:
        extras.append("blown highlights")
    return text + (f"; {', '.join(extras)}." if extras else ".")


def key_timeline(shots: list[dict], looks: list[dict | None]) -> list[dict]:
    """Runs of adjacent shots with the same lighting key: [{"t0", "t1", "label"}]."""
    out: list[dict] = []
    for s, lk in zip(shots, looks, strict=True):
        if not lk:
            continue
        lab = lk["key"]
        if out and out[-1]["label"] == lab and abs(out[-1]["t1"] - s["t0"]) < 0.1:
            out[-1]["t1"] = s["t1"]
        else:
            out.append({"t0": s["t0"], "t1": s["t1"], "label": lab})
    return out
