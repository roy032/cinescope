"""Color spaces in NumPy: RGB <-> HSV and RGB <-> CIE L*a*b*.

Why Lab matters for this project: cut detection and palettes both need a
*distance* between colors. In RGB, equal numeric steps are not equal visual
steps (our eyes are far more sensitive to changes in green than in blue, and
to lightness than to hue). Lab was designed so that Euclidean distance
approximates perceived difference (ΔE*76 ≈ 2.3 is roughly one "just noticeable
difference"). So k-means in Lab finds colors a viewer would call different;
k-means in RGB spends clusters on differences nobody can see.

Pipeline for Lab: sRGB (gamma-encoded, 0..255) -> linear RGB -> XYZ (D65) -> Lab.
Forgetting the linearisation step is the classic bug: the result "looks
plausible" and is wrong by several ΔE.
"""
from __future__ import annotations

import numpy as np

# sRGB -> XYZ (D65), IEC 61966-2-1
_M_RGB2XYZ = np.array([[0.4124564, 0.3575761, 0.1804375],
                       [0.2126729, 0.7151522, 0.0721750],
                       [0.0193339, 0.1191920, 0.9503041]])
_M_XYZ2RGB = np.linalg.inv(_M_RGB2XYZ)
_WHITE_D65 = np.array([0.95047, 1.0, 1.08883])
_EPS = 216 / 24389          # (6/29)^3
_KAPPA = 24389 / 27


def _unit(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb)
    return rgb.astype(np.float64) / 255.0 if rgb.dtype == np.uint8 else rgb.astype(np.float64)


def srgb_to_linear(c: np.ndarray) -> np.ndarray:
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c: np.ndarray) -> np.ndarray:
    c = np.clip(c, 0, None)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055)


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """rgb: uint8 0..255 or float 0..1, shape (..., 3). Returns L in 0..100, a/b ≈ -128..127."""
    xyz = srgb_to_linear(_unit(rgb)) @ _M_RGB2XYZ.T / _WHITE_D65
    f = np.where(xyz > _EPS, np.cbrt(xyz), (_KAPPA * xyz + 16) / 116)
    lab = np.empty_like(xyz)
    lab[..., 0] = 116 * f[..., 1] - 16
    lab[..., 1] = 500 * (f[..., 0] - f[..., 1])
    lab[..., 2] = 200 * (f[..., 1] - f[..., 2])
    return lab


def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    """Inverse of rgb_to_lab. Returns float sRGB in 0..1 (clipped)."""
    lab = np.asarray(lab, dtype=np.float64)
    fy = (lab[..., 0] + 16) / 116
    fx = fy + lab[..., 1] / 500
    fz = fy - lab[..., 2] / 200
    f = np.stack([fx, fy, fz], axis=-1)
    xyz = np.where(f ** 3 > _EPS, f ** 3, (116 * f - 16) / _KAPPA) * _WHITE_D65
    return np.clip(linear_to_srgb(xyz @ _M_XYZ2RGB.T), 0, 1)


def delta_e76(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """CIE76 color difference: Euclidean distance in Lab."""
    return np.linalg.norm(np.asarray(lab1) - np.asarray(lab2), axis=-1)


def rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    """H in degrees 0..360, S and V in 0..1."""
    c = _unit(rgb)
    r, g, b = c[..., 0], c[..., 1], c[..., 2]
    v = c.max(axis=-1)
    mn = c.min(axis=-1)
    delta = v - mn
    s = np.where(v > 0, delta / np.where(v > 0, v, 1), 0.0)
    safe = np.where(delta > 0, delta, 1)
    h = np.select(
        [delta == 0, v == r, v == g],
        [0.0, ((g - b) / safe) % 6, (b - r) / safe + 2],
        (r - g) / safe + 4,
    ) * 60.0
    return np.stack([h, s, v], axis=-1)


def hsv_to_rgb(hsv: np.ndarray) -> np.ndarray:
    """Inverse of rgb_to_hsv. Returns float RGB in 0..1."""
    hsv = np.asarray(hsv, dtype=np.float64)
    h, s, v = hsv[..., 0] / 60.0, hsv[..., 1], hsv[..., 2]
    c = v * s
    x = c * (1 - np.abs(h % 2 - 1))
    m = v - c
    zeros = np.zeros_like(c)
    sector = np.floor(h).astype(int) % 6
    choices = [np.stack(t, -1) for t in ((c, x, zeros), (x, c, zeros), (zeros, c, x),
                                          (zeros, x, c), (x, zeros, c), (c, zeros, x))]
    rgb = np.select([sector[..., None] == i for i in range(6)], choices)
    return rgb + m[..., None]
