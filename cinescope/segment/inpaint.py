"""Exemplar-based inpainting (Criminisi, Pérez & Toyama, 2004), from scratch,
and the PSNR / SSIM used to score it.

Diffusion inpainting (cv2.inpaint's Telea and Navier-Stokes) propagates
colours smoothly inward: fine for scratches, but a large hole comes out as a
smear because texture cannot be diffused. Exemplar-based methods copy whole
patches of real texture from elsewhere in the image instead. The question is
the *order*: copy texture blindly and straight edges (a wall, a kerb) break.
Criminisi et al. fill the boundary patch with the highest priority first,

    P(p) = C(p) * D(p)
    C(p) = mean confidence of the patch's already-filled pixels   (1 outside
           the hole; a filled pixel inherits its patch's C, so confidence
           decays toward the hole's centre)
    D(p) = |isophote(p) · normal(p)| / 255   (strong edges that hit the hole
           boundary head-on go first, so structure is continued into the hole
           before texture closes over it)

then search the known region for the patch that best matches (SSD over the
known pixels of the target patch), and copy its pixels into the hole part of
the target. Repeat until the hole is gone.

The search is restricted to a window around the hole (`search` px) — both for
speed and because nearby texture is usually the right texture.
"""
from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import ndimage as ndi


def criminisi(img: np.ndarray, hole: np.ndarray, patch: int = 9, search: int = 60) -> np.ndarray:
    """img (H, W, 3) uint8, hole (H, W) bool -> inpainted uint8 image."""
    if not hole.any():
        return img.copy()
    h, w = hole.shape
    half = patch // 2
    ys, xs = np.nonzero(hole)
    # work on a crop: hole bounding box + search margin (+ patch for borders)
    m = search + half
    cy0, cy1 = max(0, ys.min() - m), min(h, ys.max() + 1 + m)
    cx0, cx1 = max(0, xs.min() - m), min(w, xs.max() + 1 + m)
    out = img.copy()
    out[cy0:cy1, cx0:cx1] = _criminisi_crop(img[cy0:cy1, cx0:cx1], hole[cy0:cy1, cx0:cx1],
                                            patch, search)
    return out


def _criminisi_crop(img: np.ndarray, hole: np.ndarray, patch: int, search: int) -> np.ndarray:
    half = patch // 2
    # pad by half a patch so every pixel has a full patch; padding counts as
    # unknown-but-not-to-fill (never a source, never a target)
    pad = ((half, half), (half, half))
    im = np.pad(img.astype(np.float64), pad + ((0, 0),), mode="edge")
    hole = np.pad(hole, pad, constant_values=False)
    valid = np.pad(np.ones(img.shape[:2], bool), pad, constant_values=False)
    known = valid & ~hole
    conf = known.astype(np.float64)
    hp, wp = hole.shape
    st3 = np.ones((3, 3), bool)
    gray_w = np.array([0.299, 0.587, 0.114])
    n_patch = float(patch * patch)

    while hole.any():
        front = hole & ndi.binary_dilation(known, st3)
        fy, fx = np.nonzero(front)
        # confidence term: mean confidence over each front pixel's patch
        csum = ndi.uniform_filter(conf, patch, mode="constant") * n_patch
        c_term = csum[fy, fx] / n_patch
        # data term: isophote (gradient rotated 90°) at the strongest known
        # gradient in the patch, against the hole boundary's normal
        gray = im @ gray_w
        gy, gx = np.gradient(gray)
        gvalid = ndi.binary_erosion(known, st3)
        mag = np.where(gvalid, gx * gx + gy * gy, -1.0)
        nm = ndi.gaussian_filter(known.astype(np.float64), 1.0)
        ny, nx = np.gradient(nm)
        d_term = np.empty(len(fy))
        for i, (y, x) in enumerate(zip(fy, fx, strict=True)):
            y0, x0 = max(0, y - half), max(0, x - half)
            win = mag[y0:y + half + 1, x0:x + half + 1]
            k = int(np.argmax(win))
            if win.flat[k] <= 0:
                d_term[i] = 0.0
                continue
            yy, xx = np.unravel_index(k, win.shape)
            iso = np.array([-gy[y0 + yy, x0 + xx], gx[y0 + yy, x0 + xx]])
            n = np.array([nx[y, x], ny[y, x]])
            nn = np.hypot(*n)
            d_term[i] = abs(iso @ n) / (nn * 255.0) if nn > 0 else 0.0
        pri = c_term * (d_term + 1e-3)   # small floor: flat regions still fill in confidence order
        j = int(np.argmax(pri))
        py, px = int(fy[j]), int(fx[j])

        # target patch
        ty0, tx0 = py - half, px - half
        ty0 = int(np.clip(ty0, 0, hp - patch))
        tx0 = int(np.clip(tx0, 0, wp - patch))
        tgt = im[ty0:ty0 + patch, tx0:tx0 + patch]
        tknown = known[ty0:ty0 + patch, tx0:tx0 + patch].copy()
        tfill = hole[ty0:ty0 + patch, tx0:tx0 + patch].copy()

        # candidate source patches inside the search window, fully known
        sy0, sy1 = max(0, ty0 - search), min(hp, ty0 + patch + search)
        sx0, sx1 = max(0, tx0 - search), min(wp, tx0 + patch + search)
        win_im = im[sy0:sy1, sx0:sx1]
        win_ok = known[sy0:sy1, sx0:sx1]
        q = sliding_window_view(win_im, (patch, patch), axis=(0, 1))   # (Y, X, 3, p, p)
        ok = sliding_window_view(win_ok, (patch, patch)).all(axis=(-2, -1))
        if not ok.any():   # nothing fully known nearby: widen to the whole crop
            win_im, win_ok, sy0, sx0 = im, known, 0, 0
            q = sliding_window_view(win_im, (patch, patch), axis=(0, 1))
            ok = sliding_window_view(win_ok, (patch, patch)).all(axis=(-2, -1))
        if not ok.any():
            # no complete source patch anywhere: fall back to the mean known colour
            im[ty0:ty0 + patch, tx0:tx0 + patch][tfill] = im[known].mean(0)
            hole[ty0:ty0 + patch, tx0:tx0 + patch][tfill] = False
            known[ty0:ty0 + patch, tx0:tx0 + patch][tfill] = True
            continue
        kw = tknown.astype(np.float64)
        t = np.moveaxis(tgt, -1, 0)                                     # (3, p, p)
        ssd = np.einsum("yxcij,ij->yx", q * q, kw) - 2 * np.einsum("yxcij,cij->yx", q, t * kw)
        ssd = np.where(ok, ssd, np.inf)
        by, bx = np.unravel_index(int(np.argmin(ssd)), ssd.shape)
        src = im[sy0 + by:sy0 + by + patch, sx0 + bx:sx0 + bx + patch]
        tgt[tfill] = src[tfill]
        conf[ty0:ty0 + patch, tx0:tx0 + patch][tfill] = c_term[j]
        hole[ty0:ty0 + patch, tx0:tx0 + patch][tfill] = False
        known[ty0:ty0 + patch, tx0:tx0 + patch][tfill] = True

    out = im[half:-half, half:-half]
    return np.clip(out, 0, 255).round().astype(np.uint8)


def opencv_inpaint(img: np.ndarray, hole: np.ndarray, method: str = "telea", radius: int = 5) -> np.ndarray:
    import cv2

    flag = cv2.INPAINT_TELEA if method == "telea" else cv2.INPAINT_NS
    bgr = np.ascontiguousarray(img[:, :, ::-1])
    return cv2.inpaint(bgr, hole.astype(np.uint8) * 255, radius, flag)[:, :, ::-1]


# ----------------------------------------------------------------- metrics


def psnr(a: np.ndarray, b: np.ndarray, region: np.ndarray | None = None) -> float:
    d = (a.astype(np.float64) - b.astype(np.float64)) ** 2
    m = d[region].mean() if region is not None else d.mean()
    return float("inf") if m == 0 else float(10 * np.log10(255.0 ** 2 / m))


def ssim(a: np.ndarray, b: np.ndarray, sigma: float = 1.5) -> float:
    """Mean SSIM (Wang et al., 2004) with an 11-tap Gaussian window, on luma
    for colour images. Constants K1 = 0.01, K2 = 0.03, L = 255."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    if a.ndim == 3:
        w = np.array([0.299, 0.587, 0.114])
        a, b = a @ w, b @ w
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2

    def g(x):
        return ndi.gaussian_filter(x, sigma, truncate=3.5, mode="reflect")

    mu_a, mu_b = g(a), g(b)
    saa = g(a * a) - mu_a ** 2
    sbb = g(b * b) - mu_b ** 2
    sab = g(a * b) - mu_a * mu_b
    s = ((2 * mu_a * mu_b + c1) * (2 * sab + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (saa + sbb + c2))
    return float(s.mean())
