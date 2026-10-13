"""2D → stereo 3D by depth-image-based rendering (Fehn, 2004), in NumPy.

A stereo pair is two cameras a small baseline b apart. A point at depth Z
appears shifted horizontally between them by the disparity d = f·b / Z — near
things shift a lot, the far background hardly at all. Given one image and its
depth map, synthesise both eyes by shifting every pixel by ±d/2:

1. Disparity from depth. Only relative depth is known (from SfM up to scale,
   from MiDaS up to scale and shift), so the disparity range is a creative
   choice, as in a real 2D→3D conversion: `max_disp` (in % of image width) for
   the nearest point, zero at the convergence depth (the screen plane), and
   negative (behind the screen) beyond it. ~2-3% of width is comfortable.
2. Forward warping with a z-buffer. Every source pixel is splatted to its
   shifted position; where two land on the same target pixel the nearer wins
   (sorting far-to-near and letting later writes win does exactly that).
3. Hole filling. Shifting exposes background that the single camera never saw
   (disocclusions, at the edges of near objects). Holes are filled from the
   *background* side — the neighbouring pixel farther away — because the
   unseen region belongs to the background, not to the foreground object;
   filling from the foreground smears the object's edge ("rubber sheet").
"""
from __future__ import annotations

import numpy as np


def depth_range(inv_depth: np.ndarray, convergence: float = 0.25) -> tuple[float, float, float]:
    """(far, near, screen-plane) levels of an inverse-depth map: its 1st and 99th
    percentiles and the `convergence` quantile. A video smooths these over time
    so the depth budget does not pump from frame to frame."""
    v = np.asarray(inv_depth, np.float64)
    return (float(np.nanpercentile(v, 1)), float(np.nanpercentile(v, 99)),
            float(np.nanpercentile(v, 100 * convergence)))


def disparity_from_inverse_depth(inv_depth: np.ndarray, width: int, max_disp_pct: float = 2.5,
                                 convergence: float = 0.25, levels: tuple[float, float, float] | None = None
                                 ) -> np.ndarray:
    """Pixel disparity from inverse depth (any affine scale). The `convergence`
    quantile of inverse depth sits on the screen plane (zero disparity)."""
    v = np.asarray(inv_depth, np.float64)
    lo, hi, conv = levels or depth_range(v, convergence)
    n = np.clip((v - lo) / max(hi - lo, 1e-9), 0, 1)                    # 0 far … 1 near
    c = np.clip((conv - lo) / max(hi - lo, 1e-9), 0, 1)
    return (n - c) / max(1 - c, 1e-9) * max_disp_pct / 100 * width


def forward_warp(img: np.ndarray, shift: np.ndarray, nearness: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Move each pixel horizontally by `shift` pixels; where several land on one
    target pixel the one with the largest `nearness` (disparity) wins — a
    z-buffer. Returns (warped image, hole mask)."""
    h, w = shift.shape
    out = np.zeros_like(img)
    filled = np.zeros((h, w), bool)
    ys, xs = np.mgrid[0:h, 0:w]
    xt = np.rint(xs + shift).astype(int)
    ok = (xt >= 0) & (xt < w)
    order = np.argsort(-nearness[ok], kind="stable")                    # nearest first
    yy, xx, tt = ys[ok][order], xs[ok][order], xt[ok][order]
    _, first = np.unique(yy * w + tt, return_index=True)               # per target: the nearest source
    yy, xx, tt = yy[first], xx[first], tt[first]
    out[yy, tt] = img[yy, xx]
    filled[yy, tt] = True
    return out, ~filled


def fill_holes(img: np.ndarray, holes: np.ndarray, disp_warped: np.ndarray, side: str = "background") -> np.ndarray:
    """Fill each horizontal run of holes from whichever end is farther (smaller
    disparity) — or nearer, with side="foreground", to show why that is wrong."""
    out = img.copy()
    h, w = holes.shape
    for y in range(h):
        row = holes[y]
        if not row.any():
            continue
        x = 0
        while x < w:
            if not row[x]:
                x += 1
                continue
            x0 = x
            while x < w and row[x]:
                x += 1
            left, right = x0 - 1, x                                     # valid neighbours (maybe outside)
            cands = [c for c in (left, right) if 0 <= c < w]
            if not cands:
                continue
            pick = min if side == "background" else max
            src = pick(cands, key=lambda c: disp_warped[y, c])
            out[y, x0:x] = out[y, src]
    return out


def render_view(img: np.ndarray, disp: np.ndarray, shift: float, side: str = "background"
                ) -> tuple[np.ndarray, np.ndarray]:
    """Novel view with every pixel moved by shift·disp (shift = ±0.5 for the two
    eyes, -1 for a camera one full baseline to the right). Returns (view, holes)."""
    warped, holes = forward_warp(img, shift * disp, disp)
    dwarp, _ = forward_warp(disp[..., None], shift * disp, disp)
    return fill_holes(warped, holes, np.where(holes, np.inf, dwarp[..., 0]), side), holes


def stereo_pair(img: np.ndarray, disp: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """(left, right, hole fraction) views from one image and its disparity."""
    left, h1 = render_view(img, disp, +0.5)       # left eye sees near things shifted right
    right, h2 = render_view(img, disp, -0.5)
    return left, right, float((h1.mean() + h2.mean()) / 2)


def anaglyph(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Red-cyan anaglyph: red channel from the left eye, green and blue from the right."""
    out = right.copy()
    out[..., 0] = left[..., 0]
    return out


def side_by_side(left: np.ndarray, right: np.ndarray, half: bool = True) -> np.ndarray:
    """Half-width SBS (what 3D TVs expect) or full-width."""
    if half:
        left, right = left[:, ::2], right[:, ::2]
    return np.concatenate([left, right], 1)
