"""Box-prompted interactive segmentation in the GrabCut style, from scratch.

GrabCut (Rother, Kolmogorov & Blake, 2004) needs only a box around the object:

  1. everything outside the box is background for sure; inside starts as
     "probably foreground";
  2. fit a colour GMM to each side (gmm.py);
  3. graph cut (maxflow.py) on the energy
        E = sum_p -log p(colour_p | GMM of label_p)
            + gamma * sum_{p~q} exp(-beta |c_p - c_q|²) / dist(p, q) * [label_p != label_q]
     where beta = 1 / (2 <|c_p - c_q|²>) adapts the contrast term to the image;
  4. repeat 2-3: the GMMs learn from the new labels, the cut gets sharper.

Speed is the problem with doing this in pure Python, so the cut runs in two
stages over the same energy:

  superpixel stage  SLIC regions (split along the box edge, so each is wholly
                    inside or outside) are the graph nodes. A region's unary is
                    the sum of its pixels' unaries and the edge between two
                    regions is the sum of the pixel pairs crossing it — exactly
                    the pixel energy, restricted to labellings constant on each
                    region. ~10^3 nodes instead of ~10^5.
  pixel band stage  the boundary is then re-cut at pixel level inside a band a
                    few pixels wide around it. Pixels outside the band are held
                    at their label; their pull on band pixels becomes terminal
                    capacity.

Everything happens on a crop around the box (the outside-box region a margin
wide is enough to learn the background colours), downscaled if large.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi

from .gmm import GMM
from .maxflow import Graph, grid_edges
from .superpixels import slic

_HARD = 1e9


@dataclass
class GrabCutParams:
    n_iter: int = 5            # GMM / cut rounds at superpixel level
    gamma: float = 50.0        # smoothness weight (Rother et al. use 50)
    k: int = 5                 # GMM components per side
    margin: float = 0.25       # crop margin around the box, fraction of box size
    max_side: int = 200        # longest side of the working crop (pixels)
    sp_area: float = 60.0      # target pixels per superpixel (picked on training composites)
    min_segments: int = 200    # at least this many superpixels: small crops need finer regions
    band: int = 3              # half-width of the pixel refinement band (0: none)
    band_iter: int = 2         # GMM / cut rounds in the band
    min_component: float = 0.05  # drop fg islands smaller than this x the largest


def _crop_window(box, h, w, margin):
    x0, y0, x1, y1 = (float(v) for v in box)
    mx, my = margin * (x1 - x0) + 4, margin * (y1 - y0) + 4
    cx0, cy0 = int(max(0, np.floor(x0 - mx))), int(max(0, np.floor(y0 - my)))
    cx1, cy1 = int(min(w, np.ceil(x1 + mx))), int(min(h, np.ceil(y1 + my)))
    return cx0, cy0, cx1, cy1


def pairwise_weights(rgb: np.ndarray, gamma: float):
    """8-neighbour pixel pairs (i, j) and their smoothness weights."""
    h, w = rgb.shape[:2]
    i, j = grid_edges(h, w)
    c = rgb.reshape(-1, 3).astype(np.float64)
    d2 = ((c[i] - c[j]) ** 2).sum(1)
    beta = 1.0 / (2.0 * max(d2.mean(), 1e-6))
    # diagonal pairs are 1/sqrt(2) as strong (they are farther apart)
    yi, xi = np.divmod(i, w)
    yj, xj = np.divmod(j, w)
    dist = np.where((yi != yj) & (xi != xj), np.sqrt(2.0), 1.0)
    return i, j, gamma * np.exp(-beta * d2) / dist


def _unaries(pix: np.ndarray, fg_px: np.ndarray, bg_px: np.ndarray, gmms, k: int):
    """Re-fit both GMMs on the current labels and return per-pixel costs
    (cost of labelling fg, cost of labelling bg) = negative log-likelihoods."""
    fg_gmm, bg_gmm = gmms
    fg_gmm = GMM.fit(pix[fg_px], k) if fg_gmm is None else fg_gmm.refit(pix[fg_px])
    bg_gmm = GMM.fit(pix[bg_px], k) if bg_gmm is None else bg_gmm.refit(pix[bg_px])
    return -fg_gmm.loglik(pix), -bg_gmm.loglik(pix), (fg_gmm, bg_gmm)


def grabcut(rgb: np.ndarray, box, params: GrabCutParams | None = None) -> np.ndarray:
    """(H, W, 3) uint8 RGB and box (x0, y0, x1, y1) -> (H, W) bool foreground mask."""
    p = params or GrabCutParams()
    h, w = rgb.shape[:2]
    cx0, cy0, cx1, cy1 = _crop_window(box, h, w, p.margin)
    crop = rgb[cy0:cy1, cx0:cx1]
    ch, cw = crop.shape[:2]
    scale = min(1.0, p.max_side / max(ch, cw))
    if scale < 1.0:
        import cv2

        crop = cv2.resize(crop, (max(8, round(cw * scale)), max(8, round(ch * scale))),
                          interpolation=cv2.INTER_AREA)
    sh, sw = crop.shape[:2]
    bx0 = (float(box[0]) - cx0) * sw / cw
    by0 = (float(box[1]) - cy0) * sh / ch
    bx1 = (float(box[2]) - cx0) * sw / cw
    by1 = (float(box[3]) - cy0) * sh / ch
    yy, xx = np.mgrid[0:sh, 0:sw]
    inside = (xx + 0.5 >= bx0) & (xx + 0.5 <= bx1) & (yy + 0.5 >= by0) & (yy + 0.5 <= by1)
    if inside.sum() < 4 or (~inside).sum() < 4:
        out = np.zeros((h, w), bool)
        out[int(box[1]):int(box[3]), int(box[0]):int(box[2])] = True
        return out

    small = segment_crop(crop, inside, p)
    if scale < 1.0:
        import cv2

        small = cv2.resize(small.astype(np.float32), (cw, ch), interpolation=cv2.INTER_LINEAR) > 0.5
    out = np.zeros((h, w), bool)
    out[cy0:cy1, cx0:cx1] = small
    return out


def segment_crop(crop: np.ndarray, inside: np.ndarray, p: GrabCutParams) -> np.ndarray:
    """GrabCut on a crop where `inside` marks the box; pixels outside are background."""
    sh, sw = crop.shape[:2]
    n_pix = sh * sw
    pix = crop.reshape(-1, 3).astype(np.float64)
    pi, pj, pw = pairwise_weights(crop, p.gamma)
    hard_bg = ~inside.ravel()

    # ---- superpixel stage
    n_seg = max(p.min_segments, int(n_pix / p.sp_area))
    sp = slic(crop, n_segments=n_seg, compactness=10.0)
    sp = (sp.astype(np.int64) * 2 + inside).ravel()     # split along the box edge
    _, sp = np.unique(sp, return_inverse=True)
    n_sp = int(sp.max()) + 1
    sp_hard_bg = np.bincount(sp, hard_bg, minlength=n_sp) > 0
    a, b = sp[pi], sp[pj]
    cross = a != b
    lo, hi = np.minimum(a[cross], b[cross]), np.maximum(a[cross], b[cross])
    key = lo * n_sp + hi
    ukey, inv = np.unique(key, return_inverse=True)
    sp_w = np.bincount(inv, pw[cross])
    ea, eb = ukey // n_sp, ukey % n_sp

    fg_sp = ~sp_hard_bg
    gmms = (None, None)
    for _ in range(p.n_iter):
        fg_px = fg_sp[sp]
        if fg_px.sum() < 2:
            break
        d_fg, d_bg, gmms = _unaries(pix, fg_px, ~fg_px, gmms, p.k)
        u_fg = np.bincount(sp, d_fg, minlength=n_sp)
        u_bg = np.bincount(sp, d_bg, minlength=n_sp)
        # source = foreground: s->v costs what labelling v background costs
        cap_s = np.where(sp_hard_bg, 0.0, u_bg)
        cap_t = np.where(sp_hard_bg, _HARD, u_fg)
        g = Graph(n_sp)
        g.add_edges(ea, eb, sp_w)
        g.add_tweights(np.arange(n_sp), cap_s, cap_t)
        g.maxflow()
        new = g.source_side() & ~sp_hard_bg
        if not new.any():
            # collapse: mixed boundary regions taught the background GMM the
            # object's colours and the cut gave everything away. Keep the last
            # non-empty labelling instead of returning nothing.
            break
        if np.array_equal(new, fg_sp):
            fg_sp = new
            break
        fg_sp = new
    fg = fg_sp[sp].reshape(sh, sw)

    # ---- pixel band stage
    if p.band > 0 and fg.any():
        for _ in range(p.band_iter):
            st = np.ones((3, 3), bool)
            band = (ndi.binary_dilation(fg, st, p.band) & ~ndi.binary_erosion(fg, st, p.band))
            band &= inside
            if not band.any():
                break
            fgf = fg.ravel()
            d_fg, d_bg, gmms = _unaries(pix, fgf, ~fgf, gmms, p.k)
            bflat = band.ravel()
            node = np.full(n_pix, -1, np.int64)
            node[bflat] = np.arange(bflat.sum())
            cap_s = d_bg[bflat].copy()
            cap_t = d_fg[bflat].copy()
            # pairs with both ends in the band are edges; pairs with one end
            # fixed add the fixed neighbour's pull to the band node's terminal
            bi, bj = bflat[pi], bflat[pj]
            both = bi & bj
            g = Graph(int(bflat.sum()))
            g.add_edges(node[pi[both]], node[pj[both]], pw[both])
            for src_in, dst_out in ((pi, pj), (pj, pi)):
                one = bflat[src_in] & ~bflat[dst_out]
                v = node[src_in[one]]
                fixed_fg = fgf[dst_out[one]]
                np.add.at(cap_s, v[fixed_fg], pw[one][fixed_fg])
                np.add.at(cap_t, v[~fixed_fg], pw[one][~fixed_fg])
            g.add_tweights(np.arange(len(cap_s)), cap_s, cap_t)
            g.maxflow()
            new = fgf.copy()
            new[bflat] = g.source_side()
            new = new.reshape(sh, sw) & inside
            if not new.any():
                break
            if np.array_equal(new, fg):
                break
            fg = new
    return _clean(fg, p.min_component)


def _clean(mask: np.ndarray, min_frac: float) -> np.ndarray:
    """Drop small foreground islands (GrabCut has no shape prior)."""
    lab, n = ndi.label(mask, np.ones((3, 3), bool))
    if n <= 1:
        return mask
    sizes = np.bincount(lab.ravel())[1:]
    keep = np.r_[False, sizes >= min_frac * sizes.max()]
    return keep[lab]


def grabcut_opencv(rgb: np.ndarray, box, n_iter: int = 5, margin: float = 0.25,
                   max_side: int = 200) -> np.ndarray:
    """cv2.grabCut on the same crop and scale as `grabcut`, for a fair comparison."""
    import cv2

    h, w = rgb.shape[:2]
    cx0, cy0, cx1, cy1 = _crop_window(box, h, w, margin)
    crop = rgb[cy0:cy1, cx0:cx1]
    ch, cw = crop.shape[:2]
    scale = min(1.0, max_side / max(ch, cw))
    if scale < 1.0:
        crop = cv2.resize(crop, (max(8, round(cw * scale)), max(8, round(ch * scale))),
                          interpolation=cv2.INTER_AREA)
    sh, sw = crop.shape[:2]
    x0 = int(round((float(box[0]) - cx0) * sw / cw))
    y0 = int(round((float(box[1]) - cy0) * sh / ch))
    x1 = int(round((float(box[2]) - cx0) * sw / cw))
    y1 = int(round((float(box[3]) - cy0) * sh / ch))
    rect = (x0, y0, max(1, x1 - x0), max(1, y1 - y0))
    mask = np.zeros((sh, sw), np.uint8)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(np.ascontiguousarray(crop[:, :, ::-1]), mask, rect, bgd, fgd, n_iter,
                    cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        mask[y0:y1, x0:x1] = cv2.GC_PR_FGD
    small = (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)
    if scale < 1.0:
        small = cv2.resize(small.astype(np.float32), (cw, ch), interpolation=cv2.INTER_LINEAR) > 0.5
    out = np.zeros((h, w), bool)
    out[cy0:cy1, cx0:cx1] = small
    return out
