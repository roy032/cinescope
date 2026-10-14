"""SLIC superpixels (Achanta et al., 2012), vectorised in NumPy.

SLIC is k-means on (L, a, b, x, y) with two twists that make it fast and give
compact, roughly equal-sized regions:

  - centres start on a regular grid of step S, so a pixel only ever needs to
    be compared with the centres of nearby grid cells (here: the 3x3 cells
    around it), not with all k centres;
  - the distance mixes colour and position, D² = d_lab² + (m/S)² d_xy², where
    the compactness m trades boundary adherence (small m) for regular shapes.

Instead of SLIC's per-centre 2S x 2S window loops, each centre keeps the grid
cell it was seeded in, and every pixel computes its distance to the 9 centres
of the neighbouring cells as whole-image arrays. Centres drift by less than a
cell in practice, so this is the same search region.

Superpixels make graph cuts cheap: GrabCut on ~1,000 superpixels instead of
~100,000 pixels, with the same energy restricted to labellings that are
constant on each superpixel (see grabcut.py).
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from ..imgproc.color import rgb_to_lab


def slic(rgb: np.ndarray, n_segments: int = 800, compactness: float = 10.0,
         n_iter: int = 5) -> np.ndarray:
    """(H, W, 3) uint8 RGB -> (H, W) int32 labels 0..K-1, each a connected region."""
    h, w = rgb.shape[:2]
    lab = rgb_to_lab(rgb).astype(np.float32)
    step = max(2.0, np.sqrt(h * w / max(1, n_segments)))
    gy, gx = max(1, int(round(h / step))), max(1, int(round(w / step)))
    cy = (np.arange(gy) + 0.5) * h / gy
    cx = (np.arange(gx) + 0.5) * w / gx
    # centres: (gy, gx, 5) = L, a, b, y, x
    cyy, cxx = np.meshgrid(cy, cx, indexing="ij")
    iy = np.clip(cyy.astype(int), 0, h - 1)
    ix = np.clip(cxx.astype(int), 0, w - 1)
    centres = np.concatenate([lab[iy, ix], cyy[..., None], cxx[..., None]], -1).astype(np.float32)

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    # grid cell of each pixel
    cell_y = np.minimum((yy * gy / h).astype(int), gy - 1)
    cell_x = np.minimum((xx * gx / w).astype(int), gx - 1)
    sy, sx = h / gy, w / gx
    spatial_w = (compactness / np.sqrt(sy * sx)) ** 2
    labels = cell_y * gx + cell_x
    for _ in range(n_iter):
        best = np.full((h, w), np.inf, np.float32)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                ny, nx = cell_y + dy, cell_x + dx
                ok = (ny >= 0) & (ny < gy) & (nx >= 0) & (nx < gx)
                nyc, nxc = np.clip(ny, 0, gy - 1), np.clip(nx, 0, gx - 1)
                c = centres[nyc, nxc]                      # (h, w, 5)
                d = ((lab - c[..., :3]) ** 2).sum(-1) + spatial_w * (
                    (yy - c[..., 3]) ** 2 + (xx - c[..., 4]) ** 2)
                d[~ok] = np.inf
                better = d < best
                best[better] = d[better]
                labels[better] = (nyc * gx + nxc)[better]
        # move centres to their members' means
        flat = labels.ravel()
        cnt = np.bincount(flat, minlength=gy * gx).astype(np.float32)
        feats = np.concatenate([lab, yy[..., None], xx[..., None]], -1).reshape(-1, 5)
        sums = np.stack([np.bincount(flat, feats[:, k], minlength=gy * gx) for k in range(5)], 1)
        nonempty = cnt > 0
        cflat = centres.reshape(-1, 5)
        cflat[nonempty] = sums[nonempty] / cnt[nonempty, None]
        centres = cflat.reshape(gy, gx, 5)
    return _enforce_connectivity(labels, min_size=max(4, int(sy * sx / 4)))


def _enforce_connectivity(labels: np.ndarray, min_size: int) -> np.ndarray:
    """Split labels into connected components, merge the small ones into a
    touching neighbour, and relabel 0..K-1. (k-means in xy does not guarantee
    connected clusters; the graph code below assumes them only for neatness,
    but stray one-pixel islands make poor graph nodes.)"""
    h, w = labels.shape
    # connected components of "same label" in one go: a graph whose edges join
    # 4-neighbours with equal labels
    idx = np.arange(h * w).reshape(h, w)
    a = np.concatenate([idx[:, :-1].ravel(), idx[:-1, :].ravel()])
    b = np.concatenate([idx[:, 1:].ravel(), idx[1:, :].ravel()])
    flat = labels.ravel()
    same = flat[a] == flat[b]
    g = sp.coo_matrix((np.ones(same.sum(), np.int8), (a[same], b[same])), shape=(h * w, h * w))
    n, comp = connected_components(g, directed=False)
    comp = comp.reshape(h, w).astype(np.int64)
    sizes = np.bincount(comp.ravel(), minlength=n)
    small = sizes < min_size
    if small.any():
        # each small component takes the label of the largest touching component
        # (one pass over the 4-neighbour pairs, repeated until none is left small)
        for _ in range(4):
            a = np.concatenate([comp[:, :-1].ravel(), comp[:-1, :].ravel()])
            b = np.concatenate([comp[:, 1:].ravel(), comp[1:, :].ravel()])
            diff = a != b
            a, b = a[diff], b[diff]
            a, b = np.concatenate([a, b]), np.concatenate([b, a])
            sel = small[a]
            if not sel.any():
                break
            a, b = a[sel], b[sel]
            # choose the biggest neighbour for each small component
            order = np.lexsort((-sizes[b], a))
            a, b = a[order], b[order]
            firsts = np.r_[True, a[1:] != a[:-1]]
            # merge along "small -> chosen neighbour" edges; grouping by graph
            # components keeps every merged group spatially connected even when
            # chains (A -> B -> C) form
            ma, mb = a[firsts], b[firsts]
            mg = sp.coo_matrix((np.ones(len(ma), np.int8), (ma, mb)), shape=(n, n))
            n, target = connected_components(mg, directed=False)
            comp = target[comp]
            sizes = np.bincount(comp.ravel(), minlength=n)
            small = (sizes < min_size) & (sizes > 0)
    _, out = np.unique(comp, return_inverse=True)
    return out.reshape(h, w).astype(np.int32)


def adjacency(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pairs (a, b), a < b, of superpixels that touch (4-neighbourhood)."""
    a = np.concatenate([labels[:, :-1].ravel(), labels[:-1, :].ravel()])
    b = np.concatenate([labels[:, 1:].ravel(), labels[1:, :].ravel()])
    keep = a != b
    lo, hi = np.minimum(a[keep], b[keep]), np.maximum(a[keep], b[keep])
    pairs = np.unique(np.stack([lo, hi], 1), axis=0)
    return pairs[:, 0], pairs[:, 1]
