"""CenterNet training targets and decoding, in NumPy (Zhou et al., "Objects as Points", 2019).

An object is represented by its centre point. At output stride R (4 here) the
network predicts, for every feature-map cell:
  heatmap  (C,)  probability that an object *centre* of class c is here
  size     (2,)  box width and height, in feature-map cells
  offset   (2,)  sub-cell position of the centre (lost to the stride-R rounding)

Ground truth for the heatmap is not a single 1: each centre is splatted as a
Gaussian whose radius is chosen so that the same box centred anywhere inside
it still has IoU >= 0.7 with the true box. The loss (a focal-loss variant) down-weights
negatives near a centre by (1 - Y)^4, so "almost right" is barely punished.

Decoding needs no NMS in principle: a 3x3 max-pool keeps only local maxima of
the heatmap, and each surviving peak is one object.
"""
from __future__ import annotations

import numpy as np


def gaussian_radius(h: float, w: float, min_overlap: float = 0.7, exact: bool = True) -> float:
    """Radius (in cells) within which the centre can move while the box keeps
    IoU >= min_overlap with the true box.

    Shifting a w x h box by d in both x and y leaves an overlap of (w-d)(h-d),
    so IoU = (w-d)(h-d) / (2wh - (w-d)(h-d)) >= t  gives
        d = ((w+h) - sqrt((w+h)^2 - 4wh(1-t)/(1+t))) / 2.

    exact=False reproduces the function in the CenterNet/CornerNet code, which
    takes the *other* root of that quadratic (and two corner cases) and returns
    radii several times larger — a well-known quirk. Its broader Gaussians still
    train fine (they are only a soft target), so it is kept for comparison.
    """
    if exact:
        s = w + h
        return float((s - np.sqrt(max(s * s - 4 * w * h * (1 - min_overlap) / (1 + min_overlap), 0.0))) / 2)
    a1, b1 = 1.0, h + w
    c1 = w * h * (1 - min_overlap) / (1 + min_overlap)
    r1 = (b1 + np.sqrt(b1 ** 2 - 4 * a1 * c1)) / 2
    a2, b2 = 4.0, 2 * (h + w)
    c2 = (1 - min_overlap) * w * h
    r2 = (b2 + np.sqrt(b2 ** 2 - 4 * a2 * c2)) / 2
    a3, b3 = 4 * min_overlap, -2 * min_overlap * (h + w)
    c3 = (min_overlap - 1) * w * h
    r3 = (b3 + np.sqrt(b3 ** 2 - 4 * a3 * c3)) / 2
    return float(min(r1, r2, r3))


def draw_gaussian(heatmap: np.ndarray, center: tuple[int, int], radius: int) -> None:
    """Element-wise max of a (2r+1)² Gaussian (sigma = diameter / 6) into heatmap, in place."""
    d = 2 * radius + 1
    sigma = d / 6
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    g = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    g[g < np.finfo(g.dtype).eps * g.max()] = 0
    cx, cy = center
    H, W = heatmap.shape
    left, right = min(cx, radius), min(W - cx, radius + 1)
    top, bottom = min(cy, radius), min(H - cy, radius + 1)
    if right <= 0 or bottom <= 0 or left < 0 or top < 0:
        return
    region = heatmap[cy - top:cy + bottom, cx - left:cx + right]
    patch = g[radius - top:radius + bottom, radius - left:radius + right]
    np.maximum(region, patch, out=region)


def encode(boxes: np.ndarray, classes: np.ndarray, out_hw: tuple[int, int], n_classes: int = 1,
           stride: int = 4, max_objs: int = 256, min_size: float = 1.0) -> dict[str, np.ndarray]:
    """Targets for one image. boxes: (N, 4) xyxy in *input* pixels.

    Returns heatmap (C,H,W), wh (K,2), reg (K,2), ind (K,) flat index into H*W,
    mask (K,) 1 for real objects. Boxes smaller than min_size output cells are
    skipped (they would round to nothing)."""
    H, W = out_hw
    hm = np.zeros((n_classes, H, W), np.float32)
    wh = np.zeros((max_objs, 2), np.float32)
    reg = np.zeros((max_objs, 2), np.float32)
    ind = np.zeros(max_objs, np.int64)
    mask = np.zeros(max_objs, np.float32)
    b = np.asarray(boxes, np.float64).reshape(-1, 4) / stride
    b[:, [0, 2]] = b[:, [0, 2]].clip(0, W - 1)
    b[:, [1, 3]] = b[:, [1, 3]].clip(0, H - 1)
    k = 0
    # largest first, so that max_objs truncation drops the smallest (least reliable) faces
    for i in np.argsort(-(b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]), kind="stable"):
        if k >= max_objs:
            break
        w, h = b[i, 2] - b[i, 0], b[i, 3] - b[i, 1]
        if w < min_size or h < min_size:
            continue
        c = np.array([(b[i, 0] + b[i, 2]) / 2, (b[i, 1] + b[i, 3]) / 2])
        ci = c.astype(np.int64)
        r = max(0, int(gaussian_radius(np.ceil(h), np.ceil(w))))
        draw_gaussian(hm[int(classes[i])], (int(ci[0]), int(ci[1])), r)
        wh[k] = (w, h)
        reg[k] = c - ci
        ind[k] = ci[1] * W + ci[0]
        mask[k] = 1
        k += 1
    return {"heatmap": hm, "wh": wh, "reg": reg, "ind": ind, "mask": mask}


def decode(heatmap: np.ndarray, wh: np.ndarray, reg: np.ndarray, stride: int = 4, k: int = 100,
           threshold: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Boxes from network outputs for one image (NumPy reference of the PyTorch
    decoder). heatmap (C,H,W) probabilities; wh, reg (2,H,W). Returns xyxy boxes
    in input pixels, scores, classes."""
    C, H, W = heatmap.shape
    pad = np.pad(heatmap, ((0, 0), (1, 1), (1, 1)), constant_values=-np.inf)
    win = np.lib.stride_tricks.sliding_window_view(pad, (3, 3), axis=(1, 2))
    peaks = heatmap * (heatmap == win.max(axis=(-1, -2)))
    flat = peaks.reshape(-1)
    top = np.argsort(-flat, kind="stable")[:k]
    top = top[flat[top] > threshold]
    cls, rem = np.divmod(top, H * W)
    ys, xs = np.divmod(rem, W)
    cx = xs + reg[0, ys, xs]
    cy = ys + reg[1, ys, xs]
    w, h = wh[0, ys, xs], wh[1, ys, xs]
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1) * stride
    return boxes, flat[top], cls
