"""Heatmap targets, sub-pixel decoding and the person-crop transform.

A top-down pose network sees one person at a time: the person's box is
enlarged by 25%, stretched to the network's aspect ratio (3:4) and warped to
a fixed-size crop. It outputs one heatmap per keypoint at 1/4 of the crop's
resolution; the keypoint is where the heatmap peaks.

Coordinates. Pixel i covers [i - 0.5, i + 0.5], so the first and last pixel
centres of an image of width W are 0 and W - 1. Crop and heatmap are related by
x_hm = x_crop * (W_hm - 1) / (W_crop - 1) (the "unbiased" mapping of Huang et
al., 2020, UDP). With it, mirroring the crop mirrors the heatmap exactly; with
the simpler x_hm = x_crop / 4 a flipped heatmap is off by 0.75 cells, which is
why SimpleBaseline's test-time flip shifts its heatmap one cell by hand.

Decoding. A plain argmax is quantised to whole heatmap cells — 4 crop pixels.
Two fixes:
  quarter  move 0.25 cell towards the larger neighbour (SimpleBaseline, Hourglass):
           halves the worst-case error, cheap.
  taylor   (DARK, Zhang et al., 2020) the log of a Gaussian is a quadratic, so
           one Newton step on the log-heatmap from the argmax lands on the true
           centre: offset = -H^{-1} grad. Exact on a clean Gaussian; on a network
           output the heatmap is first smoothed with a Gaussian so its log is
           close to quadratic near the peak.
"""
from __future__ import annotations

import cv2
import numpy as np


def gaussian_targets(pts: np.ndarray, vis: np.ndarray, shape: tuple[int, int], sigma: float = 2.0
                     ) -> tuple[np.ndarray, np.ndarray]:
    """(K, 2) heatmap-space points -> (K, H, W) Gaussians peaking at 1 exactly at
    each point (no rounding to a cell), and (K,) loss weights: 0 for keypoints
    that are unlabelled or whose 3-sigma disc misses the map entirely."""
    pts = np.asarray(pts, np.float64)
    h, w = shape
    ys, xs = np.arange(h)[:, None], np.arange(w)[None, :]
    hm = np.zeros((len(pts), h, w), np.float32)
    weight = (np.asarray(vis) > 0).astype(np.float32)
    for k, (x, y) in enumerate(pts):
        if weight[k] == 0 or not (-3 * sigma < x < w - 1 + 3 * sigma and -3 * sigma < y < h - 1 + 3 * sigma):
            weight[k] = 0
            continue
        hm[k] = np.exp(-((xs - x) ** 2 + (ys - y) ** 2) / (2 * sigma ** 2))
    return hm, weight


def decode_heatmaps(hm: np.ndarray, method: str = "taylor", blur: float = 0.0
                    ) -> tuple[np.ndarray, np.ndarray]:
    """(N, K, H, W) heatmaps -> (N, K, 2) sub-cell (x, y) and (N, K) peak values."""
    hm = np.asarray(hm, np.float64)
    n, k, h, w = hm.shape
    flat = hm.reshape(n, k, -1)
    idx = flat.argmax(-1)
    conf = flat.max(-1)
    xy = np.stack([idx % w, idx // w], -1).astype(np.float64)
    if method == "argmax":
        return xy, conf
    if method == "quarter":
        for i in range(n):
            for j in range(k):
                x, y = int(xy[i, j, 0]), int(xy[i, j, 1])
                m = hm[i, j]
                if 0 < x < w - 1:
                    xy[i, j, 0] += 0.25 * np.sign(m[y, x + 1] - m[y, x - 1])
                if 0 < y < h - 1:
                    xy[i, j, 1] += 0.25 * np.sign(m[y + 1, x] - m[y - 1, x])
        return xy, conf
    if method != "taylor":
        raise ValueError(method)
    for i in range(n):
        for j in range(k):
            m = hm[i, j]
            if blur > 0:
                peak = m.max()
                m = cv2.GaussianBlur(m, (0, 0), blur)
                m = m * (peak / max(m.max(), 1e-12))          # keep the original peak height
            lg = np.log(np.maximum(m, 1e-10))
            x, y = int(xy[i, j, 0]), int(xy[i, j, 1])
            if not (1 <= x < w - 1 and 1 <= y < h - 1):
                continue
            dx = 0.5 * (lg[y, x + 1] - lg[y, x - 1])
            dy = 0.5 * (lg[y + 1, x] - lg[y - 1, x])
            dxx = lg[y, x + 1] - 2 * lg[y, x] + lg[y, x - 1]
            dyy = lg[y + 1, x] - 2 * lg[y, x] + lg[y - 1, x]
            dxy = 0.25 * (lg[y + 1, x + 1] - lg[y + 1, x - 1] - lg[y - 1, x + 1] + lg[y - 1, x - 1])
            hess = np.array([[dxx, dxy], [dxy, dyy]])
            if np.linalg.det(hess) <= 1e-12 or dxx >= 0 or dyy >= 0:   # not a proper maximum
                continue
            off = -np.linalg.solve(hess, [dx, dy])
            if np.all(np.abs(off) <= 1.0):                    # one cell at most; otherwise keep argmax
                xy[i, j] += off
    return xy, conf


def box_to_center_scale(box: np.ndarray, aspect: float, pad: float = 1.25) -> tuple[np.ndarray, np.ndarray]:
    """xyxy box -> centre and (w, h) of the crop window: padded and widened or
    heightened to the crop's aspect ratio (w / h), never squashed."""
    x0, y0, x1, y1 = np.asarray(box, np.float64)
    c = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
    w, h = max(x1 - x0, 1.0), max(y1 - y0, 1.0)
    if w > aspect * h:
        h = w / aspect
    else:
        w = h * aspect
    return c, np.array([w, h]) * pad


def crop_matrix(center: np.ndarray, size: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
    """2x3 affine image -> crop that maps the window's edges to the crop's
    outer pixel centres (0 and W-1), consistent with the UDP heatmap mapping."""
    oh, ow = out_hw
    sx = (ow - 1) / size[0]
    sy = (oh - 1) / size[1]
    return np.array([[sx, 0, (ow - 1) / 2 - sx * center[0]],
                     [0, sy, (oh - 1) / 2 - sy * center[1]]])


def affine_crop(img: np.ndarray, box: np.ndarray, out_hw: tuple[int, int], pad: float = 1.25
                ) -> tuple[np.ndarray, np.ndarray]:
    c, s = box_to_center_scale(box, out_hw[1] / out_hw[0], pad)
    M = crop_matrix(c, s, out_hw)
    crop = cv2.warpAffine(img, M, (out_hw[1], out_hw[0]), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    return crop, M


def transform_points(pts: np.ndarray, M: np.ndarray, inverse: bool = False) -> np.ndarray:
    pts = np.asarray(pts, np.float64)
    A = np.vstack([M, [0, 0, 1]])
    if inverse:
        A = np.linalg.inv(A)
    return pts @ A[:2, :2].T + A[:2, 2]


def crop_to_heatmap(pts: np.ndarray, crop_hw: tuple[int, int], hm_hw: tuple[int, int]) -> np.ndarray:
    s = np.array([(hm_hw[1] - 1) / (crop_hw[1] - 1), (hm_hw[0] - 1) / (crop_hw[0] - 1)])
    return np.asarray(pts, np.float64) * s


def heatmap_to_crop(pts: np.ndarray, crop_hw: tuple[int, int], hm_hw: tuple[int, int]) -> np.ndarray:
    s = np.array([(hm_hw[1] - 1) / (crop_hw[1] - 1), (hm_hw[0] - 1) / (crop_hw[0] - 1)])
    return np.asarray(pts, np.float64) / s
