"""Face alignment: five landmarks -> a canonical 112x112 crop.

Face recognition networks (ArcFace, SFace, the Week 10 network) are trained on
faces warped so the eyes, nose and mouth corners land on fixed template
points. Without alignment the network must also learn to ignore in-plane
rotation, scale and position, which wastes capacity -- and for the active
speaker detector, alignment cancels head motion so what is left in the mouth
region is mostly the mouth moving.

The warp is a similarity transform (rotation, uniform scale, translation: 4
degrees of freedom), fit to the 5 point pairs by least squares with Umeyama's
closed form (1991): centre both point sets, take the SVD of their
cross-covariance, the rotation is U V^T (with a reflection fix), the scale is
the ratio of the singular values' sum to the source variance.
"""
from __future__ import annotations

import numpy as np

# The ArcFace / InsightFace template for 112x112 crops (also SFace's).
TEMPLATE_112 = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                         [41.5493, 92.3655], [70.7299, 92.2041]])


def umeyama(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """2x3 similarity transform M minimising sum |M [src;1] - dst|^2."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    ms, md = src.mean(0), dst.mean(0)
    s0, d0 = src - ms, dst - md
    cov = d0.T @ s0 / len(src)
    U, S, Vt = np.linalg.svd(cov)
    D = np.eye(2)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:   # no reflections
        D[1, 1] = -1
    R = U @ D @ Vt
    var = (s0 ** 2).sum() / len(src)
    scale = (S * np.diag(D)).sum() / max(var, 1e-12)
    t = md - scale * R @ ms
    return np.c_[scale * R, t]


def align(rgb: np.ndarray, landmarks: np.ndarray, size: int = 112) -> np.ndarray:
    """Warp the face with these 5 landmarks to a size x size aligned crop."""
    import cv2

    M = umeyama(landmarks, TEMPLATE_112 * (size / 112.0))
    return cv2.warpAffine(rgb, M, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def asd_crop(rgb: np.ndarray, box: np.ndarray, crop_scale: float = 0.40) -> np.ndarray:
    """Light-ASD's face crop (from its Columbia_test.py): a square of side
    2*s*(1+cs) around the box (s = half the longer box side), shifted down so
    the mouth is centred, resized to 224, centre 112x112, grayscale."""
    import cv2

    x1, y1, x2, y2 = box
    bs = max(x2 - x1, y2 - y1) / 2
    cy, cx = (y1 + y2) / 2, (x1 + x2) / 2
    bsi = int(bs * (1 + 2 * crop_scale))
    pad = np.pad(rgb, ((bsi, bsi), (bsi, bsi), (0, 0)), constant_values=110)
    my, mx = cy + bsi, cx + bsi
    face = pad[int(my - bs):int(my + bs * (1 + 2 * crop_scale)),
               int(mx - bs * (1 + crop_scale)):int(mx + bs * (1 + crop_scale))]
    if face.size == 0:
        return np.full((112, 112), 110, np.uint8)
    face = cv2.resize(face, (224, 224))
    return cv2.cvtColor(face[56:168, 56:168], cv2.COLOR_RGB2GRAY)


def _asd_square(box: np.ndarray, crop_scale: float = 0.40) -> tuple[float, float, float]:
    """The frame square that asd_crop's final 112x112 covers: (centre x, centre y, side).
    asd_crop cuts a 2s(1+cs) square whose top is s above the box centre, resizes
    it to 224 and keeps the middle half: side s(1+cs), centred s*cs below the box centre."""
    x1, y1, x2, y2 = box
    bs = max(x2 - x1, y2 - y1) / 2
    return (x1 + x2) / 2, (y1 + y2) / 2 + bs * crop_scale, bs * (1 + crop_scale)


def asd_patch(gray: np.ndarray, box: np.ndarray, size: int = 168, margin: float = 1.5,
              crop_scale: float = 0.40) -> tuple[np.ndarray, tuple[float, float, float]]:
    """A grey patch around one detection, big enough to cut asd_crop's square
    later from a *smoothed* box (the Phase 9 pipeline cuts crops during its one
    decoding pass, before the tracks -- and so the smoothed boxes -- exist).

    Covers `margin` x the square's side, at size px (168 = 1.5 x 112: the final
    crop is cut at its native resolution). Returns (patch, (x0, y0, px per frame px))."""
    import cv2

    cx, cy, side = _asd_square(box, crop_scale)
    big = side * margin
    s = size / max(big, 1e-6)
    x0, y0 = cx - big / 2, cy - big / 2
    m = np.array([[s, 0, -x0 * s], [0, s, -y0 * s]])
    patch = cv2.warpAffine(gray, m, (size, size), flags=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=110)
    return patch, (float(x0), float(y0), float(s))


def asd_from_patch(patch: np.ndarray, geom: tuple[float, float, float], box: np.ndarray,
                   crop_scale: float = 0.40) -> np.ndarray:
    """asd_crop's 112x112 for `box` (frame coordinates), cut from a stored asd_patch."""
    import cv2

    x0, y0, s = geom
    cx, cy, side = _asd_square(box, crop_scale)
    k = 112 / max(side * s, 1e-6)                 # patch px -> output px
    px, py = (cx - x0) * s, (cy - y0) * s          # square centre in patch px
    m = np.array([[k, 0, 56 - px * k], [0, k, 56 - py * k]])
    return cv2.warpAffine(patch, m, (112, 112), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=110)
