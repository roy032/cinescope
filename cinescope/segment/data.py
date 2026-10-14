"""Data for the segmentation, matting and inpainting benchmarks — built from
any clips, with no hand labels.

There are no person masks for our footage, so every target here is either
*distilled* (another model's mask, which carries that model's mistakes) or
*exact by construction*:

  people sets     frames sampled from clips, with YOLOv8n-seg person boxes and
                  masks (distilled targets).
  cut-outs        a person's pixels plus their mask, lifted from a frame.
  composites      a cut-out pasted onto a frame from a different clip. The
                  mask of the pasted pixels is the ground truth *exactly*,
                  whatever errors the cut-out had: the pasted pixels are the
                  object, by definition. Seams are hard, which is easier than
                  real boundaries — a caveat for every number scored on them.
  alpha composites  the same with a soft alpha (blurred edge plus synthetic
                  hair strands), so F, B and alpha are all known: a matting
                  benchmark with exact truth.
  holes           person-shaped holes punched into real frames, whose original
                  pixels are the inpainting ground truth.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# ----------------------------------------------------------------- geometry


def mask_box(mask: np.ndarray) -> np.ndarray:
    """Tight xyxy box of a mask (pixel edges: x1 = last column + 1)."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return np.zeros(4)
    return np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], np.float64)


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 1.0


def expand_box(box, margin: float, h: int, w: int) -> tuple[int, int, int, int]:
    """Box grown by `margin` x its size on every side, as integer pixel bounds
    (not clipped: the caller pads), so the crop keeps the box's aspect."""
    x0, y0, x1, y1 = (float(v) for v in box)
    mx, my = margin * (x1 - x0), margin * (y1 - y0)
    return (int(np.floor(x0 - mx)), int(np.floor(y0 - my)),
            int(np.ceil(x1 + mx)), int(np.ceil(y1 + my)))


def crop_padded(img: np.ndarray, bounds) -> np.ndarray:
    """img[y0:y1, x0:x1] with zero padding where the window leaves the image."""
    x0, y0, x1, y1 = bounds
    h, w = img.shape[:2]
    out = np.zeros((y1 - y0, x1 - x0) + img.shape[2:], img.dtype)
    sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    if sx1 > sx0 and sy1 > sy0:
        out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = img[sy0:sy1, sx0:sx1]
    return out


def paste_window(dst: np.ndarray, src: np.ndarray, bounds) -> None:
    """Inverse of crop_padded: write src into dst over the window, clipped."""
    x0, y0, x1, y1 = bounds
    h, w = dst.shape[:2]
    sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    if sx1 > sx0 and sy1 > sy0:
        dst[sy0:sy1, sx0:sx1] = src[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0]


# ----------------------------------------------------------------- people sets


@dataclass
class PeopleFrame:
    clip: str
    t: float
    rgb: np.ndarray            # (H, W, 3)
    boxes: np.ndarray          # (N, 4)
    scores: np.ndarray         # (N,)
    masks: np.ndarray          # (N, H, W) bool, YOLOv8n-seg


def sample_people(clip: str, detector, every_s: float = 2.0, width: int | None = None,
                  min_h: int = 48, min_score: float = 0.5, max_per_frame: int = 4,
                  border: int = 2, max_frames: int | None = None) -> list[PeopleFrame]:
    """Frames every `every_s` seconds with their confident, whole (not cut by
    the frame edge), large-enough people, largest first."""
    from ..io.video import VideoReader

    out = []
    with VideoReader(clip, width=width) as vr:
        for f in vr.frames(sample_fps=1.0 / every_s):
            boxes, scores, masks = detector(f.rgb)
            h, w = f.rgb.shape[:2]
            if len(boxes):
                bh = boxes[:, 3] - boxes[:, 1]
                inside = ((boxes[:, 0] > border) & (boxes[:, 1] > border)
                          & (boxes[:, 2] < w - border) & (boxes[:, 3] < h - border))
                keep = (scores >= min_score) & (bh >= min_h) & inside & (masks.sum((1, 2)) > 0)
                idx = np.nonzero(keep)[0]
                idx = idx[np.argsort(-bh[idx])][:max_per_frame]
                boxes, scores, masks = boxes[idx], scores[idx], masks[idx]
            out.append(PeopleFrame(Path(clip).name, f.t, f.rgb, boxes, scores, masks))
            if max_frames and len(out) >= max_frames:
                break
    return out


def save_people(frames: list[PeopleFrame], path: str | Path) -> None:
    """One .npz per set: JPEG-free (frames stored raw but compressed), masks bit-packed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays, meta = {}, []
    for i, f in enumerate(frames):
        arrays[f"rgb{i}"] = f.rgb
        arrays[f"masks{i}"] = np.packbits(f.masks, axis=-1)
        meta.append({"clip": f.clip, "t": f.t, "boxes": f.boxes.tolist(),
                     "scores": f.scores.tolist(), "shape": list(f.rgb.shape[:2])})
    arrays["meta"] = np.frombuffer(json.dumps(meta).encode(), np.uint8)
    np.savez_compressed(path, **arrays)


def load_people(path: str | Path) -> list[PeopleFrame]:
    z = np.load(path)
    meta = json.loads(bytes(z["meta"]).decode())
    out = []
    for i, m in enumerate(meta):
        h, w = m["shape"]
        masks = np.unpackbits(z[f"masks{i}"], axis=-1, count=w).astype(bool)
        out.append(PeopleFrame(m["clip"], m["t"], z[f"rgb{i}"], np.asarray(m["boxes"]).reshape(-1, 4),
                               np.asarray(m["scores"]), masks.reshape(-1, h, w)))
    return out


# ----------------------------------------------------------------- composites


@dataclass
class Cutout:
    rgb: np.ndarray    # (h, w, 3) tight crop
    mask: np.ndarray   # (h, w) bool
    source: str


def make_cutout(rgb: np.ndarray, mask: np.ndarray, source: str = "") -> Cutout:
    from scipy import ndimage as ndi

    lab, n = ndi.label(mask)
    if n > 1:   # largest connected piece only: stray blobs are not part of the person
        sizes = np.bincount(lab.ravel())[1:]
        mask = lab == (1 + int(sizes.argmax()))
    x0, y0, x1, y1 = mask_box(mask).astype(int)
    return Cutout(rgb[y0:y1, x0:x1].copy(), mask[y0:y1, x0:x1].copy(), source)


def place(cut: Cutout, bg_shape, rng: np.random.Generator, height_frac=(0.25, 0.7)):
    """Random scale and position for a cut-out on a background of bg_shape:
    returns the resized cut-out and its top-left corner (fully inside)."""
    import cv2

    h, w = bg_shape[:2]
    ch, cw = cut.mask.shape
    target_h = rng.uniform(*height_frac) * h
    s = min(target_h / ch, 0.9 * w / cw, 0.95 * h / ch)
    nh, nw = max(8, round(ch * s)), max(4, round(cw * s))
    rgb = cv2.resize(cut.rgb, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    mask = cv2.resize(cut.mask.astype(np.float32), (nw, nh), interpolation=cv2.INTER_LINEAR) > 0.5
    y = int(rng.integers(0, max(1, h - nh)))
    x = int(rng.integers(0, max(1, w - nw)))
    return Cutout(rgb, mask, cut.source), (x, y)


def composite(bg: np.ndarray, cut: Cutout, xy) -> tuple[np.ndarray, np.ndarray]:
    """Paste a cut-out at (x, y): returns the image and its exact mask."""
    x, y = xy
    out = bg.copy()
    mask = np.zeros(bg.shape[:2], bool)
    h, w = cut.mask.shape
    region = out[y:y + h, x:x + w]
    region[cut.mask] = cut.rgb[cut.mask]
    mask[y:y + h, x:x + w] = cut.mask
    return out, mask


def soft_alpha(mask: np.ndarray, rng: np.random.Generator, n_strands: int = 40,
               blur: tuple[float, float] = (0.7, 2.0)) -> np.ndarray:
    """A plausible alpha matte around a binary mask: the edge is blurred (motion
    blur / defocus) and thin semi-transparent strands grow out of the upper
    contour (hair). Strands are drawn anti-aliased at 4x and averaged down, so
    their partial coverage is real fractional alpha."""
    import cv2

    h, w = mask.shape
    sigma = rng.uniform(*blur)
    a = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), sigma)
    ys, xs = np.nonzero(mask & ~cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool))
    if len(ys):
        top = ys < np.percentile(ys, 35)          # strands on the upper part of the contour
        ys, xs = ys[top], xs[top]
    up = 4
    strands = np.zeros((h * up, w * up), np.uint8)
    if len(ys):
        for _ in range(n_strands):
            k = rng.integers(len(ys))
            px, py = xs[k] * up, ys[k] * up
            ang = rng.uniform(-np.pi, 0)              # pointing up-ish
            pts = [(px, py)]
            length = rng.uniform(4, 14) * up
            for _ in range(6):
                ang += rng.normal(0, 0.3)
                px += np.cos(ang) * length / 6
                py += np.sin(ang) * length / 6
                pts.append((px, py))
            cv2.polylines(strands, [np.round(np.asarray(pts)).astype(np.int32)], False, 255,
                          thickness=int(rng.integers(1, 3)), lineType=cv2.LINE_AA)
    s = cv2.resize(strands.astype(np.float32) / 255.0, (w, h), interpolation=cv2.INTER_AREA)
    return np.clip(np.maximum(a, s * rng.uniform(0.5, 1.0)), 0, 1)


def alpha_composite(fg: np.ndarray, bg: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    a = alpha[..., None].astype(np.float64)
    return np.clip(a * fg + (1 - a) * bg, 0, 255).round().astype(np.uint8)
