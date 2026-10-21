"""WIDER FACE annotations and the training/validation datasets.

Annotation file (wider_face_split/wider_face_{train,val}_bbx_gt.txt):

    0--Parade/0_Parade_marchingband_1_849.jpg
    1
    449 330 122 149 0 0 0 0 0 0            x y w h blur expression illumination invalid occlusion pose

An image with no faces has count 0 followed by one line of zeros. Faces
flagged invalid, or with a non-positive size, are dropped from training.

Training augmentation follows RetinaFace/CenterFace practice: a random square
crop at 0.3-1.0 of the short side (this is what creates the large faces a crowd
photo lacks and the small ones a portrait lacks), faces kept if their centre is
inside the crop, resize to 512x512, horizontal flip, photometric jitter.

Two options target what the first checkpoints got wrong (p2_detection/README.md):
  keep_large  a crop may not cut through a face at least this many pixels tall
              (in the original image): clipped large faces teach undersized boxes,
              the suspect for Easy scoring below Hard
  low_light   the probability of a "night" sample: gamma-darkened, cooled and with
              sensor noise, for the dark movie frames WIDER FACE has few of
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .centernet_targets import encode

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


@dataclass
class WiderImage:
    path: str                 # relative to WIDER_{split}/images
    boxes: np.ndarray         # (N, 4) xywh, valid faces only
    n_invalid: int = 0


def parse_annotations(txt: str | Path) -> list[WiderImage]:
    lines = [ln.strip() for ln in Path(txt).read_text(encoding="utf-8").splitlines()]
    out, i = [], 0
    while i < len(lines):
        if not lines[i]:
            i += 1
            continue
        path = lines[i]
        n = int(lines[i + 1])
        rows = lines[i + 2:i + 2 + max(n, 1)]
        i += 2 + max(n, 1)
        vals = np.array([[float(v) for v in r.split()[:10]] for r in rows[:n]], np.float64).reshape(-1, 10)
        valid = (vals[:, 7] == 0) & (vals[:, 2] > 0) & (vals[:, 3] > 0)
        out.append(WiderImage(path, vals[valid, :4], int((~valid).sum())))
    return out


def _random_square_crop(img: np.ndarray, boxes: np.ndarray, rng: random.Random,
                        scales=(0.3, 0.45, 0.6, 0.8, 1.0), tries: int = 50,
                        keep_large: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    h, w = img.shape[:2]
    short = min(h, w)
    xyxy = np.c_[boxes[:, :2], boxes[:, :2] + boxes[:, 2:]]
    centres = (xyxy[:, :2] + xyxy[:, 2:]) / 2
    for _ in range(tries):
        s = int(rng.choice(scales) * short)
        x0, y0 = rng.randint(0, w - s), rng.randint(0, h - s)
        inside = ((centres[:, 0] > x0) & (centres[:, 0] < x0 + s)
                  & (centres[:, 1] > y0) & (centres[:, 1] < y0 + s))
        if len(boxes) and not inside.any():
            continue
        if keep_large is not None and len(boxes):
            # a large face that shows in the crop must show whole (one whose centre is outside
            # would otherwise leave unlabelled face pixels: a negative)
            touches = ((xyxy[:, 2] > x0) & (xyxy[:, 0] < x0 + s) & (xyxy[:, 3] > y0) & (xyxy[:, 1] < y0 + s))
            big = touches & (boxes[:, 3] >= keep_large)
            cut = ((xyxy[:, 0] < x0) | (xyxy[:, 1] < y0) | (xyxy[:, 2] > x0 + s) | (xyxy[:, 3] > y0 + s))
            if (big & cut).any():
                continue
        b = xyxy[inside] - [x0, y0, x0, y0]
        return img[y0:y0 + s, x0:x0 + s], np.clip(b, 0, s)
    # fall back: pad the whole image to a square
    side = max(h, w)
    canvas = np.zeros((side, side, 3), img.dtype) + (MEAN * 255).astype(img.dtype)
    canvas[:h, :w] = img
    return canvas, xyxy


def _jitter(img: np.ndarray, rng: random.Random) -> np.ndarray:
    x = img.astype(np.float32)
    x = x * rng.uniform(0.7, 1.3) + rng.uniform(-25, 25)                       # contrast, brightness
    gray = x.mean(-1, keepdims=True)
    x = gray + (x - gray) * rng.uniform(0.6, 1.4)                              # saturation
    return np.clip(x, 0, 255)


def _low_light(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """A night version of a day photo: gamma 1.8-3.2 (shadows sink, highlights survive),
    a cool cast, and noise that grows as the exposure drops, like a pushed sensor."""
    x = img.astype(np.float32) / 255.0
    x = x ** rng.uniform(1.8, 3.2)
    x *= np.array([rng.uniform(0.75, 0.95), rng.uniform(0.85, 1.0), 1.0], np.float32)
    nrng = np.random.default_rng(rng.getrandbits(32))
    x += nrng.normal(0, rng.uniform(0.005, 0.03), x.shape).astype(np.float32)
    return np.clip(x * 255.0, 0, 255)


class WiderTrain:
    def __init__(self, root: str | Path, items: list[WiderImage], size: int = 512, stride: int = 4,
                 seed: int | None = None, keep_large: float | None = None, low_light: float = 0.0):
        self.root, self.items, self.size, self.stride = Path(root), items, size, stride
        self.seed = seed
        self.keep_large, self.low_light = keep_large, low_light
        self.calls = 0

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict:
        import torch
        from PIL import Image

        # A fresh generator per sample: DataLoader workers get copies of this
        # object, and one shared generator would give every worker the same crops.
        self.calls += 1
        rng = (random.Random(os.urandom(8)) if self.seed is None
               else random.Random(f"{self.seed}-{i}-{self.calls}"))
        it = self.items[i]
        with Image.open(self.root / it.path) as im:
            img = np.asarray(im.convert("RGB"))
        crop, boxes = _random_square_crop(img, it.boxes, rng, keep_large=self.keep_large)
        s = self.size / crop.shape[0]
        crop = np.asarray(Image.fromarray(crop).resize((self.size, self.size), Image.BILINEAR))
        boxes = boxes * s
        if rng.random() < 0.5:
            crop = crop[:, ::-1]
            boxes = np.c_[self.size - boxes[:, 2], boxes[:, 1], self.size - boxes[:, 0], boxes[:, 3]]
        crop = _jitter(crop, rng)
        if self.low_light and rng.random() < self.low_light:
            crop = _low_light(crop, rng)
        x = (crop / 255.0 - MEAN) / STD
        t = encode(boxes, np.zeros(len(boxes), np.int64), (self.size // self.stride,) * 2, stride=self.stride)
        return {"image": torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1), dtype=np.float32)),
                **{k: torch.from_numpy(v) for k, v in t.items()}}


def load_eval_image(path: str | Path, max_side: int = 1024, multiple: int = 32) -> tuple[np.ndarray, float]:
    """Image as a normalised (3, H, W) array, resized so the long side is at most
    max_side and padded to a multiple of 32 (the backbone's total stride).
    Returns the array and the scale to map boxes back to original pixels."""
    from PIL import Image

    with Image.open(path) as im:
        im = im.convert("RGB")
        s = min(1.0, max_side / max(im.size))
        if s < 1.0:
            im = im.resize((round(im.width * s), round(im.height * s)), Image.BILINEAR)
        a = np.asarray(im, np.float32) / 255.0
    h, w = a.shape[:2]
    H, W = -(-h // multiple) * multiple, -(-w // multiple) * multiple
    out = np.zeros((H, W, 3), np.float32)
    out[:h, :w] = (a - MEAN) / STD
    return out.transpose(2, 0, 1), s
