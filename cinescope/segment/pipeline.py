"""Glue for the Phase 4 scripts: pick a segmenter by name, find the largest
person, read one frame, and turn a mask into a matted RGBA cut-out."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .data import mask_box
from .grabcut import grabcut, grabcut_opencv
from .matting import closed_form_alpha, estimate_foreground, trimap_from_mask

SEGMENTERS = ("grabcut", "cv2grabcut", "unet", "yolo", "mobilesam", "sam2")
DEFAULT_UNET = Path(__file__).resolve().parents[2] / "p4_segmentation" / "checkpoints" / "unet_person.pt"


def get_segmenter(name: str, threads: int = 1, unet_ckpt: str | Path = DEFAULT_UNET, detector=None):
    """name -> callable(rgb, boxes) -> list of (H, W) bool masks."""
    if name == "grabcut":
        return lambda rgb, boxes: [grabcut(rgb, b) for b in boxes]
    if name == "cv2grabcut":
        return lambda rgb, boxes: [grabcut_opencv(rgb, b) for b in boxes]
    if name == "unet":
        from .unet import UNetSegmenter

        if not Path(unet_ckpt).exists():
            raise FileNotFoundError(f"{unet_ckpt}: train it with p4_segmentation/train_unet.py")
        return UNetSegmenter(str(unet_ckpt))
    from .models import PersonDetector, SamSegmenter, YoloBoxSegmenter

    if name == "yolo":
        return YoloBoxSegmenter(detector or PersonDetector(threads=threads))
    if name == "mobilesam":
        return SamSegmenter("mobile_sam.pt", threads=threads)
    if name == "sam2":
        return SamSegmenter("sam2_t.pt", threads=threads)
    raise ValueError(f"unknown segmenter {name!r}; choose from {SEGMENTERS}")


def largest_person(rgb: np.ndarray, detector) -> np.ndarray | None:
    boxes, _, _ = detector(rgb)
    if len(boxes) == 0:
        return None
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return boxes[int(area.argmax())]


def frame_at(clip: str, t: float) -> tuple[np.ndarray, float]:
    from ..io.video import VideoReader

    with VideoReader(clip) as vr:
        for f in vr.frames(start_s=t):
            return f.rgb, f.t
    raise ValueError(f"{clip}: no frame at or after {t} s")


def rgba_cutout(rgb: np.ndarray, mask: np.ndarray, matting: bool = True, pad: int = 16,
                trimap_radius: int | None = None) -> tuple[np.ndarray, tuple[int, int]]:
    """Crop around the mask; alpha by closed-form matting on a trimap from the
    mask (or the hard mask), colours by foreground estimation, so the edge
    pixels do not carry the old background. Returns (RGBA uint8, (x0, y0))."""
    h, w = mask.shape
    x0, y0, x1, y1 = mask_box(mask).astype(int)
    x0, y0, x1, y1 = max(0, x0 - pad), max(0, y0 - pad), min(w, x1 + pad), min(h, y1 + pad)
    crop, m = rgb[y0:y1, x0:x1], mask[y0:y1, x0:x1]
    if matting:
        tri = trimap_from_mask(m, trimap_radius)
        alpha = closed_form_alpha(crop, tri)
        fg = estimate_foreground(crop, alpha)
        rgb_out = (fg * 255).round().astype(np.uint8)
    else:
        alpha = m.astype(np.float64)
        rgb_out = crop
    a = (alpha * 255).round().astype(np.uint8)
    return np.dstack([rgb_out, a]), (x0, y0)
