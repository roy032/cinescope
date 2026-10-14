"""Video object removal: track a mask through a shot, then fill it with the
background other frames saw — and only where no frame saw it, inpaint.

Why not inpaint each frame: a spatial method invents the hidden pixels, and it
invents them differently in every frame, so the fill shimmers. In a video the
truth is often *available*: when a person walks across a static shot, every
background pixel they cover is visible in some other frame. So:

  1. track     the object's box from frame to frame (best-IoU person detection,
               else constant-velocity prediction) and segment it in every
               frame with a box-prompted segmenter; dilate the mask a little
               (soft edges, shadows' penumbra, segmentation slop).
  2. plate     with a static camera, a clean background plate is the per-pixel
               *median over frames where that pixel is not masked*. Every other
               detected person is masked out of the samples too (passers-by
               crossing the same spot would otherwise leave ghosts); the median
               drops what remains of transient things, as long as they cover a
               pixel less than half of its samples.
  3. fallback  pixels masked in every frame (the object never moved off them)
               have no temporal sample: they are spatially inpainted, once,
               on the plate — so even the invented part is the same in every
               frame and does not flicker.
  4. composite each frame keeps its own pixels outside the mask; inside, the
               plate, blended over a 2-pixel feather.

Static-camera check: phase correlation between sampled frames (outside the
masks); if the camera moves more than ~1.5 px, the plate would be smeared, so
removal falls back to per-frame spatial inpainting and says so.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from ..detect.boxes import iou as box_iou
from .inpaint import criminisi, opencv_inpaint


@dataclass
class Track:
    boxes: np.ndarray            # (N, 4)
    masks: list[np.ndarray]      # N x (H, W) bool
    matched: np.ndarray          # (N,) bool: box came from a detection (else predicted)
    others: list[np.ndarray]     # N x (H, W) bool: every detected person (for exclusion from the plate)


def track_object(frames: list[np.ndarray], box0, detect: Callable | None,
                 segment: Callable, iou_min: float = 0.3, dilate: int = 4) -> Track:
    """Follow the object in box0 (frame 0) through `frames`.

    detect(rgb) -> (boxes, scores, masks) or None to skip detection (the box
    then only follows constant velocity). segment(rgb, boxes) -> list of masks.
    """
    import cv2

    boxes, masks, matched, others = [], [], [], []
    prev = np.asarray(box0, np.float64)
    vel = np.zeros(4)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate + 1, 2 * dilate + 1)) if dilate else None
    for i, rgb in enumerate(frames):
        guess = prev + vel if i else prev
        box, ok = guess, False
        everyone = np.zeros(rgb.shape[:2], bool)
        if detect is not None:
            dboxes, _, dmasks = detect(rgb)
            if len(dmasks):
                everyone = np.any(dmasks, 0)
            if len(dboxes):
                ov = box_iou(guess[None], dboxes)[0]
                k = int(ov.argmax())
                if ov[k] >= iou_min or (i == 0 and ov[k] > 0.1):
                    box, ok = dboxes[k].astype(np.float64), True
        if i:
            vel = 0.5 * vel + 0.5 * (box - prev) if ok else vel
        m = segment(rgb, box[None])[0]
        if kernel is not None:
            m = cv2.dilate(m.astype(np.uint8), kernel).astype(bool)
            everyone = cv2.dilate(everyone.astype(np.uint8), kernel).astype(bool)
        boxes.append(box)
        masks.append(m)
        matched.append(ok)
        others.append(everyone)
        prev = box
    return Track(np.asarray(boxes), masks, np.asarray(matched), others)


def camera_is_static(frames: list[np.ndarray], masks: list[np.ndarray], max_shift: float = 1.5,
                     samples: int = 8) -> tuple[bool, float]:
    """Largest phase-correlation shift between frame 0 and sampled frames, with
    the masked object greyed out so its own motion is not mistaken for the camera's."""
    import cv2

    idx = np.unique(np.linspace(0, len(frames) - 1, samples).round().astype(int))

    def prep(i):
        g = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY).astype(np.float32)
        g[masks[i]] = g[~masks[i]].mean() if (~masks[i]).any() else 0
        return g

    ref = prep(0)
    win = cv2.createHanningWindow(ref.shape[::-1], cv2.CV_32F)
    worst = 0.0
    for i in idx[1:]:
        (dx, dy), _ = cv2.phaseCorrelate(ref, prep(i), win)
        worst = max(worst, float(np.hypot(dx, dy)))
    return worst <= max_shift, worst


def background_plate(frames: list[np.ndarray], masks: list[np.ndarray], max_frames: int = 60):
    """Per-pixel median of unmasked samples -> (plate uint8, seen bool (H, W))."""
    idx = np.unique(np.linspace(0, len(frames) - 1, min(max_frames, len(frames))).round().astype(int))
    stack = np.stack([frames[i] for i in idx]).astype(np.float32)
    hidden = np.stack([masks[i] for i in idx])
    stack[hidden] = np.nan
    seen = ~hidden.all(0)
    with np.errstate(all="ignore"):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            plate = np.nanmedian(stack, axis=0)
    plate = np.nan_to_num(plate, nan=0.0)
    return np.clip(plate, 0, 255).round().astype(np.uint8), seen


def _spatial(img, hole, method):
    if method == "criminisi":
        return criminisi(img, hole)
    return opencv_inpaint(img, hole, method)


def remove_object(frames: list[np.ndarray], masks: list[np.ndarray], spatial: str = "telea",
                  context: list[tuple[np.ndarray, np.ndarray]] | None = None,
                  force_temporal: bool = False,
                  exclude: list[np.ndarray] | None = None) -> tuple[list[np.ndarray], dict]:
    """Return frames with the masked object removed, and a report dict.
    `context`: extra (frame, mask) pairs (e.g. from before/after the edited
    range) that only contribute background samples; their masks should
    already include anything to exclude. `exclude`: per-frame masks of other
    things (e.g. every detected person) that must not be sampled for the plate
    but are left alone in the output."""
    import cv2

    all_f = list(frames) + [c[0] for c in (context or [])]
    all_m = list(masks) + [c[1] for c in (context or [])]
    static, shift = camera_is_static(all_f, all_m)
    report = {"static_camera": static, "max_shift_px": round(shift, 2)}
    if not static and not force_temporal:
        report["mode"] = f"per-frame spatial ({spatial}): camera moves"
        return [_spatial(f, m, spatial) for f, m in zip(frames, masks, strict=True)], report
    sample_m = list(all_m)
    if exclude is not None:
        sample_m[:len(frames)] = [m | x for m, x in zip(masks, exclude, strict=True)]
    plate, seen = background_plate(all_f, sample_m)
    if exclude is not None:
        # where excluding everyone left no sample, fall back to the object-only plate
        plate2, seen2 = background_plate(all_f, all_m)
        gap = ~seen & seen2
        plate[gap] = plate2[gap]
        seen = seen | seen2
    union = np.any(np.stack(masks), 0)
    never = union & ~seen
    if never.any():
        plate = _spatial(plate, never, spatial)
    report.update(mode="temporal plate", never_seen_px=int(never.sum()),
                  masked_px=int(union.sum()))
    out = []
    for f, m in zip(frames, masks, strict=True):
        w = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 1.0)
        w = np.maximum(w, m)[..., None]           # fully replace inside, feather outside
        out.append((w * plate + (1 - w) * f).round().astype(np.uint8))
    return out, report
