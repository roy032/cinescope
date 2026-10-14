"""A small box-prompted U-Net for person masks, trainable on a CPU.

The task is the same as SAM's with a box prompt: given a frame and a box around
one person, return that person's mask. The network sees the box (plus a margin
of context) cropped and resized to a fixed square, so "which person" is
answered by the crop itself and the model only has to separate figure from
ground — a much easier job than full-frame instance segmentation, which is why
~0.5M parameters and a CPU are enough.

U-Net (Ronneberger et al., 2015): an encoder halves resolution and doubles
channels four times; a decoder upsamples back, concatenating the encoder
feature map of the same size at each step (skip connections) so fine boundary
detail lost by pooling comes back. Loss: BCE + soft Dice — BCE gives every
pixel a gradient, Dice targets the overlap metric directly and is insensitive
to how much of the crop is background.

Targets (see p4_segmentation/train_unet.py): YOLOv8n-seg masks on real frames
of the training clips (distilled: the student inherits the teacher's coarse
boundaries), and synthetic composites of cut-outs over other frames (exact).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .data import crop_padded, expand_box, paste_window

CROP_MARGIN = 0.15
CROP_SIZE = 96


class _Block(nn.Sequential):
    def __init__(self, cin: int, cout: int) -> None:
        super().__init__(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        )


class UNet(nn.Module):
    def __init__(self, base: int = 16, depth: int = 4) -> None:
        super().__init__()
        chs = [base * 2 ** i for i in range(depth + 1)]
        self.down = nn.ModuleList([_Block(3, chs[0])] + [_Block(chs[i], chs[i + 1]) for i in range(depth)])
        self.up = nn.ModuleList([_Block(chs[i + 1] + chs[i], chs[i]) for i in reversed(range(depth))])
        self.head = nn.Conv2d(chs[0], 1, 1)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, H, W) float 0..1 -> (B, 1, H, W) logits. H, W divisible by 2^depth."""
        x = (x - self.mean) / self.std
        skips = []
        for i, blk in enumerate(self.down):
            x = blk(x if i == 0 else F.max_pool2d(x, 2))
            skips.append(x)
        x = skips.pop()
        for blk in self.up:
            s = skips.pop()
            x = F.interpolate(x, size=s.shape[-2:], mode="bilinear", align_corners=False)
            x = blk(torch.cat([x, s], 1))
        return self.head(x)


def seg_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, target)
    p = torch.sigmoid(logits)
    inter = (p * target).sum((1, 2, 3))
    dice = 1 - (2 * inter + 1) / (p.sum((1, 2, 3)) + target.sum((1, 2, 3)) + 1)
    return bce + dice.mean()


def box_crop(rgb: np.ndarray, box, size: int = CROP_SIZE, margin: float = CROP_MARGIN):
    """Crop box + margin (zero-padded at frame edges), resized to size x size.
    Returns the uint8 crop and the window bounds for `paste_mask`."""
    import cv2

    h, w = rgb.shape[:2]
    bounds = expand_box(box, margin, h, w)
    c = crop_padded(rgb, bounds)
    return cv2.resize(c, (size, size), interpolation=cv2.INTER_AREA), bounds


def paste_mask(prob: np.ndarray, bounds, shape) -> np.ndarray:
    """Resize a crop-sized probability map back to the window, threshold at 0.5."""
    import cv2

    x0, y0, x1, y1 = bounds
    m = cv2.resize(prob.astype(np.float32), (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR) > 0.5
    out = np.zeros(shape[:2], bool)
    paste_window(out, m, bounds)
    return out


class UNetSegmenter:
    """Box prompts -> masks with a trained UNet checkpoint."""

    def __init__(self, ckpt: str, device: str = "cpu") -> None:
        state = torch.load(ckpt, map_location=device, weights_only=False)
        cfg = state.get("config", {})
        self.size = cfg.get("size", CROP_SIZE)
        self.margin = cfg.get("margin", CROP_MARGIN)
        self.model = UNet(base=cfg.get("base", 16)).to(device).eval()
        self.model.load_state_dict(state["model"])
        self.device = device

    @torch.inference_mode()
    def __call__(self, rgb: np.ndarray, boxes) -> list[np.ndarray]:
        boxes = np.asarray(boxes, np.float64).reshape(-1, 4)
        if len(boxes) == 0:
            return []
        crops, bounds = zip(*(box_crop(rgb, b, self.size, self.margin) for b in boxes), strict=True)
        x = torch.from_numpy(np.stack(crops)).permute(0, 3, 1, 2).float().div(255).to(self.device)
        prob = torch.sigmoid(self.model(x))[:, 0].cpu().numpy()
        return [paste_mask(p, b, rgb.shape) for p, b in zip(prob, bounds, strict=True)]
