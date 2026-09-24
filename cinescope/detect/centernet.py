"""CenterNet-style anchor-free detector: ResNet-18 + FPN + three heads (PyTorch).

    image (3, H, W)
      -> ResNet-18 stages C2..C5 (strides 4, 8, 16, 32; 64..512 channels)
      -> FPN: 1x1 lateral convs to 128 ch, top-down upsample-and-add, 3x3 smoothing
      -> P2 (stride 4, 128 ch): fine enough for 10-px faces, with C5's context
      -> heads, each 3x3 conv-ReLU-1x1 conv:
           heatmap (C)  sigmoid; bias initialised to -2.19 so p(object) starts at 0.1
           size    (2)  width, height in output cells
           offset  (2)  sub-cell centre offset

Why an FPN rather than the plain backbone: C2 has the resolution for tiny
faces but only a few conv layers of context; C5 has context but a stride of 32
— a 16-px face is half a cell. The top-down path gives the stride-4 map C5's
semantics.

Anchor-based vs. anchor-free: anchor detectors (RetinaFace, YOLOv3) tile
thousands of preset boxes and learn corrections, with IoU-based assignment and
anchor sizes to tune. CenterNet predicts one point per object and regresses the
size directly: no anchors, no IoU matching, simpler decoding — at the cost of
two objects whose centres fall in the same cell colliding (rare for faces at
stride 4, common for dense crowds of tiny faces).
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from ..framing.resnet import ResNet, resnet18


class FPN(nn.Module):
    def __init__(self, in_channels: tuple[int, ...] = (64, 128, 256, 512), out: int = 128):
        super().__init__()
        self.lateral = nn.ModuleList(nn.Conv2d(c, out, 1) for c in in_channels)
        self.smooth = nn.ModuleList(nn.Sequential(nn.Conv2d(out, out, 3, padding=1, bias=False),
                                                  nn.BatchNorm2d(out), nn.ReLU(inplace=True))
                                    for _ in in_channels)

    def forward(self, feats: list[torch.Tensor]) -> list[torch.Tensor]:
        lat = [conv(f) for conv, f in zip(self.lateral, feats, strict=True)]
        outs = [lat[-1]]
        for f in reversed(lat[:-1]):
            outs.insert(0, f + F.interpolate(outs[0], size=f.shape[-2:], mode="nearest"))
        return [s(o) for s, o in zip(self.smooth, outs, strict=True)]


def _head(c_in: int, c_out: int, mid: int = 64, bias: float = 0.0) -> nn.Sequential:
    h = nn.Sequential(nn.Conv2d(c_in, mid, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(mid, c_out, 1))
    nn.init.constant_(h[-1].bias, bias)
    return h


class CenterNet(nn.Module):
    stride = 4

    def __init__(self, n_classes: int = 1, pretrained_backbone: bool = False, fpn_channels: int = 128):
        super().__init__()
        self.backbone: ResNet = resnet18(1000, pretrained=pretrained_backbone)
        del self.backbone.fc
        self.fpn = FPN(out=fpn_channels)
        prior = 0.1
        self.heatmap = _head(fpn_channels, n_classes, bias=-math.log((1 - prior) / prior))
        self.size = _head(fpn_channels, 2)
        self.offset = _head(fpn_channels, 2)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        b = self.backbone
        x = b.maxpool(b.relu(b.bn1(b.conv1(x))))
        c2 = b.layer1(x)
        c3 = b.layer2(c2)
        c4 = b.layer3(c3)
        c5 = b.layer4(c4)
        p2 = self.fpn([c2, c3, c4, c5])[0]
        return {"heatmap": self.heatmap(p2), "size": self.size(p2), "offset": self.offset(p2)}


def focal_loss(logits: torch.Tensor, target: torch.Tensor, alpha: float = 2.0, beta: float = 4.0) -> torch.Tensor:
    """CenterNet's penalty-reduced focal loss, normalised by the number of objects.
    Positives (Y = 1): -(1-p)^a log p.  Negatives: -(1-Y)^b p^a log(1-p): cells
    near a centre (Y close to 1) are barely penalised for firing."""
    logits = logits.float()
    p = logits.sigmoid().clamp(1e-4, 1 - 1e-4)
    pos = target.eq(1).float()
    neg = 1 - pos
    pos_loss = -(1 - p).pow(alpha) * torch.log(p) * pos
    neg_loss = -(1 - target).pow(beta) * p.pow(alpha) * torch.log(1 - p) * neg
    n = pos.sum().clamp(min=1)
    return (pos_loss.sum() + neg_loss.sum()) / n


def gather(feat: torch.Tensor, ind: torch.Tensor) -> torch.Tensor:
    """feat (B, C, H, W), ind (B, K) flat indices -> (B, K, C)."""
    b, c = feat.shape[:2]
    f = feat.view(b, c, -1).permute(0, 2, 1)
    return f.gather(1, ind.unsqueeze(-1).expand(-1, -1, c))


def reg_l1(pred: torch.Tensor, ind: torch.Tensor, mask: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    p = gather(pred.float(), ind)
    m = mask.unsqueeze(-1)
    return (F.l1_loss(p * m, target * m, reduction="sum") / (m.sum() * 2).clamp(min=1))


def centernet_loss(out: dict[str, torch.Tensor], batch: dict[str, torch.Tensor],
                   w_size: float = 0.1, w_offset: float = 1.0) -> dict[str, torch.Tensor]:
    hm = focal_loss(out["heatmap"], batch["heatmap"])
    size = reg_l1(out["size"], batch["ind"], batch["mask"], batch["wh"])
    off = reg_l1(out["offset"], batch["ind"], batch["mask"], batch["reg"])
    return {"loss": hm + w_size * size + w_offset * off, "heatmap": hm, "size": size, "offset": off}


@torch.no_grad()
def decode(out: dict[str, torch.Tensor], k: int = 100, stride: int = 4, threshold: float = 0.05
           ) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Per image: (xyxy boxes in input pixels, scores, classes). A 3x3 max-pool
    keeps local maxima only — CenterNet's replacement for NMS."""
    heat = out["heatmap"].float().sigmoid()
    b, c, h, w = heat.shape
    peaks = heat * (F.max_pool2d(heat, 3, stride=1, padding=1) == heat)
    scores, idx = peaks.view(b, -1).topk(min(k, c * h * w))
    cls = idx // (h * w)
    rem = idx % (h * w)
    ys, xs = (rem // w).float(), (rem % w).float()
    size = gather(out["size"].float(), rem)
    off = gather(out["offset"].float(), rem)
    cx, cy = xs + off[..., 0], ys + off[..., 1]
    boxes = torch.stack([cx - size[..., 0] / 2, cy - size[..., 1] / 2,
                         cx + size[..., 0] / 2, cy + size[..., 1] / 2], -1) * stride
    res = []
    for i in range(b):
        keep = scores[i] > threshold
        res.append((boxes[i][keep], scores[i][keep], cls[i][keep]))
    return res
