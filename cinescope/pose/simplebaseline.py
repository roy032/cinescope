"""A SimpleBaseline pose network (Xiao et al., 2018, "Simple Baselines for
Human Pose Estimation and Tracking"), from scratch.

    person crop (H x W) -> ResNet (stride 32) -> 3 x [deconv 4x4/2, BN, ReLU]
                        -> 1x1 conv -> K heatmaps at H/4 x W/4

The paper's point was that the elaborate multi-stage designs of the time
(Hourglass, CPN) were not needed: upsampling the last ResNet stage with three
transposed convolutions and regressing Gaussian heatmaps with an MSE loss was
as accurate. The backbone layout and parameter names follow torchvision's
ResNet, so at width 64 the ImageNet ResNet-18 weights (the timm release file
in the weights folder) load into it; narrower widths are for tests.

Loss: MSE between predicted and target heatmaps, each keypoint weighted by its
target weight (0 for unlabelled keypoints, so the network is not taught that an
unlabelled wrist is "nowhere").
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ..framing.resnet import BasicBlock
from . import FLIP_PAIRS, K
from .heatmaps import affine_crop, decode_heatmaps, heatmap_to_crop, transform_points

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
IMAGENET_R18 = "resnet18_a1_0-d63eafa0.pth"     # timm's ResNet-18 (torchvision parameter names)


def weights_dir() -> Path:
    return Path(os.environ.get("CINESCOPE_WEIGHTS", Path(__file__).resolve().parents[2] / "data" / "weights"))


class SimpleBaseline(nn.Module):
    def __init__(self, n_kpts: int = K, width: int = 64, layers: tuple[int, ...] = (2, 2, 2, 2),
                 deconv: int = 256, n_deconv: int = 3):
        super().__init__()
        self.conv1 = nn.Conv2d(3, width, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        c_in = width
        for i, n in enumerate(layers):
            w = width * 2 ** i
            blocks = []
            for j in range(n):
                blocks.append(BasicBlock(c_in, w, stride=2 if (j == 0 and i > 0) else 1))
                c_in = w
            setattr(self, f"layer{i + 1}", nn.Sequential(*blocks))
        ups = []
        for _ in range(n_deconv):
            ups += [nn.ConvTranspose2d(c_in, deconv, 4, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(deconv), nn.ReLU(inplace=True)]
            c_in = deconv
        self.deconv = nn.Sequential(*ups)
        self.final = nn.Conv2d(c_in, n_kpts, 1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.normal_(self.final.weight, std=0.001)     # start near all-zero heatmaps
        nn.init.zeros_(self.final.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return self.final(self.deconv(x))

    def load_imagenet(self, path: str | Path | None = None) -> list[str]:
        """Copy ImageNet ResNet-18 weights into the backbone; returns the keys loaded."""
        path = Path(path) if path else weights_dir() / IMAGENET_R18
        state = torch.load(path, map_location="cpu", weights_only=True)
        own = self.state_dict()
        state = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape}
        self.load_state_dict(state, strict=False)
        return sorted(state)


def heatmap_loss(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Weighted MSE (SimpleBaseline's JointsMSELoss): mean over keypoints and
    cells of w_k * (pred - target)^2 / 2."""
    w = weight[:, :, None, None]
    return 0.5 * ((pred - target) ** 2 * w).mean()


def flip_heatmaps(hm: torch.Tensor, pairs=FLIP_PAIRS) -> torch.Tensor:
    """Heatmaps of a mirrored crop -> heatmaps of the original: mirror the maps
    and swap each left/right pair of channels."""
    hm = hm.flip(-1)
    idx = list(range(hm.shape[1]))
    for a, b in pairs:
        idx[a], idx[b] = b, a
    return hm[:, idx]


def to_tensor(crops: np.ndarray) -> torch.Tensor:
    """(N, H, W, 3) uint8 RGB -> normalised (N, 3, H, W) float tensor."""
    x = (crops.astype(np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(x.transpose(0, 3, 1, 2).copy())


@torch.no_grad()
def predict(net: SimpleBaseline, img: np.ndarray, boxes: np.ndarray, input_hw: tuple[int, int] = (128, 96),
            flip: bool = True, method: str = "taylor", blur: float = 1.0) -> np.ndarray:
    """Top-down inference: (B, 4) xyxy person boxes in an RGB frame -> (B, K, 3)
    keypoints (x, y, heatmap peak) in frame coordinates. With `flip`, the
    heatmaps of the mirrored crop are averaged in (standard test-time trick)."""
    boxes = np.asarray(boxes, np.float64).reshape(-1, 4)
    if not len(boxes):
        return np.zeros((0, K, 3))
    crops, mats = zip(*(affine_crop(img, b, input_hw) for b in boxes), strict=True)
    x = to_tensor(np.stack(crops))
    net.eval()
    hm = net(x)
    if flip:
        hm = 0.5 * (hm + flip_heatmaps(net(x.flip(-1))))
    hm = hm.numpy()
    xy, conf = decode_heatmaps(hm, method=method, blur=blur)
    out = np.zeros((len(boxes), hm.shape[1], 3))
    for i, M in enumerate(mats):
        out[i, :, :2] = transform_points(heatmap_to_crop(xy[i], input_hw, hm.shape[2:]), M, inverse=True)
        out[i, :, 2] = conf[i]
    return out


def load_checkpoint(path: str | Path) -> tuple[SimpleBaseline, dict]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    net = SimpleBaseline(**ck.get("arch", {}))
    net.load_state_dict(ck["model"])
    return net.eval(), ck
