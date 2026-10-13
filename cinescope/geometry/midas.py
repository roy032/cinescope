"""MiDaS v2.1 small (Ranftl et al., 2020): the library baseline for monocular depth.

The network code is vendored from github.com/isl-org/MiDaS (MIT licence,
midas/midas_net_custom.py and midas/blocks.py), trimmed to the small model,
because torch.hub would need network access at run time. The EfficientNet-
Lite3 encoder comes from timm (`tf_efficientnet_lite3`, the same architecture
MiDaS loads from rwightman's gen-efficientnet; the parameter names match).

Weights: midas_v21_small_256.pt (the official release file), looked up in
`$CINESCOPE_WEIGHTS` or `<repo>/data/weights`.

The output is *relative inverse depth* (larger = nearer), defined only up to
an unknown scale and shift — MiDaS is trained with a scale- and shift-
invariant loss so it can learn from datasets with incompatible depth units.
Comparisons with metric or SfM depth therefore fit that affine map first.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np


def weights_dir() -> Path:
    return Path(os.environ.get("CINESCOPE_WEIGHTS", Path(__file__).resolve().parents[2] / "data" / "weights"))


def _build():
    import timm
    import torch
    from torch import nn

    class ResidualConvUnit(nn.Module):
        def __init__(self, features):
            super().__init__()
            self.conv1 = nn.Conv2d(features, features, 3, 1, 1, bias=True)
            self.conv2 = nn.Conv2d(features, features, 3, 1, 1, bias=True)
            self.act = nn.ReLU(False)

        def forward(self, x):
            return self.conv2(self.act(self.conv1(self.act(x)))) + x

    class FeatureFusionBlock(nn.Module):
        def __init__(self, features, expand):
            super().__init__()
            out = features // 2 if expand else features
            self.out_conv = nn.Conv2d(features, out, 1, 1, 0, bias=True)
            self.resConfUnit1 = ResidualConvUnit(features)
            self.resConfUnit2 = ResidualConvUnit(features)

        def forward(self, *xs):
            out = xs[0]
            if len(xs) == 2:
                out = out + self.resConfUnit1(xs[1])
            out = self.resConfUnit2(out)
            out = nn.functional.interpolate(out, scale_factor=2, mode="bilinear", align_corners=True)
            return self.out_conv(out)

    class Interpolate(nn.Module):
        def forward(self, x):
            return nn.functional.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)

    class MidasSmall(nn.Module):
        def __init__(self, features=64):
            super().__init__()
            eff = timm.create_model("tf_efficientnet_lite3", pretrained=False)
            self.pretrained = nn.Module()
            self.pretrained.layer1 = nn.Sequential(eff.conv_stem, eff.bn1, nn.Identity(), *eff.blocks[0:2])
            self.pretrained.layer2 = nn.Sequential(*eff.blocks[2:3])
            self.pretrained.layer3 = nn.Sequential(*eff.blocks[3:5])
            self.pretrained.layer4 = nn.Sequential(*eff.blocks[5:9])
            ins, outs = [32, 48, 136, 384], [features, features * 2, features * 4, features * 8]
            self.scratch = nn.Module()
            for i, (a, b) in enumerate(zip(ins, outs, strict=True), 1):
                setattr(self.scratch, f"layer{i}_rn", nn.Conv2d(a, b, 3, 1, 1, bias=False))
            self.scratch.refinenet4 = FeatureFusionBlock(outs[3], True)
            self.scratch.refinenet3 = FeatureFusionBlock(outs[2], True)
            self.scratch.refinenet2 = FeatureFusionBlock(outs[1], True)
            self.scratch.refinenet1 = FeatureFusionBlock(outs[0], False)
            self.scratch.output_conv = nn.Sequential(
                nn.Conv2d(features, features // 2, 3, 1, 1), Interpolate(),
                nn.Conv2d(features // 2, 32, 3, 1, 1), nn.ReLU(False),
                nn.Conv2d(32, 1, 1, 1, 0), nn.ReLU(True), nn.Identity())

        def forward(self, x):
            l1 = self.pretrained.layer1(x)
            l2 = self.pretrained.layer2(l1)
            l3 = self.pretrained.layer3(l2)
            l4 = self.pretrained.layer4(l3)
            s = self.scratch
            p4 = s.refinenet4(s.layer4_rn(l4))
            p3 = s.refinenet3(p4, s.layer3_rn(l3))
            p2 = s.refinenet2(p3, s.layer2_rn(l2))
            p1 = s.refinenet1(p2, s.layer1_rn(l1))
            return torch.squeeze(s.output_conv(p1), 1)

    return MidasSmall()


class MiDaS:
    def __init__(self, weights: str | Path | None = None, threads: int = 2):
        import torch

        torch.set_num_threads(threads)
        path = Path(weights) if weights else weights_dir() / "midas_v21_small_256.pt"
        self.model = _build()
        sd = torch.load(path, map_location="cpu", weights_only=True)
        missing, unexpected = self.model.load_state_dict(sd, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"MiDaS weights do not match: missing {missing[:5]}, unexpected {unexpected[:5]}")
        self.model.eval()

    def __call__(self, rgb: np.ndarray, size: int = 256) -> np.ndarray:
        """Relative inverse depth at the input resolution."""
        import torch

        h, w = rgb.shape[:2]
        # MiDaS small: keep the aspect ratio, longer side ~size, both sides multiples of 32
        s = size / max(h, w)
        nh, nw = max(32, int(round(h * s / 32)) * 32), max(32, int(round(w * s / 32)) * 32)
        x = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1)[None]
        x = torch.nn.functional.interpolate(x, size=(nh, nw), mode="bicubic", align_corners=False)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        with torch.no_grad():
            out = self.model((x - mean) / std)[None]
            out = torch.nn.functional.interpolate(out, size=(h, w), mode="bicubic", align_corners=False)
        return out[0, 0].numpy()
