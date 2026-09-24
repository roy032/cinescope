"""ResNet-18 from scratch (He et al., 2015, "Deep Residual Learning").

    stem   7x7/2 conv, BN, ReLU, 3x3/2 max-pool          224 -> 56
    layer1 2 BasicBlocks,  64 ch                           56
    layer2 2 BasicBlocks, 128 ch, first one stride 2       28
    layer3 2 BasicBlocks, 256 ch, stride 2                 14
    layer4 2 BasicBlocks, 512 ch, stride 2                  7
    head   global average pool, fully connected

A BasicBlock computes relu(F(x) + shortcut(x)) with F = conv-BN-ReLU-conv-BN.
When the block changes resolution or width, the shortcut is a strided 1x1
conv + BN ("option B" in the paper); otherwise it is the identity.

Parameter names match torchvision's exactly, so ImageNet weights load straight
into this model (`load_imagenet_weights`) and tests can compare the two
networks output for output. Convs before BN have no bias: BN subtracts the
mean, which cancels any bias (the NumPy gradient check shows its gradient is
exactly zero).

Global average pooling makes the network accept any input size, so frames can
keep their 16:9 shape instead of being cropped square — cropping would change
the very thing being classified (how much of the frame the subject fills).
"""
from __future__ import annotations

import torch
from torch import nn


def conv3x3(c_in: int, c_out: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, c_in: int, c_out: int, stride: int = 1):
        super().__init__()
        self.conv1 = conv3x3(c_in, c_out, stride)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(c_out, c_out)
        self.bn2 = nn.BatchNorm2d(c_out)
        self.downsample: nn.Module | None = None
        if stride != 1 or c_in != c_out:
            self.downsample = nn.Sequential(nn.Conv2d(c_in, c_out, 1, stride=stride, bias=False),
                                            nn.BatchNorm2d(c_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class ResNet(nn.Module):
    def __init__(self, layers: tuple[int, ...] = (2, 2, 2, 2), num_classes: int = 1000,
                 zero_init_residual: bool = False, dropout: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        widths, c_in = (64, 128, 256, 512), 64
        for i, (w, n) in enumerate(zip(widths, layers, strict=True)):
            blocks = []
            for j in range(n):
                blocks.append(BasicBlock(c_in, w, stride=2 if (j == 0 and i > 0) else 1))
                c_in = w
            setattr(self, f"layer{i + 1}", nn.Sequential(*blocks))
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()
        self.fc = nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        if zero_init_residual:
            # Each block starts as the identity (last BN scale = 0): the network
            # begins "shallow" and learns how much each block should contribute.
            # Goyal et al. (2017) report it helps large-batch training.
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.zeros_(m.bn2.weight)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return torch.flatten(self.avgpool(x), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.dropout(self.features(x)))


def resnet18(num_classes: int = 1000, pretrained: bool = False, **kw) -> ResNet:
    model = ResNet((2, 2, 2, 2), num_classes=num_classes, **kw)
    if pretrained:
        load_imagenet_weights(model)
    return model


def load_imagenet_weights(model: ResNet) -> list[str]:
    """Copy torchvision's ImageNet ResNet-18 weights into `model`, except the
    classifier when the number of classes differs. Returns the skipped keys.
    The weights are downloaded once by torchvision into its cache."""
    from torchvision.models import ResNet18_Weights
    from torchvision.models import resnet18 as tv_resnet18

    state = tv_resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).state_dict()
    own = model.state_dict()
    skipped = [k for k, v in state.items() if k not in own or own[k].shape != v.shape]
    for k in skipped:
        state.pop(k)
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not unexpected, unexpected
    assert set(missing) <= {"fc.weight", "fc.bias"}, missing
    return skipped


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
