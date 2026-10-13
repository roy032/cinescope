"""A face-embedding network trained from scratch, without identity labels.

Goal: a function f(face) -> 128-d unit vector where two faces of the same
person are close and two different people are far, so tracks can be grouped
into characters.

Supervision comes from the tracks themselves (self-supervised, any clip):

  positives   two frames of the same track -- the tracker says it is one face;
  negatives   two tracks visible in the same frame -- one person cannot be in
              two places at once -- and two tracks from different clips.

Pairs of tracks from the same clip that never co-occur are *unknown* (the
same character in two shots, or two characters): they are left out of the
loss instead of being guessed. That is the difference from naive ArcFace on
track ids, which treats every other track as a different class and so pushes
the same actor's shots apart (`ArcFaceLoss` is here for that comparison).

Losses:
  batch-hard triplet (Hermans et al. 2017): for every anchor, the farthest
      positive and the closest *known* negative in the batch;
      loss = relu(d(a, p) - d(a, n) + margin) on cosine distance.
  ArcFace (Deng et al. 2019): softmax over class weight vectors with an
      additive angular margin m on the true class, cos(theta + m), scale s.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MEAN = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
STD = torch.tensor([0.25, 0.25, 0.25]).view(1, 3, 1, 1)


class _Block(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm2d(cout)
        self.skip = None if stride == 1 and cin == cout else nn.Sequential(
            nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        return F.relu(y + (x if self.skip is None else self.skip(x)))


class EmbedNet(nn.Module):
    """A small ResNet for 64x64 aligned faces (~1.2 M parameters).
    64 px keeps CPU training fast; identity cues at that size are the face's
    layout, hair and skin tone more than fine texture."""

    def __init__(self, dim: int = 128, width: int = 32, size: int = 64):
        super().__init__()
        self.size = size
        w = width
        self.stem = nn.Sequential(nn.Conv2d(3, w, 3, 1, 1, bias=False), nn.BatchNorm2d(w), nn.ReLU())
        self.layers = nn.Sequential(_Block(w, w, 2), _Block(w, 2 * w, 2), _Block(2 * w, 4 * w, 2),
                                    _Block(4 * w, 8 * w, 2))
        self.drop = nn.Dropout(0.2)
        self.fc = nn.Linear(8 * w * 16, dim)      # 4x4 map flattened: keeps *where* features are
        self.bn = nn.BatchNorm1d(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layers(self.stem(x))
        x = self.bn(self.fc(self.drop(x.flatten(1))))
        return F.normalize(x, dim=1)


def to_input(crops_uint8: np.ndarray | torch.Tensor, size: int = 64) -> torch.Tensor:
    """(N, 112, 112, 3) uint8 aligned RGB faces -> normalised (N, 3, size, size)."""
    x = torch.as_tensor(np.asarray(crops_uint8)).permute(0, 3, 1, 2).float() / 255.0
    if x.shape[-1] != size:
        x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False, antialias=True)
    return (x - MEAN) / STD


def augment(x: torch.Tensor, g: torch.Generator) -> torch.Tensor:
    """Cheap batch augmentation on normalised inputs: flip, shift/zoom,
    brightness/contrast, grayscale, blur. Frames of one track are near
    duplicates; without augmentation the positives are trivially close."""
    n = x.shape[0]
    flip = torch.rand(n, generator=g) < 0.5
    x = torch.where(flip.view(-1, 1, 1, 1), x.flip(-1), x)
    s = 1 + (torch.rand(n, generator=g) - 0.5) * 0.2
    t = (torch.rand(n, 2, generator=g) - 0.5) * 0.12
    theta = torch.zeros(n, 2, 3)
    theta[:, 0, 0] = s
    theta[:, 1, 1] = s
    theta[:, :, 2] = t
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    x = F.grid_sample(x, grid, padding_mode="border", align_corners=False)
    b = (torch.rand(n, 1, 1, 1, generator=g) - 0.5) * 0.8
    c = 1 + (torch.rand(n, 1, 1, 1, generator=g) - 0.5) * 0.6
    x = x * c + b
    gray = torch.rand(n, generator=g) < 0.2
    x = torch.where(gray.view(-1, 1, 1, 1), x.mean(1, keepdim=True).expand_as(x), x)
    blur = torch.rand(n, generator=g) < 0.2
    xb = F.avg_pool2d(x, 3, 1, 1, count_include_pad=False)
    return torch.where(blur.view(-1, 1, 1, 1), xb, x)


def batch_hard_triplet(emb: torch.Tensor, track: torch.Tensor, known_neg: torch.Tensor,
                       margin: float = 0.3) -> tuple[torch.Tensor, dict]:
    """emb (B, D) unit vectors; track (B,) track ids; known_neg (B, B) bool --
    True where the two samples are known to be different people."""
    d = 1.0 - emb @ emb.T
    same = track[:, None] == track[None, :]
    eye = torch.eye(len(emb), dtype=torch.bool)
    pos = torch.where(same & ~eye, d, torch.full_like(d, -1.0)).max(1).values
    neg = torch.where(known_neg, d, torch.full_like(d, 9.0)).min(1).values
    ok = (pos >= 0) & (neg < 9)
    loss = F.relu(pos - neg + margin)[ok]
    p, q = pos.detach()[ok], neg.detach()[ok]
    stats = {"pos": float(p.mean()) if ok.any() else 0.0,
             "neg": float(q.mean()) if ok.any() else 0.0,
             "active": float((loss.detach() > 0).float().mean()) if ok.any() else 0.0}
    return (loss.mean() if ok.any() else emb.sum() * 0.0), stats


class ArcFaceLoss(nn.Module):
    """Additive angular margin softmax over `n_classes` (here: track ids)."""

    def __init__(self, dim: int, n_classes: int, s: float = 30.0, m: float = 0.3):
        super().__init__()
        self.w = nn.Parameter(torch.randn(n_classes, dim) * 0.01)
        self.s, self.m = s, m

    def forward(self, emb: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        cos = emb @ F.normalize(self.w, dim=1).T
        theta = torch.acos(cos.clamp(-1 + 1e-6, 1 - 1e-6))
        target = torch.cos(theta + self.m)
        onehot = F.one_hot(y, cos.shape[1]).bool()
        logits = torch.where(onehot, target, cos) * self.s
        return F.cross_entropy(logits, y)


@torch.no_grad()
def embed(model: EmbedNet, crops: np.ndarray, batch: int = 256, flip_tta: bool = True) -> np.ndarray:
    """(N, 112, 112, 3) aligned RGB faces -> (N, D) unit embeddings."""
    model.eval()
    out = []
    for i in range(0, len(crops), batch):
        x = to_input(crops[i:i + batch], model.size)
        e = model(x)
        if flip_tta:
            e = F.normalize(e + model(x.flip(-1)), dim=1)
        out.append(e.numpy())
    return np.concatenate(out) if out else np.zeros((0, model.fc.out_features), np.float32)


def load(path: str, device: str = "cpu") -> EmbedNet:
    state = torch.load(path, map_location=device, weights_only=False)
    m = EmbedNet(**state.get("config", {}))
    m.load_state_dict(state["model"])
    return m.eval()


class SFaceEmbedder:
    """OpenCV zoo's SFace (Zhong et al. 2021): MobileFaceNet-style, trained on
    millions of labelled identities -- the library baseline."""

    def __init__(self, model: str = "sface.onnx"):
        import cv2

        from .detectors import weight_path

        self.rec = cv2.FaceRecognizerSF.create(str(weight_path(model)), "")

    def __call__(self, crops: np.ndarray) -> np.ndarray:
        out = [self.rec.feature(np.ascontiguousarray(c[:, :, ::-1])).ravel() for c in crops]
        e = np.asarray(out, np.float32).reshape(len(crops), -1)
        return e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-12)
