"""Pluggable image/text encoders that map into one shared embedding space.

Every encoder has the same three members, so the index, the search and the chat
code never know which one is running:

    enc.name                      str, stored with the index (an index is only valid
                                  for the encoder that built it)
    enc.encode_images(frames)     list of RGB uint8 arrays -> (N, D) float32, L2-normalised
    enc.encode_texts(texts)       list of str             -> (N, D) float32, L2-normalised

`get_encoder(spec)`:
    "openclip"                            ViT-B-32-quickgelu with the local LAION-400M file
                                          (<weights>/vit_b_32-quickgelu-laion400m_e32.pt)
    "openclip:<model>:<pretrained>"       any open_clip model; <pretrained> is a tag, a file,
                                          or e.g. "openclip:ViT-B-16-SigLIP:webli" for SigLIP
                                          where HuggingFace downloads are allowed
    "scratch:<checkpoint.pt>"             the from-scratch dual encoder (dual_encoder.py)

Wide frames. CLIP sees a 224 x 224 centre crop; on a 2.39:1 film frame that throws
away the outer 58% of the picture, where half the action can be. `tiles=True`
embeds ceil(aspect) square crops that together cover the full width and averages
them (then renormalises) — a fixed choice made before any evaluation.
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
OPENCLIP_DEFAULT = ("ViT-B-32-quickgelu", "vit_b_32-quickgelu-laion400m_e32.pt")


def weights_dir() -> Path:
    return Path(os.environ.get("CINESCOPE_WEIGHTS", REPO / "data" / "weights"))


def square_tiles(rgb: np.ndarray) -> list[np.ndarray]:
    """ceil(w/h) evenly spaced h x h crops covering the whole width (one crop for <= 1:1)."""
    h, w = rgb.shape[:2]
    if w <= h:
        return [rgb]
    n = math.ceil(w / h - 1e-6)
    xs = np.linspace(0, w - h, n).round().astype(int)
    return [rgb[:, x:x + h] for x in xs]


def _l2(x: np.ndarray) -> np.ndarray:
    return (x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)).astype(np.float32)


class _TiledMixin:
    tiles: bool = True

    def _embed_crops(self, crops: list[np.ndarray]) -> np.ndarray:   # pragma: no cover - overridden
        raise NotImplementedError

    def encode_images(self, frames: list[np.ndarray], batch: int = 32) -> np.ndarray:
        crops, owner = [], []
        for i, f in enumerate(frames):
            parts = square_tiles(f) if self.tiles else [f]
            crops += parts
            owner += [i] * len(parts)
        emb = np.concatenate([self._embed_crops(crops[s:s + batch]) for s in range(0, len(crops), batch)]) \
            if crops else np.zeros((0, self.dim), np.float32)
        out = np.zeros((len(frames), emb.shape[1]), np.float32)
        np.add.at(out, np.asarray(owner), emb)
        return _l2(out)


class OpenClipEncoder(_TiledMixin):
    def __init__(self, model: str = OPENCLIP_DEFAULT[0], pretrained: str | None = None, tiles: bool = True,
                 device: str = "cpu") -> None:
        import open_clip
        import torch

        if pretrained is None:
            pretrained = str(weights_dir() / OPENCLIP_DEFAULT[1])
            if not Path(pretrained).exists():
                raise FileNotFoundError(f"{pretrained} not found: set CINESCOPE_WEIGHTS or pass "
                                        "--encoder openclip:<model>:<pretrained>")
        self.torch = torch
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model, pretrained=pretrained,
                                                                               device=device)
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(model)
        self.device = device
        self.tiles = tiles
        self.name = f"openclip:{model}:{Path(pretrained).name}"
        with torch.no_grad():
            self.dim = int(self.model.encode_text(self.tokenizer(["x"]).to(device)).shape[-1])

    def _embed_crops(self, crops: list[np.ndarray]) -> np.ndarray:
        from PIL import Image

        x = self.torch.stack([self.preprocess(Image.fromarray(c)) for c in crops]).to(self.device)
        with self.torch.no_grad():
            return _l2(self.model.encode_image(x).float().cpu().numpy())

    def encode_texts(self, texts: list[str], batch: int = 256) -> np.ndarray:
        out = []
        with self.torch.no_grad():
            for s in range(0, len(texts), batch):
                out.append(self.model.encode_text(self.tokenizer(texts[s:s + batch]).to(self.device)).float().cpu().numpy())
        return _l2(np.concatenate(out)) if out else np.zeros((0, self.dim), np.float32)


class ScratchEncoder(_TiledMixin):
    """The from-scratch dual encoder, resized to its training resolution."""

    def __init__(self, checkpoint: str | Path, tiles: bool = True) -> None:
        import torch

        from .dual_encoder import DualEncoder

        self.torch = torch
        self.model = DualEncoder.load(checkpoint)
        self.size = self.model.cfg.image_size
        self.dim = self.model.cfg.embed_dim
        self.tiles = tiles
        self.name = f"scratch:{Path(checkpoint).name}"

    def _embed_crops(self, crops: list[np.ndarray]) -> np.ndarray:
        from PIL import Image

        from .dual_encoder import to_tensor

        arr = [np.asarray(Image.fromarray(c).resize((self.size, self.size), Image.BILINEAR)) for c in crops]
        with self.torch.no_grad():
            return self.model.encode_image(to_tensor(arr)).numpy()

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        with self.torch.no_grad():
            return self.model.encode_text(self.model.tokenizer(texts)).numpy().astype(np.float32)

    def coverage(self, text: str) -> float:
        return self.model.tokenizer.coverage(text)


def get_encoder(spec: str = "openclip", tiles: bool = True):
    if spec == "openclip":
        return OpenClipEncoder(tiles=tiles)
    if spec.startswith("openclip:"):
        _, model, *rest = spec.split(":", 2)
        return OpenClipEncoder(model, rest[0] if rest else None, tiles=tiles)
    if spec.startswith("scratch:"):
        return ScratchEncoder(spec.split(":", 1)[1], tiles=tiles)
    raise ValueError(f"unknown encoder {spec!r} (openclip | openclip:<model>:<pretrained> | scratch:<ckpt>)")
