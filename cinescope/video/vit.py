"""Vision Transformer, from scratch (Dosovitskiy et al., 2021), for images and
short clips.

    image (C, H, W) -> p x p patches -> linear embedding (patch_dim -> D)
    video (C, T, H, W) -> t x p x p "tubelets" -> linear embedding (VideoMAE, ViViT)
    + position embeddings, [CLS] token prepended
    -> L x [x + MHSA(LN(x)); x + MLP(LN(x))]      (pre-norm blocks)
    -> LN -> head on the [CLS] token (or on the mean of all tokens)

Multi-head self-attention, written out: each token x_i is projected to a
query, key and value per head (one Linear producing all three for all heads);
head h computes softmax(Q K^T / sqrt(d)) V, so every token's output is a
weighted mix of every token's value, the weights being query-key similarity.
The heads' outputs are concatenated and mixed by a final Linear. Dividing by
sqrt(d) keeps the dot products' variance near 1 regardless of head width, so
the softmax does not saturate at initialisation.

Why a [CLS] token: attention is permutation-equivariant, so the model needs a
designated slot whose output summarises the image; it attends to all patches
and has no patch of its own. Mean-pooling all tokens works as well (VideoMAE
fine-tunes that way) and is used for the video encoder here.

Position embeddings: attention ignores order, so each token gets a vector
added that says where it came from. Learned (ViT) or fixed sinusoids
(`sincos_embedding`, the Transformer's, as VideoMAE uses): fixed ones need no
training data, which matters with the few thousand clips a CPU can see.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def patchify(x: torch.Tensor, p: int, t: int = 1) -> torch.Tensor:
    """(B, C, H, W) -> (B, N, C*p*p), or (B, C, T, H, W) -> (B, N, C*t*p*p),
    tokens ordered time-major then row-major. Pure reshapes: equal to a
    Conv2d/Conv3d with kernel = stride = patch size followed by flattening."""
    if x.dim() == 4:
        x = x.unsqueeze(2)
        t = 1
    B, C, T, H, W = x.shape
    x = x.reshape(B, C, T // t, t, H // p, p, W // p, p)
    x = x.permute(0, 2, 4, 6, 1, 3, 5, 7)               # B, T', H', W', C, t, p, p
    return x.reshape(B, (T // t) * (H // p) * (W // p), C * t * p * p)


class PatchEmbed(nn.Module):
    def __init__(self, patch: int, dim: int, in_ch: int = 3, tubelet: int = 1):
        super().__init__()
        self.p, self.t = patch, tubelet
        self.proj = nn.Linear(in_ch * tubelet * patch * patch, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(patchify(x, self.p, self.t))


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float = 0.0):
        super().__init__()
        assert dim % heads == 0
        self.h, self.d = heads, dim // heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)
        self.last_attn: torch.Tensor | None = None       # (B, heads, N, N), kept for visualisation

    def forward(self, x: torch.Tensor, keep_attn: bool = False) -> torch.Tensor:
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.h, self.d).permute(2, 0, 3, 1, 4)   # 3, B, h, N, d
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.d)
        attn = attn.softmax(dim=-1)
        if keep_attn:
            self.last_attn = attn.detach()
        out = self.drop(attn) @ v                        # B, h, N, d
        return self.proj(out.transpose(1, 2).reshape(B, N, D))


class Block(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(int(dim * mlp_ratio), dim), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor, keep_attn: bool = False) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), keep_attn)
        return x + self.mlp(self.norm2(x))


def sincos_embedding(n: int, dim: int) -> torch.Tensor:
    """(n, dim) fixed sinusoidal position codes (Vaswani et al., 2017)."""
    pos = torch.arange(n, dtype=torch.float32)[:, None]
    i = torch.arange(dim // 2, dtype=torch.float32)[None, :]
    ang = pos / (10000 ** (2 * i / dim))
    return torch.cat([ang.sin(), ang.cos()], dim=1)


class ViT(nn.Module):
    """ViT for (B, 3, H, W) images (tubelet=1, frames=1) or (B, 3, T, H, W) clips.

    pool="cls": [CLS] token, learned position embeddings (the original ViT).
    pool="mean": no [CLS], fixed sin-cos positions, mean of tokens (VideoMAE).
    """

    def __init__(self, img_hw: tuple[int, int] = (64, 112), patch: int = 8, frames: int = 1, tubelet: int = 1,
                 dim: int = 192, depth: int = 6, heads: int = 3, num_classes: int = 0, pool: str = "cls",
                 dropout: float = 0.0):
        super().__init__()
        self.img_hw, self.patch, self.frames, self.tubelet = img_hw, patch, frames, tubelet
        self.grid = (frames // tubelet, img_hw[0] // patch, img_hw[1] // patch)
        self.n_tokens = self.grid[0] * self.grid[1] * self.grid[2]
        self.pool = pool
        self.embed = PatchEmbed(patch, dim, 3, tubelet)
        if pool == "cls":
            self.cls = nn.Parameter(torch.zeros(1, 1, dim))
            self.pos = nn.Parameter(torch.randn(1, self.n_tokens + 1, dim) * 0.02)
        else:
            self.register_buffer("pos", sincos_embedding(self.n_tokens, dim)[None], persistent=False)
        self.blocks = nn.ModuleList(Block(dim, heads, dropout=dropout) for _ in range(depth))
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes) if num_classes else nn.Identity()
        self.apply(_init)

    def tokens(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4 and self.frames > 1:
            raise ValueError("this ViT expects clips (B, C, T, H, W)")
        z = self.embed(x)
        if self.pool == "cls":
            z = torch.cat([self.cls.expand(len(z), -1, -1), z], 1)
        return z + self.pos

    def features(self, x: torch.Tensor, keep_attn: bool = False) -> torch.Tensor:
        z = self.tokens(x)
        for blk in self.blocks:
            z = blk(z, keep_attn)
        z = self.norm(z)
        return z[:, 0] if self.pool == "cls" else z.mean(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


def _init(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


def reference_attention(x: torch.Tensor, mhsa: MultiHeadSelfAttention) -> torch.Tensor:
    """The same computation through PyTorch's fused kernel (for the tests)."""
    B, N, D = x.shape
    qkv = mhsa.qkv(x).reshape(B, N, 3, mhsa.h, mhsa.d).permute(2, 0, 3, 1, 4)
    out = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2])
    return mhsa.proj(out.transpose(1, 2).reshape(B, N, D))
