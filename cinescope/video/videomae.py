"""VideoMAE (Tong et al., 2022): self-supervised pre-training by reconstructing
masked video, from scratch.

    clip (3, T, H, W) -> tubelets (2 frames x p x p) -> drop 90% of them
    encoder (ViT)  sees only the ~10% visible tubelets
    decoder (small ViT) gets the encoded visible tokens + a shared learned
                   [MASK] token at every hidden position (all with positions)
                   and predicts the pixels of the hidden tubelets
    loss           MSE on hidden tubelets only, each target patch normalised
                   by its own mean and std (so the model learns structure, not
                   brightness)

Two choices make video different from images (MAE):

- **Tube masking.** The same spatial positions are hidden in every frame.
  Video is very redundant in time; with random per-frame masks a hidden patch
  could be copied from the same place in the neighbouring frame, and the task
  would teach nothing. Masking whole tubes forces the model to infer content
  from *other places*, i.e. to understand the scene.
- **A very high mask ratio (90%).** Same reason: the redundancy makes 75% (MAE's
  image ratio) too easy. And the encoder only runs on 10% of the tokens, which
  makes pre-training cheap — what makes it feasible on a CPU at all.

After pre-training the decoder is thrown away; the encoder, given all tokens,
is fine-tuned or linear-probed on labels.
"""
from __future__ import annotations

import torch
from torch import nn

from .vit import Block, ViT, _init, patchify, sincos_embedding


def tube_mask(batch: int, n_time: int, n_space: int, ratio: float, generator: torch.Generator | None = None
              ) -> torch.Tensor:
    """(B, n_time * n_space) bool, True = hidden. One random spatial mask per
    clip, repeated over every time slot; exactly round(ratio * n_space) hidden."""
    n_hide = int(round(ratio * n_space))
    noise = torch.rand(batch, n_space, generator=generator)
    order = noise.argsort(1)
    m = torch.zeros(batch, n_space, dtype=torch.bool)
    m.scatter_(1, order[:, :n_hide], True)
    return m.repeat(1, n_time)


class VideoMAE(nn.Module):
    def __init__(self, img_hw: tuple[int, int] = (64, 112), patch: int = 8, frames: int = 4, tubelet: int = 2,
                 dim: int = 192, depth: int = 6, heads: int = 3, dec_dim: int = 96, dec_depth: int = 2,
                 dec_heads: int = 3, mask_ratio: float = 0.9):
        super().__init__()
        self.encoder = ViT(img_hw, patch, frames, tubelet, dim, depth, heads, num_classes=0, pool="mean")
        self.mask_ratio = mask_ratio
        n = self.encoder.n_tokens
        self.enc_to_dec = nn.Linear(dim, dec_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dec_dim))
        self.register_buffer("dec_pos", sincos_embedding(n, dec_dim)[None], persistent=False)
        self.decoder = nn.ModuleList(Block(dec_dim, dec_heads) for _ in range(dec_depth))
        self.dec_norm = nn.LayerNorm(dec_dim)
        self.pred = nn.Linear(dec_dim, 3 * tubelet * patch * patch)
        for m in (self.enc_to_dec, self.decoder, self.dec_norm, self.pred):
            m.apply(_init)
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def targets(self, clip: torch.Tensor) -> torch.Tensor:
        """Per-tubelet normalised pixels: (B, N, 3*t*p*p)."""
        x = patchify(clip, self.encoder.patch, self.encoder.tubelet)
        mu, sd = x.mean(-1, keepdim=True), x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / (sd + 1e-6).sqrt()

    def forward(self, clip: torch.Tensor, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor,
                                                                                    torch.Tensor]:
        """clip (B, 3, T, H, W) normalised -> (loss, predictions for hidden tokens, mask)."""
        enc = self.encoder
        B = clip.shape[0]
        if mask is None:
            mask = tube_mask(B, enc.grid[0], enc.grid[1] * enc.grid[2], self.mask_ratio).to(clip.device)
        z = enc.embed(clip) + enc.pos                         # B, N, D (no [CLS])
        n_vis = int((~mask[0]).sum())
        vis = z[~mask].reshape(B, n_vis, -1)
        for blk in enc.blocks:
            vis = blk(vis)
        vis = enc.norm(vis)
        d = self.enc_to_dec(vis)
        full = self.mask_token.expand(B, mask.shape[1], -1).clone()
        full[~mask] = d.reshape(-1, d.shape[-1])
        full = full + self.dec_pos
        for blk in self.decoder:
            full = blk(full)
        pred = self.pred(self.dec_norm(full))[mask].reshape(B, -1, self.pred.out_features)
        tgt = self.targets(clip)[mask].reshape(B, -1, self.pred.out_features)
        loss = ((pred - tgt) ** 2).mean()
        return loss, pred, mask

    @torch.no_grad()
    def reconstruct(self, clip: torch.Tensor, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """For visualisation: (B, N, patch_dim) tubelets with hidden ones replaced
        by predictions (de-normalised with the true patch statistics), and the mask."""
        _, pred, mask = self.forward(clip, mask)
        x = patchify(clip, self.encoder.patch, self.encoder.tubelet)
        mu, sd = x.mean(-1, keepdim=True), (x.var(-1, keepdim=True, unbiased=False) + 1e-6).sqrt()
        out = x.clone()
        out[mask] = (pred.reshape(-1, pred.shape[-1]) * sd[mask] + mu[mask])
        return out, mask


def unpatchify(tokens: torch.Tensor, grid: tuple[int, int, int], p: int, t: int, c: int = 3) -> torch.Tensor:
    """Inverse of patchify for clips: (B, N, c*t*p*p) -> (B, c, T, H, W)."""
    B = tokens.shape[0]
    gt, gh, gw = grid
    x = tokens.reshape(B, gt, gh, gw, c, t, p, p).permute(0, 4, 1, 5, 2, 6, 3, 7)
    return x.reshape(B, c, gt * t, gh * p, gw * p)
