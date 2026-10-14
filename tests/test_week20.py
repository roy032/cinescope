"""Week 20: ViT pieces, VideoMAE tube masking and reconstruction, scene-type scoring."""
from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from cinescope.video.scenes import CLASSES, accuracy, softmax
from cinescope.video.videomae import VideoMAE, tube_mask, unpatchify
from cinescope.video.vit import (
    MultiHeadSelfAttention,
    PatchEmbed,
    ViT,
    patchify,
    reference_attention,
    sincos_embedding,
)


@pytest.fixture(autouse=True)
def _one_thread():
    n = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(n)


def test_patch_embedding_is_a_strided_conv():
    torch.manual_seed(0)
    pe = PatchEmbed(8, 16)
    conv = nn.Conv2d(3, 16, 8, stride=8)
    with torch.no_grad():
        conv.weight.copy_(pe.proj.weight.reshape(16, 3, 8, 8))
        conv.bias.copy_(pe.proj.bias)
    x = torch.randn(2, 3, 32, 48)
    want = conv(x).flatten(2).transpose(1, 2)
    torch.testing.assert_close(pe(x), want, atol=1e-5, rtol=1e-5)


def test_tubelet_embedding_is_a_3d_conv():
    torch.manual_seed(0)
    pe = PatchEmbed(8, 16, tubelet=2)
    conv = nn.Conv3d(3, 16, (2, 8, 8), stride=(2, 8, 8))
    with torch.no_grad():
        conv.weight.copy_(pe.proj.weight.reshape(16, 3, 2, 8, 8))
        conv.bias.copy_(pe.proj.bias)
    x = torch.randn(2, 3, 4, 16, 24)
    torch.testing.assert_close(pe(x), conv(x).flatten(2).transpose(1, 2), atol=1e-5, rtol=1e-5)


def test_patchify_round_trip():
    x = torch.randn(2, 3, 4, 16, 24)
    tok = patchify(x, 8, 2)
    assert tok.shape == (2, 2 * 2 * 3, 3 * 2 * 8 * 8)
    torch.testing.assert_close(unpatchify(tok, (2, 2, 3), 8, 2), x)


def test_attention_matches_pytorch():
    torch.manual_seed(0)
    mhsa = MultiHeadSelfAttention(24, 4)
    x = torch.randn(3, 10, 24)
    torch.testing.assert_close(mhsa(x), reference_attention(x, mhsa), atol=1e-5, rtol=1e-5)
    ref = nn.MultiheadAttention(24, 4, batch_first=True)
    with torch.no_grad():
        ref.in_proj_weight.copy_(mhsa.qkv.weight)
        ref.in_proj_bias.copy_(mhsa.qkv.bias)
        ref.out_proj.weight.copy_(mhsa.proj.weight)
        ref.out_proj.bias.copy_(mhsa.proj.bias)
    torch.testing.assert_close(mhsa(x), ref(x, x, x, need_weights=False)[0], atol=1e-5, rtol=1e-5)
    mhsa(x, keep_attn=True)
    torch.testing.assert_close(mhsa.last_attn.sum(-1), torch.ones(3, 4, 10))


def test_vit_needs_positions_to_see_order():
    """Without position embeddings a ViT is blind to where patches are:
    shuffling the patches leaves the [CLS] output unchanged."""
    torch.manual_seed(0)
    vit = ViT((16, 16), patch=4, dim=32, depth=2, heads=4, num_classes=5).eval()
    x = torch.randn(1, 3, 16, 16)
    tiles = x.unfold(2, 4, 4).unfold(3, 4, 4)                     # 1, 3, 4, 4, 4, 4
    perm = torch.randperm(16)
    t = tiles.reshape(1, 3, 16, 4, 4)[:, :, perm].reshape(1, 3, 4, 4, 4, 4)
    shuffled = t.permute(0, 1, 2, 4, 3, 5).reshape(1, 3, 16, 16)
    assert vit(x).shape == (1, 5)
    assert not torch.allclose(vit(x), vit(shuffled), atol=1e-4)
    with torch.no_grad():
        vit.pos[:, 1:] = 0
    torch.testing.assert_close(vit(x), vit(shuffled), atol=1e-4, rtol=1e-4)


def test_sincos_embedding():
    e = sincos_embedding(10, 16)
    assert e.shape == (10, 16)
    torch.testing.assert_close(e[0], torch.cat([torch.zeros(8), torch.ones(8)]))
    assert torch.all((e.norm(dim=1) - np.sqrt(8)).abs() < 1e-5)


def test_tube_mask_same_in_every_frame():
    g = torch.Generator().manual_seed(0)
    m = tube_mask(3, 2, 20, 0.9, g)
    assert m.shape == (3, 40)
    assert torch.equal(m[:, :20], m[:, 20:])
    assert torch.all(m[:, :20].sum(1) == 18)
    assert not torch.equal(m[0], m[1])


def test_videomae_learns_to_reconstruct():
    torch.manual_seed(0)
    mae = VideoMAE(img_hw=(16, 16), patch=4, frames=4, tubelet=2, dim=32, depth=2, heads=2, dec_dim=32,
                   dec_depth=1, dec_heads=2, mask_ratio=0.75)
    # a smooth pattern that is the same in every frame: hidden tubes are predictable from visible ones
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, 16), torch.linspace(-1, 1, 16), indexing="ij")
    base = torch.stack([torch.sin(3 * xx), torch.cos(2 * yy), xx * yy])
    clip = base[:, None].expand(3, 4, 16, 16)[None].repeat(4, 1, 1, 1, 1).contiguous()
    opt = torch.optim.AdamW(mae.parameters(), 3e-3)
    first = None
    for _ in range(60):
        loss, pred, mask = mae(clip)
        first = first if first is not None else loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert pred.shape == (4, int(mask[0].sum()), 3 * 2 * 4 * 4)
    assert float(loss.detach()) < 0.5 * first
    out, m = mae.reconstruct(clip)
    full = unpatchify(out, mae.encoder.grid, 4, 2)
    assert full.shape == clip.shape


def test_vit_video_encoder_mean_pool():
    vit = ViT((16, 16), patch=4, frames=4, tubelet=2, dim=32, depth=1, heads=2, num_classes=len(CLASSES),
              pool="mean")
    assert vit(torch.randn(2, 3, 4, 16, 16)).shape == (2, len(CLASSES))


def test_scene_accuracy_and_softmax():
    truth = {"0": "text", "1": "dialogue", "2": "none", "3": "dialogue"}
    pred = {"0": "text", "1": "action", "2": "text", "3": "dialogue"}
    r = accuracy(pred, truth)
    assert r["n"] == 3 and r["accuracy"] == pytest.approx(2 / 3)
    assert r["confusion"]["dialogue"] == {"action": 1, "dialogue": 1}
    p = softmax(np.array([[1.0, 2.0, 3.0]]))
    np.testing.assert_allclose(p.sum(1), 1)
