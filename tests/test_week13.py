"""Week 13: closed-form matting, foreground estimation, Criminisi inpainting,
PSNR/SSIM and temporal background reconstruction."""
import numpy as np
import pytest

from cinescope.segment.inpaint import criminisi, psnr, ssim
from cinescope.segment.matting import (
    closed_form_alpha,
    estimate_foreground,
    matting_laplacian,
    trimap_from_mask,
)
from cinescope.segment.video_remove import background_plate, camera_is_static, remove_object


def _matting_scene(h=40, w=48):
    """Two flat-ish colours mixed by a known alpha ramp across the middle."""
    rng = np.random.default_rng(0)
    xx = np.tile(np.arange(w, dtype=np.float64), (h, 1))
    alpha = np.clip((xx - 18) / 12, 0, 1)
    fg = np.array([0.9, 0.3, 0.2]) + rng.normal(0, 0.01, (h, w, 3))
    bg = np.array([0.1, 0.4, 0.8]) + rng.normal(0, 0.01, (h, w, 3))
    img = alpha[..., None] * fg + (1 - alpha[..., None]) * bg
    return img, alpha, fg


def test_laplacian_is_symmetric_with_zero_row_sums():
    img, _, _ = _matting_scene(12, 14)
    lap = matting_laplacian(img)
    assert abs(lap - lap.T).max() < 1e-9
    assert np.allclose(np.asarray(lap.sum(1)).ravel(), 0, atol=1e-8)
    # constant alpha costs nothing; the Laplacian is positive semi-definite
    v = np.random.default_rng(1).normal(size=lap.shape[0])
    assert v @ (lap @ v) >= -1e-8
    assert abs(np.ones(lap.shape[0]) @ (lap @ np.ones(lap.shape[0]))) < 1e-8


def test_laplacian_matches_pymatting():
    pymatting = pytest.importorskip("pymatting")
    img, _, _ = _matting_scene(12, 14)
    ref = pymatting.cf_laplacian(img, epsilon=1e-7, radius=1)
    assert abs(matting_laplacian(img) - ref).max() < 1e-5 * abs(ref).max()


def test_closed_form_recovers_alpha_ramp():
    img, alpha, fg = _matting_scene()
    tri = np.where(alpha <= 0, 0.0, np.where(alpha >= 1, 1.0, 0.5))
    tri[:, 14:34] = 0.5                         # widen the unknown band
    est = closed_form_alpha(img, tri)
    unk = tri == 0.5
    assert np.abs(est - alpha)[unk].mean() < 0.03
    f = estimate_foreground(img, est)
    sel = est > 0.3
    assert np.abs(f[sel] - fg[sel]).mean() < 0.08


def test_trimap_from_mask():
    m = np.zeros((40, 40), bool)
    m[10:30, 10:30] = True
    tri = trimap_from_mask(m, radius=3)
    assert tri[20, 20] == 1 and tri[0, 0] == 0 and tri[10, 20] == 0.5
    assert set(np.unique(tri)) == {0.0, 0.5, 1.0}


def test_psnr_ssim_basics():
    rng = np.random.default_rng(2)
    a = rng.integers(0, 255, (32, 32, 3)).astype(np.uint8)
    assert psnr(a, a) == float("inf")
    assert ssim(a, a) == pytest.approx(1.0)
    b = np.clip(a.astype(int) + 10, 0, 255).astype(np.uint8)
    assert 25 < psnr(a, b) < 30
    noisy = np.clip(a + rng.normal(0, 40, a.shape), 0, 255).astype(np.uint8)
    assert ssim(a, noisy) < ssim(a, b) < 1.0


def test_criminisi_continues_stripes():
    h, w = 48, 48
    stripes = np.zeros((h, w, 3), np.uint8)
    stripes[:, (np.arange(w) // 4) % 2 == 0] = 220
    hole = np.zeros((h, w), bool)
    hole[18:30, 18:30] = True
    damaged = stripes.copy()
    damaged[hole] = 0
    out = criminisi(damaged, hole, patch=7, search=20)
    assert np.array_equal(out[~hole], stripes[~hole])
    assert psnr(out, stripes, hole) > 25           # periodic texture is copied, not smeared


def test_criminisi_beats_diffusion_on_texture():
    cv2 = pytest.importorskip("cv2")  # noqa: F841
    from cinescope.segment.inpaint import opencv_inpaint

    rng = np.random.default_rng(3)
    tile = rng.integers(0, 255, (6, 6, 3)).astype(np.uint8)
    tex = np.tile(tile, (9, 9, 1))
    hole = np.zeros(tex.shape[:2], bool)
    hole[20:32, 22:34] = True
    a = criminisi(tex, hole, patch=7, search=20)
    b = opencv_inpaint(tex, hole, "telea")
    assert psnr(a, tex, hole) > psnr(b, tex, hole) + 5


def _walker(n=12, h=40, w=60):
    rng = np.random.default_rng(4)
    bg = rng.integers(0, 255, (h, w, 3)).astype(np.uint8)
    frames, masks = [], []
    for t in range(n):
        f = bg.copy()
        m = np.zeros((h, w), bool)
        x = 4 + 4 * t
        m[10:30, x:x + 6] = True
        f[m] = (255, 0, 0)
        frames.append(f)
        masks.append(m)
    return bg, frames, masks


def test_plate_recovers_background_and_removal_is_exact():
    bg, frames, masks = _walker()
    plate, seen = background_plate(frames, masks)
    assert seen.all() and np.array_equal(plate, bg)
    assert camera_is_static(frames, masks)[0]
    out, rep = remove_object(frames, masks)
    assert rep["mode"] == "temporal plate" and rep["never_seen_px"] == 0
    for o, m in zip(out, masks, strict=True):
        assert np.array_equal(o[m], bg[m])


def test_moving_camera_is_detected():
    rng = np.random.default_rng(5)
    big = rng.integers(0, 255, (60, 120, 3)).astype(np.uint8)
    big = np.asarray(np.repeat(np.repeat(big, 2, 0), 2, 1))          # smoother texture
    frames = [big[20:100, 4 * t:4 * t + 120] for t in range(8)]
    masks = [np.zeros(frames[0].shape[:2], bool) for _ in frames]
    static, shift = camera_is_static(frames, masks)
    assert not static and shift > 10
