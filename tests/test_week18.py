"""Week 18: OKS, keypoint AP, heatmap targets/decoding, the SimpleBaseline network."""
from __future__ import annotations

import contextlib
import copy
import io

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from cinescope.pose import COCO_SIGMAS, FLIP_PAIRS, K
from cinescope.pose.heatmaps import (
    affine_crop,
    decode_heatmaps,
    gaussian_targets,
    transform_points,
)
from cinescope.pose.oks import KeypointEvaluator, oks, oks_matrix
from cinescope.pose.simplebaseline import SimpleBaseline, flip_heatmaps, heatmap_loss


def _random_keypoint_set(rng, n_images=14):
    images, anns, dets = [], [], []
    for i in range(1, n_images + 1):
        images.append({"id": i, "width": 640, "height": 480})
        for _ in range(rng.integers(0, 5)):
            x, y = rng.uniform(0, 400, 2)
            w, h = rng.uniform(5, 200, 2)
            kp = np.stack([x + rng.uniform(0, w, K), y + rng.uniform(0, h, K), rng.integers(0, 3, K)], 1)
            if rng.random() < 0.1:
                kp[:, 2] = 0
            anns.append({"id": len(anns) + 1, "image_id": i, "category_id": 1, "bbox": [x, y, w, h],
                         "area": w * h, "keypoints": kp.ravel().tolist(),
                         "num_keypoints": int((kp[:, 2] > 0).sum()), "iscrowd": int(rng.random() < 0.05)})
            if rng.random() < 0.8:
                dk = kp.copy()
                dk[:, :2] += rng.normal(0, rng.uniform(1, 15), (K, 2))
                dk[:, 2] = 1
                dets.append({"image_id": i, "category_id": 1, "keypoints": dk.ravel().tolist(),
                             "score": float(rng.random())})
        for _ in range(rng.integers(0, 3)):           # false positives
            x, y = rng.uniform(0, 400, 2)
            dk = np.stack([x + rng.uniform(0, 100, K), y + rng.uniform(0, 100, K), np.ones(K)], 1)
            dets.append({"image_id": i, "category_id": 1, "keypoints": dk.ravel().tolist(),
                         "score": float(rng.random())})
    return {"images": images, "annotations": anns, "categories": [{"id": 1, "name": "person"}]}, dets


def test_oks_hand_cases():
    gt = np.array([[100.0, 100.0]] * K)
    vis = np.ones(K)
    assert oks(gt, gt, vis, area=1000) == pytest.approx(1.0)
    # one keypoint off by d: exp(-d^2 / (2 * area * (2 sigma)^2)) for that one, 1 elsewhere
    pred = gt.copy()
    pred[0, 0] += 10
    want = (K - 1 + np.exp(-100 / (2 * 1000 * (2 * COCO_SIGMAS[0]) ** 2))) / K
    assert oks(pred, gt, vis, area=1000) == pytest.approx(want)
    # invisible keypoints do not count, however wrong
    vis2 = vis.copy()
    vis2[0] = 0
    assert oks(pred, gt, vis2, area=1000) == pytest.approx(1.0)
    # the same pixel error matters less on a bigger person
    assert oks(pred, gt, vis, area=10000) > oks(pred, gt, vis, area=1000)


def test_oks_matrix_matches_scalar():
    rng = np.random.default_rng(1)
    gt, dets = _random_keypoint_set(rng, 3)
    gts = [a for a in gt["annotations"] if a["num_keypoints"] > 0][:3]
    m = oks_matrix(dets[:4], gts)
    for i, d in enumerate(dets[:4]):
        for j, g in enumerate(gts):
            gk, dk = np.reshape(g["keypoints"], (-1, 3)), np.reshape(d["keypoints"], (-1, 3))
            assert m[i, j] == pytest.approx(oks(dk[:, :2], gk[:, :2], gk[:, 2], g["area"]))


def test_keypoint_ap_matches_pycocotools():
    pytest.importorskip("pycocotools")
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    rng = np.random.default_rng(0)
    for _ in range(3):
        gt, dets = _random_keypoint_set(rng)
        with contextlib.redirect_stdout(io.StringIO()):
            c = COCO()
            c.dataset = copy.deepcopy(gt)
            c.createIndex()
            e = COCOeval(c, c.loadRes(copy.deepcopy(dets)), "keypoints")
            e.evaluate()
            e.accumulate()
            e.summarize()
        ours = KeypointEvaluator(gt).evaluate(dets)
        got = [ours[k] for k in ("AP", "AP50", "AP75", "AP_medium", "AP_large",
                                 "AR20", "AR_medium", "AR_large")]
        want = [e.stats[i] for i in (0, 1, 2, 3, 4, 5, 8, 9)]
        np.testing.assert_allclose(got, want, atol=1e-10)


def test_gaussian_targets_peak_and_weights():
    pts = np.array([[10.0, 7.0], [3.3, 20.6], [-10.0, 4.0]])
    vis = np.array([1, 1, 1])
    hm, w = gaussian_targets(pts, vis, (32, 24), sigma=2.0)
    assert hm.shape == (3, 32, 24)
    assert hm[0, 7, 10] == pytest.approx(1.0)
    assert np.unravel_index(hm[1].argmax(), hm[1].shape) == (21, 3)
    assert w[2] == 0 and hm[2].max() == 0           # outside the map: no target, no loss


@pytest.mark.parametrize("method", ["quarter", "taylor"])
def test_subpixel_decoding(method):
    rng = np.random.default_rng(3)
    pts = rng.uniform(4, 20, (K, 2))
    hm, _ = gaussian_targets(pts, np.ones(K), (32, 24), sigma=2.0)
    xy, conf = decode_heatmaps(hm[None], method=method)
    err = np.abs(xy[0] - pts).max()
    int_err = np.abs(np.round(pts) - pts).max()
    if method == "taylor":
        assert err < 0.02                            # exact for a Gaussian, up to float error
    else:
        assert err <= int_err + 0.26                 # a quarter pixel towards the higher neighbour
    assert conf.shape == (1, K) and np.all(conf > 0.8)


def test_affine_crop_round_trip():
    img = np.zeros((100, 160, 3), np.uint8)
    box = np.array([40.0, 20.0, 80.0, 90.0])
    crop, M = affine_crop(img, box, (64, 48))
    assert crop.shape == (64, 48, 3)
    pts = np.array([[50.0, 30.0], [70.0, 80.0]])
    back = transform_points(transform_points(pts, M), M, inverse=True)
    np.testing.assert_allclose(back, pts, atol=1e-6)


def test_simplebaseline_shapes_and_flip():
    torch.manual_seed(0)
    net = SimpleBaseline(width=16, deconv=32).eval()
    x = torch.randn(2, 3, 64, 64)
    out = net(x)
    assert out.shape == (2, K, 16, 16)
    # flipping a heatmap twice is the identity, and left/right channels swap
    f = flip_heatmaps(out, FLIP_PAIRS)
    np.testing.assert_allclose(flip_heatmaps(f, FLIP_PAIRS).detach().numpy(), out.detach().numpy())
    np.testing.assert_allclose(f[:, 1].detach().numpy(), out[:, 2].flip(-1).detach().numpy())
    tgt = torch.zeros_like(out)
    wts = torch.ones(2, K)
    loss = heatmap_loss(out, tgt, wts)
    loss.backward()
    assert torch.isfinite(loss)


def test_simplebaseline_learns_a_single_point():
    """Overfit: a bright dot in the crop -> every heatmap peaks on it."""
    threads = torch.get_num_threads()
    torch.set_num_threads(1)                     # tiny tensors: threads only add overhead
    try:
        _overfit_dot()
    finally:
        torch.set_num_threads(threads)


def _overfit_dot():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    net = SimpleBaseline(n_kpts=2, width=8, layers=(1, 1, 1, 1), deconv=16)
    opt = torch.optim.Adam(net.parameters(), 3e-3)
    imgs, tgts = [], []
    for _ in range(12):
        y, x = rng.integers(4, 28), rng.integers(4, 28)
        im = np.zeros((3, 32, 32), np.float32)
        im[:, y - 1:y + 2, x - 1:x + 2] = 1
        hm, _ = gaussian_targets(np.tile([[x / 4, y / 4]], (2, 1)), np.ones(2), (8, 8), sigma=1.0)
        imgs.append(im)
        tgts.append(hm)
    X, T = torch.tensor(np.stack(imgs)), torch.tensor(np.stack(tgts), dtype=torch.float32)
    for _ in range(120):
        opt.zero_grad()
        loss = heatmap_loss(net(X), T, torch.ones(12, 2))
        loss.backward()
        opt.step()
    xy, _ = decode_heatmaps(net(X).detach().numpy(), method="argmax")
    want = np.stack([np.argwhere(t[0] == t[0].max())[0][::-1] for t in T.numpy()])
    assert np.mean(np.abs(xy[:, 0] - want).sum(-1) <= 1) >= 0.75
