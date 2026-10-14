"""Week 12: max-flow / min-cut, GMMs, SLIC, GrabCut, the U-Net and composites."""
import numpy as np
import pytest

from cinescope.segment.data import (
    Cutout,
    composite,
    crop_padded,
    mask_box,
    mask_iou,
    paste_window,
)
from cinescope.segment.gmm import GMM
from cinescope.segment.grabcut import grabcut
from cinescope.segment.maxflow import Graph, grid_edges
from cinescope.segment.superpixels import adjacency, slic


def _random_graph(rng):
    n = int(rng.integers(2, 25))
    m = int(rng.integers(1, 60))
    i, j = rng.integers(0, n, m), rng.integers(0, n, m)
    keep = i != j
    i, j = i[keep], j[keep]
    c, r = rng.integers(0, 10, len(i)), rng.integers(0, 10, len(i))
    cs = rng.integers(0, 10, n) * (rng.random(n) < 0.4)
    ct = rng.integers(0, 10, n) * (rng.random(n) < 0.4)
    return n, i, j, c, r, cs, ct


def test_maxflow_matches_scipy_and_cut_equals_flow():
    sp = pytest.importorskip("scipy.sparse")
    from scipy.sparse.csgraph import maximum_flow

    rng = np.random.default_rng(0)
    for _ in range(150):
        n, i, j, c, r, cs, ct = _random_graph(rng)
        g = Graph(n)
        g.add_edges(i, j, c, r)
        g.add_tweights(np.arange(n), cs, ct)
        flow = g.maxflow()
        cap = np.zeros((n + 2, n + 2), np.int64)        # node n = source, n+1 = sink
        np.add.at(cap, (i, j), c)
        np.add.at(cap, (j, i), r)
        cap[n, :n] += cs
        cap[:n, n + 1] += ct
        ref = maximum_flow(sp.csr_matrix(cap), n, n + 1).flow_value
        assert flow == pytest.approx(ref)
        side = np.r_[g.source_side(), True, False]
        assert cap[side][:, ~side].sum() == pytest.approx(ref)   # min cut = max flow


def test_maxflow_hand_example():
    # s -> 0 (3), s -> 1 (2), 0 -> 1 (1), 0 -> t (2), 1 -> t (3): max flow 5
    g = Graph(2)
    g.add_edges([0], [1], [1], [0])
    g.add_tweights([0, 1], [3, 2], [2, 3])
    assert g.maxflow() == pytest.approx(5.0)


def test_graph_cut_denoises_binary_image():
    rng = np.random.default_rng(1)
    h = w = 40
    truth = np.zeros((h, w), bool)
    truth[10:30, 12:28] = True
    noisy = truth ^ (rng.random((h, w)) < 0.2)
    i, j = grid_edges(h, w)
    g = Graph(h * w)
    g.add_edges(i, j, 0.8)
    obs = noisy.ravel()
    g.add_tweights(np.arange(h * w), obs * 1.0, (~obs) * 1.0)
    g.maxflow()
    out = g.source_side().reshape(h, w)
    assert (out != truth).mean() < 0.02 < (noisy != truth).mean()


def test_gmm_single_component_matches_gaussian():
    stats = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(2)
    cov = np.array([[4.0, 1.0, 0.0], [1.0, 3.0, 0.5], [0.0, 0.5, 2.0]])
    x = rng.multivariate_normal([10, 20, 30], cov, 5000)
    g = GMM.fit(x, k=1, reg=0.0)
    ref = stats.multivariate_normal(x.mean(0), np.cov(x, rowvar=False, bias=True)).logpdf(x[:20])
    assert np.allclose(g.loglik(x[:20]), ref)


def test_gmm_separates_two_colours():
    rng = np.random.default_rng(3)
    x = np.r_[rng.normal(50, 3, (300, 3)), rng.normal(200, 3, (300, 3))]
    g = GMM.fit(x, k=2)
    assert sorted(np.round(g.means[:, 0] / 10)) == [5, 20]


def _scene(rng, h=80, w=80):
    img = rng.normal(90, 12, (h, w, 3))
    img[..., 2] += 40                                    # bluish textured background
    yy, xx = np.mgrid[0:h, 0:w]
    disk = (yy - 40) ** 2 / 22 ** 2 + (xx - 40) ** 2 / 14 ** 2 <= 1
    img[disk] = rng.normal([200, 120, 60], 10, (disk.sum(), 3))   # orange object
    return np.clip(img, 0, 255).astype(np.uint8), disk


def test_slic_regions_are_connected_and_cover_image():
    from scipy import ndimage as ndi

    rng = np.random.default_rng(4)
    img, _ = _scene(rng)
    lab = slic(img, n_segments=60)
    assert lab.shape == img.shape[:2]
    assert 20 <= lab.max() + 1 <= 120
    for k in range(lab.max() + 1):
        _, n = ndi.label(lab == k)
        assert n == 1
    a, b = adjacency(lab)
    assert len(a) > 0 and np.all(a < b)


def test_grabcut_finds_object_in_box():
    rng = np.random.default_rng(5)
    img, disk = _scene(rng)
    m = grabcut(img, mask_box(disk) + np.array([-3, -3, 3, 3]))
    assert mask_iou(m, disk) > 0.9


def test_grabcut_close_to_opencv():
    pytest.importorskip("cv2")
    from cinescope.segment.grabcut import grabcut_opencv

    rng = np.random.default_rng(6)
    img, disk = _scene(rng)
    box = mask_box(disk) + np.array([-3, -3, 3, 3])
    assert mask_iou(grabcut(img, box), grabcut_opencv(img, box)) > 0.85


def test_composite_mask_is_exact_and_crop_roundtrip():
    rng = np.random.default_rng(7)
    bg = rng.integers(0, 255, (50, 60, 3), dtype=np.uint8)
    cm = np.zeros((10, 6), bool)
    cm[2:9, 1:5] = True
    cut = Cutout(np.full((10, 6, 3), 7, np.uint8), cm, "x")
    img, mask = composite(bg, cut, (20, 30))
    assert mask.sum() == cm.sum()
    assert np.all(img[mask] == 7) and np.array_equal(img[~mask], bg[~mask])
    assert np.array_equal(mask_box(mask), [21, 32, 25, 39])
    win = (-5, -4, 30, 20)                           # leaves the image: zero padding
    c = crop_padded(bg, win)
    assert c.shape == (24, 35, 3) and np.all(c[:4] == 0)
    back = np.zeros_like(bg)
    paste_window(back, c, win)
    assert np.array_equal(back[:20, :30], bg[:20, :30])


def test_unet_shapes_and_overfits_one_batch():
    torch = pytest.importorskip("torch")
    from cinescope.segment.unet import UNet, box_crop, paste_mask, seg_loss

    torch.manual_seed(0)
    net = UNet(base=8, depth=3)
    x = torch.rand(4, 3, 32, 32)
    y = torch.zeros(4, 1, 32, 32)
    y[:, :, 8:24, 10:22] = 1
    x[:, 0] += y[:, 0]                       # the object is redder
    assert net(x).shape == (4, 1, 32, 32)
    opt = torch.optim.Adam(net.parameters(), 1e-2)
    first = None
    for _ in range(40):
        loss = seg_loss(net(x), y)
        first = first or loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.5 * first
    img = np.zeros((60, 80, 3), np.uint8)
    crop, bounds = box_crop(img, [10, 20, 30, 50], size=32)
    assert crop.shape == (32, 32, 3)
    m = paste_mask(np.ones((32, 32)), bounds, img.shape)
    assert m.shape == (60, 80) and m[20:50, 10:30].all()
