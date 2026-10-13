"""Week 16: plane-sweep stereo, depth metrics, depth-image-based rendering, MiDaS wiring."""
import unittest

import numpy as np

from cinescope.geometry import dibr, metrics, mvs
from cinescope.geometry.camera import intrinsics, look_at
from cinescope.geometry.synthetic import Room, render

RNG = np.random.default_rng(16)


class TestMVSPieces(unittest.TestCase):
    def test_box_mean_matches_brute_force(self):
        img = RNG.random((13, 17))
        r = 2
        ref = np.zeros_like(img)
        for y in range(13):
            for x in range(17):
                ref[y, x] = img[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1].mean()
        np.testing.assert_allclose(mvs.box_mean(img, r), ref, atol=1e-12)

    def test_zncc_is_invariant_to_gain_and_offset(self):
        a = RNG.random((20, 20)) * 100
        z = mvs.zncc(a, 2.5 * a + 40, 2)
        self.assertGreater(z[3:-3, 3:-3].min(), 0.999)

    def test_plane_homography_maps_plane_points(self):
        k = intrinsics(100, 40, 30)
        rot, t = look_at([0.5, 0.1, 0.0], [0, 0, 5.0])
        x = np.c_[RNG.uniform(-1, 1, (10, 2)), np.full(10, 4.0)]          # on the plane z = 4 (ref camera = world)
        hm = mvs.plane_homography(k, k, rot, t, 4.0)
        u_ref = x @ k.T
        u_ref = u_ref[:, :2] / u_ref[:, 2:]
        u_src = (x @ rot.T + t) @ k.T
        u_src = u_src[:, :2] / u_src[:, 2:]
        q = np.c_[u_ref, np.ones(10)] @ hm.T
        np.testing.assert_allclose(q[:, :2] / q[:, 2:], u_src, atol=1e-9)


class TestPlaneSweep(unittest.TestCase):
    def test_recovers_room_depth(self):
        room = Room()
        w, h = 96, 72
        k = intrinsics(0.9 * w, w / 2, h / 2)
        poses = [look_at([x, -0.2, -0.5], [0.0, 0.2, 6.0]) for x in (-0.5, -0.25, 0.0, 0.25, 0.5)]
        imgs, depths = zip(*(render(room, k, r, t, w, h) for r, t in poses), strict=True)
        gray = [im.astype(float).mean(-1) for im in imgs]
        res = mvs.plane_sweep(gray[2], [gray[i] for i in (0, 1, 3, 4)], k, poses[2],
                              [poses[i] for i in (0, 1, 3, 4)], near=2.5, far=15, n_planes=40, window=2)
        m = res.confident
        self.assertGreater(m.mean(), 0.6)
        rel = np.abs(res.depth[m] - depths[2][m]) / depths[2][m]
        self.assertLess(np.median(rel), 0.05)


class TestDepthMetrics(unittest.TestCase):
    def test_scale_invariance(self):
        gt = RNG.uniform(1, 10, 500)
        pred = gt * np.exp(RNG.normal(0, 0.1, 500))
        self.assertAlmostEqual(metrics.scale_invariant_log_rmse(pred, gt),
                               metrics.scale_invariant_log_rmse(7.3 * pred, gt), places=10)
        self.assertAlmostEqual(metrics.abs_rel(3 * gt, gt), 0.0, places=10)
        self.assertEqual(metrics.delta_accuracy(0.5 * gt, gt), 1.0)

    def test_affine_inverse_depth_fit(self):
        depth = RNG.uniform(2, 20, 300)
        inv_pred = (1 / depth - 0.01) / 0.02          # what an affine-invariant network might output
        a, b = metrics.fit_inverse_depth_affine(inv_pred, depth)
        np.testing.assert_allclose(a * inv_pred + b, 1 / depth, atol=1e-12)
        self.assertAlmostEqual(metrics.spearman(inv_pred, -depth), 1.0)


class TestDIBR(unittest.TestCase):
    def test_constant_disparity_is_a_shift(self):
        img = RNG.integers(0, 255, (10, 30, 3)).astype(np.uint8)
        disp = np.full((10, 30), 4.0)
        view, holes = dibr.render_view(img, disp, -1.0)
        np.testing.assert_array_equal(view[:, :26], img[:, 4:])
        self.assertTrue(holes[:, 26:].all() and not holes[:, :26].any())

    def test_near_wins_and_holes_take_the_background(self):
        img = np.zeros((1, 20, 3), np.uint8)
        img[0, :, 0] = 50                      # background
        img[0, 8:12, 0] = 200                  # a near object
        disp = np.zeros((1, 20))
        disp[0, 8:12] = 4.0
        view, holes = dibr.render_view(img, disp, 1.0)                 # near object moves right by 4
        self.assertTrue((view[0, 12:16, 0] == 200).all())              # it covers the background there
        self.assertTrue((view[0, 8:12, 0] == 50).all())                # the uncovered strip is background
        fg, _ = dibr.render_view(img, disp, 1.0, side="foreground")
        self.assertTrue((fg[0, 8:12, 0] == 200).all())                 # the wrong choice smears the object

    def test_disparity_range_and_anaglyph(self):
        inv = np.linspace(0, 1, 100).reshape(10, 10)
        d = dibr.disparity_from_inverse_depth(inv, 200, max_disp_pct=2.5, convergence=0.25)
        self.assertAlmostEqual(d.max(), 5.0, places=6)
        self.assertLess(d.min(), 0)                                    # behind the screen plane
        left = np.full((2, 2, 3), 10, np.uint8)
        right = np.full((2, 2, 3), 99, np.uint8)
        a = dibr.anaglyph(left, right)
        self.assertEqual(a[0, 0].tolist(), [10, 99, 99])
        self.assertEqual(dibr.side_by_side(left, right).shape, (2, 2, 3))


class TestMiDaSWiring(unittest.TestCase):
    def test_architecture_builds_and_runs(self):
        try:
            import timm  # noqa: F401
            import torch
        except ImportError:
            self.skipTest("timm/torch not installed")
        from cinescope.geometry.midas import _build

        torch.manual_seed(0)
        net = _build().eval()
        names = set(net.state_dict())
        self.assertIn("pretrained.layer1.3.0.conv_dw.weight", names)    # official checkpoint key names
        self.assertIn("scratch.output_conv.4.weight", names)
        with torch.no_grad():
            out = net(torch.rand(1, 3, 64, 96))
        self.assertEqual(tuple(out.shape), (1, 64, 96))


if __name__ == "__main__":
    unittest.main()
