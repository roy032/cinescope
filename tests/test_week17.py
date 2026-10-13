"""Week 17: the Gaussian splatting renderer — exactness, gradients, optimisation."""
import unittest

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from cinescope.geometry.camera import intrinsics

if torch is not None:
    from cinescope.geometry import splat


def small_scene(n=40, seed=0, opacity=0.4):
    rng = np.random.default_rng(seed)
    pts = rng.uniform([-1, -0.8, 3], [1, 0.8, 5], (n, 3))
    cols = rng.random((n, 3))
    g = splat.Gaussians(pts, cols, rng.uniform(0.05, 0.15, n), opacity=opacity)
    with torch.no_grad():
        g.quats.copy_(torch.nn.functional.normalize(torch.randn(n, 4, generator=torch.Generator().manual_seed(seed)),
                                                    dim=1))
        g.log_scales.add_(torch.as_tensor(rng.normal(0, 0.3, (n, 3)), dtype=torch.float32))   # anisotropic
    cam = splat.Camera(intrinsics(60, 24, 20), np.eye(3), np.zeros(3), 48, 40)
    return g, cam


@unittest.skipIf(torch is None, "torch not installed")
class TestSplatting(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_quaternion_rotation(self):
        r = splat.quat_to_rot(torch.nn.functional.normalize(torch.randn(10, 4), dim=1))
        np.testing.assert_allclose((r @ r.transpose(1, 2)).numpy(), np.broadcast_to(np.eye(3), (10, 3, 3)),
                                   atol=1e-5)
        self.assertTrue(np.allclose(torch.linalg.det(r).numpy(), 1, atol=1e-5))

    def test_tiled_equals_dense(self):
        g, cam = small_scene(opacity=0.3)
        with torch.no_grad():
            tiled = splat.render(g, cam, background=(0.2, 0.3, 0.4))
            dense = splat.render_dense_reference(g, cam, background=(0.2, 0.3, 0.4))
            tiled_small_batches = splat.render(g, cam, background=(0.2, 0.3, 0.4), budget=8)
        # identical up to the splats' contribution outside their 3σ ellipse
        self.assertLess((tiled - dense).abs().max().item(), 5e-3)
        self.assertLess((tiled - tiled_small_batches).abs().max().item(), 1e-6)

    def test_depth_order(self):
        """A near opaque splat hides a far one wherever it covers it."""
        g = splat.Gaussians(np.array([[0, 0, 2.0], [0, 0, 4.0]]), np.array([[0.99, 0.01, 0.01], [0.01, 0.01, 0.99]]),
                            np.array([0.3, 0.6]), opacity=0.99)
        cam = splat.Camera(intrinsics(40, 16, 16), np.eye(3), np.zeros(3), 32, 32)
        with torch.no_grad():
            img = splat.render(g, cam)
        self.assertGreater(img[16, 16, 0].item(), 0.9)
        self.assertLess(img[16, 16, 2].item(), 0.1)

    def test_gradients_match_finite_differences(self):
        g, cam = small_scene()
        g = g.double()                          # float64 so finite differences are meaningful
        target = torch.rand(cam.height, cam.width, 3, generator=torch.Generator().manual_seed(1), dtype=torch.float64)

        def loss():
            return ((splat.render(g, cam) - target) ** 2).sum()

        loss().backward()
        for param, i, j in ((g.means, 3, 0), (g.means, 7, 2), (g.color_logit, 5, 1), (g.log_scales, 2, 1),
                            (g.quats, 1, 2), (g.opacity_logit, 4, None)):
            ana = (param.grad[i, j] if j is not None else param.grad[i]).item()
            with torch.no_grad():
                idx = (i, j) if j is not None else (i,)
                h = 1e-6
                param[idx] += h
                lp = loss().item()
                param[idx] -= 2 * h
                lm_ = loss().item()
                param[idx] += h
            num = (lp - lm_) / (2 * h)
            self.assertAlmostEqual(ana, num, delta=1e-4 * abs(num) + 1e-6)

    def test_training_fits_a_view(self):
        g_true, cam = small_scene(seed=3)
        with torch.no_grad():
            target = (splat.render(g_true, cam).numpy() * 255).astype(np.uint8)
        rng = np.random.default_rng(4)
        start = g_true.means.detach().numpy() + rng.normal(0, 0.05, (len(g_true), 3))
        g = splat.Gaussians(start, np.full((len(g_true), 3), 0.5), np.full(len(g_true), 0.1))
        before = splat.psnr(splat.render(g, cam).detach(), target / 255.0)
        cfg = splat.TrainConfig(iters=60, densify_from=10_000, ssim_weight=0.0, lr_color=5e-2)
        g, log = splat.train(g, [cam], [target], cfg, log_every=1000)
        after = splat.psnr(splat.render(g, cam).detach(), target / 255.0)
        self.assertGreater(after, before + 5)

    def test_densify_clones_and_prunes(self):
        g, cam = small_scene(n=20)
        acc = torch.zeros(20)
        acc[:5] = 1.0                           # five "under-reconstructed" Gaussians
        with torch.no_grad():
            g.opacity_logit[-3:] = -10.0        # three invisible ones
        new = splat._densify(g, acc, torch.ones(20), splat.TrainConfig(), extent=100.0, rng=np.random.default_rng(0))
        self.assertEqual(len(new), 20 - 3 + 5)

    def test_point_baseline_zbuffer(self):
        cam = splat.Camera(intrinsics(10, 5, 5), np.eye(3), np.zeros(3), 10, 10)
        pts = np.array([[0, 0, 2.0], [0, 0, 1.0]])
        img = splat.render_points(pts, np.array([[1.0, 0, 0], [0, 1.0, 0]]), cam, radius=0)
        self.assertEqual(img[4, 4].tolist(), [0, 1.0, 0])     # the nearer (green) point wins


if __name__ == "__main__":
    unittest.main()
