"""Week 14: rotations, Levenberg-Marquardt, Zhang calibration, self-calibration."""
import unittest

import numpy as np

from cinescope.geometry import camera, lm
from cinescope.geometry.calib import calibrate, chessboard_points
from cinescope.geometry.selfcalib import (
    estimate_focal,
    focal_from_vps,
    focal_prior,
    ransac_vanishing_points,
)

try:
    import cv2
except ImportError:
    cv2 = None

RNG = np.random.default_rng(14)


def synthetic_board_views(k, dist, n_views=8, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    obj = chessboard_points(9, 6, square=1.0)
    obj3 = np.c_[obj - obj.mean(0), np.zeros(len(obj))]
    views = []
    for _ in range(n_views):
        r = camera.rodrigues(rng.uniform(-0.5, 0.5, 3))
        t = np.array([rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(12, 18)])
        img = camera.project(obj3, k, r, t, dist) + rng.normal(0, noise, (len(obj), 2))
        views.append(img)
    return [obj - obj.mean(0)] * n_views, views


class TestRotations(unittest.TestCase):
    def test_exp_log_roundtrip(self):
        for _ in range(20):
            r = RNG.normal(size=3)
            r *= RNG.uniform(0.01, 3.0) / np.linalg.norm(r)
            rot = camera.rodrigues(r)
            np.testing.assert_allclose(rot @ rot.T, np.eye(3), atol=1e-12)
            np.testing.assert_allclose(camera.log_rotation(rot), r, atol=1e-9)
        np.testing.assert_allclose(camera.rodrigues_batch(np.eye(3)), [camera.rodrigues(v) for v in np.eye(3)])

    def test_right_jacobian(self):
        r, d, v = RNG.normal(size=3), RNG.normal(size=3) * 1e-6, RNG.normal(size=3)
        lhs = camera.rodrigues(r + d) @ v
        rhs = camera.rodrigues(r) @ camera.rodrigues(camera.right_jacobian(r) @ d) @ v
        np.testing.assert_allclose(lhs, rhs, atol=1e-11)

    def test_distortion_inverse(self):
        xy = RNG.uniform(-0.5, 0.5, (50, 2))
        dist = (-0.28, 0.08)
        np.testing.assert_allclose(camera.undistort(camera.distort(xy, dist), dist), xy, atol=1e-9)

    def test_look_at_identity(self):
        rot, t = camera.look_at([0, 0, 0], [0, 0, 5])
        np.testing.assert_allclose(rot, np.eye(3), atol=1e-12)


class TestLevenbergMarquardt(unittest.TestCase):
    def test_rosenbrock(self):
        res = lm.levenberg_marquardt(lambda x: np.array([10 * (x[1] - x[0] ** 2), 1 - x[0]]), np.array([-1.2, 1.0]))
        np.testing.assert_allclose(res.x, [1, 1], atol=1e-6)

    def test_curve_fit(self):
        t = np.linspace(0, 3, 40)
        y = 2.5 * np.exp(-1.3 * t) + 0.4
        res = lm.levenberg_marquardt(lambda p: p[0] * np.exp(-p[1] * t) + p[2] - y, np.array([1.0, 0.5, 0.0]))
        np.testing.assert_allclose(res.x, [2.5, 1.3, 0.4], atol=1e-6)


class TestZhang(unittest.TestCase):
    K = np.array([[800.0, 0, 330], [0, 805.0, 235], [0, 0, 1]])
    DIST = np.array([-0.2, 0.05])

    def test_exact_without_noise(self):
        obj, img = synthetic_board_views(self.K, self.DIST)
        cal = calibrate(obj, img, (640, 480))
        np.testing.assert_allclose(cal.k, self.K, rtol=1e-5, atol=1e-3)
        np.testing.assert_allclose(cal.dist, self.DIST, atol=1e-5)
        self.assertLess(cal.rms, 1e-4)

    def test_closed_form_is_close_and_lm_fits_the_noise(self):
        obj, img = synthetic_board_views(self.K, self.DIST, noise=0.3, n_views=12, seed=1)
        cal = calibrate(obj, img, (640, 480))
        self.assertLess(abs(cal.closed_form_k[0, 0] - 800) / 800, 0.1)       # distortion ignored, still close
        self.assertLess(abs(cal.k[0, 0] - 800) / 800, 0.01)
        self.assertAlmostEqual(cal.rms, 0.3 * np.sqrt(2), delta=0.06)     # residual ≈ the pixel noise (2-D)

    @unittest.skipIf(cv2 is None, "OpenCV not installed")
    def test_matches_opencv(self):
        obj, img = synthetic_board_views(self.K, self.DIST, noise=0.2, n_views=10, seed=2)
        cal = calibrate(obj, img, (640, 480))
        o3 = [np.c_[o, np.zeros(len(o))].astype(np.float32) for o in obj]
        rms, k, d, _, _ = cv2.calibrateCamera(o3, [i.astype(np.float32) for i in img], (640, 480), None, None,
                                              flags=cv2.CALIB_FIX_K3 | cv2.CALIB_ZERO_TANGENT_DIST)
        np.testing.assert_allclose(cal.k, k, rtol=1e-3)
        np.testing.assert_allclose(cal.dist, d.ravel()[:2], atol=1e-3)
        self.assertAlmostEqual(cal.rms, rms, delta=1e-3)


class TestSelfCalibration(unittest.TestCase):
    def test_prior(self):
        self.assertEqual(focal_prior(1280, 544), 1.2 * 1280)

    def _box_segments(self, f, w=640, h=480):
        """Segments of a grid of 3D lines along three orthogonal directions, seen
        by a camera rotated so that all three vanishing points are finite."""
        k = camera.intrinsics(f, w / 2, h / 2)
        rot = camera.rodrigues(np.array([0.35, 0.6, 0.1]))
        t = np.array([0, 0, 12.0])
        segs = []
        for axis in range(3):
            for a in np.linspace(-2, 2, 4):
                for b in np.linspace(-2, 2, 4):
                    p = np.zeros((2, 3))
                    others = [i for i in range(3) if i != axis]
                    p[:, others[0]], p[:, others[1]] = a, b
                    p[:, axis] = [-1.5, 1.5]
                    uv = camera.project(p, k, rot, t)
                    segs.append(uv.ravel())
        return np.asarray(segs) + RNG.normal(0, 0.2, (len(segs), 4))

    def test_vanishing_points_give_the_focal_length(self):
        segs = self._box_segments(700.0)
        vps = ransac_vanishing_points(segs)
        self.assertEqual(len(vps), 3)
        self.assertEqual(sorted(s for _, s in vps), [16, 16, 16])
        est = estimate_focal(segs, 640, 480)
        self.assertEqual(est.method, "vanishing-points")
        self.assertLess(abs(est.f - 700) / 700, 0.03)

    def test_focal_from_two_vps(self):
        k = camera.intrinsics(500, 320, 240)
        rot = camera.rodrigues(np.array([0.2, 0.7, 0.0]))
        v1, v2 = k @ rot @ np.array([1.0, 0, 0]), k @ rot @ np.array([0, 0, 1.0])
        self.assertAlmostEqual(focal_from_vps(v1, v2, (320, 240)), 500, places=6)

    def test_no_lines_falls_back_to_prior(self):
        est = estimate_focal(np.zeros((0, 4)), 640, 480)
        self.assertEqual(est.method, "prior")
        self.assertEqual(est.f, focal_prior(640, 480))


if __name__ == "__main__":
    unittest.main()
