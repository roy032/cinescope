"""Week 3: Harris corners, Lucas-Kanade, RANSAC homography, camera moves."""
import unittest

import numpy as np

from cinescope.imgproc.filters import gaussian_blur
from cinescope.motion import harris, homography, lk
from cinescope.motion.camera import classify_shot

try:
    import cv2
except ImportError:
    cv2 = None

RNG = np.random.default_rng(3)


def texture(h=240, w=320, seed=0):
    return gaussian_blur(np.random.default_rng(seed).random((h, w)) * 255, 2.0)


def shift(img, dx, dy):
    """Sub-pixel translation by bilinear resampling (content moves by +dx, +dy)."""
    h, w = img.shape
    ys, xs = np.mgrid[0:h, 0:w].astype(float)
    return lk.bilinear(img, xs - dx, ys - dy)


class TestHarris(unittest.TestCase):
    def test_checkerboard_corners(self):
        board = np.kron((np.indices((6, 6)).sum(0) % 2), np.ones((20, 20))) * 255
        pts = harris.detect_corners(board, max_corners=100, min_distance=5)
        # Interior grid corners sit at multiples of 20; every detection should be near one.
        d = np.abs(pts - np.round(pts / 20) * 20).max(1)
        self.assertGreaterEqual(len(pts), 20)
        self.assertTrue((d <= 2).all())

    def test_flat_and_edge_have_no_corners(self):
        self.assertEqual(len(harris.detect_corners(np.full((50, 50), 7.0))), 0)
        edge = np.zeros((50, 50))
        edge[:, 25:] = 255
        r = harris.corner_response(edge)
        self.assertLessEqual(r.max(), 1e-6 * np.abs(r).max() + 1e-9)   # edges score <= 0


class TestLucasKanade(unittest.TestCase):
    def test_subpixel_translation(self):
        a = texture()
        b = shift(a, 2.4, -1.7)
        pts = harris.detect_corners(a, max_corners=80, border=20)
        new, ok = lk.track(a, b, pts)
        self.assertGreater(ok.mean(), 0.9)
        np.testing.assert_allclose(np.median(new[ok] - pts[ok], 0), [2.4, -1.7], atol=0.05)

    def test_large_motion_needs_the_pyramid(self):
        a = texture()
        b = shift(a, 11.0, 6.0)
        pts = harris.detect_corners(a, max_corners=80, border=30)
        new, ok = lk.track(a, b, pts, levels=4)
        np.testing.assert_allclose(np.median(new[ok] - pts[ok], 0), [11, 6], atol=0.1)
        flat, _ = lk.track(a, b, pts, levels=1)            # single level: too far to converge
        self.assertGreater(np.abs(np.median(flat - pts, 0) - [11, 6]).max(), 1.0)

    @unittest.skipIf(cv2 is None, "opencv not installed")
    def test_matches_opencv(self):
        a = texture()
        b = shift(a, 3.3, 1.2)
        pts = harris.detect_corners(a, max_corners=60, border=20)
        ours, ok = lk.track(a, b, pts)
        ref, st, _ = cv2.calcOpticalFlowPyrLK(a.astype(np.uint8), b.astype(np.uint8),
                                              pts.astype(np.float32).reshape(-1, 1, 2), None)
        both = ok & (st.ravel() == 1)
        self.assertLess(np.median(np.abs(ours[both] - ref.reshape(-1, 2)[both])), 0.1)


class TestHomography(unittest.TestCase):
    H = np.array([[1.03, 0.02, 4.0], [-0.015, 0.98, -6.0], [2e-5, -1e-5, 1.0]])

    def test_dlt_exact(self):
        src = RNG.random((10, 2)) * 400
        h = homography.dlt(src, homography.apply(self.H, src))
        np.testing.assert_allclose(h, self.H, atol=1e-8)

    def test_ransac_with_40_percent_outliers(self):
        src = RNG.random((200, 2)) * 400
        dst = homography.apply(self.H, src) + RNG.normal(0, 0.3, (200, 2))
        dst[:80] = RNG.random((80, 2)) * 400
        res = homography.ransac_homography(src, dst, thresh=1.5)
        self.assertGreater(res.inliers[80:].mean(), 0.95)
        self.assertLess(res.inliers[:80].mean(), 0.05)
        err = homography.reprojection_error(res.h, src[80:], homography.apply(self.H, src[80:]))
        self.assertLess(np.median(err), 0.3)

    def test_ransac_iterations_adapt(self):
        src = RNG.random((100, 2)) * 400
        res = homography.ransac_homography(src, homography.apply(self.H, src))
        self.assertLess(res.iterations, 20)                 # w = 1 -> stops almost at once


def synthetic_shot(kind, n=28, seed=5):
    big = texture(420, 560, seed)
    frames = []
    for i in range(n):
        if kind == "static":
            f = big[100:220, 100:260]
        elif kind == "pan":                                # window slides right: camera pans right
            f = big[100:220, 100 + 2 * i:260 + 2 * i]
        elif kind == "tilt":
            f = big[100 + 2 * i:220 + 2 * i, 100:260]
        elif kind == "zoom":                               # window shrinks around the centre
            m = 30 * i / n
            y0, y1, x0, x1 = 100 + m * 0.75, 220 - m * 0.75, 100 + m, 260 - m
            ys = np.linspace(y0, y1, 120)
            xs = np.linspace(x0, x1, 160)
            f = lk.bilinear(big, xs[None, :].repeat(120, 0), ys[:, None].repeat(160, 1))
        else:                                              # handheld: random shake
            dx, dy = RNG.normal(0, 3, 2)
            f = lk.bilinear(big, np.arange(160)[None, :] + 100 + dx + 0 * np.arange(120)[:, None],
                            np.arange(120)[:, None] + 100 + dy + 0 * np.arange(160)[None, :])
        frames.append(np.asarray(f, float))
    return np.stack(frames)


class TestCameraMoves(unittest.TestCase):
    def test_labels(self):
        expected = {"static": ("static", ""), "pan": ("pan", "right"), "tilt": ("tilt", "down"),
                    "zoom": ("zoom", "in"), "handheld": ("handheld", "")}
        for kind, (label, direction) in expected.items():
            m = classify_shot(synthetic_shot(kind), fps=24)
            self.assertEqual((m.label, m.direction), (label, direction), f"{kind}: {m.to_dict()}")


if __name__ == "__main__":
    unittest.main()
