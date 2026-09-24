"""Week 1: filters, color spaces and video I/O, each checked against a library.

Runs with `python -m pytest` or `python -m unittest discover -s tests`.
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from cinescope.imgproc import color, filters
from cinescope.io.video import VideoReader, write_video

try:
    import cv2
except ImportError:          # the library comparisons are skipped without OpenCV
    cv2 = None

RNG = np.random.default_rng(0)


def img(h=37, w=53, c=None):
    shape = (h, w) if c is None else (h, w, c)
    return RNG.integers(0, 256, size=shape).astype(np.uint8)


class TestConvolution(unittest.TestCase):
    def test_identity_and_shift(self):
        x = img().astype(float)
        ident = np.zeros((3, 3))
        ident[1, 1] = 1
        np.testing.assert_allclose(filters.convolve2d(x, ident), x)
        shift = np.zeros((3, 3))
        shift[1, 2] = 1          # correlation with this reads the right neighbour
        np.testing.assert_allclose(filters.correlate2d(x, shift, border="edge")[:, :-1], x[:, 1:])
        # convolution flips the kernel, so it reads the LEFT neighbour
        np.testing.assert_allclose(filters.convolve2d(x, shift, border="edge")[:, 1:], x[:, :-1])

    def test_matches_scipy(self):
        from scipy import ndimage

        x = img().astype(float)
        k = RNG.normal(size=(5, 3))
        ours = filters.convolve2d(x, k, border="reflect")
        ref = ndimage.convolve(x, k, mode="mirror")      # mirror == reflect-101
        np.testing.assert_allclose(ours, ref, atol=1e-9)

    @unittest.skipIf(cv2 is None, "opencv not installed")
    def test_matches_opencv_filter2d(self):
        x = img().astype(np.float64)
        k = RNG.normal(size=(3, 5))
        ref = cv2.filter2D(x, -1, k, borderType=cv2.BORDER_REFLECT_101)   # correlation
        np.testing.assert_allclose(filters.correlate2d(x, k), ref, atol=1e-9)

    def test_rejects_even_kernel(self):
        with self.assertRaises(ValueError):
            filters.convolve2d(img(), np.ones((2, 2)))


class TestGaussianSobel(unittest.TestCase):
    def test_kernel_normalised_and_separable(self):
        k = filters.gaussian_kernel1d(1.5)
        self.assertAlmostEqual(k.sum(), 1.0)
        self.assertEqual(len(k), 2 * 5 + 1)
        x = img(20, 24).astype(float)
        full = filters.correlate2d(x, filters.gaussian_kernel2d(1.5))
        np.testing.assert_allclose(filters.gaussian_blur(x, 1.5), full, atol=1e-9)

    @unittest.skipIf(cv2 is None, "opencv not installed")
    def test_gaussian_matches_opencv(self):
        x = img(40, 60).astype(np.float64)
        sigma = 2.0
        size = 2 * int(np.ceil(3 * sigma)) + 1
        ref = cv2.GaussianBlur(x, (size, size), sigma, borderType=cv2.BORDER_REFLECT_101)
        np.testing.assert_allclose(filters.gaussian_blur(x, sigma), ref, atol=1e-6)

    @unittest.skipIf(cv2 is None, "opencv not installed")
    def test_sobel_matches_opencv(self):
        x = img(30, 40).astype(np.float64)
        gx, gy = filters.sobel(x)
        np.testing.assert_allclose(gx, cv2.Sobel(x, cv2.CV_64F, 1, 0, ksize=3), atol=1e-9)
        np.testing.assert_allclose(gy, cv2.Sobel(x, cv2.CV_64F, 0, 1, ksize=3), atol=1e-9)

    def test_sobel_orientation_on_a_ramp(self):
        ramp = np.tile(np.arange(20, dtype=float), (10, 1))     # brighter to the right
        gx, gy = filters.sobel(ramp)
        self.assertTrue(np.all(gx[:, 2:-2] > 0))
        np.testing.assert_allclose(gy[2:-2, 2:-2], 0)


class TestColor(unittest.TestCase):
    def test_lab_known_values(self):
        lab = color.rgb_to_lab(np.array([[255, 255, 255], [0, 0, 0], [255, 0, 0]], np.uint8))
        np.testing.assert_allclose(lab[0], [100, 0, 0], atol=0.01)
        np.testing.assert_allclose(lab[1], [0, 0, 0], atol=0.01)
        np.testing.assert_allclose(lab[2], [53.24, 80.09, 67.20], atol=0.05)   # sRGB red

    def test_lab_roundtrip(self):
        x = img(10, 10, 3)
        back = color.lab_to_rgb(color.rgb_to_lab(x)) * 255
        np.testing.assert_allclose(back, x, atol=0.5)

    @unittest.skipIf(cv2 is None, "opencv not installed")
    def test_lab_matches_opencv(self):
        x = img(16, 16, 3).astype(np.float32) / 255
        ref = cv2.cvtColor(x, cv2.COLOR_RGB2Lab)        # float input -> L 0..100
        # OpenCV approximates the sRGB gamma curve with a spline, so it differs
        # from the exact formula by up to ~0.3 ΔE — about a tenth of a
        # just-noticeable difference. The exact reference values are checked in
        # test_lab_known_values.
        diff = color.delta_e76(color.rgb_to_lab(x), ref)
        self.assertLess(diff.max(), 0.5)

    def test_hsv_roundtrip_and_values(self):
        hsv = color.rgb_to_hsv(np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255], [128, 128, 128]],
                                        np.uint8))
        np.testing.assert_allclose(hsv[:3, 0], [0, 120, 240])
        self.assertEqual(hsv[3, 1], 0)                  # grey has no saturation
        x = img(9, 9, 3)
        np.testing.assert_allclose(color.hsv_to_rgb(color.rgb_to_hsv(x)) * 255, x, atol=1e-6)

    @unittest.skipIf(cv2 is None, "opencv not installed")
    def test_hsv_matches_opencv(self):
        x = img(16, 16, 3).astype(np.float32) / 255
        ref = cv2.cvtColor(x, cv2.COLOR_RGB2HSV)        # float: H 0..360
        np.testing.assert_allclose(color.rgb_to_hsv(x), ref, atol=1e-3)

    def test_delta_e_is_perceptual_not_rgb(self):
        # Two pairs with the same RGB distance (40 levels) but different visibility:
        # a step in green is far more visible than the same step in blue.
        base = np.array([100, 100, 100], np.uint8)
        g = color.delta_e76(color.rgb_to_lab(base), color.rgb_to_lab(np.array([100, 140, 100], np.uint8)))
        b = color.delta_e76(color.rgb_to_lab(base), color.rgb_to_lab(np.array([100, 100, 140], np.uint8)))
        self.assertGreater(g, b)


class TestVideoIO(unittest.TestCase):
    def test_write_read_and_sample(self):
        frames = np.zeros((48, 32, 48, 3), np.uint8)
        for i in range(48):
            frames[i, :, :, :] = (i * 5) % 256             # brightness encodes the frame index
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "clip.mp4"
            write_video(path, frames, fps=24)
            with VideoReader(path) as vr:
                self.assertEqual((vr.info.width, vr.info.height), (48, 32))
                self.assertAlmostEqual(vr.info.fps, 24, delta=0.1)
                got, times = vr.read_all()
            self.assertEqual(len(got), 48)
            np.testing.assert_allclose(got[:, 16, 24, 0].astype(int), frames[:, 16, 24, 0], atol=6)
            with VideoReader(path, width=24) as vr:
                sampled = list(vr.frames(sample_fps=4, start_s=0.5))
            self.assertEqual(sampled[0].rgb.shape, (16, 24, 3))
            self.assertAlmostEqual(sampled[0].t, 0.5, delta=1 / 24 + 1e-6)
            gaps = np.diff([f.t for f in sampled])
            np.testing.assert_allclose(gaps, 0.25, atol=1 / 24 + 1e-6)


if __name__ == "__main__":
    unittest.main()
