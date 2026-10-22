"""Composition (cinescope.cinema.composition) and the pipeline's composition stage."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from cinescope.cinema.composition import (
    _resize,
    frame_composition,
    placement,
    saliency,
    shot_composition,
    summarise,
    trim_bars,
)


def frame(cx: float, cy: float, sharp_bg: bool = True, bars: int = 0, seed: int = 0,
          bg_blur: float = 0.0) -> np.ndarray:
    """A grey frame (crisp 8 px blocks, blurred blocks, or flat) with one bright orange subject at (cx, cy)."""
    from cinescope.imgproc.filters import gaussian_blur

    rng = np.random.default_rng(seed)
    f = np.full((180, 320, 3), 60, np.int64)
    if sharp_bg:
        bg = np.kron(rng.random((23, 40)), np.ones((8, 8)))[:180] * 60
        if bg_blur:
            bg = gaussian_blur(bg, bg_blur)
        f += bg[:, :, None].astype(np.int64)
    x, y = int(cx * 320), int(cy * 180)
    f[y - 20:y + 20, x - 15:x + 15] = (230, 180, 40)
    f[y - 20:y + 20, x - 15:x + 15] += (rng.random((40, 30, 1)) * 25).astype(np.int64)
    f = f.clip(0, 255).astype(np.uint8)
    if bars:
        z = np.zeros((bars, 320, 3), np.uint8)
        f = np.concatenate([z, f, z])
    return f


class TestBasics(unittest.TestCase):
    def test_resize_is_area_mean(self):
        img = np.arange(16.0).reshape(4, 4)
        np.testing.assert_allclose(_resize(img, 2), [[2.5, 4.5], [10.5, 12.5]])
        self.assertIs(_resize(img, 8), img)

    def test_trim_bars(self):
        f = frame(0.5, 0.5, bars=30) / 255.0
        self.assertEqual(trim_bars(f).shape[0], 180)
        dark = np.zeros((100, 50, 3))
        self.assertEqual(trim_bars(dark).shape[0], 100)          # an all-black frame is not all bars

    def test_saliency_peaks_on_subject(self):
        s = saliency(frame(0.25, 0.5))
        h, w = s.shape
        y, x = np.unravel_index(np.argmax(s), s.shape)
        self.assertLess(abs(x / w - 0.25), 0.1)
        self.assertLess(abs(y / h - 0.5), 0.15)
        self.assertAlmostEqual(float(s.max()), 1.0)

    def test_placement(self):
        self.assertEqual(placement(0.5, 0.5), "centre")
        self.assertEqual(placement(0.34, 0.5), "thirds")
        self.assertEqual(placement(0.66, 0.68), "power point")
        self.assertEqual(placement(0.1, 0.5), "edge")


class TestFrames(unittest.TestCase):
    def test_centre_of_interest(self):
        for cx, cy, place in ((0.5, 0.5, "centre"), (1 / 3, 0.4, "power point"), (0.1, 0.5, "edge")):
            c = frame_composition(frame(cx, cy))
            self.assertAlmostEqual(c["x"], cx, delta=0.03)
            self.assertAlmostEqual(c["y"], cy, delta=0.03)
            self.assertEqual(c["placement"], place)

    def test_balance_and_headroom(self):
        left, right = frame_composition(frame(0.2, 0.5)), frame_composition(frame(0.8, 0.5))
        self.assertLess(left["balance"], -0.1)
        self.assertGreater(right["balance"], 0.1)
        high, low = frame_composition(frame(0.5, 0.25)), frame_composition(frame(0.5, 0.75))
        self.assertLess(high["headroom"], low["headroom"])

    def test_letterbox_does_not_move_the_subject(self):
        a, b = frame_composition(frame(0.5, 0.3)), frame_composition(frame(0.5, 0.3, bars=30))
        self.assertAlmostEqual(a["y"], b["y"], delta=0.02)

    def test_focus(self):
        self.assertTrue(frame_composition(frame(0.4, 0.5, sharp_bg=False))["shallow"])
        self.assertFalse(frame_composition(frame(0.4, 0.5, sharp_bg=True))["shallow"])
        soft = frame_composition(frame(0.4, 0.5, bg_blur=6.0))
        self.assertTrue(soft["shallow"], soft)


class TestShotsAndSummary(unittest.TestCase):
    def test_shot_mean(self):
        c = shot_composition([frame(0.3, 0.5, seed=k) for k in range(3)] + [frame(0.36, 0.5)])
        self.assertEqual(c["placement"], "thirds")
        self.assertIsNone(shot_composition([]))

    def test_summary(self):
        per = [{"x": 0.5, "y": 0.5, "placement": "centre", "balance": 0.0, "headroom": 0.2, "shallow": True},
               {"x": 0.3, "y": 0.3, "placement": "power point", "balance": -0.4, "headroom": 0.1, "shallow": False},
               None]
        s = summarise(per, [9.0, 1.0, 5.0])
        self.assertAlmostEqual(s["share"]["centre"], 0.9, places=2)
        self.assertAlmostEqual(sum(s["share"].values()), 1.0, places=2)
        self.assertAlmostEqual(s["shallow_share"], 0.9, places=2)
        self.assertAlmostEqual(float(np.sum(s["heatmap"])), 1.0, places=2)
        self.assertEqual(s["heatmap"][3][3], 0.9)
        self.assertIn("centred", s["description"])
        self.assertIn("shallow focus", s["description"])
        self.assertEqual(summarise([None], [1.0]), {})


class TestCompositionStage(unittest.TestCase):
    def setUp(self):
        __import__("pytest").importorskip("av")

    def test_stage(self):
        from test_week25 import synthetic_clip

        from cinescope.pipeline import run

        with tempfile.TemporaryDirectory() as d:
            video = Path(d) / "synth.mp4"
            synthetic_clip(video)
            rep = run(video, ["shots", "frames", "composition"], Path(d) / "out", log=None)
        c = rep["extra"]["composition"]
        self.assertEqual(len(c["per_shot"]), len(rep["shots"]))
        self.assertIn("placement", rep["shots"][0]["composition"])
        self.assertIn("description", c["summary"])


class TestCompositionBreakdown(unittest.TestCase):
    def test_facts_and_tags(self):
        from test_editorial import make_report

        from cinescope.editorial.breakdown import breakdown_html

        rep = make_report()
        per = [shot_composition([frame(0.33, 0.5, seed=i)]) for i in range(len(rep["shots"]))]
        for s, c in zip(rep["shots"], per, strict=True):
            s["composition"] = c
        rep.setdefault("extra", {})["composition"] = {"per_shot": per, "summary": summarise(
            per, [s["t1"] - s["t0"] for s in rep["shots"]])}
        page = breakdown_html(rep, "/nonexistent")
        for text in ("On the thirds", "Shallow focus", "Mostly framed on the thirds", "thirds"):
            self.assertIn(text, page)
