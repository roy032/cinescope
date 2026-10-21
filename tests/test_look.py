"""Look and grade per shot (cinescope.cinema.look) and the pipeline's look stage."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from cinescope.cinema.look import (
    film_look,
    key_timeline,
    ratio_name,
    shot_look,
    symmetry_score,
    teal_orange_score,
)
from cinescope.imgproc.color import rgb_to_hsv


def solid(rgb, h=36, w=64):
    return np.broadcast_to(np.asarray(rgb, np.uint8), (h, w, 3)).copy()


class TestShotLook(unittest.TestCase):
    def test_low_key_and_high_key(self):
        dark = solid((10, 10, 12))
        dark[10:20, 20:30] = 200                      # one hard practical light in the dark
        self.assertEqual(shot_look(dark)["key"], "low-key")
        bright = solid((215, 215, 210))
        bright[:, :8] = 190                           # bright and flat
        self.assertEqual(shot_look(bright)["key"], "high-key")
        grad = np.repeat(np.linspace(0, 255, 64, dtype=np.uint8)[None, :, None], 36, 0).repeat(3, 2)
        self.assertEqual(shot_look(grad)["key"], "normal")

    def test_temperature(self):
        self.assertEqual(shot_look(solid((230, 150, 60)))["temperature"], "warm")
        self.assertEqual(shot_look(solid((60, 120, 200)))["temperature"], "cool")
        self.assertEqual(shot_look(solid((128, 128, 128)))["temperature"], "neutral")
        self.assertAlmostEqual(shot_look(solid((128, 128, 128)))["saturation"], 0.0)

    def test_teal_orange_needs_both_bands(self):
        img = solid((230, 140, 60))                   # orange
        img[:, 32:] = (40, 150, 170)                  # teal
        both = teal_orange_score(rgb_to_hsv(img))
        self.assertGreater(both, 0.9)
        self.assertEqual(teal_orange_score(rgb_to_hsv(solid((230, 140, 60)))), 0.0)
        self.assertEqual(teal_orange_score(rgb_to_hsv(solid((120, 120, 120)))), 0.0)

    def test_symmetry(self):
        rng = np.random.default_rng(0)
        half = rng.random((20, 16))
        sym = np.concatenate([half, half[:, ::-1]], 1)
        self.assertAlmostEqual(symmetry_score(sym), 1.0)
        lopsided = np.zeros((20, 32))
        lopsided[:, :8] = 1.0
        self.assertLess(symmetry_score(lopsided), 0.2)
        self.assertEqual(symmetry_score(np.full((8, 8), 0.5)), 0.0)

    def test_frames_and_clipping(self):
        stack = np.stack([solid((0, 0, 0)), solid((255, 255, 255))])
        lk = shot_look(stack)
        self.assertAlmostEqual(lk["crushed"], 0.5)
        self.assertAlmostEqual(lk["clipped"], 0.5)
        self.assertAlmostEqual(lk["brightness"], 0.5, places=3)


    def test_letterbox_is_found_and_ignored(self):
        img = solid((40, 120, 200), h=72, w=128)        # 16:9 frame
        img[:9] = 0
        img[-9:] = 0                                    # bars leave 128x54 = 2.37:1
        lk = shot_look(img)
        self.assertTrue(lk["bars"])
        self.assertAlmostEqual(lk["aspect"], 128 / 54, places=2)
        self.assertEqual(lk["crushed"], 0.0)            # the bars are not "crushed blacks"
        self.assertEqual(ratio_name(lk["aspect"]), "2.39:1 scope")
        self.assertFalse(shot_look(solid((40, 120, 200), h=72, w=128))["bars"])
        dark = solid((2, 2, 2), h=72, w=128)            # a black frame is not all bars
        self.assertAlmostEqual(shot_look(dark)["aspect"], 128 / 72, places=2)


class TestFilmLook(unittest.TestCase):
    def test_weighted_by_screen_time(self):
        a = shot_look(solid((230, 150, 60)))
        b = shot_look(solid((60, 120, 200)))
        s = film_look([a, b, None], [9.0, 1.0, 5.0])
        self.assertEqual(s["temperature"], "warm")
        self.assertAlmostEqual(s["temperature_share"]["warm"], 0.9, places=3)
        self.assertIn("warm", s["description"])
        self.assertEqual(s["aspect_name"], "16:9")
        self.assertEqual(s["letterboxed"], 0.0)
        self.assertEqual(film_look([None], [1.0]), {})

    def test_key_timeline_merges_runs(self):
        shots = [{"t0": 0.0, "t1": 1.0}, {"t0": 1.0, "t1": 2.0}, {"t0": 2.0, "t1": 3.0}]
        looks = [{"key": "low-key"}, {"key": "low-key"}, {"key": "normal"}]
        self.assertEqual(key_timeline(shots, looks), [{"t0": 0.0, "t1": 2.0, "label": "low-key"},
                                                      {"t0": 2.0, "t1": 3.0, "label": "normal"}])


class TestLookStage(unittest.TestCase):
    def test_stage_in_the_pipeline(self):
        __import__("pytest").importorskip("av")
        from test_week25 import synthetic_clip

        from cinescope.pipeline import run

        with tempfile.TemporaryDirectory() as d:
            video = Path(d) / "synth.mp4"
            synthetic_clip(video)                       # red, green, blue shots
            rep = run(video, ["shots", "look"], Path(d) / "out", log=None)
        looks = [s["look"] for s in rep["shots"]]
        self.assertEqual(len(looks), 3)
        self.assertEqual(looks[0]["temperature"], "warm")
        self.assertEqual(looks[2]["temperature"], "cool")
        self.assertIn("description", rep["extra"]["look"]["summary"])
        self.assertIn("look", rep["timelines"])
