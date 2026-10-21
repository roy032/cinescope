"""Story structure (cinescope.cinema.story) and the pipeline's story stage."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from cinescope.cinema.story import (
    _ranks,
    analyse,
    cut_similarity,
    intensity,
    scene_starts,
    turning_points,
)


def hist(k: int, bins: int = 56) -> np.ndarray:
    """A colour histogram concentrated in bin k (two scenes = two disjoint histograms)."""
    h = np.zeros(bins)
    h[k % bins] = 0.7
    h[(k + 1) % bins] = 0.3
    return h


def shots_of(durations: list[float]) -> list[dict]:
    out, t = [], 0.0
    for i, d in enumerate(durations):
        out.append({"index": i, "start": int(t * 24), "end": int((t + d) * 24) - 1, "t0": t, "t1": t + d,
                    "duration_s": d})
        t += d
    return out


class TestScenes(unittest.TestCase):
    def test_colour_change_starts_a_scene(self):
        shots = shots_of([3.0] * 9)
        hists = [hist(0)] * 3 + [hist(20)] * 3 + [hist(40)] * 3
        self.assertEqual(scene_starts(shots, hists), [0, 3, 6])
        self.assertAlmostEqual(cut_similarity(hists, 1), 1.0)
        self.assertAlmostEqual(cut_similarity(hists, 3), 0.0)

    def test_shared_setup_keeps_the_scene(self):
        shots = shots_of([3.0] * 6)
        hists = [hist(0), hist(20), hist(0), hist(20), hist(0), hist(20)]
        # alternating colours would look like breaks everywhere, but they are two set-ups cut A B A B
        self.assertEqual(scene_starts(shots, hists, setups=[0, 1, 0, 1, 0, 1]), [0])

    def test_min_scene_length(self):
        shots = shots_of([1.0] * 6)
        hists = [hist(0), hist(10), hist(20), hist(30), hist(40), hist(50)]
        starts = scene_starts(shots, hists)
        self.assertEqual(starts[0], 0)
        self.assertTrue(all(b - a >= 2 for a, b in zip(starts, starts[1:], strict=False)))
        times = [shots[i]["t0"] for i in starts]
        self.assertTrue(all(b - a >= 4.0 for a, b in zip(times, times[1:], strict=False)))

    def test_silence_lowers_the_bar(self):
        shots = shots_of([3.0] * 6)
        a, b = hist(0), 0.5 * hist(0) + 0.5 * hist(20)          # overlap 0.5: same scene unless silent
        hists = [a] * 3 + [b] * 3
        self.assertEqual(scene_starts(shots, hists), [0])
        self.assertEqual(scene_starts(shots, hists, silences=[(8.5, 9.5)]), [0, 3])


class TestIntensity(unittest.TestCase):
    def test_ranks(self):
        np.testing.assert_allclose(_ranks([3, 1, 2]), [1.0, 0.0, 0.5])
        np.testing.assert_allclose(_ranks([5, 5, 5]), [0.5] * 3)

    def test_fast_cutting_is_intense(self):
        cuts = list(np.arange(5, 60, 10.0)) + list(np.arange(60, 80, 0.5))
        c = intensity(100.0, cuts)
        hot = c["t"][int(np.argmax(c["value"]))]
        self.assertTrue(55 <= hot <= 85, hot)
        self.assertAlmostEqual(float(c["value"].max()), 1.0)
        self.assertEqual(c["signals"], ["cutting"])

    def test_signals_are_combined(self):
        t = np.linspace(0, 100, 101)
        c = intensity(100.0, [50.0], loud=(t, -40 + 30 * (t > 70)), motion=(t, t))
        self.assertEqual(c["signals"], ["cutting", "loudness", "motion"])
        self.assertGreater(c["value"][-1], c["value"][0])

    def test_turning_points_in_their_windows(self):
        t = np.linspace(0, 1000, 400)
        curve = {"t": t, "value": np.clip((t - 200) / 600, 0, 1), "window_s": 50.0}
        beats = turning_points(curve, 1000.0, [100.0, 250.0, 500.0, 700.0, 950.0])
        by = {b["label"]: b for b in beats}
        self.assertEqual(by["plot point 1"]["t"], 250.0)
        self.assertEqual(by["midpoint"]["t"], 500.0)
        self.assertEqual(by["plot point 2"]["t"], 700.0)
        self.assertTrue(all(b["on"] == "scene change" for b in beats[:3]))
        self.assertEqual(turning_points(curve, 1000.0, [], [300.0])[0]["on"], "cut")
        self.assertIsNone(turning_points(curve, 1000.0, [])[1]["on"])
        self.assertGreaterEqual(by["climax"]["t"], 790.0)


class TestAnalyse(unittest.TestCase):
    def test_full(self):
        shots = shots_of([10.0] * 12)
        hists = [hist(4 * (i // 3)) for i in range(12)]
        r = analyse(shots, hists, 120.0)
        self.assertEqual(r["summary"]["scenes"], 4)
        self.assertEqual([s["first_shot"] for s in r["scenes"]], [0, 3, 6, 9])
        self.assertEqual(len(r["per_shot"]), 12)
        self.assertEqual([p["scene"] for p in r["per_shot"]], [i // 3 for i in range(12)])
        acts = [p["act"] for p in r["per_shot"]]
        self.assertEqual(acts, sorted(acts))
        self.assertEqual({a["act"] for a in r["acts"]}, {1, 2, 3})
        self.assertAlmostEqual(sum(r["summary"]["act_shares"]), 1.0, places=2)
        self.assertEqual(r["summary"]["structure"], "short")
        self.assertEqual(r["timeline"][0]["label"], "scene")

    def test_empty(self):
        self.assertEqual(analyse([], [], 0.0)["per_shot"], [])

    def test_one_shot(self):
        r = analyse(shots_of([20.0]), [hist(0)], 20.0)
        self.assertEqual(r["summary"]["scenes"], 1)
        self.assertEqual(len(r["beats"]), 4)


class TestStoryStage(unittest.TestCase):
    def setUp(self):
        __import__("pytest").importorskip("av")

    def test_stage(self):
        from test_week25 import synthetic_clip

        from cinescope.pipeline import run

        with tempfile.TemporaryDirectory() as d:
            video = Path(d) / "synth.mp4"
            synthetic_clip(video)
            rep = run(video, ["shots", "editing", "sound", "story"], Path(d) / "out", log=None)
        st = rep["extra"]["story"]
        self.assertGreaterEqual(st["summary"]["scenes"], 1)
        self.assertEqual(len(st["beats"]), 4)
        self.assertIn("motion", st["summary"]["signals"])
        self.assertIn("scene", rep["shots"][0]["story"])
        self.assertIn("story", rep["timelines"])


class TestStoryBreakdown(unittest.TestCase):
    def test_story_lane(self):
        from test_editorial import make_report

        from cinescope.editorial.breakdown import breakdown_html

        rep = make_report()
        shots = rep["shots"]
        rep.setdefault("extra", {})["story"] = analyse(shots, [hist(i // 2 * 9) for i in range(len(shots))],
                                                       rep["info"]["duration_s"])
        page = breakdown_html(rep, "/nonexistent")
        for text in ("Story", "plot point 1", "climax", "Scenes", "Arc"):
            self.assertIn(text, page)
        self.assertNotIn("Story", breakdown_html(make_report(), "/nonexistent"))
