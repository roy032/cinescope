"""Editing grammar (cinescope.cinema.editing) and the pipeline's editing stage."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from cinescope.cinema.editing import (
    analyse,
    exchanges,
    find_setups,
    long_takes,
    montages,
    pacing_curve,
    same_setup,
    shot_descriptor,
)


def angle(seed: int, colour) -> np.ndarray:
    """A 'camera set-up': a fixed random layout of light and dark in one colour cast."""
    rng = np.random.default_rng(seed)
    layout = np.kron(rng.random((6, 8)), np.ones((6, 6)))[..., None]
    return np.clip(255 * layout * np.asarray(colour) / 255, 0, 255).astype(np.uint8)


def take(img: np.ndarray, seed: int) -> np.ndarray:
    """Three frames of a shot from that set-up: small noise and a small exposure change."""
    rng = np.random.default_rng(seed)
    gain = 1.0 + 0.05 * rng.standard_normal()
    return np.stack([np.clip(img * gain + rng.normal(0, 4, img.shape), 0, 255).astype(np.uint8) for _ in range(3)])


A, B, C = angle(1, (230, 180, 150)), angle(2, (120, 140, 230)), angle(3, (90, 220, 110))


class TestSetups(unittest.TestCase):
    def test_same_and_different_setup(self):
        self.assertTrue(same_setup(shot_descriptor(take(A, 0)), shot_descriptor(take(A, 1))))
        self.assertFalse(same_setup(shot_descriptor(take(A, 0)), shot_descriptor(take(B, 1))))
        black = np.zeros((3, 36, 48, 3), np.uint8)
        self.assertFalse(same_setup(shot_descriptor(black), shot_descriptor(black)))

    def test_conversation_is_found(self):
        seq = [C, A, B, A, B, A, C]
        descs = [shot_descriptor(take(img, k)) for k, img in enumerate(seq)]
        setups = find_setups(descs)
        self.assertEqual(setups, [0, 1, 2, 1, 2, 1, 0])
        self.assertEqual(exchanges(setups), [(1, 5)])

    def test_window_limits_matching(self):
        descs = [shot_descriptor(take(A, 0))] + [shot_descriptor(take(angle(10 + k, (200, 200, 200)), k))
                                                  for k in range(5)] + [shot_descriptor(take(A, 9))]
        self.assertEqual(find_setups(descs, window=3)[-1], 6)
        self.assertEqual(find_setups(descs, window=10)[-1], 0)


class TestPatterns(unittest.TestCase):
    def test_exchanges(self):
        self.assertEqual(exchanges([0, 1, 0, 1, 2, 3, 2, 3, 2]), [(0, 3), (4, 8)])
        self.assertEqual(exchanges([0, 1, 0]), [])
        self.assertEqual(exchanges([0, 0, 0, 0]), [])

    def test_montage_and_long_take(self):
        d = [4.0] * 6 + [0.5] * 6 + [4.0] * 4 + [40.0]
        self.assertEqual(montages(d), [(6, 11)])
        self.assertEqual(montages([0.5] * 4 + [4.0] * 4), [])          # four is not a montage
        self.assertEqual(long_takes(d), [16])
        self.assertEqual(long_takes([3.0, 3.0, 11.0]), [])               # 15 s at least

    def test_pacing_curve(self):
        cuts = list(np.arange(1, 60, 1.0))                               # a cut every second
        p = pacing_curve(cuts, 60.0, window=10.0, points=61)
        self.assertAlmostEqual(p["cpm"][30], 66.0, delta=0.1)            # 11 cuts in [25, 35]
        self.assertEqual(pacing_curve([], 0.0)["t"], [])

    def test_analyse(self):
        shots, t = [], 0.0
        for d in [3.0, 3.0, 3.0, 3.0, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 30.0]:
            shots.append({"t0": t, "t1": t + d, "duration_s": d})
            t += d
        seq = [A, B, A, B] + [angle(20 + k, (200, 200, 200)) for k in range(7)]
        out = analyse(shots, [shot_descriptor(take(img, k)) for k, img in enumerate(seq)], t)
        self.assertEqual([e["label"] for e in out["timeline"]], ["shot/reverse-shot", "montage", "long take"])
        self.assertEqual(out["summary"]["recurring_setups"], 2)
        self.assertTrue(out["per_shot"][0]["recurs"])
        self.assertLess(out["summary"]["fastest_t"], 20.0)                # not in the long take
        self.assertGreater(out["summary"]["fastest_cpm"], 60 * 10 / t)  # faster than the average


class TestEditingStage(unittest.TestCase):
    def test_stage_in_the_pipeline(self):
        __import__("pytest").importorskip("av")
        from test_week25 import synthetic_clip

        from cinescope.pipeline import run

        with tempfile.TemporaryDirectory() as d:
            video = Path(d) / "synth.mp4"
            synthetic_clip(video)
            rep = run(video, ["shots", "editing"], Path(d) / "out", log=None)
        self.assertEqual([s["editing"]["setup"] for s in rep["shots"]], [0, 1, 2])
        self.assertEqual(rep["extra"]["editing"]["summary"]["setups"], 3)
        self.assertEqual(len(rep["extra"]["editing"]["pacing"]["t"]), 200)
