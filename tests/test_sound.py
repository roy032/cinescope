"""The soundtrack (cinescope.cinema.sound) and the pipeline's sound stage."""
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

import numpy as np

from cinescope.cinema.sound import (
    _block_power,
    analyse,
    cut_sync,
    integrated,
    k_weight_gain,
    loudness_range,
    onsets,
    runs,
)

SR = 16000


def tone(seconds: float, dbfs: float, f: float = 1000.0) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return (10 ** (dbfs / 20) * np.sin(2 * np.pi * f * t)).astype(np.float32)


class TestLoudness(unittest.TestCase):
    def test_reference_sine(self):
        # BS.1770: a full-scale 1 kHz sine in one channel reads -3.01 LUFS
        self.assertAlmostEqual(integrated(_block_power(tone(5, 0), SR, 0.4, 0.1)), -3.01, delta=0.05)
        self.assertAlmostEqual(integrated(_block_power(tone(5, -20), SR, 0.4, 0.1)), -23.01, delta=0.05)

    def test_k_weighting_shape(self):
        g = 10 * np.log10(k_weight_gain([20.0, 1000.0, 6000.0]))
        self.assertLess(g[0], -10)                     # rumble is turned down
        self.assertAlmostEqual(g[1], 0.7, delta=0.1)
        self.assertAlmostEqual(g[2], 4.0, delta=0.3)   # the head-shadow shelf

    def test_gating_ignores_silence(self):
        x = np.concatenate([tone(5, -20), np.zeros(5 * SR, np.float32)])
        # the 400 ms blocks that straddle the end of the tone pass the gate too, as in BS.1770
        self.assertAlmostEqual(integrated(_block_power(x, SR, 0.4, 0.1)), -23.01, delta=0.2)
        self.assertEqual(integrated(_block_power(np.zeros(SR, np.float32), SR, 0.4, 0.1)), float("-inf"))

    def test_loudness_range(self):
        x = np.concatenate([tone(10, -30), tone(10, -10)])
        lra = loudness_range(_block_power(x, SR, 3.0, 1.0))
        self.assertGreater(lra, 15)
        self.assertLess(loudness_range(_block_power(tone(20, -20), SR, 3.0, 1.0)), 0.5)

    def test_runs(self):
        self.assertEqual(runs(np.array([0, 1, 1, 1, 0, 1, 0, 1, 1], bool), 2), [(1, 4), (7, 9)])


class TestEvents(unittest.TestCase):
    def test_peak_and_silence(self):
        x = np.concatenate([tone(6, -30), tone(1, -10), tone(3, -30), np.zeros(3 * SR, np.float32), tone(3, -30)])
        shots = [{"t0": 0.0, "t1": 8.0}, {"t0": 8.0, "t1": 16.0}]
        out = analyse(x, shots, 16.0)
        labels = [(e["label"], round(e["t0"])) for e in out["timeline"]]
        self.assertIn(("peak", 6), labels)
        self.assertIn(("silence", 10), labels)
        self.assertEqual(out["summary"]["peaks"], 1)
        self.assertGreater(out["per_shot"][0]["max_lufs"], out["per_shot"][1]["max_lufs"])
        self.assertLessEqual(len(out["envelope"]["lufs"]), 600)

    def test_no_audio(self):
        out = analyse(None, [{"t0": 0.0, "t1": 1.0}], 1.0)
        self.assertFalse(out["audio"])
        self.assertEqual(out["per_shot"], [None])

    def test_onsets_find_clicks(self):
        rng = np.random.default_rng(0)
        x = 0.001 * rng.standard_normal(8 * SR).astype(np.float32)
        hits = [1.0, 2.5, 4.0, 6.2]
        for h in hits:
            a = int(h * SR)
            x[a:a + 800] += 0.5 * rng.standard_normal(800) * np.exp(-np.arange(800) / 200)
        on = onsets(x, SR)
        for h in hits:
            self.assertLess(np.min(np.abs(on - h)), 0.05)
        self.assertLessEqual(len(on), 2 * len(hits))

    def test_cut_sync(self):
        o = np.array([1.0, 3.0, 5.0, 7.0])
        s = cut_sync([1.02, 3.05, 6.0], o, 10.0)
        self.assertAlmostEqual(s["on_sound"], 2 / 3, places=3)
        self.assertAlmostEqual(s["chance"], 0.08, delta=0.01)       # 4 onsets x 0.2 s of 10 s
        self.assertEqual(cut_sync([], o, 10.0)["on_sound"], 0.0)


def clip_with_audio(path: Path, audio: np.ndarray, fps: float = 24.0) -> None:
    """A grey video with a PCM soundtrack, muxed with PyAV into Matroska."""
    import av

    n = int(len(audio) / SR * fps)
    with av.open(str(path), "w") as out:
        v = out.add_stream("mpeg4", rate=Fraction(fps).limit_denominator(1001))
        v.width, v.height, v.pix_fmt = 64, 48, "yuv420p"
        a = out.add_stream("pcm_s16le", rate=SR)
        a.layout = "mono"
        pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
        for k in range(0, len(pcm), 1024):
            fr = av.AudioFrame.from_ndarray(pcm[None, k:k + 1024], format="s16", layout="mono")
            fr.sample_rate = SR
            fr.pts = k
            for p in a.encode(fr):
                out.mux(p)
        for p in a.encode():
            out.mux(p)
        for i in range(n):
            img = np.full((48, 64, 3), 40 if i < n // 2 else 200, np.uint8)
            for p in v.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                out.mux(p)
        for p in v.encode():
            out.mux(p)


class TestSoundStage(unittest.TestCase):
    def setUp(self):
        __import__("pytest").importorskip("av")

    def test_stage_with_audio(self):
        from cinescope.pipeline import run

        x = np.concatenate([tone(2, -30), tone(1, -8), tone(3, -30)])
        with tempfile.TemporaryDirectory() as d:
            video = Path(d) / "beep.mkv"
            clip_with_audio(video, x)
            rep = run(video, ["shots", "sound"], Path(d) / "out", log=None)
        snd = rep["extra"]["sound"]
        self.assertTrue(snd["audio"])
        self.assertIsNotNone(snd["summary"]["integrated_lufs"])
        self.assertEqual([e["label"] for e in snd["timeline"]], ["peak"])
        self.assertAlmostEqual(snd["timeline"][0]["t0"], 2.0, delta=0.3)
        self.assertIn("lufs", rep["shots"][0]["sound"])

    def test_stage_without_audio(self):
        from test_week25 import synthetic_clip

        from cinescope.pipeline import run

        with tempfile.TemporaryDirectory() as d:
            video = Path(d) / "synth.mp4"
            synthetic_clip(video)
            rep = run(video, ["shots", "sound"], Path(d) / "out", log=None)
        self.assertFalse(rep["extra"]["sound"]["audio"])
