"""Rhythm (cinescope.cinema.rhythm) and the pipeline's rhythm stage."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from cinescope.cinema.rhythm import (
    PULSE,
    SR,
    analyse,
    cuts_on_beat,
    local_clarity,
    onset_envelope,
    rhythmic_sections,
    tempo,
    track_beats,
)


def clicks(bpm: float, seconds: float, phase: float = 0.1, noise: float = 0.01, seed: int = 0) -> np.ndarray:
    """A click track: a short 1.5 kHz ping every beat over a little noise."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0, noise, int(seconds * SR))
    n = int(0.03 * SR)
    ping = 0.5 * np.exp(-np.arange(n) / 200) * np.sin(2 * np.pi * 1500 * np.arange(n) / SR)
    for t in np.arange(phase, seconds - 0.05, 60.0 / bpm):
        i = int(t * SR)
        x[i:i + n] += ping[:len(x) - i]
    return x


class TestTempo(unittest.TestCase):
    def test_tempo_of_click_tracks(self):
        for bpm in (70, 90, 120, 150):
            tp = tempo(onset_envelope(clicks(bpm, 30)))
            self.assertAlmostEqual(tp["bpm"], bpm, delta=1.5)
            self.assertGreater(tp["clarity"], 0.6)

    def test_noise_has_no_pulse(self):
        e = onset_envelope(np.random.default_rng(1).normal(0, 0.1, 30 * SR))
        self.assertLess(tempo(e)["clarity"], PULSE)

    def test_too_short(self):
        self.assertEqual(len(onset_envelope(np.zeros(100))), 0)
        self.assertIsNone(tempo(np.zeros(50))["bpm"])


class TestBeats(unittest.TestCase):
    def test_beats_land_on_clicks(self):
        e = onset_envelope(clicks(120, 20, phase=0.3))
        b = track_beats(e, tempo(e)["period_s"])
        truth = np.arange(0.3, 19.95, 0.5)
        self.assertGreaterEqual(len(b), len(truth) - 2)
        err = np.array([np.abs(truth - t).min() for t in b])
        self.assertLess(float(np.median(err)), 0.04)

    def test_rhythmic_sections(self):
        x = np.concatenate([np.random.default_rng(2).normal(0, 0.05, 20 * SR), clicks(100, 30)])
        t, c, _ = local_clarity(onset_envelope(x))
        spans = rhythmic_sections(t, c, 50.0)
        self.assertEqual(len(spans), 1)
        self.assertAlmostEqual(spans[0][0], 20.0, delta=4.0)
        self.assertAlmostEqual(spans[0][1], 50.0, delta=1.0)

    def test_cuts_on_beat(self):
        beats = np.arange(0, 10, 0.5)
        r = cuts_on_beat([1.0, 2.52, 3.25, 20.0], beats, [(0.0, 10.0)], 0.5)
        self.assertEqual(r["cuts"], 3)
        self.assertAlmostEqual(r["on_beat"], 2 / 3, places=3)
        self.assertAlmostEqual(r["chance"], 0.332, places=3)
        self.assertIsNone(cuts_on_beat([1.0], beats, [], 0.5)["on_beat"])


class TestAnalyse(unittest.TestCase):
    def test_music_video(self):
        shots = [{"t0": float(a), "t1": float(a + 2.0)} for a in range(0, 30, 2)]
        r = analyse(clicks(120, 30), shots, 30.0)
        s = r["summary"]
        self.assertAlmostEqual(s["bpm"], 120.0, delta=1.5)
        self.assertGreater(s["rhythmic_share"], 0.9)
        self.assertGreaterEqual(s["cuts_on_beat"]["on_beat"], 0.9)       # every cut is on a beat here
        self.assertEqual(r["per_shot"][3]["beats"], 4)
        self.assertAlmostEqual(r["per_shot"][3]["length_beats"], 4.0, delta=0.1)
        self.assertEqual(r["timeline"][0]["label"], "rhythmic")

    def test_no_audio(self):
        r = analyse(None, [{"t0": 0.0, "t1": 1.0}], 1.0)
        self.assertFalse(r["audio"])
        self.assertEqual(r["per_shot"], [None])

    def test_speech_like_noise_has_no_bpm(self):
        r = analyse(np.random.default_rng(3).normal(0, 0.1, 20 * SR), [{"t0": 0.0, "t1": 20.0}], 20.0)
        self.assertIsNone(r["summary"]["bpm"])
        self.assertEqual(r["beats"], [])


class TestRhythmStage(unittest.TestCase):
    def setUp(self):
        __import__("pytest").importorskip("av")

    def test_stage_with_audio(self):
        from test_sound import clip_with_audio

        from cinescope.pipeline import run

        with tempfile.TemporaryDirectory() as d:
            video = Path(d) / "beat.mkv"
            clip_with_audio(video, clicks(120, 12).astype(np.float32))
            rep = run(video, ["shots", "rhythm"], Path(d) / "out", log=None)
        r = rep["extra"]["rhythm"]
        self.assertTrue(r["audio"])
        self.assertAlmostEqual(r["summary"]["bpm"], 120.0, delta=2.0)
        self.assertIn("beats", rep["shots"][0]["rhythm"])

    def test_stage_without_audio(self):
        from test_week25 import synthetic_clip

        from cinescope.pipeline import run

        with tempfile.TemporaryDirectory() as d:
            video = Path(d) / "synth.mp4"
            synthetic_clip(video)
            rep = run(video, ["shots", "rhythm"], Path(d) / "out", log=None)
        self.assertFalse(rep["extra"]["rhythm"]["audio"])


class TestRhythmBreakdown(unittest.TestCase):
    def test_beat_lane_and_facts(self):
        from test_editorial import make_report

        from cinescope.editorial.breakdown import breakdown_html

        rep = make_report()
        dur = rep["info"]["duration_s"]
        rep.setdefault("extra", {})["rhythm"] = analyse(clicks(120, dur), rep["shots"], dur)
        page = breakdown_html(rep, "/nonexistent")
        for text in ("Beat", "Tempo", "120 BPM"):
            self.assertIn(text, page)
