"""Highlights: shot scores, picking moments, the EDL and the rendered reel."""
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from cinescope.cinema.highlights import analyse, pick, ranks, shot_scores
from cinescope.editorial.exports import highlights_edl


def shots_of(lengths):
    out, t = [], 0.0
    for i, d in enumerate(lengths):
        out.append({"index": i, "t0": t, "t1": t + d, "start": int(t * 24), "end": int((t + d) * 24) - 1,
                    "duration_s": d})
        t += d
    return out


class TestScores(unittest.TestCase):
    def test_ranks(self):
        np.testing.assert_allclose(ranks([3, 1, 2]), [1.0, 0.0, 0.5])
        np.testing.assert_allclose(ranks([1, 1, 2]), [0.25, 0.25, 1.0])
        np.testing.assert_allclose(ranks([np.nan, 1, 2]), [0.5, 0.0, 1.0])

    def test_intensity_and_loudness_drive_the_score(self):
        shots = shots_of([4.0] * 10)
        v = np.zeros(400)
        v[200:240] = 1.0                                  # shot 5 is the intense one
        intensity = {"t0": 0.0, "dt": 0.1, "value": v.tolist()}
        lufs = np.full(400, -40.0)
        lufs[205:215] = -10.0
        env = {"t0": 0.0, "dt": 0.1, "lufs": lufs.tolist()}
        sc = shot_scores(shots, intensity, env)
        self.assertEqual(int(np.argmax(sc)), 5)

    def test_fast_cutting_scores_higher_without_other_signals(self):
        shots = shots_of([6, 6, 6, 1, 1, 1, 1, 6, 6, 6])
        sc = shot_scores(shots)
        self.assertGreater(sc[4], sc[0])


class TestPick(unittest.TestCase):
    def test_budget_share_and_order(self):
        shots = shots_of([3.0] * 100)                    # 300 s
        sc = np.linspace(0, 1, 100)[::-1].copy()
        ms = pick(shots, 300.0, sc, target_s=30.0)
        total = sum(m["t1"] - m["t0"] for m in ms)
        self.assertLessEqual(total, 30.0 + 1e-6)
        self.assertGreater(total, 24.0)
        self.assertEqual(ms, sorted(ms, key=lambda m: m["t0"]))
        for a, b in zip(ms, ms[1:], strict=False):
            self.assertLessEqual(a["t1"], b["t0"])       # no overlaps
        self.assertTrue(all(m["t1"] - m["t0"] <= 6.0 + 1e-6 for m in ms))

    def test_short_clip_gets_a_summary_not_itself(self):
        shots = shots_of([1.0] * 12)
        ms = pick(shots, 12.0, np.linspace(0, 1, 12))
        self.assertLessEqual(sum(m["t1"] - m["t0"] for m in ms), 0.25 * 12 + 1e-6)
        self.assertTrue(ms)

    def test_climax_is_in(self):
        shots = shots_of([3.0] * 50)
        sc = np.linspace(1, 0, 50)                       # the start scores best
        ms = pick(shots, 150.0, sc, climax_t=140.0, target_s=12.0)
        self.assertTrue(any(m["climax"] and m["t0"] <= 140.0 <= m["t1"] for m in ms))

    def test_long_shot_is_trimmed_around_its_peak_on_the_beat(self):
        shots = shots_of([5.0, 40.0, 5.0])
        lufs = np.full(500, -40.0)
        lufs[300:310] = -5.0                             # loudest at 30-31 s, inside shot 1
        env = {"t0": 0.0, "dt": 0.1, "lufs": lufs.tolist()}
        beats = np.arange(0.0, 50.0, 0.5)
        ms = pick(shots, 50.0, np.array([0.0, 1.0, 0.0]), envelope=env, beats=beats.tolist(),
                  target_s=6.0, max_share=1.0)
        m = ms[0]
        self.assertLess(m["t0"], 30.5)
        self.assertGreater(m["t1"], 30.5)
        self.assertLessEqual(m["t1"] - m["t0"], 6.0 + 0.5)
        self.assertAlmostEqual(m["t0"] * 2, round(m["t0"] * 2), places=6)   # on a beat

    def test_analyse_result_shape(self):
        shots = shots_of([2.0] * 40)
        r = analyse(shots, 80.0)
        self.assertEqual(len(r["per_shot"]), 40)
        self.assertEqual(len(r["timeline"]), len(r["moments"]))
        self.assertEqual(r["summary"]["signals"], ["cutting rate"])
        self.assertEqual(analyse([], 10.0)["moments"], [])


class TestEDL(unittest.TestCase):
    def test_events_are_laid_end_to_end(self):
        rep = {"clip": "c", "video": "c.mp4", "info": {"fps": 25.0, "duration_s": 60.0},
               "extra": {"highlights": {"moments": [
                   {"t0": 2.0, "t1": 4.0, "shots": [1, 1], "score": 0.9, "climax": False},
                   {"t0": 40.0, "t1": 43.0, "shots": [9, 10], "score": 0.8, "climax": True}]}}}
        text = highlights_edl(rep)
        ev = [ln for ln in text.splitlines() if ln[:3].isdigit()]
        self.assertEqual(len(ev), 2)
        self.assertIn("00:00:02:00 00:00:04:00 01:00:00:00 01:00:02:00", ev[0])
        self.assertIn("00:00:40:00 00:00:43:00 01:00:02:00 01:00:05:00", ev[1])
        self.assertIn("CLIMAX", text)
        with self.assertRaises(ValueError):
            highlights_edl({**rep, "extra": {}})


def clip_with_sound(path: Path, seconds: float = 6.0, fps: int = 24, sr: int = 48000) -> None:
    """Colour changes every second, and a tone whose pitch changes with it."""
    av = pytest.importorskip("av")
    with av.open(str(path), "w") as out:
        vs = out.add_stream("mpeg4", rate=fps)
        vs.width, vs.height, vs.pix_fmt = 64, 48, "yuv420p"
        aus = out.add_stream("aac", rate=sr)
        aus.layout = "mono"
        for i in range(int(seconds * fps)):
            f = np.full((48, 64, 3), (i // fps) * 40 % 256, np.uint8)
            vf = av.VideoFrame.from_ndarray(f, format="rgb24")
            vf.pts, vf.time_base = i, Fraction(1, fps)
            for p in vs.encode(vf):
                out.mux(p)
        t = np.arange(int(seconds * sr)) / sr
        x = (0.3 * np.sin(2 * np.pi * (220 + 110 * (t // 1)) * t)).astype(np.float32)
        for k in range(0, len(x), 1024):
            af = av.AudioFrame.from_ndarray(x[None, k:k + 1024], format="fltp", layout="mono")
            af.sample_rate, af.pts, af.time_base = sr, k, Fraction(1, sr)
            for p in aus.encode(af):
                out.mux(p)
        for st in (vs, aus):
            for p in st.encode():
                out.mux(p)


class TestReel(unittest.TestCase):
    def test_render_reel_with_sound(self):
        av = pytest.importorskip("av")
        from cinescope.editorial.reel import render_reel

        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "src.mp4"
            clip_with_sound(src)
            moments = [{"t0": 1.0, "t1": 2.0}, {"t0": 4.0, "t1": 5.5}]
            dest = render_reel(src, moments, Path(d) / "reel.mp4")
            with av.open(str(dest)) as c:
                vs = c.streams.video[0]
                n = sum(1 for _ in c.decode(vs))
                self.assertTrue(c.streams.audio)
            self.assertAlmostEqual(n, 24 * 2.5, delta=2)
            from cinescope.speaker.audio import load_audio

            x = load_audio(str(dest), 16000)
            self.assertAlmostEqual(len(x) / 16000, 2.5, delta=0.1)   # sound as long as the picture
            with self.assertRaises(ValueError):
                render_reel(src, [], Path(d) / "none.mp4")

    def test_pipeline_stage_and_exports(self):
        pytest.importorskip("av")
        from cinescope.pipeline import run

        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "src.mp4"
            clip_with_sound(src, seconds=8.0)
            rep = run(src, ["highlights"], Path(d) / "out", log=None)
            hl = rep["extra"]["highlights"]
            self.assertTrue(hl["moments"])
            self.assertIn("highlights", rep["timelines"])
            self.assertLessEqual(hl["summary"]["duration_s"], 0.25 * 8.0 + 1e-6)
            self.assertTrue(highlights_edl(rep).startswith("TITLE: src CineScope highlights"))
            from cinescope.editorial.__main__ import main

            self.assertEqual(main([str(Path(d) / "out" / "src"), "-f", "highlights", "mp4", "html"]), 0)
            ex = Path(d) / "out" / "src" / "exports"
            self.assertTrue((ex / "src_highlights.mp4").stat().st_size > 0)
            self.assertIn("Highlights", (ex / "src_breakdown.html").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
