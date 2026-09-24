"""Week 2: shot boundaries, evaluation, k-means palettes."""
import unittest

import numpy as np

from cinescope.io.video import Frame
from cinescope.palette.kmeans import barcode, kmeans, palette, palette_strip
from cinescope.shots import cuts, features
from cinescope.shots.evaluate import score

RNG = np.random.default_rng(0)


def shot_frames(color, n, noise=6.0, texture_seed=None, h=60, w=80):
    """n frames of a textured shot whose dominant colour is `color`."""
    rng = np.random.default_rng(texture_seed)
    tex = rng.normal(0, 25, size=(h, w, 1))
    base = np.clip(np.asarray(color, float) + tex, 0, 255)
    return [np.clip(base + RNG.normal(0, noise, size=(h, w, 3)), 0, 255).astype(np.uint8)
            for _ in range(n)]


def to_features(frames, fps=24.0):
    return features.extract(Frame(i, i / fps, f) for i, f in enumerate(frames))


def synthetic_clip():
    """shot A | cut | shot B | flash | shot B | dissolve | shot C | fade | shot D"""
    a = shot_frames((200, 60, 50), 40, texture_seed=1)
    b = shot_frames((40, 150, 60), 40, texture_seed=2)
    flash = [np.full_like(b[0], 250)]
    b2 = shot_frames((40, 150, 60), 30, texture_seed=2)
    c_ = shot_frames((50, 60, 210), 40, texture_seed=3)
    dissolve = [((1 - a_) * b2[-1] + a_ * c_[0]).astype(np.uint8) for a_ in np.linspace(0, 1, 16)[1:-1]]
    fade = [(c_[-1] * s).astype(np.uint8) for s in np.linspace(1, 0, 10)] + \
           [np.zeros_like(c_[0])] * 4
    d = shot_frames((220, 200, 60), 40, texture_seed=4)
    fade_in = [(d[0] * s).astype(np.uint8) for s in np.linspace(0, 1, 8)]
    frames = a + b + flash + b2 + dissolve + c_ + fade + fade_in + d
    marks = {"cut": 40, "flash": 80, "dissolve": (111, 124),
             "fade": len(a + b + flash + b2 + dissolve + c_) + 12}
    return frames, marks


class TestBoundaries(unittest.TestCase):
    def setUp(self):
        self.frames, self.marks = synthetic_clip()
        self.feat = to_features(self.frames)

    def test_histogram_is_normalised(self):
        self.assertEqual(self.feat.hist.shape[1], 256)
        np.testing.assert_allclose(self.feat.hist.sum(1), 1.0)

    def test_hard_cut_found_and_flash_ignored(self):
        found = [t.frame for t in cuts.detect_cuts(self.feat)]
        self.assertIn(self.marks["cut"], found)
        self.assertNotIn(self.marks["flash"], found)
        self.assertNotIn(self.marks["flash"] + 1, found)

    def test_dissolve_and_fade_found(self):
        tr = cuts.detect_transitions(self.feat)
        kinds = {t.kind for t in tr}
        self.assertIn("fade", kinds)
        a, b = self.marks["dissolve"]
        self.assertTrue(any(a - 3 <= t.frame <= b + 3 for t in tr), [t.to_dict() for t in tr])
        fade = [t for t in tr if t.kind == "fade"][0]
        self.assertLessEqual(abs(fade.frame - self.marks["fade"]), 4)

    def test_no_false_boundaries_inside_static_shots(self):
        tr = cuts.detect_transitions(self.feat)
        self.assertLessEqual(len(tr), 4)          # cut, dissolve, fade (+ at most one extra)

    def test_shots_partition_the_clip(self):
        tr = cuts.detect_transitions(self.feat)
        shots = cuts.shots_from_transitions(tr, len(self.feat), self.feat.times)
        self.assertEqual(shots[0].start, 0)
        self.assertEqual(shots[-1].end, len(self.feat) - 1)
        for s, t in zip(shots[:-1], shots[1:], strict=True):
            self.assertEqual(s.end + 1, t.start)


class TestScoring(unittest.TestCase):
    def test_tolerance_and_gradual_spans(self):
        labels = [{"kind": "cut", "frame": 100, "start": 100, "end": 100},
                  {"kind": "cut", "frame": 200, "start": 200, "end": 200},
                  {"kind": "dissolve", "start": 300, "end": 320, "frame": 310}]
        s = score([101, 250, 305], labels, tol=2)
        self.assertEqual((s["overall"]["tp"], s["overall"]["fp"], s["overall"]["fn"]), (2, 1, 1))
        self.assertAlmostEqual(s["overall"]["f1"], 2 / 3, places=3)
        self.assertEqual(s["gradual"]["recall"], 1.0)
        self.assertEqual(s["cut"]["recall"], 0.5)

    def test_one_to_one_matching(self):
        labels = [{"kind": "cut", "frame": 100, "start": 100, "end": 100}]
        s = score([99, 100, 101], labels, tol=2)
        self.assertEqual((s["overall"]["tp"], s["overall"]["fp"]), (1, 2))


class TestKMeans(unittest.TestCase):
    def test_recovers_separated_clusters(self):
        centres = np.array([[0, 0], [10, 10], [-10, 8]], float)
        x = np.concatenate([c + RNG.normal(0, 0.5, (200, 2)) for c in centres])
        res = kmeans(x, 3, seed=1)
        got = res.centres[np.argsort(res.centres[:, 0])]
        np.testing.assert_allclose(got, centres[np.argsort(centres[:, 0])], atol=0.2)
        self.assertEqual(len(set(res.labels[:200])), 1)

    def test_matches_sklearn_inertia(self):
        try:
            from sklearn.cluster import KMeans
        except ImportError:
            self.skipTest("scikit-learn not installed")
        x = RNG.normal(size=(500, 3))
        ours = kmeans(x, 4, n_init=8, seed=0).inertia
        ref = KMeans(4, n_init=8, random_state=0).fit(x).inertia_
        self.assertLess(ours, ref * 1.05)

    def test_palette_weights_and_order(self):
        img = np.zeros((40, 40, 3), np.uint8)
        img[:, :30] = (220, 30, 30)          # 75% red
        img[:, 30:] = (30, 30, 220)          # 25% blue
        p = palette(img, k=2)
        self.assertGreater(p.rgb[0][0], 180)
        np.testing.assert_allclose(p.weights, [0.75, 0.25], atol=0.02)
        strip = palette_strip(p, width=100, height=5)
        self.assertEqual(strip.shape, (5, 100, 3))
        self.assertTrue((strip[:, :70, 0] > 180).all())

    def test_black_bars_ignored(self):
        img = np.zeros((40, 40, 3), np.uint8)
        img[10:30] = (30, 160, 60)           # letterboxed green frame
        p = palette(img, k=1)
        self.assertGreater(int(p.rgb[0][1]), 120)

    def test_barcode_shape(self):
        frames = np.stack([np.full((10, 10, 3), v, np.uint8) for v in (20, 120, 220)])
        bc = barcode(frames, height=4)
        self.assertEqual(bc.shape, (4, 3, 3))
        self.assertTrue(bc[0, 0, 0] < bc[0, 1, 0] < bc[0, 2, 0])


class TestPipeline(unittest.TestCase):
    def test_end_to_end_on_a_written_video(self):
        import tempfile
        from pathlib import Path

        from cinescope.io.video import write_video
        from cinescope.shots.pipeline import analyse

        frames = np.stack(shot_frames((200, 60, 50), 30, texture_seed=1)
                          + shot_frames((40, 150, 60), 30, texture_seed=2))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "two_shots.mp4"
            write_video(path, frames, fps=24)
            res = analyse(path, with_camera=True, k=3)
            res.save(Path(tmp) / "shots.json")
        self.assertEqual(len(res.shots), 2)
        self.assertLessEqual(abs(res.shots[1].start - 30), 1)
        self.assertEqual(len(res.palettes), 2)
        self.assertEqual(res.camera[0]["label"], "static")
        self.assertEqual(res.barcode.shape[2], 3)


if __name__ == "__main__":
    unittest.main()
