"""Week 9: Hungarian assignment, Kalman box filter, ByteTrack-style tracker, MOT metrics."""
import unittest

import numpy as np

from cinescope.track.hungarian import linear_sum_assignment, match
from cinescope.track.kalman import BoxKalman, xyah_to_xyxy, xyxy_to_xyah
from cinescope.track.mot import evaluate, tracks_to_frames
from cinescope.track.synthetic import make_sequence
from cinescope.track.tracker import Tracker, TrackerParams, track_detections


def box(cx, cy, w=40, h=50):
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], float)


class TestHungarian(unittest.TestCase):
    def test_against_scipy(self):
        so = __import__("pytest").importorskip("scipy.optimize")
        rng = np.random.default_rng(0)
        for t in range(200):
            n, m = rng.integers(1, 9, 2)
            c = rng.random((n, m)) * 10
            if t % 3 == 0:
                c = np.round(c)                    # ties
            for maximize in (False, True):
                r, k = linear_sum_assignment(c, maximize)
                r2, k2 = so.linear_sum_assignment(c, maximize=maximize)
                self.assertAlmostEqual(c[r, k].sum(), c[r2, k2].sum(), places=9)
                self.assertEqual(len(r), min(n, m))
                self.assertEqual(len(set(k.tolist())), len(k))

    def test_beats_greedy(self):
        # Greedy takes the 1 first and is then forced into the 10: total 11 vs optimum 4.
        c = np.array([[1.0, 2.0], [2.0, 10.0]])
        r, k = linear_sum_assignment(c)
        self.assertEqual(c[r, k].sum(), 4.0)

    def test_forbidden_and_infeasible(self):
        r, k = linear_sum_assignment(np.array([[np.inf, 1.0], [2.0, np.inf]]))
        self.assertEqual(k.tolist(), [1, 0])
        with self.assertRaises(ValueError):
            linear_sum_assignment(np.array([[np.inf, np.inf], [1.0, 2.0]]))

    def test_match_gates_after_optimum(self):
        pairs, ur, uc = match(np.array([[0.1, 0.9], [0.95, 0.99]]), max_cost=0.8)
        self.assertEqual(pairs, [(0, 0)])
        self.assertEqual((ur, uc), ([1], [1]))


class TestKalman(unittest.TestCase):
    def test_roundtrip(self):
        b = np.array([10, 20, 50, 100.0])
        np.testing.assert_allclose(xyah_to_xyxy(xyxy_to_xyah(b)), b)

    def test_learns_velocity(self):
        kf = BoxKalman(box(100, 100))
        for t in range(1, 15):
            kf.predict()
            kf.update(box(100 + 5 * t, 100 - 2 * t))
        pred = kf.predict()                         # frame 15, no detection
        np.testing.assert_allclose((pred[:2] + pred[2:]) / 2, [175, 70], atol=1.0)

    def test_uncertainty_grows_without_detections(self):
        kf = BoxKalman(box(100, 100))
        kf.predict()
        kf.update(box(100, 100))
        p0 = np.trace(kf.P)
        for _ in range(5):
            kf.predict()
        self.assertGreater(np.trace(kf.P), p0)
        # A detection near the prediction is close in Mahalanobis distance, a far one is not.
        d = kf.mahalanobis(np.stack([box(100, 100), box(300, 300)]))
        self.assertLess(d[0], 9.49)
        self.assertGreater(d[1], 9.49)


class TestTracker(unittest.TestCase):
    def test_crossing_faces_keep_ids(self):
        dets = []
        for t in range(40):
            dets.append((np.stack([box(50 + 8 * t, 100), box(370 - 8 * t, 110)]), np.array([0.9, 0.9])))
        tracks = track_detections(dets)
        self.assertEqual(len(tracks), 2)
        start = {round((t.boxes[0][0] + t.boxes[0][2]) / 2) for t in tracks}
        self.assertEqual(start, {50, 370})
        for t in tracks:                          # each kept moving in its own direction
            c = [(b[0] + b[2]) / 2 for b in t.boxes]
            self.assertEqual(len(t), 40)
            self.assertTrue(np.all(np.diff(c) > 0) or np.all(np.diff(c) < 0))

    def test_cut_ends_tracks(self):
        dets = [(box(100, 100)[None], np.array([0.9]))] * 20
        self.assertEqual(len(track_detections(dets)), 1)
        self.assertEqual(len(track_detections(dets, cuts={10})), 2)

    def test_low_score_dip_does_not_split(self):
        # A face turning to profile: scores dip to 0.4 for 6 frames.
        scores = [0.9] * 10 + [0.4] * 6 + [0.9] * 10
        dets = [(box(100 + t, 100)[None], np.array([s])) for t, s in enumerate(scores)]
        self.assertEqual(len(track_detections(dets)), 1)
        # Without the second (low-score) round the dip is a gap; the track survives it as
        # "lost", but its detections in the dip are not used.
        sort = track_detections(dets, params=TrackerParams(low=0.6))
        self.assertEqual(sum(len(t) for t in sort), 20)

    def test_one_frame_false_positive_is_not_a_track(self):
        dets = [(np.zeros((0, 4)), np.zeros(0))] * 5
        dets[2] = (box(50, 50)[None], np.array([0.95]))
        dets[0] = (np.zeros((0, 4)), np.zeros(0))
        trk = Tracker()
        trk.update(0, *dets[0])                     # first frame of the shot is empty
        for k in range(1, 5):
            trk.update(k, *dets[k])
        self.assertEqual(trk.finish(), [])

    def test_synthetic_quality(self):
        rng = np.random.default_rng(3)
        gt, dets, cuts = make_sequence(rng, n_frames=120, cut_every=60)
        r = evaluate(dict(enumerate(gt)), tracks_to_frames(track_detections(dets, cuts)))
        self.assertGreater(r["mota"], 0.6)
        self.assertGreater(r["idf1"], 0.7)


class TestMOTMetrics(unittest.TestCase):
    def setUp(self):
        self.gt = {k: (["a"], box(10 * k, 50)[None]) for k in range(10)}

    def test_perfect(self):
        r = evaluate(self.gt, self.gt)
        self.assertEqual((r["mota"], r["idf1"], r["idsw"]), (1.0, 1.0, 0))

    def test_swap_halfway(self):
        hyp = {k: ([1 if k < 5 else 2], b) for k, (_, b) in self.gt.items()}
        r = evaluate(self.gt, hyp)
        self.assertEqual(r["idsw"], 1)
        self.assertAlmostEqual(r["mota"], 0.9)
        self.assertAlmostEqual(r["idf1"], 0.5)        # best single mapping covers half the frames

    def test_misses_and_false_positives(self):
        hyp = {k: ([1], b) for k, (_, b) in self.gt.items() if k % 2 == 0}
        hyp[1] = ([7], box(500, 500)[None])
        r = evaluate(self.gt, hyp)
        self.assertEqual((r["fn"], r["fp"], r["idsw"]), (5, 1, 0))
        self.assertAlmostEqual(r["mota"], 1 - 6 / 10)
        self.assertAlmostEqual(r["idf1"], 2 * 5 / (10 + 6))

    def test_ignore_region(self):
        hyp = {0: ([1, 2], np.stack([box(0, 50), box(300, 300)]))}
        r = evaluate({0: self.gt[0]}, hyp, ignore={0: box(300, 300)[None]})
        self.assertEqual((r["fp"], r["mota"]), (0, 1.0))

    def test_against_motmetrics(self):
        mm = __import__("pytest").importorskip("motmetrics")
        from cinescope.detect.boxes import iou

        rng = np.random.default_rng(5)
        gt, dets, cuts = make_sequence(rng, n_frames=80, cut_every=40)
        G = dict(enumerate(gt))
        H = tracks_to_frames(track_detections(dets, cuts))
        ours = evaluate(G, H)
        acc = mm.MOTAccumulator(auto_id=False)
        for k in G:
            gi, gb = G[k]
            hi, hb = H.get(k, ([], np.zeros((0, 4))))
            d = 1 - iou(gb, hb)
            d[d > 0.5] = np.nan
            acc.update(gi, hi, d, frameid=k)
        ref = mm.metrics.create().compute(acc, metrics=["mota", "idf1", "num_switches"], name="x")
        self.assertAlmostEqual(ours["mota"], float(ref["mota"].iloc[0]), places=9)
        self.assertAlmostEqual(ours["idf1"], float(ref["idf1"].iloc[0]), places=9)
        self.assertEqual(ours["idsw"], int(ref["num_switches"].iloc[0]))


class TestUltralyticsWrapper(unittest.TestCase):
    def test_runs_on_our_detections(self):
        pytest = __import__("pytest")
        pytest.importorskip("ultralytics")
        pytest.importorskip("lap")
        from cinescope.track.ultra import UltralyticsTracker

        trk = UltralyticsTracker("bytetrack")
        for k in range(10):
            trk.update(k, box(100 + 3 * k, 100)[None], np.array([0.9]))
        trk.reset()
        for k in range(10, 15):
            trk.update(k, box(300, 100)[None], np.array([0.9]))
        tracks = trk.finish()
        self.assertEqual(len(tracks), 2)


if __name__ == "__main__":
    unittest.main()
