"""Week 7: CenterNet targets/decoding, WIDER FACE parsing and evaluation, the detector."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from cinescope.detect.boxes import iou
from cinescope.detect.centernet_targets import decode, draw_gaussian, encode, gaussian_radius
from cinescope.detect.wider import parse_annotations
from cinescope.detect.wider_eval import evaluate, image_eval, voc_ap

try:
    import torch
except ImportError:
    torch = None


def perfect_outputs(t, shape=(128, 128)):
    wh, reg = np.zeros((2, *shape)), np.zeros((2, *shape))
    for k in range(int(t["mask"].sum())):
        y, x = divmod(int(t["ind"][k]), shape[1])
        wh[:, y, x], reg[:, y, x] = t["wh"][k], t["reg"][k]
    return t["heatmap"], wh, reg


class TestTargets(unittest.TestCase):
    def test_radius_keeps_iou(self):
        for h, w in [(20, 20), (40, 12), (100, 60), (8, 8)]:
            r = gaussian_radius(h, w)
            box = np.array([[0, 0, w, h]])
            # moving the centre by r in x and y (the worst direction) leaves IoU at exactly 0.7
            for dx, dy in [(r, r), (-r, -r), (r, -r)]:
                shifted = np.array([[dx, dy, w + dx, h + dy]])
                self.assertAlmostEqual(iou(box, shifted)[0, 0], 0.7, places=6)
            self.assertGreater(gaussian_radius(h, w, exact=False), r)     # the original code's radius

    def test_gaussian_peak_and_border(self):
        hm = np.zeros((10, 10))
        draw_gaussian(hm, (0, 9), 3)                     # corner: must not wrap or crash
        self.assertEqual(hm[9, 0], 1.0)
        self.assertEqual(hm.argmax(), 90)
        self.assertLess(hm[9, 3], hm[9, 1])

    def test_encode_decode_round_trip(self):
        boxes = np.array([[40, 40, 120, 140], [200, 60, 233, 101], [10, 300, 30, 322]], float)
        t = encode(boxes, np.zeros(3, int), (128, 128))
        self.assertEqual(t["mask"].sum(), 3)
        got, scores, _ = decode(*perfect_outputs(t), threshold=0.5)
        order = np.argsort(got[:, 0])
        np.testing.assert_allclose(got[order], boxes[np.argsort(boxes[:, 0])], atol=1e-4)
        np.testing.assert_allclose(scores, 1.0)

    def test_tiny_boxes_are_skipped(self):
        t = encode(np.array([[0, 0, 3, 3], [10, 10, 60, 60]], float), np.zeros(2, int), (32, 32))
        self.assertEqual(t["mask"].sum(), 1)


class TestWiderParsing(unittest.TestCase):
    def test_format(self):
        txt = ("0--Parade/a.jpg\n2\n10 10 20 30 0 0 0 0 0 0\n5 5 8 9 2 0 0 1 0 0\n"
               "1--Handshaking/b.jpg\n0\n0 0 0 0 0 0 0 0 0 0\n2--Demo/c.jpg\n1\n1 2 3 4 1 0 0 0 0 0\n")
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "gt.txt"
            p.write_text(txt)
            items = parse_annotations(p)
        self.assertEqual([i.path for i in items], ["0--Parade/a.jpg", "1--Handshaking/b.jpg", "2--Demo/c.jpg"])
        np.testing.assert_array_equal(items[0].boxes, [[10, 10, 20, 30]])     # invalid face dropped
        self.assertEqual(items[0].n_invalid, 1)
        self.assertEqual(len(items[1].boxes), 0)
        self.assertEqual(len(items[2].boxes), 1)


class TestWiderEval(unittest.TestCase):
    def setUp(self):
        self.gts = {"e/a": np.array([[10, 10, 50, 60], [100, 100, 12, 14]], float),
                    "e/b": np.array([[30, 30, 80, 90]], float)}
        self.keeps = {"easy": {"e/a": [0], "e/b": [0]}, "hard": {"e/a": [0, 1], "e/b": [0]}}

    def test_perfect_detector(self):
        preds = {k: np.c_[v, np.linspace(0.9, 0.8, len(v))] for k, v in self.gts.items()}
        r = evaluate(preds, self.gts, self.keeps)
        self.assertAlmostEqual(r["easy"], 1.0, places=2)
        self.assertAlmostEqual(r["hard"], 1.0, places=2)

    def test_missing_small_face_hurts_hard_only(self):
        preds = {"e/a": np.array([[10, 10, 50, 60, 0.9]]), "e/b": np.array([[30, 30, 80, 90, 0.8]])}
        r = evaluate(preds, self.gts, self.keeps)
        self.assertAlmostEqual(r["easy"], 1.0, places=2)
        self.assertAlmostEqual(r["hard"], 2 / 3, places=2)

    def test_detection_on_ignored_face_is_not_a_false_positive(self):
        pred = np.array([[100, 100, 12, 14, 1.0], [10, 10, 50, 60, 0.5]])
        rec, prop = image_eval(pred, self.gts["e/a"], np.array([True, False]))
        self.assertEqual(prop.tolist(), [-1, 1])
        self.assertEqual(rec.tolist(), [0, 1])

    def test_voc_ap(self):
        self.assertAlmostEqual(voc_ap(np.array([0.5, 0.5, 1.0]), np.array([1.0, 0.5, 2 / 3])), 0.5 + 0.5 * 2 / 3)


@unittest.skipIf(torch is None, "PyTorch not installed")
class TestCenterNet(unittest.TestCase):
    def test_shapes_and_prior(self):
        from cinescope.detect.centernet import CenterNet

        m = CenterNet().eval()
        with torch.no_grad():
            out = m(torch.randn(2, 3, 128, 192))
        self.assertEqual(tuple(out["heatmap"].shape), (2, 1, 32, 48))
        self.assertEqual(tuple(out["size"].shape), (2, 2, 32, 48))
        self.assertAlmostEqual(out["heatmap"].sigmoid().mean().item(), 0.1, delta=0.05)

    def test_torch_decode_matches_numpy(self):
        from cinescope.detect.centernet import decode as tdecode

        boxes = np.array([[40, 40, 120, 140], [200, 60, 233, 101]], float)
        t = encode(boxes, np.zeros(2, int), (128, 128))
        hm, wh, reg = perfect_outputs(t)
        logits = torch.logit(torch.from_numpy(hm).clamp(1e-6, 1 - 1e-6))[None]
        out = {"heatmap": logits, "size": torch.from_numpy(wh)[None].float(),
               "offset": torch.from_numpy(reg)[None].float()}
        b, s, _ = tdecode(out, threshold=0.5)[0]
        ref, _, _ = decode(hm, wh, reg, threshold=0.5)
        np.testing.assert_allclose(np.sort(b.numpy(), 0), np.sort(ref, 0), atol=1e-3)

    def test_focal_loss_is_low_for_the_right_heatmap(self):
        from cinescope.detect.centernet import focal_loss

        t = torch.from_numpy(encode(np.array([[40, 40, 120, 140]], float), np.zeros(1, int), (64, 64))["heatmap"])[None]
        good = torch.logit(t.clamp(1e-4, 1 - 1e-4))
        bad = torch.zeros_like(t)
        self.assertLess(focal_loss(good, t).item(), 0.05 * focal_loss(bad, t).item())

    def test_synthetic_training_reduces_loss(self):
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "p2_detection"))
        from train_faces import main

        with tempfile.TemporaryDirectory() as t:
            res = main(["--synthetic", "--epochs", "4", "--bs", "8", "--lr", "2e-3",
                        "--out", t, "--ckpt-dir", t, "--tag", "smoke"])
        hist = res["history"]
        self.assertLess(hist[-1]["heatmap"], hist[0]["heatmap"])


if __name__ == "__main__":
    unittest.main()
