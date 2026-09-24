"""Week 6: IoU, NMS, COCO-style mAP."""
import unittest

import numpy as np

from cinescope.detect.boxes import giou, iou, xywh_to_xyxy, xyxy_to_xywh
from cinescope.detect.coco_eval import COCOEvaluator, ap_at_iou
from cinescope.detect.nms import batched_nms, nms, soft_nms

try:
    import torch
    import torchvision
except ImportError:
    torch = torchvision = None
try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
except ImportError:
    COCO = COCOeval = None

RNG = np.random.default_rng(0)


def gt_dict(anns, n_images=1, cats=(1,)):
    out = []
    for i, a in enumerate(anns):
        a = {"id": i + 1, "iscrowd": 0, "category_id": 1, "image_id": 1, **a}
        a.setdefault("area", a["bbox"][2] * a["bbox"][3])
        out.append(a)
    return {"images": [{"id": i + 1, "width": 640, "height": 480} for i in range(n_images)],
            "annotations": out, "categories": [{"id": c, "name": str(c)} for c in cats]}


def det(bbox, score, image_id=1, cat=1):
    return {"image_id": image_id, "category_id": cat, "bbox": list(bbox), "score": score}


class TestBoxes(unittest.TestCase):
    def test_iou_hand_worked(self):
        a = np.array([[0, 0, 10, 10]])
        b = np.array([[5, 0, 15, 10], [20, 20, 30, 30], [0, 0, 10, 10]])
        np.testing.assert_allclose(iou(a, b), [[50 / 150, 0.0, 1.0]])

    def test_crowd_iou_uses_detection_area(self):
        crowd_region = np.array([[0, 0, 100, 100]])
        inside = np.array([[10, 10, 20, 20]])
        self.assertAlmostEqual(iou(inside, crowd_region, crowd=[True])[0, 0], 1.0)
        self.assertAlmostEqual(iou(inside, crowd_region)[0, 0], 0.01)

    def test_giou_is_negative_when_apart(self):
        g = giou(np.array([[0, 0, 1, 1]]), np.array([[2, 0, 3, 1], [0, 0, 1, 1]]))[0]
        self.assertAlmostEqual(g[0], 0 - (3 - 2) / 3)
        self.assertAlmostEqual(g[1], 1.0)

    def test_format_round_trip(self):
        b = RNG.uniform(0, 100, (5, 4))
        np.testing.assert_allclose(xyxy_to_xywh(xywh_to_xyxy(b)), b)

    @unittest.skipIf(torchvision is None, "torchvision not installed")
    def test_iou_matches_torchvision(self):
        a, b = self._rand(30), self._rand(40)
        ref = torchvision.ops.box_iou(torch.from_numpy(a), torch.from_numpy(b)).numpy()
        np.testing.assert_allclose(iou(a, b), ref, atol=1e-12)

    @staticmethod
    def _rand(n):
        xy = RNG.uniform(0, 100, (n, 2))
        return np.c_[xy, xy + RNG.uniform(1, 40, (n, 2))]


class TestNMS(unittest.TestCase):
    def test_keeps_best_of_duplicates(self):
        boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60], [0, 0, 10, 9]])
        scores = np.array([0.8, 0.9, 0.7, 0.3])
        self.assertEqual(nms(boxes, scores, 0.5).tolist(), [1, 2])

    def test_crowd_failure_and_soft_nms(self):
        # two real, heavily overlapping faces (IoU 0.6) — greedy NMS at 0.5 deletes one
        boxes = np.array([[0, 0, 10, 10], [2.5, 0, 12.5, 10]])
        scores = np.array([0.95, 0.9])
        self.assertAlmostEqual(iou(boxes[:1], boxes[1:])[0, 0], 0.6)
        self.assertEqual(len(nms(boxes, scores, 0.5)), 1)
        keep, s = soft_nms(boxes, scores, sigma=0.5)
        self.assertEqual(keep.tolist(), [0, 1])
        self.assertAlmostEqual(s[1], 0.9 * np.exp(-0.36 / 0.5))

    def test_batched_nms_is_per_class(self):
        boxes = np.array([[0, 0, 10, 10], [0, 0, 10, 10]])
        self.assertEqual(sorted(batched_nms(boxes, np.array([0.9, 0.8]), np.array([0, 1])).tolist()), [0, 1])
        self.assertEqual(batched_nms(boxes, np.array([0.9, 0.8]), np.array([0, 0])).tolist(), [0])

    @unittest.skipIf(torchvision is None, "torchvision not installed")
    def test_matches_torchvision(self):
        for seed in range(5):
            rng = np.random.default_rng(seed)
            xy = rng.uniform(0, 60, (200, 2))
            b = np.c_[xy, xy + rng.uniform(5, 30, (200, 2))]
            s = rng.uniform(size=200)
            for t in (0.3, 0.5, 0.7):
                ref = torchvision.ops.nms(torch.from_numpy(b), torch.from_numpy(s), t).numpy()
                self.assertEqual(nms(b, s, t).tolist(), ref.tolist())


class TestCOCOEval(unittest.TestCase):
    def test_perfect(self):
        g = gt_dict([{"bbox": [10, 10, 50, 50]}, {"bbox": [100, 100, 20, 20]}])
        r = COCOEvaluator(g).evaluate([det([10, 10, 50, 50], 0.9), det([100, 100, 20, 20], 0.8)])
        for k in ("AP", "AP50", "AR100"):
            self.assertAlmostEqual(r[k], 1.0)       # precision is tp / (tp + fp + eps), as in pycocotools

    def test_false_positive_ranked_first_halves_ap(self):
        g = gt_dict([{"bbox": [10, 10, 50, 50]}])
        r = COCOEvaluator(g).evaluate([det([10, 10, 50, 50], 0.5), det([200, 200, 50, 50], 0.9)])
        self.assertAlmostEqual(r["AP"], 0.5)
        r = COCOEvaluator(g).evaluate([det([10, 10, 50, 50], 0.9), det([200, 200, 50, 50], 0.5)])
        self.assertAlmostEqual(r["AP"], 1.0)                       # FP after the last TP costs nothing

    def test_loose_box_scores_only_low_thresholds(self):
        g = gt_dict([{"bbox": [0, 0, 100, 100]}])
        d = det([0, 0, 100, 62], 0.9)                              # IoU 0.62
        r = COCOEvaluator(g).evaluate([d])
        self.assertAlmostEqual(r["AP50"], 1.0)
        self.assertEqual(r["AP75"], 0.0)
        self.assertAlmostEqual(r["AP"], 0.3)                       # matches at .50 .55 .60

    def test_area_ranges(self):
        g = gt_dict([{"bbox": [0, 0, 20, 20]}, {"bbox": [100, 100, 60, 60]}, {"bbox": [300, 300, 150, 150]}])
        r = COCOEvaluator(g).evaluate([det([0, 0, 20, 20], 0.9), det([300, 300, 150, 150], 0.8)])
        self.assertAlmostEqual(r["AP_small"], 1.0)
        self.assertAlmostEqual(r["AP_large"], 1.0)
        self.assertEqual(r["AP_medium"], 0.0)
        self.assertAlmostEqual(r["AR100"], 2 / 3)

    def test_crowd_absorbs_detections(self):
        g = gt_dict([{"bbox": [0, 0, 20, 20]}, {"bbox": [100, 100, 200, 200], "iscrowd": 1}])
        dets = [det([0, 0, 20, 20], 0.5)] + [det([110 + 20 * i, 110, 20, 20], 0.9 - 0.01 * i) for i in range(5)]
        self.assertAlmostEqual(COCOEvaluator(g).evaluate(dets)["AP"], 1.0)   # the in-crowd boxes are ignored

    def test_max_dets(self):
        g = gt_dict([{"bbox": [0, 0, 20, 20]}, {"bbox": [50, 50, 20, 20]}])
        r = COCOEvaluator(g).evaluate([det([0, 0, 20, 20], 0.9), det([50, 50, 20, 20], 0.8)])
        self.assertAlmostEqual(r["AR1"], 0.5)
        self.assertAlmostEqual(r["AR10"], 1.0)

    def test_categories_without_gt_are_skipped(self):
        g = gt_dict([{"bbox": [0, 0, 20, 20]}], cats=(1, 2))
        r = COCOEvaluator(g).evaluate([det([0, 0, 20, 20], 0.9), det([0, 0, 20, 20], 0.9, cat=2)])
        self.assertAlmostEqual(r["AP"], 1.0)
        self.assertEqual(r["per_category_AP"][2], -1.0)

    @unittest.skipIf(COCOeval is None, "pycocotools not installed")
    def test_matches_pycocotools(self):
        import contextlib
        import io

        for seed in range(3):
            rng = np.random.default_rng(seed)
            images, anns, dets = [], [], []
            for img in range(1, 21):
                images.append({"id": img, "width": 640, "height": 480})
                for _ in range(rng.integers(0, 8)):
                    xy, wh = rng.uniform(0, 500, 2), rng.uniform(4, 200, 2)
                    c = int(rng.integers(1, 4))
                    anns.append({"id": len(anns) + 1, "image_id": img, "category_id": c,
                                 "bbox": [*xy, *wh], "area": float(wh.prod()),
                                 "iscrowd": int(rng.uniform() < 0.05)})
                    for _ in range(rng.integers(0, 3)):           # noisy copies of the truth
                        dets.append({"image_id": img, "category_id": c, "score": float(rng.uniform()),
                                     "bbox": [*(xy + rng.normal(0, 0.1, 2) * wh), *(wh * rng.uniform(0.7, 1.3, 2))]})
                for _ in range(rng.integers(0, 5)):               # pure false positives
                    dets.append({"image_id": img, "category_id": int(rng.integers(1, 4)), "score": float(rng.uniform()),
                                 "bbox": [*rng.uniform(0, 500, 2), *rng.uniform(4, 200, 2)]})
            gt = {"images": images, "annotations": anns, "categories": [{"id": c, "name": str(c)} for c in (1, 2, 3)]}
            ours = COCOEvaluator(gt).evaluate(dets)
            with contextlib.redirect_stdout(io.StringIO()):
                cg = COCO()
                cg.dataset = gt
                cg.createIndex()
                ev = COCOeval(cg, cg.loadRes(dets), "bbox")
                ev.evaluate()
                ev.accumulate()
                ev.summarize()
            names = ["AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large",
                     "AR1", "AR10", "AR100", "AR_small", "AR_medium", "AR_large"]
            for name, ref in zip(names, ev.stats, strict=True):
                self.assertAlmostEqual(ours[name], ref, places=10, msg=f"seed {seed} {name}")


class TestSingleIoUAP(unittest.TestCase):
    def test_all_point_interpolation(self):
        gts = [np.array([[0, 0, 10, 10], [20, 20, 30, 30]])]
        dets = [np.array([[0, 0, 10, 10], [50, 50, 60, 60], [20, 20, 30, 30]])]
        r = ap_at_iou(gts, dets, [np.array([0.9, 0.8, 0.7])])
        # PR points: (0.5, 1), (0.5, 0.5), (1, 2/3) -> envelope area 0.5*1 + 0.5*(2/3)
        self.assertAlmostEqual(r["ap"], 0.5 + 0.5 * 2 / 3)

    def test_ignored_faces_neither_help_nor_hurt(self):
        gts = [np.array([[0, 0, 10, 10], [40, 40, 42, 42]])]
        ign = [np.array([False, True])]
        dets = [np.array([[40, 40, 42, 42], [0, 0, 10, 10]])]
        r = ap_at_iou(gts, dets, [np.array([0.9, 0.8])], gt_ignore=ign)
        self.assertEqual((r["ap"], r["n_pos"]), (1.0, 1))


if __name__ == "__main__":
    unittest.main()
