"""Week 8: hand-drawn clip labels -> COCO ground truth -> evaluation."""
import unittest

from cinescope.detect.cliplabels import to_coco
from cinescope.detect.coco_eval import COCOEvaluator

LABELS = {
    "name": "clip_dark", "classes": ["face", "car"],
    "frames": [
        {"file": "0000.jpg", "t": 0.0, "index": 0, "width": 640, "height": 360,
         "boxes": [{"bbox": [10, 10, 40, 50], "label": "face", "ignore": 0},
                   {"bbox": [300, 20, 8, 9], "label": "face", "ignore": 1},
                   {"bbox": [100, 200, 200, 100], "label": "car", "ignore": 0}]},
        {"file": "0001.jpg", "t": 2.0, "index": 48, "width": 640, "height": 360, "boxes": []},
    ],
}


class TestClipLabels(unittest.TestCase):
    def test_to_coco(self):
        gt = to_coco(LABELS, ["face"])
        self.assertEqual(len(gt["images"]), 2)
        self.assertEqual([a["iscrowd"] for a in gt["annotations"]], [0, 1])       # car dropped, ignore -> crowd
        self.assertEqual(gt["categories"], [{"id": 1, "name": "face"}])

    def test_ignored_face_neither_helps_nor_hurts(self):
        gt = to_coco(LABELS, ["face"])
        found_both = [{"image_id": 1, "category_id": 1, "bbox": [10, 10, 40, 50], "score": 0.9},
                      {"image_id": 1, "category_id": 1, "bbox": [300, 20, 8, 9], "score": 0.95}]
        self.assertAlmostEqual(COCOEvaluator(gt).evaluate(found_both)["AP"], 1.0)
        self.assertAlmostEqual(COCOEvaluator(gt).evaluate(found_both[:1])["AP"], 1.0)


if __name__ == "__main__":
    unittest.main()
