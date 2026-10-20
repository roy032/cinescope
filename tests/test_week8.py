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


class TestPickedFramesAndYuNet(unittest.TestCase):
    def test_label_boxes_times(self):
        """--times labels the first frame at or after each moment, in one decoding pass."""
        import json
        import sys
        import tempfile
        from pathlib import Path

        import numpy as np

        from cinescope.io.video import write_video

        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "p2_detection"))
        import label_boxes

        frames = np.zeros((48, 32, 48, 3), np.uint8)
        for i in range(48):
            frames[i] = (i * 5) % 256
        with tempfile.TemporaryDirectory() as tmp:
            write_video(Path(tmp) / "clip.mp4", frames, fps=24)
            out = label_boxes.main([str(Path(tmp) / "clip.mp4"), "--times", "1.5,0.25,0.25", "--tag", "x",
                                    "--width", "48", "--out", tmp])
            page = (out / "index.html").read_text()
            meta = json.loads(page.split("const META = ", 1)[1].split(";\n", 1)[0])
        ts = [f["t"] for f in meta["frames"]]
        self.assertEqual(len(ts), 2)                      # a repeated moment is one frame
        self.assertAlmostEqual(ts[0], 0.25, delta=1 / 24 + 1e-6)
        self.assertAlmostEqual(ts[1], 1.5, delta=1 / 24 + 1e-6)

    def test_yunet_boxes_are_xyxy(self):
        import tempfile
        from pathlib import Path

        import numpy as np

        from cinescope.detect.yunet import YuNet

        class Fake:
            def setInputSize(self, size):
                pass

            def detect(self, bgr):
                row = np.zeros(15, np.float32)
                row[:4], row[14] = (10, 20, 30, 40), 0.9          # x, y, w, h ... score
                return 1, row[None]

        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "yunet.onnx").write_bytes(b"")
            det = YuNet(Path(tmp) / "yunet.onnx")
        det.det = Fake()
        r = det.detect([np.zeros((100, 100, 3), np.uint8)])[0]
        np.testing.assert_allclose(r["boxes"], [[10, 20, 40, 60]])
        np.testing.assert_allclose(r["scores"], [0.9])
        with self.assertRaises(FileNotFoundError):
            YuNet("/nonexistent/yunet.onnx")


if __name__ == "__main__":
    unittest.main()


def test_week8_tables_in_readme(tmp_path, monkeypatch):
    """update_results fills the faces/props blocks from the comparison JSONs; our row stays '—' without a run."""
    import json
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    import update_results

    res = tmp_path / "p2_detection" / "results"
    res.mkdir(parents=True)
    clip = {"AP": 0.4, "AP50": 0.9, "AR100": 0.5, "AP_small": -1.0}
    (res / "compare_faces.json").write_text(json.dumps(
        {"OpenCV YuNet": {"x_dark": clip, "y_profile": clip, "all": clip, "ms_per_frame": 12.0}}))
    (res / "compare_props.json").write_text(json.dumps(
        {"yolo11n.pt": {"clips": {"x_dark": {"AP": 0.5, "AP50": 0.75}}, "ms_per_frame": 50.0}}))
    readme = tmp_path / "p2_detection" / "README.md"
    readme.write_text("a\n<!-- auto:p2-faces -->\n<!-- /auto:p2-faces -->\n<!-- auto:p2-props -->\nold\n"
                      "<!-- /auto:p2-props -->\n")
    monkeypatch.setattr(update_results, "ROOT", tmp_path)
    assert update_results.week8()
    text = readme.read_text(encoding="utf-8")
    assert "| CenterNet-R18 (ours) | — |" in text
    assert "| OpenCV YuNet | 0.900 | 0.900 | — | 0.900 | 0.400 | 12.0 |" in text
    assert "| YOLO11n | 0.750 | — | — | 0.500 | — | — | 50.0 |" in text and "old" not in text
