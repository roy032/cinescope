"""Week 5: ResNet-18, MovieShots index, training loop pieces.

The PyTorch tests skip cleanly without torch; the torchvision comparisons skip
without torchvision. The data-index and metric tests need neither.
"""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from cinescope.framing import SCALES
from cinescope.framing.classify import keyframe_indices
from cinescope.framing.data import (
    ShotRecord,
    assign_splits,
    canonical_scale,
    read_index,
    read_movieshots,
    write_index,
)
from cinescope.framing.metrics import confusion_matrix, metrics
from cinescope.shots.cuts import Shot

try:
    import torch
except ImportError:
    torch = None
try:
    import torchvision
except ImportError:
    torchvision = None


class TestMovieShotsIndex(unittest.TestCase):
    def _fake_root(self, tmp: Path) -> Path:
        ann = {"train": {"tt001": {"0000": {"scale": {"label": "CS", "value": 1},
                                            "movement": {"label": "Static", "value": 0}},
                                   "0001": {"scale": {"label": "LS"}, "movement": {"label": "Push"}}}},
               "val": {"tt002": {"0000": {"scale": {"label": "MS"}}}},
               "test": {"tt003": {"0005": {"scale": {"label": "ECS"}},
                                  "0006": {"scale": {"label": "??"}},
                                  "0007": {"scale": {"label": "FS"}}}}}
        (tmp / "v1_split_trailer.json").write_text(json.dumps(ann))
        for tid, sid in [("tt001", "0000"), ("tt001", "0001"), ("tt002", "0000"), ("tt003", "0005"),
                         ("tt003", "0006")]:          # 0007 has no frames on disk
            d = tmp / "trailer" / tid
            d.mkdir(parents=True, exist_ok=True)
            for k in range(3):
                (d / f"shot_{sid}_img_{k}.jpg").write_bytes(b"")
        return tmp

    def test_reads_nested_json_and_reports_problems(self):
        with tempfile.TemporaryDirectory() as t:
            recs, problems = read_movieshots(self._fake_root(Path(t)))
        got = {(r.split, r.trailer, r.shot, r.scale) for r in recs}
        self.assertEqual(got, {("train", "tt001", "0000", "CS"), ("train", "tt001", "0001", "LS"),
                               ("val", "tt002", "0000", "MS"), ("test", "tt003", "0005", "ECS")})
        self.assertEqual(problems["unknown scale label"], 1)
        self.assertEqual(problems["frames not found"], 1)
        r = next(r for r in recs if r.shot == "0001")
        self.assertEqual(r.movement, "Push")
        self.assertEqual(r.frames, [f"trailer/tt001/shot_0001_img_{k}.jpg" for k in range(3)])

    def test_index_round_trip(self):
        with tempfile.TemporaryDirectory() as t:
            recs, _ = read_movieshots(self._fake_root(Path(t)))
            write_index(recs, Path(t) / "idx" / "index.csv")
            back = read_index(Path(t) / "idx" / "index.csv")
            train = read_index(Path(t) / "idx" / "index.csv", "train")
        self.assertEqual([(r.shot, r.scale, r.frames) for r in back], [(r.shot, r.scale, r.frames) for r in recs])
        self.assertEqual(len(train), 2)

    def test_label_aliases(self):
        self.assertEqual(canonical_scale("close-up"), "CS")
        self.assertEqual(canonical_scale(" ls "), "LS")
        with self.assertRaises(ValueError):
            canonical_scale("dutch angle")

    def test_splits_are_by_trailer(self):
        recs = [ShotRecord("all", f"tt{t:03d}", str(s), "CS", "", ["x.jpg"]) for t in range(50) for s in range(4)]
        assign_splits(recs, val=0.1, test=0.2, seed=3)
        by_trailer = {}
        for r in recs:
            by_trailer.setdefault(r.trailer, set()).add(r.split)
        self.assertTrue(all(len(v) == 1 for v in by_trailer.values()))      # no trailer spans splits
        counts = {s: sum(1 for v in by_trailer.values() if s in v) for s in ("train", "val", "test")}
        self.assertEqual(counts, {"train": 35, "val": 5, "test": 10})


class TestMetrics(unittest.TestCase):
    def test_confusion_and_per_class(self):
        y = np.array([0, 0, 1, 1, 1, 2])
        p = np.eye(3)[[0, 1, 1, 1, 2, 2]]
        m = metrics(p, y, ("a", "b", "c"))
        self.assertEqual(m["confusion"], [[1, 1, 0], [0, 2, 1], [0, 0, 1]])
        self.assertAlmostEqual(m["accuracy"], 4 / 6)
        self.assertAlmostEqual(m["per_class"]["a"], 0.5)
        self.assertAlmostEqual(m["mean_class_accuracy"], (0.5 + 2 / 3 + 1) / 3)
        self.assertEqual({(c["true"], c["pred"]) for c in m["top_confusions"]}, {("a", "b"), ("b", "c")})
        self.assertEqual(confusion_matrix(np.array([2]), np.array([0]), 3)[2, 0], 1)


class TestKeyframes(unittest.TestCase):
    def test_three_frames_inside_each_shot(self):
        shots = [Shot(0, 0, 99, 0.0, 4.1), Shot(1, 100, 101, 4.2, 4.3)]
        k = keyframe_indices(shots)
        self.assertEqual(k[0], [25, 50, 74])
        self.assertTrue(all(100 <= i <= 101 for i in k[1]))


@unittest.skipIf(torch is None, "PyTorch not installed")
class TestResNet(unittest.TestCase):
    def setUp(self):
        from cinescope.framing.resnet import count_parameters, resnet18

        self.resnet18, self.count = resnet18, count_parameters

    def test_parameter_count_matches_the_paper_model(self):
        self.assertEqual(self.count(self.resnet18(1000)), 11_689_512)

    def test_any_input_size(self):
        m = self.resnet18(len(SCALES)).eval()
        with torch.no_grad():
            self.assertEqual(tuple(m(torch.randn(2, 3, 224, 384)).shape), (2, 5))
            self.assertEqual(tuple(m(torch.randn(1, 3, 64, 112)).shape), (1, 5))

    def test_zero_init_residual_makes_blocks_identity(self):
        from cinescope.framing.resnet import BasicBlock

        m = self.resnet18(5, zero_init_residual=True).eval()
        block = m.layer1[0]
        self.assertIsInstance(block, BasicBlock)
        x = torch.rand(1, 64, 8, 8)                       # non-negative, like post-ReLU activations
        with torch.no_grad():
            torch.testing.assert_close(block(x), x)

    @unittest.skipIf(torchvision is None, "torchvision not installed")
    def test_same_network_as_torchvision(self):
        ref = torchvision.models.resnet18(weights=None).eval()
        ours = self.resnet18(1000).eval()
        self.assertEqual(list(ours.state_dict()), list(ref.state_dict()))
        ours.load_state_dict(ref.state_dict())
        x = torch.randn(2, 3, 96, 160)
        with torch.no_grad():
            torch.testing.assert_close(ours(x), ref(x))


@unittest.skipIf(torch is None, "PyTorch not installed")
class TestTrainingPieces(unittest.TestCase):
    def test_warmup_cosine(self):
        from cinescope.framing.train import warmup_cosine

        f = warmup_cosine(total_steps=100, warmup_steps=10)
        self.assertAlmostEqual(f(0), 0.1)
        self.assertAlmostEqual(f(9), 1.0)
        self.assertAlmostEqual(f(10), 1.0)
        self.assertAlmostEqual(f(55), 0.5)
        self.assertAlmostEqual(f(100), 0.0)
        self.assertTrue(all(f(s) >= f(s + 1) for s in range(10, 100)))

    def test_no_weight_decay_on_bn_and_bias(self):
        from cinescope.framing.resnet import resnet18
        from cinescope.framing.train import param_groups

        m = resnet18(5)
        decay, no_decay = param_groups(m, 5e-4)
        self.assertTrue(all(p.ndim == 4 or p.ndim == 2 for p in decay["params"]))
        self.assertTrue(all(p.ndim == 1 for p in no_decay["params"]))       # BN scales/shifts, fc bias
        self.assertEqual(sum(p.numel() for g in (decay, no_decay) for p in g["params"]),
                         sum(p.numel() for p in m.parameters()))

    def test_synthetic_run_end_to_end(self):
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "p1_cnn"))
        from train_framing import main

        with tempfile.TemporaryDirectory() as t:
            res = main(["--synthetic", "--synthetic-n", "96", "--epochs", "2", "--bs", "16",
                        "--tag", "smoke", "--out", t, "--ckpt-dir", t])
            self.assertTrue((Path(t) / "smoke" / "metrics.json").exists())
            self.assertTrue((Path(t) / "smoke" / "confusion.png").exists())
            self.assertTrue((Path(t) / "smoke.pt").exists())
        self.assertEqual(len(res["test"]["confusion"]), len(SCALES))
        self.assertEqual(res["test"]["n"], 24)


if __name__ == "__main__":
    unittest.main()
