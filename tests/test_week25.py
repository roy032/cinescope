"""Week 25: the full-clip pipeline (stage registry, shared decoding, cache, resume) and the speed-ups."""
import json
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np

from cinescope.pipeline import STAGES, Interrupted, Options, Stage, Stream, register, run
from cinescope.pipeline.core import _Sampler, clip_hash, resolve, stage_key
from cinescope.pipeline.store import DiskArray

CHEAP = ["shots", "palette", "camera", "frames"]


def synthetic_clip(path: Path, fps: float = 24.0) -> None:
    """Three 'shots' of 30 frames: flat colours with a moving bright square, so
    cuts are obvious and the camera stage has something to track."""
    from cinescope.io.video import write_video

    rng = np.random.default_rng(0)
    frames = []
    for colour in ((200, 40, 40), (30, 160, 60), (40, 60, 200)):
        base = np.clip(np.array(colour) + rng.normal(0, 6, (72, 128, 3)), 0, 255).astype(np.uint8)
        for k in range(30):
            f = base.copy()
            f[20:40, 10 + 2 * k:30 + 2 * k] = 250
            frames.append(f)
    write_video(path, np.stack(frames), fps)


class PipelineCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        __import__("pytest").importorskip("av")
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.video = cls.root / "synth.mp4"
        synthetic_clip(cls.video)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def out(self, name: str) -> Path:
        return self.root / name


class TestRun(PipelineCase):
    def test_report_matches_phase0(self):
        from cinescope.shots.pipeline import analyse

        rep = run(self.video, CHEAP, self.out("a"), log=None)
        ref = analyse(self.video, with_camera=True)
        self.assertEqual([(s.start, s.end) for s in ref.shots], [(s["start"], s["end"]) for s in rep["shots"]])
        self.assertEqual(len(rep["shots"]), 3)
        self.assertEqual([c["label"] for c in ref.camera], [s["camera"]["label"] for s in rep["shots"]])
        self.assertEqual(ref.palettes[1]["rgb"], rep["shots"][1]["palette"]["rgb"])
        self.assertTrue((self.out("a") / "synth" / "report.json").exists())
        self.assertTrue(all(s["thumb"] for s in rep["shots"]))
        self.assertGreater(rep["timing"]["video_s_per_wall_s"], 0)

    def test_cache_and_invalidation(self):
        out = self.out("b")
        run(self.video, CHEAP, out, log=None)
        again = run(self.video, CHEAP, out, log=None)
        self.assertTrue(all(t["cached"] for t in again["timing"]["stages"].values()))
        self.assertEqual(again["timing"]["frames_decoded"], 0)          # nothing to decode
        changed = run(self.video, CHEAP, out, params={"palette": {"k": 3}}, log=None)
        st = changed["timing"]["stages"]
        self.assertFalse(st["palette"]["cached"])
        self.assertTrue(st["shots"]["cached"] and st["camera"]["cached"])
        self.assertEqual(len(changed["shots"][0]["palette"]["rgb"]), 3)

    def test_resume_after_interruption(self):
        out = self.out("c")
        opts = Options(checkpoint_s=0.5)
        with self.assertRaises(Interrupted):
            run(self.video, ["shots", "palette"], out, options=opts, log=None, stop=lambda n, t: n == 50)
        self.assertTrue((out / "synth" / "stages" / "_decode" / "checkpoint.pkl").exists())
        resumed = run(self.video, ["shots", "palette"], out, options=opts, log=None)
        self.assertLess(resumed["timing"]["frames_decoded"], 90)        # did not start over
        self.assertGreater(resumed["timing"]["frames_decoded"], 0)
        straight = run(self.video, ["shots", "palette"], self.out("c2"), log=None)
        self.assertEqual(json.dumps(resumed["shots"], sort_keys=True), json.dumps(straight["shots"], sort_keys=True))
        self.assertFalse((out / "synth" / "stages" / "_decode" / "checkpoint.pkl").exists())

    def test_threads_give_the_same_result(self):
        a = run(self.video, ["shots", "frames", "camera"], self.out("d1"), log=None)
        b = run(self.video, ["shots", "frames", "camera"], self.out("d2"), options=Options(workers=2), log=None)
        self.assertEqual([s["camera"] for s in a["shots"]], [s["camera"] for s in b["shots"]])
        self.assertEqual([s["end"] for s in a["shots"]], [s["end"] for s in b["shots"]])

    def test_new_stage_plugs_in(self):
        @register
        class CountFrames(Stage):
            name = "_count"
            requires = ("shots",)
            streams = {"slow": Stream(width=32, fps=6.0)}
            defaults = {"scale": 1}

            def begin(self, ctx):
                self.n, self.widths = 0, set()

            def frame(self, ctx, views):
                self.n += self.params["scale"]
                self.widths.add(views["slow"].rgb.shape[1])

            def end(self, ctx):
                k = len(ctx.results["shots"]["shots"])
                return {"n": self.n, "widths": sorted(self.widths), "per_shot": [self.n / k] * k,
                        "timeline": [{"t0": 0.0, "t1": 1.0, "label": "start"}]}

        try:
            rep = run(self.video, ["_count"], self.out("e"), log=None)
            r = json.loads((self.out("e") / "synth" / "stages" / "_count" / "result.json").read_text())["result"]
            self.assertEqual(r["widths"], [32])
            self.assertAlmostEqual(r["n"], 90 / 24 * 6, delta=1)            # 6 of every 24 frames
            self.assertIn("shots", rep["stages"])                           # pulled in by requires
            self.assertEqual(rep["shots"][0]["_count"], r["n"] / 3)        # report conventions
            self.assertEqual(rep["timelines"]["_count"][0]["label"], "start")
            self.assertEqual(rep["extra"]["_count"]["n"], r["n"])
        finally:
            STAGES.pop("_count")


    def test_pose_stage_with_a_stand_in_detector(self):
        """Phase 6 pose stage: batched detection during the pass, per-shot people counts."""
        from unittest import mock

        from cinescope.pose.yolo import People

        class FakePose:
            def __init__(self, *a, **k):
                pass

            def __call__(self, frames):
                return [People(np.array([[10.0, 10, 40, 60]]), np.array([0.9]), np.zeros((1, 17, 3)))
                        for _ in frames]

        with mock.patch("cinescope.pose.yolo.YoloPose", FakePose):
            rep = run(self.video, ["pose"], self.out("f"), params={"pose": {"model": "yolo", "batch": 5}}, log=None)
        self.assertEqual([s["pose"]["max_people"] for s in rep["shots"]], [1, 1, 1])
        self.assertEqual(rep["extra"]["pose"]["n_samples"], 15)          # 4 fps over 3.75 s


class TestPlanning(unittest.TestCase):
    def test_resolve_orders_dependencies(self):
        order = resolve(["characters"])
        self.assertLess(order.index("shots"), order.index("faces"))
        self.assertLess(order.index("speaking"), order.index("characters"))
        with self.assertRaises(KeyError):
            resolve(["no-such-stage"])

    def test_cycle_is_an_error(self):
        @register
        class A(Stage):
            name, requires = "_a", ("_b",)

        @register
        class B(Stage):
            name, requires = "_b", ("_a",)

        try:
            with self.assertRaises(ValueError):
                resolve(["_a"])
        finally:
            STAGES.pop("_a")
            STAGES.pop("_b")

    def test_key_depends_on_version_params_and_inputs(self):
        cls = STAGES["palette"]
        o = Options()
        k = stage_key("h", cls(), ["dep"], o)
        self.assertEqual(k, stage_key("h", cls(), ["dep"], o))
        self.assertNotEqual(k, stage_key("h", cls({"k": 4}), ["dep"], o))
        self.assertNotEqual(k, stage_key("h", cls(), ["dep2"], o))
        self.assertNotEqual(k, stage_key("h2", cls(), ["dep"], o))
        with self.assertRaises(ValueError):
            cls({"no_such_param": 1})

    def test_backend_only_matters_to_model_stages(self):
        o1, o2 = Options(backend="torch"), Options(backend="onnx")
        self.assertEqual(stage_key("h", STAGES["palette"](), [], o1), stage_key("h", STAGES["palette"](), [], o2))
        self.assertNotEqual(stage_key("h", STAGES["search"](), [], o1), stage_key("h", STAGES["search"](), [], o2))

    def test_sampler_matches_videoreader_rule(self):
        s = _Sampler(fps=8.0)
        t = np.arange(48) / 24.0
        took = [i for i, ti in enumerate(t) if s.take(ti)]
        self.assertEqual(took, list(range(0, 48, 3)))

    def test_clip_hash_reads_content(self):
        with tempfile.TemporaryDirectory() as d:
            a, b = Path(d) / "a.bin", Path(d) / "b.bin"
            a.write_bytes(b"x" * 3_000_000)
            b.write_bytes(b"x" * 2_999_999 + b"y")
            self.assertNotEqual(clip_hash(a), clip_hash(b))
            (Path(d) / "c.bin").write_bytes(a.read_bytes())
            self.assertEqual(clip_hash(a), clip_hash(Path(d) / "c.bin"))


class TestDiskArray(unittest.TestCase):
    def test_append_memmap_and_checkpoint_truncation(self):
        with tempfile.TemporaryDirectory() as d:
            arr = DiskArray(Path(d) / "x.u8", (4, 3))
            for i in range(5):
                arr.append(np.full((4, 3), i, np.uint8))
            blob = pickle.dumps(arr)                     # checkpoint at 5 items
            arr.append(np.full((4, 3), 9, np.uint8))
            arr.close()
            back = pickle.loads(blob)                    # resume: item 6 is dropped
            self.assertEqual(len(back), 5)
            back.append(np.full((4, 3), 7, np.uint8))
            a = back.array()
            self.assertEqual(a.shape, (6, 4, 3))
            self.assertEqual(a[:, 0, 0].tolist(), [0, 1, 2, 3, 4, 7])
            del a                                        # release the memmap: Windows cannot delete a mapped file
            back.close()
            with self.assertRaises(ValueError):
                DiskArray(Path(d) / "y.u8", (2, 2)).append(np.zeros((3, 3), np.uint8))


class TestFaceCropsInOnePass(unittest.TestCase):
    def test_asd_crop_from_stored_patch(self):
        """Cutting Light-ASD's crop from the stored patch reproduces asd_crop on the
        frame, also for a box moved by track smoothing and a box at the frame edge."""
        cv2 = __import__("pytest").importorskip("cv2")
        from cinescope.faces.align import asd_crop, asd_from_patch, asd_patch

        rng = np.random.default_rng(0)
        img = cv2.GaussianBlur(rng.integers(0, 255, (300, 500, 3)).astype(np.uint8), (0, 0), 3)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        for box, shift in (([200, 80, 260, 160], [0, 0, 0, 0]), ([200, 80, 260, 160], [4, -3, 4, -3]),
                           ([440, 230, 500, 300], [0, 0, 0, 0])):
            box = np.asarray(box, float)
            patch, geom = asd_patch(gray, box)
            moved = box + np.asarray(shift, float)
            diff = np.abs(asd_crop(img, moved).astype(int) - asd_from_patch(patch, geom, moved).astype(int))
            self.assertLess(diff.mean(), 1.5)


class TestNoFaces(unittest.TestCase):
    def test_timeline_without_tracks(self):
        """A clip where no face is tracked long enough gives an empty timeline, not an error."""
        from cinescope.faces.characters import timeline_from_crops

        r = timeline_from_crops("x.mp4", np.arange(10) / 24, 24.0, [0], [], emb=lambda c: c, speak={})
        self.assertEqual((r["characters"], r["n_tracks"], r["_thumbs"]), ([], 0, []))


class TestLightASDBatched(unittest.TestCase):
    def test_batched_equals_reference_recipe(self):
        """The batched recipe (each distinct chunk length once, weighted; full chunks
        in one batch) gives the reference recipe's scores. A cheap stand-in network
        whose output depends on the whole chunk, so a wrong chunking would show."""
        __import__("pytest").importorskip("torch")
        from cinescope.speaker.lightasd import LightASD

        def fake(a, v):                 # (4n, 13), (n, 112, 112) -> (n,), or batched with a leading B
            ctx = v.mean(axis=(-1, -2, -3), keepdims=True)[..., 0, 0] + a.mean(axis=(-1, -2))[..., None]
            frame = v[..., :8, :8].mean(axis=(-1, -2))
            return (frame - ctx).astype(np.float32)

        m = LightASD.__new__(LightASD)
        m.runner = fake
        rng = np.random.default_rng(0)
        n = 190                                           # 7.6 s: every duration with a remainder
        crops = rng.integers(0, 255, (n, 112, 112)).astype(np.uint8)
        t = np.arange(n) / 25.0 + 0.5
        audio = rng.normal(0, 0.1, 16000 * 9).astype(np.float32)
        ref = m.score(crops, t, audio, batched=False)
        fast = m.score(crops, t, audio)
        self.assertEqual(ref.shape, (n,))
        np.testing.assert_allclose(fast, ref, atol=1e-4)

    def test_network_batch_equals_one_at_a_time(self):
        """The real network: two chunks as one batch == each chunk on its own."""
        torch = __import__("pytest").importorskip("torch")
        from cinescope.speaker.lightasd import LightASD

        torch.manual_seed(0)
        torch.set_num_threads(2)
        m = LightASD(weights=None)
        rng = np.random.default_rng(0)
        a = rng.normal(0, 1, (2, 24, 13)).astype(np.float32)
        v = rng.integers(0, 255, (2, 6, 112, 112)).astype(np.float32)
        both = m._run(a, v)
        self.assertEqual(both.shape, (2, 6))
        for i in range(2):
            np.testing.assert_allclose(both[i], m._run(a[i], v[i]), atol=1e-4)


class TestOnnx(unittest.TestCase):
    def test_exported_module_matches_torch(self):
        import pytest

        torch = pytest.importorskip("torch")
        pytest.importorskip("onnxruntime")
        pytest.importorskip("onnx")
        from cinescope.framing.resnet import ResNet
        from cinescope.pipeline.accel import OnnxModule, compare

        torch.manual_seed(0)
        m = ResNet((1, 1, 1, 1), num_classes=5).eval()
        with tempfile.TemporaryDirectory() as d:
            om = OnnxModule(m, (torch.randn(2, 3, 64, 96),), ["x"], ["y"], {"x": {0: "b"}, "y": {0: "b"}},
                            tag="tiny", options=Options(onnx_dir=d))
            r = compare(m, om, (torch.randn(5, 3, 64, 96),))           # another batch size than traced
            self.assertLess(r["max_abs"], 1e-4)
            self.assertGreater(r["min_cosine"], 0.99999)
            again = OnnxModule(m, (torch.randn(2, 3, 64, 96),), ["x"], ["y"], {"x": {0: "b"}, "y": {0: "b"}},
                               tag="tiny", options=Options(onnx_dir=d))
            self.assertEqual(again.path, om.path)                       # cached by weight fingerprint


if __name__ == "__main__":
    unittest.main()
