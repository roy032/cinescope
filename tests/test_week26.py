"""Week 26: the web app's back end (FastAPI), end to end on a synthetic clip.

A real pipeline job runs the cheap stages plus scene search with a stand-in
encoder (an image embeds as its mean colour), so no weights are needed.
"""
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np
import pytest

from tests.test_week25 import CHEAP, synthetic_clip

SPEC = "openclip:colour-test"      # an "openclip" spec, so the scenes stage runs its prompts
COLOURS = {"red": (200, 40, 40), "green": (30, 160, 60), "blue": (40, 60, 200)}


class ColourEncoder:
    """Stand-in for CLIP: an image embeds as its mean colour, "red ..." as red."""

    name = "colour-test"
    dim = 3

    def _n(self, x):
        return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)

    def encode_images(self, frames):
        return self._n(np.stack([f.reshape(-1, 3).mean(0) for f in frames]) + 1e-3)

    def encode_texts(self, texts):
        out = []
        for t in texts:
            v = np.full(3, 0.05)
            for i, c in enumerate(COLOURS):
                if c in t.lower():
                    v[i] = 1.0
            out.append(v)
        return self._n(np.stack(out))


def wait(client, job_id: str, timeout: float = 60.0) -> dict:
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = client.get(f"/api/jobs/{job_id}").json()
        if j["status"] not in ("queued", "running"):
            return j
        time.sleep(0.05)
    raise TimeoutError(job_id)


class TestBackend(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pytest.importorskip("fastapi")
        pytest.importorskip("httpx")
        pytest.importorskip("multipart")
        pytest.importorskip("av")
        from fastapi.testclient import TestClient

        from app.backend.main import create_app
        from cinescope.pipeline import accel

        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        (root / "clips").mkdir()
        synthetic_clip(root / "clips" / "synth.mp4")
        # the search stage asks accel for its encoder by spec; seed the cache with the stand-in
        accel._CACHE[("clip", SPEC, "torch")] = ColourEncoder()
        cls.app = create_app(clip_dirs=[root / "clips"], out_dir=root / "out", default_backend="torch",
                             encoder_factory=lambda spec: ColourEncoder(), frontend_dist=root / "no-dist")
        cls.c = TestClient(cls.app)
        r = cls.c.post("/api/jobs", json={"clip": "synth", "stages": [*CHEAP, "search", "scenes"],
                                          "params": {"search": {"encoder": SPEC, "tags": False}}})
        assert r.status_code == 200, r.text
        cls.job = wait(cls.c, r.json()["id"])

    @classmethod
    def tearDownClass(cls):
        from cinescope.pipeline import accel

        accel._CACHE.pop(("clip", SPEC, "torch"), None)
        cls.tmp.cleanup()

    def test_job_ran_and_clip_is_listed(self):
        self.assertEqual(self.job["status"], "done", self.job.get("error"))
        self.assertEqual(self.job["progress"], 1.0)
        rows = {r["clip"]: r for r in self.c.get("/api/clips").json()}
        self.assertTrue(rows["synth"]["processed"])
        self.assertEqual(rows["synth"]["n_shots"], 3)
        self.assertTrue(self.c.get("/api/health").json()["ok"])
        names = {s["name"] for s in self.c.get("/api/stages").json()}
        self.assertTrue({"shots", "faces", "subtitles", "search"} <= names)

    def test_report_and_assets(self):
        rep = self.c.get("/api/clips/synth/report").json()
        self.assertEqual(len(rep["shots"]), 3)
        self.assertEqual(len(rep["barcode"]["colors"]), 4)       # 3.75 s at 1 colour per second
        thumb = self.c.get(f"/api/clips/synth/files/{rep['shots'][1]['thumb']}")
        self.assertEqual(thumb.status_code, 200)
        self.assertEqual(thumb.content[:2], b"\xff\xd8")         # JPEG
        self.assertEqual(self.c.get(f"/api/clips/synth/files/{rep['barcode']['image']}").status_code, 200)
        self.assertEqual(self.c.get("/api/clips/synth/files/../../etc/passwd").status_code, 404)
        self.assertEqual(self.c.get("/api/clips/nope/report").status_code, 404)
        self.assertEqual(self.c.get("/api/clips/synth/campath").status_code, 404)  # stage not run
        # Phase 6 scene types through the report's generic conventions
        self.assertTrue(all(s["scenes"]["type"] for s in rep["shots"]))
        self.assertEqual(rep["timelines"]["scenes"][0]["t0"], 0.0)

    def test_frame_and_video(self):
        f = self.c.get("/api/clips/synth/frame", params={"t": 1.5, "width": 64})
        self.assertEqual(f.status_code, 200)
        self.assertEqual(f.headers["content-type"], "image/jpeg")
        import io

        from PIL import Image

        rgb = np.asarray(Image.open(io.BytesIO(f.content)).convert("RGB"))
        self.assertEqual(rgb.shape[1], 64)
        self.assertGreater(rgb[..., 1].mean(), rgb[..., 0].mean())   # 1.5 s is in the green shot
        v = self.c.get("/api/clips/synth/video", headers={"Range": "bytes=0-99"})
        self.assertEqual(v.status_code, 206)
        self.assertEqual(len(v.content), 100)

    def test_search_finds_the_shot(self):
        r = self.c.get("/api/clips/synth/search", params={"q": "something blue", "k": 3}).json()
        self.assertEqual(r["hits"][0]["shot"], 2)
        self.assertTrue(r["hits"][0]["frames"][0].startswith("/api/clips/synth/files/"))
        a = self.c.get("/api/clips/synth/ask", params={"q": "where is the red one?"})
        self.assertEqual(a.status_code, 200)

    def test_cutout_on_demand(self):
        """Phase 4 through the app: GrabCut (ours) + matting on the bright square, box given."""
        import io

        from PIL import Image

        r = self.c.get("/api/clips/synth/cutout", params={"t": 0.0, "box": "4,14,36,46", "width": 128})
        self.assertEqual(r.status_code, 200, r.text)
        im = np.asarray(Image.open(io.BytesIO(r.content)))
        self.assertEqual(im.shape[2], 4)                                # RGBA
        opaque = im[..., 3] > 128
        self.assertGreater(opaque.sum(), 0.6 * 20 * 20)
        self.assertGreater(im[..., :3][opaque].mean(), 200)             # the square, not the background
        self.assertEqual(self.c.get("/api/clips/synth/cutout", params={"method": "nope"}).status_code, 400)

    def test_job_plan_and_filter(self):
        """The job lists its steps in run order, so the app can show a checklist."""
        plan = self.job["plan"]
        self.assertEqual(plan[0], "decode")
        self.assertLess(plan.index("shots"), plan.index("search"))
        self.assertTrue({"shots", "frames", "search", "scenes"} <= set(plan))
        self.assertEqual(self.job["stage_progress"].get("shots"), 1.0)
        self.assertIn(self.job["id"], [j["id"] for j in self.c.get("/api/jobs", params={"clip": "synth"}).json()])
        self.assertEqual(self.c.get("/api/jobs", params={"clip": "nope"}).json(), [])
        stages = {s["name"]: s for s in self.c.get("/api/stages").json()}
        self.assertTrue(stages["shots"]["default"])
        self.assertFalse(stages["cutouts"]["default"])

    def test_second_job_is_cached(self):
        r = self.c.post("/api/jobs", json={"clip": "synth", "stages": ["palette"]}).json()
        j = wait(self.c, r["id"])
        self.assertEqual(j["status"], "done")
        rep = self.c.get("/api/clips/synth/report").json()
        self.assertTrue(rep["timing"]["stages"]["palette"]["cached"])
        self.assertEqual(rep["timing"]["frames_decoded"], 0)       # nothing left to decode

    def test_upload_and_errors(self):
        data = (Path(self.tmp.name) / "clips" / "synth.mp4").read_bytes()
        r = self.c.post("/api/clips/upload", files={"file": ("my clip.mp4", data, "video/mp4")})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["clip"], "my_clip")
        self.assertIn("my_clip", {x["clip"] for x in self.c.get("/api/clips").json()})
        # a second upload of the same name, or one named like a clip in a clip folder, gets a new
        # name: it must neither overwrite the first upload nor hide the folder's clip
        again = self.c.post("/api/clips/upload", files={"file": ("my clip.mp4", data, "video/mp4")}).json()
        self.assertEqual(again["clip"], "my_clip-2")
        clash = self.c.post("/api/clips/upload", files={"file": ("synth.mp4", data, "video/mp4")}).json()
        self.assertEqual(clash["clip"], "synth-2")
        rows = {x["clip"]: x for x in self.c.get("/api/clips").json()}
        self.assertFalse(rows["synth"]["uploaded"])
        self.assertTrue(rows["synth-2"]["uploaded"])
        bad = self.c.post("/api/clips/upload", files={"file": ("notes.txt", b"hi", "text/plain")})
        self.assertEqual(bad.status_code, 400)
        self.assertEqual(self.c.post("/api/jobs", json={"clip": "nope"}).status_code, 404)
        self.assertEqual(self.c.get("/api/jobs/nope").status_code, 404)


class TestProgress(unittest.TestCase):
    def test_plan_progress(self):
        pytest.importorskip("fastapi")
        from app.backend.main import plan_progress

        plan = ["decode", "shots", "palette", "search"]
        self.assertEqual(plan_progress(plan, {}), 0.0)
        self.assertEqual(plan_progress(plan, {"decode": 0.5}), 0.125)
        # decode and shots never reported before palette did: cached, so counted as done
        self.assertEqual(plan_progress(plan, {"palette": 0.5}), 0.625)
        self.assertEqual(plan_progress(plan, {"decode": 1, "shots": 1, "palette": 1, "search": 1}), 0.99)
        self.assertEqual(plan_progress([], {"x": 1}), 0.0)
        # never goes backwards as more steps report
        seen, last = {}, 0.0
        for step in plan:
            for f in (0.0, 0.5, 1.0):
                seen[step] = f
                p = plan_progress(plan, seen)
                self.assertGreaterEqual(p, last)
                last = p


class TestCancel(unittest.TestCase):
    def test_cancel_stops_the_job(self):
        pytest.importorskip("fastapi")
        pytest.importorskip("httpx")
        pytest.importorskip("multipart")
        from fastapi.testclient import TestClient

        from app.backend.main import create_app
        from cinescope.pipeline import Interrupted

        def runner(stop, progress, **kw):
            n = 0
            while not stop(n, n / 24):
                progress("decode", min(n / 1e4, 0.9), "")
                n += 1
                time.sleep(0.001)
            raise Interrupted("stopped")

        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "a.mp4").write_bytes(b"\0" * 10)
            c = TestClient(create_app(clip_dirs=[d], out_dir=Path(d) / "out", runner=runner,
                                      frontend_dist=Path(d) / "x"))
            j = c.post("/api/jobs", json={"clip": "a"}).json()
            again = c.post("/api/jobs", json={"clip": "a"}).json()
            self.assertEqual(again["id"], j["id"])                  # one job per clip at a time
            c.post(f"/api/jobs/{j['id']}/cancel")
            self.assertEqual(wait(c, j["id"])["status"], "cancelled")


if __name__ == "__main__":
    unittest.main()
