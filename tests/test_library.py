"""Batch runs, the library page and the cache commands."""
import tempfile
import unittest
from pathlib import Path

import pytest

from cinescope.pipeline.library import cache_table, clear, human_bytes, library_html, scan, videos
from tests.test_week25 import synthetic_clip


class TestVideos(unittest.TestCase):
    def test_files_and_folders(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "sub").mkdir()
            for n in ("b.mp4", "a.MKV", "sub/c.mov", "notes.txt"):
                (root / n).write_bytes(b"x")
            found = videos([root, root / "b.mp4"])
            self.assertEqual([p.name for p in found], ["a.MKV", "b.mp4", "c.mov"])   # no duplicates
            self.assertEqual([p.name for p in videos([root], recursive=False)], ["a.MKV", "b.mp4"])
            with self.assertRaises(FileNotFoundError):
                videos([root / "missing.mp4"])

    def test_human_bytes(self):
        self.assertEqual(human_bytes(512), "512 B")
        self.assertEqual(human_bytes(1536), "1.5 KB")
        self.assertEqual(human_bytes(3 * 2**30), "3.0 GB")


class TestBatch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pytest.importorskip("av")
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        clips = cls.root / "clips"
        clips.mkdir()
        synthetic_clip(clips / "one.mp4")
        synthetic_clip(clips / "two.mp4", fps=25.0)
        (clips / "broken.mp4").write_bytes(b"not a video")
        cls.out = cls.root / "out"
        from cinescope.pipeline.__main__ import main

        cls.result = main([str(clips), "--stages", "shots", "palette", "--out", str(cls.out), "--quiet"])

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_a_broken_file_does_not_stop_the_batch(self):
        r = self.result
        self.assertEqual(sorted(x["clip"] for x in r["reports"]), ["one", "two"])
        self.assertEqual(list(r["failed"]), [str(self.root / "clips" / "broken.mp4")])
        self.assertTrue(r["errors"])                       # -> exit status 1

    def test_library_page(self):
        page = (self.out / "library.html").read_text(encoding="utf-8")
        self.assertIn("2 analysed clips", page)
        self.assertIn('href="one/exports/one_breakdown.html"', page)
        self.assertTrue((self.out / "one" / "exports" / "one_breakdown.html").exists())
        self.assertIn("data:image/jpeg;base64,", page)      # the barcodes are embedded
        custom = library_html(scan(self.out), link=lambda c: f"/x/{c}")
        self.assertIn('href="/x/two"', custom)

    def test_scan_and_cache_table(self):
        entries = scan(self.out)
        self.assertEqual({e["clip"] for e in entries}, {"one", "two"})
        e = next(e for e in entries if e["clip"] == "one")
        self.assertEqual(e["facts"]["shots"], 3)
        self.assertTrue(e["video_found"])
        self.assertGreater(e["disk"]["stages"]["shots"], 0)
        table = cache_table(entries)
        self.assertIn("one", table)
        self.assertIn("total", table)

    def test_clear(self):
        import shutil

        work = self.root / "clear_out"
        shutil.copytree(self.out / "two", work / "two")
        removed = clear(work, "two", ["palette"])
        self.assertFalse((work / "two" / "stages" / "palette").exists())
        self.assertTrue((work / "two" / "stages" / "shots").exists())
        self.assertFalse((work / "two" / "report.json").exists())       # it described the palette
        self.assertEqual(len(removed), 2)
        with self.assertRaises(FileNotFoundError):
            clear(work, "two")                                             # no report any more
        with self.assertRaises(FileNotFoundError):
            clear(work, "nope")

    def test_library_api(self):
        pytest.importorskip("fastapi")
        pytest.importorskip("httpx")
        pytest.importorskip("multipart")
        from fastapi.testclient import TestClient

        from app.backend.main import create_app

        c = TestClient(create_app(clip_dirs=[self.root / "clips"], out_dir=self.out, frontend_dist=self.root / "x"))
        r = c.get("/api/library")
        self.assertEqual(r.status_code, 200)
        self.assertIn('href="/api/clips/one/export/html?download=false"', r.text)


if __name__ == "__main__":
    unittest.main()
