"""The shareable one-file HTML breakdown (cinescope.editorial.breakdown)."""
import json
import re
import tempfile
import unittest
from pathlib import Path

import pytest
from test_editorial import make_report

from cinescope.editorial.breakdown import breakdown_html, save_breakdown


def full_report() -> dict:
    """make_report() plus what the barcode, look, editing and sound stages add."""
    rep = make_report()
    rep["barcode"] = {"colors": [[10, 20, 30], [200, 100, 50]] * 5}
    rep["characters"]["list"][0].update({"name": "Ana <lead>", "screen_time_s": 3.0, "speaking_time_s": 1.0})
    for i, s in enumerate(rep["shots"]):
        s["look"] = {"brightness": 0.2 + 0.1 * i, "warmth": -10 + 8 * i, "key": "low-key", "temperature": "cool"}
        s["editing"] = {"setup": i % 2, "recurs": True}
    rep["extra"] = {
        "look": {"summary": {"description": "Low-key, cool & moody.", "aspect_name": "2.39:1 scope",
                             "key": "low-key", "temperature": "cool"}},
        "editing": {"summary": {"setups": 2, "exchanges": 1}},
        "sound": {"audio": True, "summary": {"integrated_lufs": -24.0, "lra_lu": 7.5, "speech_share": 0.4},
                  "envelope": {"t0": 0.2, "dt": 0.1, "lufs": [-40.0, -20.0, -30.0]},
                  "timeline": [{"t0": 1.0, "t1": 2.0, "label": "peak"}]},
    }
    return rep


class TestBreakdown(unittest.TestCase):
    def test_page_has_every_section_and_is_self_contained(self):
        page = breakdown_html(full_report(), "/nonexistent")
        self.assertTrue(page.startswith("<!doctype html>"))
        for text in ("Low-key, cool &amp; moody.", "2.39:1 scope", "Movie barcode", "Pace", "Exposure",
                     "Colour temperature", "Loudness", "-24.0 LUFS", "Characters", "Ana &lt;lead&gt;",
                     "#3", "00:00:07:21", "set-up 2", "23.976 fps"):
            self.assertIn(text, page)
        # nothing is loaded from elsewhere: no external scripts, styles or images
        self.assertIsNone(re.search(r'(src|href)="(https?:)?//', page))
        self.assertNotIn("<script", page)

    def test_only_what_ran_is_shown(self):
        page = breakdown_html(make_report(), "/nonexistent")
        self.assertNotIn("Exposure", page)
        self.assertNotIn("Loudness", page)
        self.assertIn("Shots", page)

    def test_thumbnails_are_embedded(self):
        pytest.importorskip("PIL")
        from PIL import Image

        rep = make_report()
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "assets").mkdir()
            Image.new("RGB", (400, 200), (200, 30, 30)).save(root / "assets" / "0.jpg")
            rep["shots"][0]["thumb"] = "assets/0.jpg"
            dest = save_breakdown(rep, root, root / "exports" / "demo_breakdown.html")
            page = dest.read_text(encoding="utf-8")
        self.assertEqual(page.count("data:image/jpeg;base64,"), 1)

    def test_cli_and_api(self):
        from cinescope.editorial.__main__ import main

        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "demo"
            root.mkdir()
            (root / "report.json").write_text(json.dumps(full_report()), encoding="utf-8")
            self.assertEqual(main([str(root), "-f", "html"]), 0)
            self.assertTrue((root / "exports" / "demo_breakdown.html").exists())

        pytest.importorskip("fastapi")
        pytest.importorskip("httpx")
        pytest.importorskip("multipart")
        from fastapi.testclient import TestClient

        from app.backend.main import create_app

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "clips").mkdir()
            (root / "clips" / "demo.mp4").write_bytes(b"")
            (root / "out" / "demo").mkdir(parents=True)
            (root / "out" / "demo" / "report.json").write_text(json.dumps(full_report()), encoding="utf-8")
            c = TestClient(create_app(clip_dirs=[root / "clips"], out_dir=root / "out", frontend_dist=root / "x"))
            r = c.get("/api/clips/demo/export/html")
            self.assertEqual(r.status_code, 200)
            self.assertTrue(r.headers["content-type"].startswith("text/html"))
            self.assertIn('filename="demo_breakdown.html"', r.headers["content-disposition"])
            self.assertIn("Low-key, cool &amp; moody.", r.text)
