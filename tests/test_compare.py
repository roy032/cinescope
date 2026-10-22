"""Comparing two reports (cinescope.editorial.compare): numbers, CLI, page and API."""
import json
import re
import tempfile
import unittest
from pathlib import Path

import pytest
from test_breakdown import full_report

from cinescope.editorial.compare import EXTENSIVE, compare, compare_html, curves, main, profile


def other_report() -> dict:
    rep = full_report()
    rep["clip"] = "other"
    rep["extra"]["look"]["summary"].update({"key": "high-key", "temperature": "warm"})
    rep["extra"]["sound"]["summary"]["integrated_lufs"] = -16.0
    rep["extra"]["rhythm"] = {"summary": {"bpm": 128.0, "rhythmic_share": 0.8,
                                          "cuts_on_beat": {"on_beat": 0.5, "chance": 0.3}}}
    for s in rep["shots"]:
        s["t1"] = s["t0"] + (s["t1"] - s["t0"]) / 2
    rep["pacing"] = None
    return rep


class TestCompare(unittest.TestCase):
    def test_profile(self):
        p = profile(full_report())
        self.assertIn("Shots", p)
        self.assertEqual(p["Lighting"], "low-key")
        self.assertEqual(p["Loudness"], -24.0)
        self.assertNotIn("Tempo", p)

    def test_self_comparison_is_identical(self):
        c = compare(full_report(), full_report())
        self.assertEqual(c["similarity"], 1.0)
        self.assertTrue(all(r["delta"] in (None, 0) for r in c["rows"]))

    def test_differences(self):
        a, b = full_report(), other_report()
        c = compare(a, b)
        self.assertLess(c["similarity"], 1.0)
        rows = {r["metric"]: r for r in c["rows"]}
        self.assertEqual(rows["Loudness"]["delta"], 8.0)
        self.assertIsNone(rows["Tempo"]["a"])
        self.assertEqual(rows["Tempo"]["b"], 128.0)
        self.assertEqual(rows["Lighting"]["b"], "high-key")
        self.assertTrue(c["headline"])
        self.assertFalse(any(h.split(":")[0] in EXTENSIVE for h in c["headline"]))

    def test_curves_are_on_relative_time(self):
        cv = curves(full_report())
        self.assertEqual(len(cv["pace"]), 100)

    def test_page(self):
        page = compare_html(full_report(), other_report())
        self.assertTrue(page.startswith("<!doctype html>"))
        for text in ("demo", "other", "Style similarity", "Side by side", "Pace", "128 BPM", "-16 LUFS"):
            self.assertIn(text, page)
        self.assertIsNone(re.search(r'(src|href)="(https?:)?//', page))
        self.assertNotIn("<script", page)


class TestCompareCliAndApi(unittest.TestCase):
    def test_cli(self):
        with tempfile.TemporaryDirectory() as d:
            for name, rep in (("a", full_report()), ("b", other_report())):
                (Path(d) / name).mkdir()
                (Path(d) / name / "report.json").write_text(json.dumps(rep), encoding="utf-8")
            dest = Path(d) / "cmp.html"
            self.assertEqual(main([str(Path(d) / "a"), str(Path(d) / "b"), "-o", str(dest)]), 0)
            self.assertIn("Side by side", dest.read_text(encoding="utf-8"))
            self.assertEqual(main([str(Path(d) / "a"), str(Path(d) / "missing")]), 2)

    def test_api(self):
        pytest.importorskip("fastapi")
        pytest.importorskip("httpx")
        pytest.importorskip("multipart")
        from fastapi.testclient import TestClient

        from app.backend.main import create_app

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "clips").mkdir()
            for name, rep in (("demo", full_report()), ("other", other_report())):
                (root / "out" / name).mkdir(parents=True)
                (root / "out" / name / "report.json").write_text(json.dumps(rep), encoding="utf-8")
            c = TestClient(create_app(clip_dirs=[root / "clips"], out_dir=root / "out", frontend_dist=root / "x"))
            r = c.get("/api/compare", params={"a": "demo", "b": "other"})
            self.assertEqual(r.status_code, 200)
            self.assertEqual(r.json()["b"], "other")
            r = c.get("/api/compare", params={"a": "demo", "b": "other", "format": "html"})
            self.assertTrue(r.headers["content-type"].startswith("text/html"))
            self.assertEqual(c.get("/api/compare", params={"a": "demo", "b": "nope"}).status_code, 404)
            self.assertEqual(c.get("/api/compare", params={"a": "demo", "b": "../out/demo"}).status_code, 400)
