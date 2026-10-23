"""Editorial hand-off: timecode, the EDL / FCPXML / OTIO / CSV / WebVTT exports,
the storyboard and the pace, on a hand-made report (no video or models needed)."""
import csv
import io
import json
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from cinescope.editorial import (
    Rate,
    cut_list,
    edl,
    fcpxml,
    from_timecode,
    otio,
    pacing,
    shot_csv,
    to_timecode,
    webvtt,
)


def make_report(fps: float = 24000 / 1001, times=(0.0, 2.5, 3.1, 7.9), duration: float = 10.0) -> dict:
    t = [*times, duration]
    shots = []
    for i in range(len(times)):
        shots.append({"index": i, "start": round(t[i] * fps), "end": round(t[i + 1] * fps) - 1, "t0": t[i],
                      "t1": t[i + 1], "duration_s": round(t[i + 1] - t[i], 3), "thumb": None,
                      "framing": {"label": ["CU", "MS", "WS", "MCU"][i % 4], "name": ["close-up", "medium",
                                                                                        "wide", "medium close-up"][i % 4]},
                      "camera": {"label": ["static", "pan left", "unknown", "tilt up"][i % 4]},
                      "scenes": {"type": "dialogue" if i < 2 else "action"}})
    return {
        "clip": "demo", "video": "demo.mp4", "video_path": "/films/demo & co.mp4",
        "info": {"width": 1920, "height": 800, "fps": round(fps, 4), "duration_s": duration},
        "stages": ["shots"], "shots": shots,
        "characters": {"list": [{"id": "C1", "on_screen": [[0.0, 3.0]]}, {"id": "C2", "on_screen": [[7.5, 10.0]]}]},
        "subtitles": {"cues": [{"t0": 0.5, "t1": 2.0, "text": "Hello <there> & bye"},
                               {"t0": 3.2, "t1": 4.0, "text": "Second"}]},
    }


class TestTimecode(unittest.TestCase):
    def test_rates(self):
        self.assertEqual(Rate.from_fps(23.976023976), Rate(24000, 1001))
        self.assertEqual(Rate.from_fps(29.97), Rate(30000, 1001))
        self.assertEqual(Rate.from_fps(25.0), Rate(25, 1))
        self.assertEqual(Rate.from_fps(12.5), Rate(25, 2))
        self.assertEqual(Rate(24000, 1001).frame_duration, "1001/24000s")
        self.assertEqual(Rate(25, 1).frame_duration, "1/25s")
        self.assertEqual(Rate(24000, 1001).rational(48), "1001/500s")
        self.assertEqual(Rate(25, 1).rational(50), "2s")
        with self.assertRaises(ValueError):
            Rate.from_fps(0)

    def test_non_drop(self):
        r = Rate(24000, 1001)
        self.assertEqual(to_timecode(0, r), "00:00:00:00")
        self.assertEqual(to_timecode(24 * 3600 + 23, r), "01:00:00:23")
        for f in (0, 1, 23, 24, 86399, 123457):
            self.assertEqual(from_timecode(to_timecode(f, r), r), f)

    def test_drop_frame_known_values(self):
        r = Rate(30000, 1001)
        self.assertEqual(to_timecode(1799, r, True), "00:00:59;29")
        self.assertEqual(to_timecode(1800, r, True), "00:01:00;02")     # ;00 and ;01 are skipped
        self.assertEqual(to_timecode(17982, r, True), "00:10:00;00")    # ... except every tenth minute
        self.assertEqual(to_timecode(107892, r, True), "01:00:00;00")
        for f in range(0, 40000, 997):
            self.assertEqual(from_timecode(to_timecode(f, r, True), r), f)
        r60 = Rate(60000, 1001)
        self.assertEqual(to_timecode(3600, r60, True), "00:01:00;04")
        with self.assertRaises(ValueError):
            to_timecode(10, Rate(25, 1), True)


class TestExports(unittest.TestCase):
    def setUp(self):
        self.rep = make_report()

    def test_cut_list_tiles_the_clip(self):
        rows = cut_list(self.rep)
        self.assertEqual([r["f0"] for r in rows], [0, 60, 74, 189])
        for a, b in zip(rows, rows[1:], strict=False):
            self.assertEqual(a["f1"], b["f0"])
        self.assertEqual(rows[-1]["f1"], 240)
        self.assertEqual(rows[0]["characters"], ["C1"])
        self.assertEqual(rows[3]["characters"], ["C2"])
        self.assertIsNone(rows[2]["camera"])            # "unknown" is not a label

    def test_edl(self):
        text = edl(self.rep)
        lines = text.splitlines()
        self.assertEqual(lines[0], "TITLE: demo CineScope cuts")
        self.assertEqual(lines[1], "FCM: NON-DROP FRAME")
        ev = [ln for ln in lines if ln[:3].isdigit()]
        self.assertEqual(len(ev), 4)
        self.assertEqual(ev[0], "001  AX       V     C        00:00:00:00 00:00:02:12 01:00:00:00 01:00:02:12")
        self.assertTrue(ev[3].endswith("01:00:07:21 01:00:10:00"))
        self.assertIn("* FROM CLIP NAME: demo.mp4", lines)
        self.assertIn("* COMMENT: SHOT 0 | close-up | static | dialogue | C1", lines)
        # columns stay fixed width, as CMX parsers expect
        self.assertEqual({len(e) for e in ev}, {len(ev[0])})

    def test_edl_drop_frame(self):
        rep = make_report(fps=29.97)
        text = edl(rep)
        self.assertIn("FCM: DROP FRAME", text)
        self.assertIn("01:00:00;00", text)
        self.assertIn("NON-DROP", edl(rep, drop_frame=False))

    def test_fcpxml(self):
        root = ET.fromstring(fcpxml(self.rep).split("\n", 2)[2])
        self.assertEqual(root.tag, "fcpxml")
        self.assertEqual(root.get("version"), "1.10")
        fmt = root.find("resources/format")
        self.assertEqual(fmt.get("frameDuration"), "1001/24000s")
        src = root.find("resources/asset/media-rep").get("src")
        self.assertEqual(src, "file:///films/demo%20%26%20co.mp4")
        clips = root.findall(".//spine/asset-clip")
        self.assertEqual(len(clips), 4)
        self.assertEqual(clips[1].get("offset"), "1001/400s")   # frame 60 at 24000/1001
        self.assertEqual(clips[1].get("offset"), clips[1].get("start"))
        self.assertEqual(clips[0].find("keyword").get("value"), "close-up, static, dialogue, C1")
        chapters = [c.find("chapter-marker") for c in clips]
        self.assertEqual([c.get("value") if c is not None else None for c in chapters],
                         ["dialogue", None, "action", None])
        seq = root.find(".//sequence")
        self.assertEqual(seq.get("duration"), Rate(24000, 1001).rational(240))

    def test_fcpxml_windows_path(self):
        rep = make_report()
        rep["video_path"] = r"C:\clips\my film.mp4"
        self.assertIn('src="file:///C:/clips/my%20film.mp4"', fcpxml(rep))

    def test_otio_json(self):
        tl = json.loads(otio(self.rep))
        self.assertEqual(tl["OTIO_SCHEMA"], "Timeline.1")
        clips = tl["tracks"]["children"][0]["children"]
        self.assertEqual(len(clips), 4)
        self.assertEqual(clips[1]["source_range"]["start_time"]["value"], 60.0)
        self.assertEqual(clips[1]["source_range"]["duration"]["value"], 14.0)
        self.assertEqual(clips[0]["metadata"]["cinescope"]["characters"], ["C1"])
        self.assertEqual(clips[2]["markers"][0]["name"], "action")

    def test_otio_loads_in_opentimelineio(self):
        o = pytest.importorskip("opentimelineio")
        tl = o.adapters.read_from_string(otio(self.rep), "otio_json")
        clips = list(tl.find_clips())
        self.assertEqual(len(clips), 4)
        self.assertAlmostEqual(tl.duration().to_seconds(), 240 * 1001 / 24000, places=6)
        self.assertEqual(clips[2].source_range.start_time.value, 74)

    def test_csv(self):
        rows = list(csv.DictReader(io.StringIO(shot_csv(self.rep))))
        self.assertEqual(len(rows), 4)
        self.assertEqual(rows[1]["tc_in"], "00:00:02:12")
        self.assertEqual(rows[1]["frames"], "14")
        self.assertEqual(rows[0]["characters"], "C1")
        self.assertEqual(rows[2]["framing"], "wide")

    def test_webvtt(self):
        text = webvtt(self.rep)
        self.assertTrue(text.startswith("WEBVTT\n\n1\n00:00:00.500 --> 00:00:02.000\n"))
        self.assertIn("Hello &lt;there&gt; &amp; bye", text)
        self.assertEqual(webvtt({"info": {}}), "WEBVTT\n")

    def test_no_shots(self):
        rep = make_report()
        rep["shots"] = []
        self.assertEqual(cut_list(rep), [])
        self.assertEqual(len([ln for ln in edl(rep).splitlines() if ln[:3].isdigit()]), 0)
        ET.fromstring(fcpxml(rep).split("\n", 2)[2])
        self.assertIsNone(pacing(rep))


class TestPacing(unittest.TestCase):
    def test_numbers(self):
        p = pacing(make_report())
        self.assertEqual(p["n_shots"], 4)
        self.assertAlmostEqual(p["asl_s"], 2.5)
        self.assertAlmostEqual(p["median_s"], (2.1 + 2.5) / 2)
        self.assertAlmostEqual(p["cuts_per_min"], 18.0)
        self.assertEqual(len(p["curve"]["t"]), 200)
        self.assertEqual(sum(b["n"] for b in p["length_histogram"]), 4)
        self.assertGreaterEqual(p["fastest"]["cuts_per_min"], p["slowest"]["cuts_per_min"])

    def test_fast_passage_is_found(self):
        times = [0.0, 20.0] + [20.0 + 0.5 * k for k in range(1, 20)] + [40.0]
        p = pacing(make_report(fps=25, times=times, duration=60.0))
        self.assertTrue(20 <= p["fastest"]["t"] <= 31, p["fastest"])
        self.assertLess(p["slowest"]["cuts_per_min"], 1e-6)


class TestSheetAndCli(unittest.TestCase):
    def test_storyboard_pages(self):
        pytest.importorskip("PIL")
        from PIL import Image

        from cinescope.editorial.sheet import contact_sheet, save_sheet

        rep = make_report(fps=25, times=[float(i) for i in range(20)], duration=21.0)
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "assets").mkdir()
            Image.new("RGB", (192, 80), (200, 30, 30)).save(root / "assets" / "0.jpg")
            rep["shots"][0]["thumb"] = "assets/0.jpg"
            pages = contact_sheet(rep, root)
            self.assertEqual(len(pages), 2)                      # 20 shots at 16 a page
            self.assertEqual(pages[0].size, (2339, 1653))
            # the first cell shows the thumbnail
            self.assertGreater(pages[0].getpixel((100, 260))[0], 180)
            pdf = save_sheet(rep, root, root / "x" / "sb.pdf")
            self.assertEqual(pdf[0].read_bytes()[:5], b"%PDF-")
            pngs = save_sheet(rep, root, root / "sb.png")
            self.assertEqual([p.name for p in pngs], ["sb.png", "sb-2.png"])

    def test_cli(self):
        from cinescope.editorial.__main__ import main

        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "demo"
            root.mkdir()
            (root / "report.json").write_text(json.dumps(make_report()), encoding="utf-8")
            self.assertEqual(main([str(root), "-f", "edl", "fcpxml", "otio", "csv", "vtt"]), 0)
            self.assertEqual(sorted(p.name for p in (root / "exports").iterdir()),
                             ["demo.csv", "demo.edl", "demo.fcpxml", "demo.otio", "demo.vtt"])
            self.assertEqual(main([str(Path(d) / "missing")]), 2)


class TestExportApi(unittest.TestCase):
    def test_endpoints(self):
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
            (root / "out" / "demo" / "report.json").write_text(json.dumps(make_report()), encoding="utf-8")
            c = TestClient(create_app(clip_dirs=[root / "clips"], out_dir=root / "out", frontend_dist=root / "x"))
            r = c.get("/api/clips/demo/export/edl")
            self.assertEqual(r.status_code, 200)
            self.assertIn('attachment; filename="demo.edl"', r.headers["content-disposition"])
            self.assertTrue(r.text.startswith("TITLE: demo"))
            r = c.get("/api/clips/demo/export/vtt?download=false")
            self.assertTrue(r.headers["content-type"].startswith("text/vtt"))
            self.assertTrue(r.headers["content-disposition"].startswith("inline"))
            self.assertEqual(c.get("/api/clips/demo/export/fcpxml").status_code, 200)
            self.assertEqual(c.get("/api/clips/demo/export/otio").status_code, 200)
            self.assertEqual(c.get("/api/clips/demo/export/csv").status_code, 200)
            self.assertEqual(c.get("/api/clips/demo/export/pdf").content[:5], b"%PDF-")
            self.assertEqual(c.get("/api/clips/demo/export/mov").status_code, 400)
            self.assertEqual(c.get("/api/clips/nope/export/edl").status_code, 404)


def test_edl_and_fcpxml_read_back_by_opentimelineio_adapters():
    """The EDL and FCPXML land on the same frames when OpenTimelineIO's own readers parse them
    (otio-cmx3600-adapter, otio-fcpx-xml-adapter). The FCPXML reader truncates NTSC rates
    (24000/1001 -> 23 fps), so that half runs on a 25 fps report."""
    o = pytest.importorskip("opentimelineio")
    names = o.adapters.available_adapter_names()

    def ranges(tl):
        return [(round(c.source_range.start_time.value), round(c.source_range.duration.value))
                for c in tl.find_clips()]

    def want(rep):
        return [(s["start"], s["end"] - s["start"] + 1) for s in rep["shots"]]

    if "cmx_3600" in names:
        rep = make_report()
        tl = o.adapters.read_from_string(edl(rep), "cmx_3600", rate=24000 / 1001)
        assert ranges(tl) == want(rep)
    if "fcpx_xml" in names:
        rep = make_report(fps=25.0)
        tl = o.adapters.read_from_string(fcpxml(rep), "fcpx_xml")
        assert ranges(tl) == want(rep)


@pytest.mark.parametrize("path, url", [
    ("/films/demo & co.mp4", "file:///films/demo%20%26%20co.mp4"),
    ("C:\\Users\\me\\My Films\\a.mkv", "file:///C:/Users/me/My%20Films/a.mkv"),
    ("\\\\nas\\media\\a.mp4", "file://nas/media/a.mp4"),
    ("clips\\a b.mp4", "clips/a%20b.mp4"),
    (None, ""),
])
def test_file_url_does_not_depend_on_the_os(path, url):
    """A report written on Linux may be exported on Windows and the other way round."""
    from cinescope.editorial.exports import _file_url

    assert _file_url(path) == url
