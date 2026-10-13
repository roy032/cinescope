"""Week 22: text map, subtitle tracking and .srt output, credits scrolling,
scoring. Synthetic frames only; the recogniser is replaced by a stub."""
import random
import unittest

import numpy as np
from PIL import Image, ImageDraw

from cinescope.ocr.credits import CreditsReader, score_lines, scroll_shift
from cinescope.ocr.srt import Cue, match, parse_srt, schedule, score, wrap, write_srt
from cinescope.ocr.subtitles import SubtitleParams, SubtitleReader, merge_cues
from cinescope.ocr.synth import load_font
from cinescope.ocr.textmap import TextParams, detect_lines, split_rows

H, W = 270, 480


def _background(rng: np.random.Generator, t: int) -> np.ndarray:
    """Smooth moving texture: brightness ramps plus a drifting blob."""
    yy, xx = np.mgrid[0:H, 0:W]
    img = 60 + 40 * np.sin((xx + 7 * t) / 37.0) + 30 * np.cos((yy - 3 * t) / 23.0)
    img = np.clip(img + rng.normal(0, 3, img.shape), 0, 255)
    return np.repeat(img[:, :, None], 3, 2).astype(np.uint8)


def _draw(img: np.ndarray, lines: list[str], y_bottom: int, size: int = 16) -> np.ndarray:
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    font = load_font(None, size)
    for k, text in enumerate(reversed(lines)):
        y = y_bottom - (k + 1) * int(size * 1.35)
        d.text((W / 2, y), text, font=font, fill=(255, 255, 255), anchor="mt",
               stroke_width=2, stroke_fill=(0, 0, 0))
    return np.asarray(im)


def _by_width(crops):
    """Stub recogniser: names a crop by its width bucket (lines of different
    lengths -> different 'texts'), always confident."""
    return [(f"line{int(round(c.shape[1] / 40.0))}", 0.99) for c in crops]


class TestTextMap(unittest.TestCase):
    def test_finds_subtitle_and_not_texture(self):
        rng = np.random.default_rng(0)
        bg = _background(rng, 0)
        img = _draw(bg, ["The quick brown fox jumps"], y_bottom=H - 10)
        boxes, _ = detect_lines(img, TextParams(), frame_h=H)
        self.assertTrue(boxes)
        x0, y0, x1, y1 = max(boxes, key=lambda b: b[2] - b[0])
        self.assertGreater(y0, H * 0.75)
        self.assertGreater(x1 - x0, 120)
        self.assertLess(abs((x0 + x1) / 2 - W / 2), 30)
        self.assertEqual(detect_lines(bg, TextParams(), frame_h=H)[0], [])

    def test_split_rows(self):
        prof = np.array([0, 5, 9, 9, 8, 1, 0, 6, 9, 9, 7, 2], float)
        self.assertEqual(split_rows(prof, min_h=4), [(0, 6), (6, 12)])
        self.assertEqual(split_rows(np.array([1, 8, 9, 9, 9, 7, 1], float), min_h=4), [(0, 7)])


class TestSrt(unittest.TestCase):
    def test_roundtrip_and_tags(self):
        cues = [Cue(1.5, 3.25, "Hello there"), Cue(3661.001, 3662.0, "two\nlines")]
        back = parse_srt(write_srt(cues))
        self.assertEqual([(c.start, c.end, c.text) for c in back], [(c.start, c.end, c.text) for c in cues])
        self.assertEqual(parse_srt("1\n00:00:01,000 --> 00:00:02,000\n<i>Hi</i> {\\an8}you\n")[0].text,
                         "Hi you")

    def test_schedule_and_wrap(self):
        cues = schedule(["a" * 10, "b" * 60, "c" * 20], 30.0, random.Random(0))
        self.assertEqual(len(cues), 3)
        self.assertTrue(all(c.end > c.start for c in cues))
        self.assertTrue(all(a.end < b.start for a, b in zip(cues, cues[1:], strict=False)))
        self.assertEqual(wrap("one two three four five six seven eight nine ten"),
                         "one two three four five\nsix seven eight nine ten")

    def test_score(self):
        truth = [Cue(0, 2, "Hello"), Cue(3, 5, "World"), Cue(6, 8, "Missed")]
        pred = [Cue(0.1, 2.2, "Hallo"), Cue(3, 5, "World"), Cue(9, 10, "ghost")]
        self.assertEqual(match(truth, pred), [(0, 0), (1, 1)])
        r = score(truth, pred)
        self.assertEqual(r["matched"], 2)
        self.assertAlmostEqual(r["cer"], (1 + 6 + 5) / 16)
        self.assertAlmostEqual(r["start_err_s"], 0.05)
        self.assertAlmostEqual(r["end_err_s"], 0.1)
        self.assertEqual(score(truth, truth)["cer"], 0.0)

    def test_merge_duplicates(self):
        cues = [Cue(0, 1, "Same text"), Cue(1.2, 2, "Same text"), Cue(2.1, 3, "Other text here")]
        out = merge_cues(cues, max_gap=0.5, max_cer=0.3)
        self.assertEqual([(c.start, c.end, c.text) for c in out], [(0, 2, "Same text"), (2.1, 3, "Other text here")])


class TestSubtitleReader(unittest.TestCase):
    def test_two_subtitles_timed(self):
        """Subtitle A on samples 4-13, B (two lines) on 18-29, at 8 fps,
        over a moving background."""
        rng = np.random.default_rng(1)
        fps = 8.0
        frames = []
        for i in range(36):
            img = _background(rng, i)
            if 4 <= i <= 13:
                img = _draw(img, ["Short line"], H - 10)
            elif 18 <= i <= 29:
                img = _draw(img, ["A much longer second line", "and its second row"], H - 10)
            frames.append((i / fps, img))
        cues = SubtitleReader(_by_width, SubtitleParams()).run(frames, 1 / fps)
        self.assertEqual(len(cues), 2, cues)
        a, b = cues
        self.assertAlmostEqual(a.start, 4 / fps - 0.5 / fps, places=6)
        self.assertAlmostEqual(a.end, 13 / fps + 0.5 / fps, places=6)
        self.assertAlmostEqual(b.start, 18 / fps - 0.5 / fps, places=6)
        self.assertEqual(len(b.text.split("\n")), 2)
        self.assertNotEqual(a.text, b.text)


class TestCredits(unittest.TestCase):
    def test_scroll_shift(self):
        prof = np.random.default_rng(0).random(200)
        self.assertEqual(scroll_shift(prof, np.roll(prof, -7), 20), 7)
        self.assertEqual(scroll_shift(prof, prof, 20), 0)

    def test_rolling_credits(self):
        """Three lines of different lengths roll up 12 px per sample on black:
        each must come out once, in order."""
        page_h = 3 * H
        page = Image.new("RGB", (W, page_h), (0, 0, 0))
        d = ImageDraw.Draw(page)
        font = load_font(None, 18)
        texts = ["Short", "A medium credit line", "A rather long line of credits text"]
        for k, t in enumerate(texts):
            d.text((W / 2, H + 60 * k), t, font=font, fill=(255, 255, 255), anchor="mm")
        page = np.asarray(page)
        frames = [(i * 0.25, page[12 * i:12 * i + H]) for i in range(0, (page_h - H) // 12)]
        lines = CreditsReader(_by_width).run(frames)
        got = [ln["text"] for ln in lines]
        self.assertEqual(len(got), 3, got)
        self.assertEqual(len(set(got)), 3)
        widths = [int(g[4:]) for g in got]
        self.assertEqual(widths, sorted(widths))

    def test_score_lines(self):
        r = score_lines(["Directed by", "Jane Doe"], ["Directed by", "Jane Dee", "noise"])
        self.assertEqual(r["found"], 2)
        self.assertAlmostEqual(r["precision"], 2 / 3)


class TestLibraries(unittest.TestCase):
    def test_split_words(self):
        from cinescope.ocr.libs import split_words

        img = np.zeros((32, 200, 3), np.uint8)
        im = Image.fromarray(img)
        ImageDraw.Draw(im).text((10, 6), "two    words", font=load_font(None, 18), fill=(255, 255, 255))
        self.assertEqual(len(split_words(np.asarray(im))), 2)


if __name__ == "__main__":
    unittest.main()
