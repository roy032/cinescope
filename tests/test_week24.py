"""Week 24: shot index on disk, hook fields, and retrieval-augmented answers (no weights, no network)."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from cinescope.search.rag import (
    BM25,
    RagParams,
    answer,
    fmt_t,
    question_kind,
    question_to_query,
    retrieve,
    template_answer,
)
from cinescope.search.shotindex import (
    IndexParams,
    ShotIndex,
    attach_characters,
    attach_subtitles,
    build_clip_index,
    parse_srt,
)
from cinescope.search.vecindex import IndexSpec, normalize

COLOURS = {"red": (230, 30, 30), "green": (30, 200, 40), "blue": (40, 60, 230)}


class ColourEncoder:
    """Stand-in for CLIP: an image embeds as its mean colour, the text "red ..." as red."""

    name = "colour-test"
    dim = 3

    def encode_images(self, frames):
        return normalize(np.stack([f.reshape(-1, 3).mean(0) for f in frames]) + 1e-3)

    def encode_texts(self, texts):
        out = []
        for t in texts:
            v = np.full(3, 0.05)
            for i, c in enumerate(COLOURS):
                if c in t.lower():
                    v[i] = 1.0
            out.append(v)
        return normalize(np.stack(out))


def fake_index(n=6):
    rng = np.random.default_rng(0)
    docs = [{"id": i, "clip": "c", "shot": i, "part": 0, "n_parts": 1, "t0": 2.0 * i, "t1": 2.0 * i + 2,
             "frames": [], "tags": [], "subtitles": [], "characters": []} for i in range(n)]
    vec = normalize(rng.standard_normal((n, 3)) * 0.02 + 1.0)
    vec[3] = normalize(np.array([1.0, 0.05, 0.05]))
    return ShotIndex(docs, vec, {"encoder": "colour-test"})


class TestQuestions(unittest.TestCase):
    def test_question_to_query(self):
        self.assertEqual(question_to_query("When does the car chase start?"), "the car chase")
        self.assertEqual(question_to_query("Is there a plane in the clip?"), "a plane")
        self.assertEqual(question_to_query("a dog on a beach"), "a dog on a beach")

    def test_question_kind(self):
        self.assertEqual(question_kind("When does the car chase start?"), "when")
        self.assertEqual(question_kind("who is at the computer?"), "who")
        self.assertEqual(question_kind("How many times do we see snow?"), "count")
        self.assertEqual(question_kind("Is there a plane?"), "yesno")
        self.assertEqual(question_kind("what happens in the snow"), "what")

    def test_fmt_t(self):
        self.assertEqual(fmt_t(75.25), "1:15.2")


class TestLexical(unittest.TestCase):
    def test_bm25_prefers_rare_matching_terms(self):
        bm = BM25(["the man walks", "a man and a dog", "a car explodes", "man man man"])
        s = bm.scores("dog")
        self.assertEqual(int(np.argmax(s)), 1)
        self.assertEqual(s[0], 0.0)
        # term-frequency saturation: three "man" do not score 3x one "man"
        s = bm.scores("man")
        self.assertLess(s[3], 3 * s[0])

    def test_srt_and_hooks(self):
        srt = "1\n00:00:01,000 --> 00:00:02,500\n<i>Hello</i> there\n\n2\n00:00:05,000 --> 00:00:06,000\nFor England\n"
        cues = parse_srt(srt)
        self.assertEqual(cues, [{"t0": 1.0, "t1": 2.5, "text": "Hello there"},
                                {"t0": 5.0, "t1": 6.0, "text": "For England"}])
        docs = [{"t0": 0.0, "t1": 3.0}, {"t0": 3.0, "t1": 5.5}]
        attach_subtitles(docs, cues)
        self.assertEqual([len(d["subtitles"]) for d in docs], [1, 1])
        attach_characters(docs, [{"name": "Boris", "t0": 2.0, "t1": 4.0}, {"name": "Natalya", "t0": 0, "t1": 1.5}])
        self.assertEqual(docs[0]["characters"][0], {"name": "Natalya", "seconds": 1.5})
        self.assertEqual(docs[1]["characters"], [{"name": "Boris", "seconds": 1.0}])


class TestRag(unittest.TestCase):
    def test_retrieve_and_answer_when(self):
        ix, enc = fake_index(), ColourEncoder()
        ans = answer("When does the red scene start?", ix, enc, RagParams(k=3, min_score=0.5), use_llm=False)
        self.assertTrue(ans.confident)
        self.assertEqual(ans.evidence[0].doc["shot"], 3)
        self.assertIn("0:06.0", ans.text)
        self.assertEqual(ans.kind, "when")
        self.assertEqual(ans.to_json()["evidence"][0]["shot"], 3)

    def test_no_match_says_so(self):
        ix, enc = fake_index(), ColourEncoder()
        ans = answer("When does the car chase start?", ix, enc, RagParams(min_score=1.01), use_llm=False)
        self.assertFalse(ans.confident)
        self.assertIn("could not find", ans.text)

    def test_subtitles_make_dialogue_findable(self):
        ix, enc = fake_index(), ColourEncoder()
        attach_subtitles(ix.docs, [{"t0": 8.2, "t1": 9.0, "text": "for England, James"}])
        query, ev = retrieve("When does he say for England?", ix, enc, RagParams(k=3, lexical_weight=1.0))
        self.assertEqual(ev[0].doc["shot"], 4)
        ans = template_answer("When does he say for England?", query, ev, RagParams(min_score=9.0))
        self.assertTrue(ans.confident)                        # a subtitle hit is evidence by itself
        self.assertIn("for England", ans.text)

    def test_ann_search_on_index(self):
        ix, enc = fake_index(40), ColourEncoder()
        hits = ix.search_text("red", enc, k=3)
        self.assertEqual(hits[0]["shot"], 3)
        hits2 = ix.search_text("red", enc, k=3, spec=IndexSpec("ivfpq", nlist=4, m=1, nbits=8))
        self.assertEqual(len(hits2), 3)


class TestBuildIndex(unittest.TestCase):
    def test_build_save_load_merge_on_a_synthetic_clip(self):
        from cinescope.io.video import write_video

        frames = []
        for c in ("red", "green", "blue"):                     # three 1-second shots
            img = np.zeros((48, 64, 3), np.uint8)
            img[:] = COLOURS[c]
            frames += [img] * 24
        with tempfile.TemporaryDirectory() as d:
            video = Path(d) / "colours.avi"
            write_video(video, np.stack(frames), fps=24)
            enc = ColourEncoder()
            ix = build_clip_index(video, enc, Path(d) / "index", IndexParams(decode_width=64, thumb_width=32),
                                  tag_vocab=["red", "green", "blue"], log=None)
            self.assertEqual(len(ix), 3)
            loaded = ShotIndex.load(Path(d) / "index")
            hit = loaded.search_text("the blue shot", enc, k=1)[0]
            self.assertEqual(hit["shot"], 2)
            self.assertAlmostEqual(hit["t0"], 2.0, places=1)
            self.assertEqual(loaded.docs[0]["tags"][0]["tag"], "red")
            self.assertTrue(loaded.frame_path(loaded.docs[0]).exists())
            self.assertEqual(set(loaded.docs[0]) >= {"subtitles", "characters", "tags", "frames"}, True)
            lib = ShotIndex.merge([loaded, loaded])
            self.assertEqual(len(lib), 6)
            self.assertTrue(lib.frame_path(lib.docs[4]).exists())


if __name__ == "__main__":
    unittest.main()
