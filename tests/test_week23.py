"""Week 23: vector search from scratch (exact, IVF, IVF-PQ) and the contrastive dual encoder."""
import unittest

import numpy as np

from cinescope.search.metrics import first_relevant_rank, overlaps, summarise
from cinescope.search.shotindex import keyframe_indices, plan_segments
from cinescope.search.vecindex import (
    FlatIndex,
    IndexSpec,
    IVFFlatIndex,
    IVFPQIndex,
    ProductQuantizer,
    kmeans_fast,
    load_index,
    normalize,
    one_recall_at_k,
    recall_at_k,
    save_index,
    topk,
)


def clustered(n, d=32, k=20, noise=0.3, seed=0):
    rng = np.random.default_rng(seed)
    c = rng.standard_normal((k, d))
    return normalize(c[rng.integers(k, size=n)] + noise * rng.standard_normal((n, d)))


class TestExact(unittest.TestCase):
    def test_topk_matches_argsort(self):
        s = np.random.default_rng(1).standard_normal((5, 100))
        v, i = topk(s, 7)
        np.testing.assert_array_equal(i, np.argsort(-s, 1)[:, :7])
        np.testing.assert_allclose(v, -np.sort(-s, 1)[:, :7])

    def test_flat_is_brute_force(self):
        x, q = clustered(500), clustered(20, seed=2)
        ix = FlatIndex(32)
        ix.add(x)
        D, I = ix.search(q, 5)  # noqa: E741
        np.testing.assert_array_equal(I, np.argsort(-(q @ x.T), 1)[:, :5])
        np.testing.assert_allclose(D[:, 0], (q @ x.T).max(1), rtol=1e-5)

    def test_flat_matches_faiss(self):
        faiss = __import__("pytest").importorskip("faiss")
        x, q = clustered(400), clustered(10, seed=3)
        ix, fx = FlatIndex(32), faiss.IndexFlatIP(32)
        ix.add(x)
        fx.add(x)
        self.assertEqual(recall_at_k(ix.search(q, 10)[1], fx.search(q, 10)[1], 10), 1.0)

    def test_fewer_than_k_pads(self):
        ix = FlatIndex(4)
        ix.add(normalize(np.eye(4)[:2]))
        D, I = ix.search(np.ones((1, 4)), 3)  # noqa: E741
        self.assertEqual(I[0, 2], -1)
        self.assertEqual(D[0, 2], -np.inf)


class TestKMeansAndPQ(unittest.TestCase):
    def test_kmeans_recovers_separated_clusters(self):
        rng = np.random.default_rng(0)
        centres = np.array([[0, 0], [10, 0], [0, 10], [10, 10]], np.float32)
        x = centres[rng.integers(4, size=800)] + 0.3 * rng.standard_normal((800, 2)).astype(np.float32)
        c = kmeans_fast(x, 4, seed=1)
        d = np.linalg.norm(c[:, None] - centres[None], axis=-1).min(0)
        self.assertLess(d.max(), 0.2)

    def test_pq_reconstruction_improves_with_more_subquantisers(self):
        x = clustered(2000, d=32, noise=0.5)
        errs = []
        for m in (2, 8):
            pq = ProductQuantizer(32, m=m, nbits=5)
            pq.train(x, n_iter=10)
            errs.append(float(((pq.decode(pq.encode(x)) - x) ** 2).sum(1).mean()))
        self.assertLess(errs[1], errs[0])
        self.assertLess(errs[1], 0.5 * float((x ** 2).sum(1).mean()))

    def test_adc_table_equals_inner_product_with_reconstruction(self):
        x, q = clustered(600, d=16), clustered(3, d=16, seed=5)
        pq = ProductQuantizer(16, m=4, nbits=4)
        pq.train(x, n_iter=5)
        codes = pq.encode(x[:50])
        lut = pq.ip_tables(q)                                         # (Q, M, ksub)
        adc = np.stack([lut[:, j, codes[:, j]] for j in range(4)], 0).sum(0)
        np.testing.assert_allclose(adc, q @ pq.decode(codes).T, rtol=1e-4, atol=1e-5)


class TestIVF(unittest.TestCase):
    def setUp(self):
        self.x, self.q = clustered(3000, d=32), clustered(50, d=32, seed=9)
        self.gt = np.argsort(-(self.q @ self.x.T), 1)[:, :10]

    def test_ivf_with_all_lists_probed_is_exact(self):
        ix = IVFFlatIndex(32, nlist=16, nprobe=16)
        ix.train(self.x)
        ix.add(self.x)
        self.assertEqual(recall_at_k(ix.search(self.q, 10)[1], self.gt, 10), 1.0)
        self.assertEqual(ix.inv.sizes().sum(), 3000)

    def test_more_probes_more_recall(self):
        ix = IVFFlatIndex(32, nlist=32, nprobe=1)
        ix.train(self.x)
        ix.add(self.x)
        r1 = recall_at_k(ix.search(self.q, 10)[1], self.gt, 10)
        ix.nprobe = 8
        r8 = recall_at_k(ix.search(self.q, 10)[1], self.gt, 10)
        self.assertGreaterEqual(r8, r1)
        self.assertGreater(r8, 0.8)

    def test_ivfpq_scores_are_coarse_plus_adc(self):
        ix = IVFPQIndex(32, nlist=8, m=8, nbits=6, nprobe=8)
        ix.train(self.x)
        ix.add(self.x)
        D, I = ix.search(self.q[:3], 5)  # noqa: E741
        rows = np.argsort(ix.inv.ids)                        # payload row of every id
        for qi in range(3):
            for s, i in zip(D[qi], I[qi], strict=True):
                r = rows[i]
                recon = ix.ivf.centroids[ix.inv.lists[r]] + ix.pq.decode(ix.inv.payload[r:r + 1])[0]
                self.assertAlmostEqual(float(s), float(self.q[qi] @ recon), places=4)

    def test_ivfpq_recall_and_rerank(self):
        ix = IVFPQIndex(32, nlist=16, m=8, nbits=6, nprobe=16)
        ix.train(self.x)
        ix.add(self.x)
        r = one_recall_at_k(ix.search(self.q, 10)[1], self.gt, 10)
        ix2 = IVFPQIndex(32, nlist=16, m=8, nbits=6, nprobe=16, refine=50)
        ix2.train(self.x)
        ix2.add(self.x)
        r_exact = one_recall_at_k(ix2.search(self.q, 1)[1], self.gt, 1)
        self.assertGreater(r, 0.7)           # 8 bytes per vector instead of 128
        self.assertGreater(r_exact, r)       # exact re-ranking of 50 candidates recovers most of the rest
        self.assertLess(ix.bytes_per_vector(), 32 * 4)

    def test_save_load_roundtrip(self):
        import tempfile
        from pathlib import Path

        for spec in (IndexSpec("flat"), IndexSpec("ivf", nlist=8), IndexSpec("ivfpq", nlist=8, m=4, nbits=4)):
            ix = spec.build(32)
            ix.train(self.x)
            ix.add(self.x[:500])
            with tempfile.TemporaryDirectory() as d:
                p = Path(d) / "ix.npz"
                save_index(ix, p)
                ix2 = load_index(p)
            np.testing.assert_array_equal(ix.search(self.q, 5)[1], ix2.search(self.q, 5)[1])

    def test_ivfpq_close_to_faiss(self):
        faiss = __import__("pytest").importorskip("faiss")
        ours = IVFPQIndex(32, nlist=16, m=8, nbits=6, nprobe=4)
        ours.train(self.x)
        ours.add(self.x)
        fx = faiss.IndexIVFPQ(faiss.IndexFlatIP(32), 32, 16, 8, 6, faiss.METRIC_INNER_PRODUCT)
        fx.train(self.x)
        fx.add(self.x)
        fx.nprobe = 4
        r_ours = recall_at_k(ours.search(self.q, 10)[1], self.gt, 10)
        r_faiss = recall_at_k(fx.search(self.q, 10)[1], self.gt, 10)
        self.assertGreater(r_ours, r_faiss - 0.1)


class TestDualEncoder(unittest.TestCase):
    def setUp(self):
        global torch
        import pytest

        torch = pytest.importorskip("torch")

    def test_tokenizer(self):
        from cinescope.search.dual_encoder import UNK, WordTokenizer

        tok = WordTokenizer.build(["a red circle", "a blue square"], max_len=5)
        ids = tok(["A red triangle!"])
        self.assertEqual(ids.shape, (1, 5))
        self.assertEqual(int(ids[0, 2]), UNK)
        self.assertAlmostEqual(tok.coverage("a red triangle"), 2 / 3)

    def test_info_nce(self):
        from cinescope.search.dual_encoder import caption_groups, info_nce

        u = torch.nn.functional.normalize(torch.randn(8, 16), dim=-1)
        scale = torch.tensor(np.log(1 / 0.07))
        aligned = info_nce(u, u, scale)
        shuffled = info_nce(u, u[torch.randperm(8)], scale)
        self.assertLess(float(aligned), float(shuffled))
        # all captions distinct: the multi-positive form is plain InfoNCE
        g = caption_groups([str(i) for i in range(8)])
        self.assertAlmostEqual(float(info_nce(u, u, scale, g)), float(aligned), places=5)
        self.assertEqual(caption_groups(["x", "x", "y"]).tolist(), [0, 0, 1])
        # two images with one caption: the multi-positive target is uniform over both
        v = u.clone()
        v[1] = v[0]
        dup = caption_groups(["x", "x"] + [str(i) for i in range(6)])
        self.assertTrue(np.isfinite(float(info_nce(u, v, scale, dup))))

    def test_shapes_and_relation_swap(self):
        from cinescope.search.dual_encoder import shapes_sample, swap_relation

        rng = np.random.default_rng(0)
        held = frozenset({("purple", "triangle")})
        for _ in range(50):
            img, cap = shapes_sample(rng, 32, held)
            self.assertEqual(img.shape, (32, 32, 3))
            self.assertNotIn("purple triangle", cap)
        img, cap = shapes_sample(rng, 32, held, only_held_out=True)
        self.assertIn("purple triangle", cap)
        self.assertEqual(swap_relation("a red circle left of a blue square"), "a blue square left of a red circle")
        self.assertIsNone(swap_relation("a big red circle"))

    def test_training_lowers_loss(self):
        from cinescope.search.dual_encoder import (
            DualEncoder,
            DualEncoderConfig,
            shapes_sample,
            shapes_vocab,
            train_contrastive,
        )

        torch.manual_seed(0)
        rng = np.random.default_rng(0)
        data = [shapes_sample(rng, 16) for _ in range(32)]
        m = DualEncoder(DualEncoderConfig(image_size=16, embed_dim=16, widths=(8, 16), text_width=16,
                                          text_layers=1, text_heads=2, max_len=10, vocab=shapes_vocab()))
        losses = train_contrastive(m, lambda s: ([d[0] for d in data], [d[1] for d in data]), 40, lr=5e-3,
                                   log=None)
        self.assertLess(np.mean(losses[-5:]), np.mean(losses[:5]))


class TestSegmentsAndMetrics(unittest.TestCase):
    def test_plan_segments_splits_long_shots(self):
        shots = [{"index": 0, "start": 0, "end": 23, "t0": 0.0, "t1": 1.0},
                 {"index": 1, "start": 24, "end": 24 * 14 - 1, "t0": 1.0, "t1": 14.0}]
        segs = plan_segments(shots, 6.0, 24.0)
        self.assertEqual([s["shot"] for s in segs], [0, 1, 1, 1])
        self.assertEqual(segs[1]["start"], 24)
        self.assertEqual(segs[-1]["end"], 24 * 14 - 1)
        self.assertAlmostEqual(segs[-1]["t1"], 14.0)
        for a, b in zip(segs[1:], segs[2:], strict=False):
            self.assertEqual(b["start"], a["end"] + 1)
            self.assertAlmostEqual(b["t0"], a["t1"])

    def test_keyframes_inside_segment(self):
        self.assertEqual(keyframe_indices(10, 15, 3), [11, 13, 15])
        self.assertEqual(keyframe_indices(5, 5, 3), [5])

    def test_overlap_rule(self):
        self.assertTrue(overlaps(10, 16, [[12, 14]]))          # short event inside a segment
        self.assertFalse(overlaps(10, 16, [[15.6, 30]]))      # grazes by 0.4 s
        self.assertTrue(overlaps(10, 11, [[0, 100]]))

    def test_ranks_and_summary(self):
        docs = [{"clip": "a", "t0": 0, "t1": 5}, {"clip": "b", "t0": 10, "t1": 15}, {"clip": "a", "t0": 10, "t1": 15}]
        self.assertEqual(first_relevant_rank(docs, [[10, 15]]), 2)
        self.assertEqual(first_relevant_rank(docs, [[10, 15]], clip="a"), 3)
        self.assertIsNone(first_relevant_rank(docs, [[50, 60]]))
        s = summarise([1, 3, None, 6])
        self.assertEqual(s["R@1"], 0.25)
        self.assertEqual(s["R@5"], 0.5)
        self.assertAlmostEqual(s["MRR"], (1 + 1 / 3 + 0 + 1 / 6) / 4, places=4)


if __name__ == "__main__":
    unittest.main()
