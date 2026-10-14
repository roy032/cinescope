"""Week 10: alignment, embedding losses, agglomerative clustering and its metrics."""
import unittest

import numpy as np

from cinescope.faces.align import TEMPLATE_112, umeyama
from cinescope.faces.cluster import (
    agglomerative,
    auto_threshold,
    cooccurring_pairs,
    cosine_distances,
    inverse_purity,
    pairwise_f1,
    purity,
)


class TestAlign(unittest.TestCase):
    def test_umeyama_recovers_similarity(self):
        a = np.deg2rad(17)
        R = 1.7 * np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        src = TEMPLATE_112 + 5
        dst = src @ R.T + [30, -12]
        M = umeyama(src, dst)
        np.testing.assert_allclose(M[:, :2], R, atol=1e-9)
        np.testing.assert_allclose(src @ M[:, :2].T + M[:, 2], dst, atol=1e-9)

    def test_against_opencv(self):
        cv2 = __import__("pytest").importorskip("cv2")
        rng = np.random.default_rng(0)
        src = TEMPLATE_112 + rng.normal(0, 3, TEMPLATE_112.shape)
        dst = TEMPLATE_112 * 0.8 + 20 + rng.normal(0, 2, TEMPLATE_112.shape)
        ref, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
        ours = umeyama(src, dst)
        # LMEDS is robust (not least squares); on 5 inlying points it lands close.
        np.testing.assert_allclose(ours, ref, atol=0.15 * np.abs(ref).max())


class TestAgglomerative(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        centers = rng.normal(0, 1, (3, 16))
        self.X = np.concatenate([c + rng.normal(0, 0.15, (6, 16)) for c in centers])
        self.y = np.repeat([0, 1, 2], 6)

    def test_matches_scipy_average_linkage(self):
        hier = __import__("pytest").importorskip("scipy.cluster.hierarchy")
        from scipy.spatial.distance import squareform

        D = cosine_distances(self.X)
        for k in (2, 3, 5, 9):
            ours = agglomerative(D, 0, n_clusters=k)
            ref = hier.fcluster(hier.linkage(squareform(D, checks=False), "average"), k, "maxclust")
            self.assertEqual(pairwise_f1(ours, list(ref))["f1"], 1.0)   # same partition

    def test_threshold_finds_three(self):
        lab = agglomerative(cosine_distances(self.X), threshold=0.5)
        self.assertEqual(len(set(lab.tolist())), 3)
        self.assertEqual(purity(lab, list(self.y)), 1.0)

    def test_cannot_link_blocks_merge(self):
        D = np.array([[0, 0.1, 0.9], [0.1, 0, 0.9], [0.9, 0.9, 0]])
        self.assertEqual(len(set(agglomerative(D, 0.5).tolist())), 2)
        lab = agglomerative(D, 0.5, cannot_link=[(0, 1)])
        self.assertNotEqual(lab[0], lab[1])
        # ...and it propagates: 2 may join 0 or 1, never both.
        lab = agglomerative(D, 1.0, cannot_link=[(0, 1)])
        self.assertEqual(len(set(lab.tolist())), 2)

    def test_metrics(self):
        pred = np.array([0, 0, 0, 1, 1, 2])
        truth = ["a", "a", "b", "b", "b", "c"]
        self.assertAlmostEqual(purity(pred, truth), 5 / 6)
        self.assertAlmostEqual(purity(np.arange(6), truth), 1.0)          # singletons are "pure"
        self.assertAlmostEqual(inverse_purity(np.arange(6), truth), 3 / 6)
        pf = pairwise_f1(pred, truth)
        # same-cluster pairs: (0,1)(0,2)(1,2)(3,4) -> TP (0,1),(3,4); same-identity pairs: (0,1),(2,3),(2,4),(3,4)
        self.assertAlmostEqual(pf["precision"], 2 / 4)
        self.assertAlmostEqual(pf["recall"], 2 / 4)

    def test_cooccurrence_and_auto_threshold(self):
        self.assertEqual(cooccurring_pairs([[1, 2, 3], [3, 4], [5]]), [(0, 1)])
        thr = auto_threshold(np.array([0.1, 0.15, 0.2]), np.array([0.6, 0.7, 0.9]))
        self.assertTrue(0.2 <= thr <= 0.6)


class TestEmbedding(unittest.TestCase):
    def setUp(self):
        self.torch = __import__("pytest").importorskip("torch")
        self.torch.set_num_threads(1)

    def test_net_shapes_and_norm(self):
        from cinescope.faces.embed import EmbedNet, embed

        m = EmbedNet()
        e = embed(m, np.random.default_rng(0).integers(0, 255, (5, 112, 112, 3)).astype(np.uint8))
        self.assertEqual(e.shape, (5, 128))
        np.testing.assert_allclose(np.linalg.norm(e, axis=1), 1, atol=1e-5)

    def test_triplet_uses_only_known_negatives(self):
        from cinescope.faces.embed import batch_hard_triplet

        torch = self.torch
        e = torch.nn.functional.normalize(torch.tensor([[1.0, 0], [1, 0.01], [1, 0.02], [0, 1]]), dim=1)
        track = torch.tensor([0, 0, 1, 2])
        # Track 1 is almost on top of track 0 but its relation is unknown: no loss from it.
        known = torch.zeros(4, 4, dtype=torch.bool)
        known[0, 3] = known[3, 0] = known[1, 3] = known[3, 1] = True
        loss, _ = batch_hard_triplet(e, track, known, margin=0.3)
        self.assertAlmostEqual(float(loss), 0.0, places=5)
        known[0, 2] = known[2, 0] = True                              # now it is a known negative
        loss, _ = batch_hard_triplet(e, track, known, margin=0.3)
        self.assertGreater(float(loss), 0.1)

    def test_triplet_training_separates_toy_identities(self):
        from cinescope.faces.embed import batch_hard_triplet

        torch = self.torch
        torch.manual_seed(0)
        rng = np.random.default_rng(0)
        protos = rng.normal(0, 1, (4, 20)).astype(np.float32)
        net = torch.nn.Sequential(torch.nn.Linear(20, 32), torch.nn.ReLU(), torch.nn.Linear(32, 8))
        opt = torch.optim.Adam(net.parameters(), 1e-2)
        ident = np.repeat(np.arange(4), 8)
        known = torch.as_tensor(ident[:, None] != ident[None, :])
        for _ in range(150):
            x = torch.as_tensor(protos[ident] + rng.normal(0, 0.8, (32, 20)).astype(np.float32))
            e = torch.nn.functional.normalize(net(x), dim=1)
            loss, _ = batch_hard_triplet(e, torch.as_tensor(ident), known)
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            x = torch.as_tensor(protos[ident] + rng.normal(0, 0.8, (32, 20)).astype(np.float32))
            e = torch.nn.functional.normalize(net(x), dim=1).numpy()
        lab = agglomerative(cosine_distances(e), 0, n_clusters=4)
        self.assertGreater(purity(lab, list(ident)), 0.9)

    def test_arcface_margin_raises_loss(self):
        from cinescope.faces.embed import ArcFaceLoss

        torch = self.torch
        torch.manual_seed(0)
        h = ArcFaceLoss(8, 3, m=0.0)
        e = torch.nn.functional.normalize(torch.randn(6, 8), dim=1)
        y = torch.tensor([0, 1, 2, 0, 1, 2])
        l0 = float(h(e, y).detach())
        h.m = 0.5
        self.assertGreater(float(h(e, y).detach()), l0)


if __name__ == "__main__":
    unittest.main()
