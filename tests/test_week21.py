"""Week 21: CTC loss and decoding from scratch, CER/WER, the CRNN and its
synthetic training lines."""
import itertools
import math
import random
import unittest

import numpy as np

from cinescope.ocr import charset
from cinescope.ocr.ctc import ctc_numpy, greedy_decode, prefix_beam_search
from cinescope.ocr.metrics import cer, corpus_cer, levenshtein, wer


class TestMetrics(unittest.TestCase):
    def test_levenshtein(self):
        self.assertEqual(levenshtein("kitten", "sitting"), 3)
        self.assertEqual(levenshtein("", "abc"), 3)
        self.assertEqual(levenshtein("abc", "abc"), 0)
        self.assertEqual(levenshtein(["a", "b"], ["b"]), 1)

    def test_cer_wer(self):
        self.assertAlmostEqual(cer("hello", "helo"), 0.2)
        self.assertAlmostEqual(wer("the cat sat", "the hat sat"), 1 / 3)
        self.assertEqual(cer("", ""), 0.0)
        self.assertAlmostEqual(corpus_cer(["ab", "abcd"], ["ab", "abc"]), 1 / 6)

    def test_charset(self):
        s = charset.normalize("“What’s  that?” — café")
        self.assertEqual(s, "\"What's that?\" - cafe")
        self.assertEqual(charset.decode(charset.encode(s)), s)
        self.assertEqual(charset.fold("Hello, World 42!"), "helloworld42")


def _brute_force_prob(lp: np.ndarray, label: list[int]) -> float:
    """Sum over every path of length T that collapses to `label`."""
    t_max, c = lp.shape
    total = 0.0
    for path in itertools.product(range(c), repeat=t_max):
        if greedy_decode(np.eye(c)[list(path)]) == label:
            total += math.exp(sum(lp[t, k] for t, k in enumerate(path)))
    return total


class TestCTC(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)

    def test_numpy_matches_brute_force(self):
        logits = self.rng.normal(size=(5, 3))
        lp = logits - np.log(np.exp(logits).sum(1, keepdims=True))
        for label in ([1], [1, 1], [1, 2], [2, 1, 2]):
            loss, _ = ctc_numpy(logits, label)
            self.assertAlmostEqual(loss, -math.log(_brute_force_prob(lp, label)), places=9)

    def test_numpy_gradient_finite_differences(self):
        logits = self.rng.normal(size=(6, 4))
        label = [1, 1, 3]
        _, grad = ctc_numpy(logits, label)
        eps = 1e-6
        for t, k in [(0, 0), (2, 1), (5, 3), (3, 2)]:
            a, b = logits.copy(), logits.copy()
            a[t, k] += eps
            b[t, k] -= eps
            num = (ctc_numpy(a, label)[0] - ctc_numpy(b, label)[0]) / (2 * eps)
            self.assertAlmostEqual(grad[t, k], num, places=5)

    def test_torch_matches_pytorch(self):
        import torch
        import torch.nn.functional as F

        from cinescope.ocr.ctc import ctc_loss

        torch.manual_seed(0)
        t, b, c = 20, 4, 6
        logits = torch.randn(t, b, c, dtype=torch.float64, requires_grad=True)
        targets = torch.tensor([[1, 1, 2, 0, 0], [3, 4, 5, 1, 2], [2, 0, 0, 0, 0], [5, 5, 5, 0, 0]])
        tl = torch.tensor([3, 5, 1, 3])
        il = torch.tensor([20, 15, 7, 20])
        for red in ("none", "sum", "mean"):
            ours = ctc_loss(logits.log_softmax(2), targets, il, tl, reduction=red)
            ref = F.ctc_loss(logits.log_softmax(2), targets, il, tl, reduction=red)
            self.assertTrue(torch.allclose(ours, ref, atol=1e-10), red)
        ctc_loss(logits.log_softmax(2), targets, il, tl).backward()
        g_ours = logits.grad.clone()
        logits.grad = None
        F.ctc_loss(logits.log_softmax(2), targets, il, tl).backward()
        self.assertTrue(torch.allclose(g_ours, logits.grad, atol=1e-10))
        # concatenated targets, and a label too long for its input -> inf like PyTorch
        flat = torch.cat([targets[i, :tl[i]] for i in range(b)])
        self.assertTrue(torch.allclose(ctc_loss(logits.log_softmax(2), flat, il, tl, reduction="sum"),
                                       F.ctc_loss(logits.log_softmax(2), flat, il, tl, reduction="sum")))
        short = ctc_loss(logits.log_softmax(2), targets, torch.tensor([2, 2, 2, 2]), tl, reduction="none")
        self.assertTrue(torch.isinf(short[1]) and torch.isfinite(short[2]))

    def test_numpy_grad_matches_pytorch(self):
        import torch
        import torch.nn.functional as F

        logits = self.rng.normal(size=(12, 5))
        label = [2, 2, 4, 1]
        loss, grad = ctc_numpy(logits, label)
        x = torch.tensor(logits[:, None], requires_grad=True)
        ref = F.ctc_loss(x.log_softmax(2), torch.tensor([label]), [12], [4], reduction="sum")
        ref.backward()
        self.assertAlmostEqual(loss, ref.item(), places=9)
        self.assertTrue(np.allclose(grad, x.grad[:, 0].numpy(), atol=1e-9))

    def test_greedy(self):
        # path a a - a b b - -> "aab"
        path = [1, 1, 0, 1, 2, 2, 0, 0]
        self.assertEqual(greedy_decode(np.eye(3)[path]), [1, 1, 2])

    def test_beam_beats_best_path(self):
        # Best path is "-" (blank, blank) with p=0.36, but label "a" collects
        # a-, -a, aa = 0.24+0.24+0.16 = 0.64.
        lp = np.log(np.array([[0.6, 0.4], [0.6, 0.4]]))
        self.assertEqual(greedy_decode(lp), [])
        best, logp = prefix_beam_search(lp, beam=4)[0]
        self.assertEqual(best, [1])
        self.assertAlmostEqual(math.exp(logp), 0.64)

    def test_beam_probabilities_are_exact(self):
        logits = self.rng.normal(size=(4, 3))
        lp = logits - np.log(np.exp(logits).sum(1, keepdims=True))
        for labels, logp in prefix_beam_search(lp, beam=50, prune=-1e9)[:5]:
            self.assertAlmostEqual(math.exp(logp), _brute_force_prob(lp, labels), places=9)


class TestCRNN(unittest.TestCase):
    def test_shapes_and_collate(self):
        import torch

        from cinescope.ocr.crnn import CRNN, collate

        imgs = [np.zeros((32, 50), np.uint8), np.zeros((32, 97), np.uint8)]
        x, widths, tg, tl = collate(imgs, ["ab", "hello"])
        self.assertEqual(tuple(x.shape), (2, 1, 32, 100))
        self.assertEqual(tl.tolist(), [2, 5])
        model = CRNN(width=4, hidden=8).eval()
        with torch.no_grad():
            out = model(x)
        self.assertEqual(tuple(out.shape), (25, 2, charset.N_CLASSES))
        self.assertTrue(torch.allclose(out.exp().sum(-1), torch.ones(25, 2), atol=1e-5))

    def test_render_line_default_font(self):
        from cinescope.ocr.synth import corpus_lines, random_strings, render_line

        rng = random.Random(0)
        im = render_line("Hello there", rng, fonts=(None,), frames=None)
        self.assertEqual(im.height, 32)
        self.assertGreater(im.width, 32)
        lines = corpus_lines("It is a truth universally acknowledged, that a single man", rng, 20)
        self.assertEqual(len(lines), 20)
        self.assertTrue(all(0 < len(s) <= 44 for s in lines + random_strings(rng, 20, ["word"])))

    def test_tiny_crnn_learns(self):
        """Two fixed images, trained with our CTC: past the all-blank plateau,
        greedy decoding starts reading the letters back."""
        import torch

        from cinescope.ocr.crnn import CRNN, batch_loss, collate
        from cinescope.ocr.synth import render_line

        torch.set_num_threads(1)
        torch.manual_seed(0)
        rng = random.Random(2)
        texts = ["cab", "bad"]
        imgs = [np.asarray(render_line(t, rng, (None,), None, degrade=False)) for t in texts]
        x, widths, tg, tl = collate(imgs, texts)
        model = CRNN(width=8, hidden=32)
        opt = torch.optim.Adam(model.parameters(), lr=3e-3)
        first = None
        for _ in range(250):
            loss = batch_loss(model, x, widths, tg, tl)
            opt.zero_grad()
            loss.backward()
            opt.step()
            first = first if first is not None else loss.item()
        self.assertLess(loss.item(), 0.05 * first)
        with torch.no_grad():               # batch statistics: 250 steps are too few for BN's running ones
            lp = model(x)
        reads = [charset.decode(greedy_decode(lp[:int(w) // 4, i].numpy())) for i, w in enumerate(widths)]
        self.assertLess(corpus_cer(texts, reads), 0.34)


if __name__ == "__main__":
    unittest.main()
