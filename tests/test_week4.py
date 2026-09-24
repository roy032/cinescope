"""Week 4: autograd engine, NumPy layers and their backward passes."""
import math
import unittest

import numpy as np

from cinescope.imgproc.filters import correlate2d
from cinescope.nn.autograd import MLP, Value
from cinescope.nn.gradcheck import check_module, numeric_grad, rel_error
from cinescope.nn.layers import (
    SGD,
    BatchNorm2d,
    Conv2d,
    GlobalAvgPool,
    Linear,
    MaxPool2d,
    ReLU,
    Residual,
    Sequential,
    softmax_cross_entropy,
)
from cinescope.nn.toy import make_framing_toy, tiny_resnet

try:
    import torch
except ImportError:          # the NumPy tests stand on their own
    torch = None

RNG = np.random.default_rng(0)
TOL = 1e-6


def f_scalar(a, b, c):
    return ((a * b + c ** 2).tanh() * a.exp() / (b + 3) - (a - c).relu() + (b * b).log()) * 2


class TestAutograd(unittest.TestCase):
    def test_matches_finite_differences(self):
        vals = [0.7, -1.3, 0.4]
        xs = [Value(v) for v in vals]
        out = f_scalar(*xs)
        out.backward()
        for i in range(3):
            def g(d, i=i):
                p = [Value(v + (d if j == i else 0)) for j, v in enumerate(vals)]
                return f_scalar(*p).data
            num = (g(1e-6) - g(-1e-6)) / 2e-6
            self.assertAlmostEqual(xs[i].grad, num, places=5)

    def test_gradients_accumulate_over_reuse(self):
        x = Value(3.0)
        y = x * x + x               # dy/dx = 2x + 1
        y.backward()
        self.assertEqual(x.grad, 7.0)

    def test_backward_twice_does_not_double_count(self):
        x = Value(2.0)
        y = x * x
        y.backward()
        y.backward()
        self.assertEqual(x.grad, 4.0)

    def test_deep_chain_has_no_recursion_limit(self):
        x = Value(1.0)
        y = x
        for _ in range(5000):
            y = y * 1.0001
        y.backward()
        self.assertAlmostEqual(x.grad, 1.0001 ** 5000, places=6)

    @unittest.skipIf(torch is None, "PyTorch not installed")
    def test_matches_pytorch(self):
        vals = [0.7, -1.3, 0.4]
        xs = [Value(v) for v in vals]
        f_scalar(*xs).backward()
        ts = [torch.tensor(v, dtype=torch.float64, requires_grad=True) for v in vals]
        a, b, c = ts
        out = ((a * b + c ** 2).tanh() * a.exp() / (b + 3) - (a - c).relu() + (b * b).log()) * 2
        out.backward()
        for x, t in zip(xs, ts, strict=True):
            self.assertAlmostEqual(x.grad, t.grad.item(), places=10)

    def test_mlp_learns(self):
        pts = [(-1, -1, -1), (-1, 1, 1), (1, -1, 1), (1, 1, -1)] * 2      # XOR
        net = MLP(2, [8, 1], seed=1)
        losses = []
        for _ in range(150):
            loss = sum(((net([x, y])[0] - t) ** 2 for x, y, t in pts), Value(0.0)) / len(pts)
            loss.backward()
            for p in net.parameters():
                p.data -= 0.1 * p.grad
            losses.append(loss.data)
        self.assertLess(losses[-1], 0.05)
        self.assertLess(losses[-1], losses[0] / 10)


class TestLayerGradients(unittest.TestCase):
    x = RNG.normal(size=(2, 3, 7, 7))

    def assert_grads(self, module, x):
        for name, err in check_module(module, x).items():
            self.assertLess(err, TOL, f"{type(module).__name__} d{name}: {err:.2e}")

    def test_linear(self):
        self.assert_grads(Linear(5, 4), RNG.normal(size=(3, 5)))

    def test_conv_padding_and_stride(self):
        for k, s, p in [(3, 1, 1), (3, 2, 1), (1, 1, 0), (5, 2, 2), (2, 2, 0)]:
            self.assert_grads(Conv2d(3, 4, k, stride=s, padding=p), self.x)

    def test_relu_maxpool_avgpool(self):
        x = RNG.permutation(np.arange(2 * 3 * 6 * 6, dtype=float)).reshape(2, 3, 6, 6) / 10  # no ties
        for m in (ReLU(), MaxPool2d(2), MaxPool2d(3, 1), GlobalAvgPool()):
            self.assert_grads(m, x - 3.05)

    def test_batchnorm(self):
        self.assert_grads(BatchNorm2d(3), self.x * 3 + 1)

    def test_residual_block(self):
        body = Sequential(Conv2d(3, 3, 3, padding=1, bias=False), BatchNorm2d(3), ReLU(),
                          Conv2d(3, 3, 3, padding=1, bias=False), BatchNorm2d(3))
        self.assert_grads(Residual(body), self.x)
        down = Residual(Sequential(Conv2d(3, 5, 3, stride=2, padding=1), BatchNorm2d(5)),
                        shortcut=Sequential(Conv2d(3, 5, 1, stride=2), BatchNorm2d(5)))
        self.assert_grads(down, self.x)

    def test_softmax_cross_entropy(self):
        logits = RNG.normal(size=(4, 5)) * 3
        y = np.array([0, 3, 1, 4])
        _, d = softmax_cross_entropy(logits, y)
        num = numeric_grad(lambda: softmax_cross_entropy(logits, y)[0], logits)
        self.assertLess(rel_error(d, num), TOL)
        big = softmax_cross_entropy(np.array([[1000.0, 0.0]]), np.array([0]))[0]
        self.assertTrue(math.isfinite(big))            # no overflow

    def test_conv_is_correlation_like_phase0(self):
        img = RNG.normal(size=(9, 11))
        conv = Conv2d(1, 1, 3, padding=1)
        conv.params["W"][:] = RNG.normal(size=(1, 1, 3, 3))
        out = conv(img[None, None])[0, 0]
        ref = correlate2d(img, conv.params["W"][0, 0], border="constant")
        np.testing.assert_allclose(out, ref, atol=1e-12)


class TestBatchNormModes(unittest.TestCase):
    def test_eval_uses_running_statistics(self):
        bn = BatchNorm2d(2, momentum=1.0)             # running stats = last batch
        x = RNG.normal(3, 2, size=(8, 2, 4, 4))
        train_out = bn(x)
        bn.eval()
        np.testing.assert_allclose(bn(x), train_out, atol=0.05)     # unbiased vs biased var
        one = bn(x[:1])
        self.assertFalse(np.allclose(one.mean((0, 2, 3)), 0, atol=1e-3))   # not re-normalised per batch


@unittest.skipIf(torch is None, "PyTorch not installed")
class TestAgainstPyTorch(unittest.TestCase):
    def test_conv_forward_and_backward(self):
        x = RNG.normal(size=(2, 3, 9, 9))
        conv = Conv2d(3, 4, 3, stride=2, padding=1)
        tconv = torch.nn.Conv2d(3, 4, 3, stride=2, padding=1).double()
        with torch.no_grad():
            tconv.weight.copy_(torch.from_numpy(conv.params["W"]))
            tconv.bias.copy_(torch.from_numpy(conv.params["b"]))
        tx = torch.from_numpy(x).requires_grad_()
        tout = tconv(tx)
        out = conv(x)
        np.testing.assert_allclose(out, tout.detach().numpy(), atol=1e-10)
        r = RNG.normal(size=out.shape)
        dx = conv.backward(r)
        tout.backward(torch.from_numpy(r))
        np.testing.assert_allclose(dx, tx.grad.numpy(), atol=1e-10)
        np.testing.assert_allclose(conv.grads["W"], tconv.weight.grad.numpy(), atol=1e-10)
        np.testing.assert_allclose(conv.grads["b"], tconv.bias.grad.numpy(), atol=1e-10)

    def test_batchnorm_matches(self):
        x = RNG.normal(1, 2, size=(4, 3, 5, 5))
        bn = BatchNorm2d(3)
        tbn = torch.nn.BatchNorm2d(3).double()
        tx = torch.from_numpy(x).requires_grad_()
        out, tout = bn(x), tbn(tx)
        np.testing.assert_allclose(out, tout.detach().numpy(), atol=1e-10)
        r = RNG.normal(size=out.shape)
        tout.backward(torch.from_numpy(r))
        np.testing.assert_allclose(bn.backward(r), tx.grad.numpy(), atol=1e-10)
        np.testing.assert_allclose(bn.running_var, tbn.running_var.numpy(), atol=1e-10)
        np.testing.assert_allclose(bn.running_mean, tbn.running_mean.numpy(), atol=1e-10)

    def test_maxpool_matches(self):
        x = RNG.normal(size=(2, 3, 8, 8))
        tx = torch.from_numpy(x).requires_grad_()
        pool = MaxPool2d(2)
        tout = torch.nn.functional.max_pool2d(tx, 2)
        np.testing.assert_allclose(pool(x), tout.detach().numpy())
        r = RNG.normal(size=tout.shape)
        tout.backward(torch.from_numpy(r))
        np.testing.assert_allclose(pool.backward(r), tx.grad.numpy())

    def test_cross_entropy_matches(self):
        logits = RNG.normal(size=(6, 4))
        y = RNG.integers(0, 4, 6)
        tl = torch.from_numpy(logits).requires_grad_()
        tloss = torch.nn.functional.cross_entropy(tl, torch.from_numpy(y))
        tloss.backward()
        loss, d = softmax_cross_entropy(logits, y)
        self.assertAlmostEqual(loss, tloss.item(), places=10)
        np.testing.assert_allclose(d, tl.grad.numpy(), atol=1e-12)


class TestTraining(unittest.TestCase):
    def test_tiny_resnet_learns_shot_scale(self):
        xtr, ytr = make_framing_toy(600, seed=0)
        xte, yte = make_framing_toy(200, seed=1)
        net = tiny_resnet(seed=0)
        opt = SGD(net.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-4)
        rng = np.random.default_rng(0)
        for _ in range(2):
            for b in np.array_split(rng.permutation(len(xtr)), len(xtr) // 32):
                _, d = softmax_cross_entropy(net(xtr[b]), ytr[b])
                net.backward(d)
                opt.step()
        net.eval()
        self.assertGreater((net(xte).argmax(1) == yte).mean(), 0.9)


if __name__ == "__main__":
    unittest.main()
