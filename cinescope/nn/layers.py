"""Neural-network layers in NumPy with hand-derived backward passes.

Each layer caches what its backward pass needs during `forward`, and `backward`
takes dL/d(output) and returns dL/d(input), storing parameter gradients in
`self.grads`. Arrays are NCHW like PyTorch, so weights can be copied across and
compared one to one (tests/test_week4.py).

Convolution uses im2col: every receptive field becomes a row of a matrix, and
the convolution becomes one matrix multiply. Its backward pass is the reverse:
the gradient of each row is scattered back ("col2im") onto the pixels it came
from, *adding* where receptive fields overlap. Like every deep-learning "conv",
it computes cross-correlation (no kernel flip).
"""
from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


class Module:
    training = True

    def __init__(self):
        self.params: dict[str, np.ndarray] = {}
        self.grads: dict[str, np.ndarray] = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def backward(self, dout: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    def children(self) -> list[Module]:
        return []

    def parameters(self) -> list[tuple[Module, str]]:
        """(module, name) pairs, so an optimiser can update params and read grads."""
        out = [(self, k) for k in self.params]
        for c in self.children():
            out += c.parameters()
        return out

    def train(self, mode: bool = True) -> Module:
        self.training = mode
        for c in self.children():
            c.train(mode)
        return self

    def eval(self) -> Module:
        return self.train(False)


def _he(rng: np.random.Generator, shape: tuple[int, ...], fan_in: int) -> np.ndarray:
    # He/Kaiming normal init: keeps activation variance constant through ReLUs.
    return rng.normal(0.0, np.sqrt(2.0 / fan_in), size=shape)


class Linear(Module):
    def __init__(self, n_in: int, n_out: int, seed: int = 0):
        super().__init__()
        rng = np.random.default_rng(seed)
        self.params = {"W": _he(rng, (n_out, n_in), n_in), "b": np.zeros(n_out)}

    def forward(self, x):
        self.x = x
        return x @ self.params["W"].T + self.params["b"]

    def backward(self, dout):
        self.grads = {"W": dout.T @ self.x, "b": dout.sum(0)}
        return dout @ self.params["W"]


class Conv2d(Module):
    def __init__(self, c_in: int, c_out: int, k: int, stride: int = 1, padding: int = 0,
                 bias: bool = True, seed: int = 0):
        super().__init__()
        rng = np.random.default_rng(seed)
        self.k, self.stride, self.padding = k, stride, padding
        self.params = {"W": _he(rng, (c_out, c_in, k, k), c_in * k * k)}
        if bias:
            self.params["b"] = np.zeros(c_out)

    def _windows(self, xp: np.ndarray) -> np.ndarray:
        s = self.stride
        # (N, C, Ho, Wo, k, k) view — no copy until the reshape below
        return sliding_window_view(xp, (self.k, self.k), axis=(2, 3))[:, :, ::s, ::s]

    def forward(self, x):
        p = self.padding
        xp = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p))) if p else x
        win = self._windows(xp)
        n, c, ho, wo = win.shape[:4]
        cols = win.transpose(0, 2, 3, 1, 4, 5).reshape(n * ho * wo, c * self.k * self.k)
        w = self.params["W"].reshape(len(self.params["W"]), -1)
        out = cols @ w.T
        if "b" in self.params:
            out = out + self.params["b"]
        self.cache = (x.shape, xp.shape, cols, ho, wo)
        return out.reshape(n, ho, wo, -1).transpose(0, 3, 1, 2)

    def backward(self, dout):
        x_shape, xp_shape, cols, ho, wo = self.cache
        n, c = x_shape[:2]
        k, s, p = self.k, self.stride, self.padding
        w = self.params["W"]
        d2 = dout.transpose(0, 2, 3, 1).reshape(-1, w.shape[0])          # (N*Ho*Wo, O)
        self.grads = {"W": (d2.T @ cols).reshape(w.shape)}
        if "b" in self.params:
            self.grads["b"] = d2.sum(0)
        dcols = (d2 @ w.reshape(w.shape[0], -1)).reshape(n, ho, wo, c, k, k)
        dxp = np.zeros(xp_shape)
        for i in range(k):                 # col2im: k² vectorised scatter-adds
            for j in range(k):
                dxp[:, :, i:i + s * ho:s, j:j + s * wo:s] += dcols[:, :, :, :, i, j].transpose(0, 3, 1, 2)
        return dxp[:, :, p:p + x_shape[2], p:p + x_shape[3]] if p else dxp


class ReLU(Module):
    def forward(self, x):
        self.mask = x > 0
        return x * self.mask

    def backward(self, dout):
        return dout * self.mask


class MaxPool2d(Module):
    def __init__(self, k: int = 2, stride: int | None = None):
        super().__init__()
        self.k, self.stride = k, stride or k

    def forward(self, x):
        k, s = self.k, self.stride
        win = sliding_window_view(x, (k, k), axis=(2, 3))[:, :, ::s, ::s]
        flat = win.reshape(*win.shape[:4], k * k)
        self.arg = flat.argmax(-1)          # the one input that gets the gradient (first on ties)
        self.x_shape = x.shape
        return flat.max(-1)

    def backward(self, dout):
        k, s = self.k, self.stride
        dx = np.zeros(self.x_shape)
        ho, wo = dout.shape[2:]
        for i in range(k):
            for j in range(k):
                dx[:, :, i:i + s * ho:s, j:j + s * wo:s] += dout * (self.arg == i * k + j)
        return dx


class BatchNorm2d(Module):
    """Normalise each channel over (N, H, W) with batch statistics while training;
    with the running averages at inference. That switch is why model.eval()
    matters: a batch of one image has no meaningful batch variance."""

    def __init__(self, c: int, momentum: float = 0.1, eps: float = 1e-5):
        super().__init__()
        self.params = {"gamma": np.ones(c), "beta": np.zeros(c)}
        self.running_mean, self.running_var = np.zeros(c), np.ones(c)
        self.momentum, self.eps = momentum, eps

    def forward(self, x):
        g = self.params["gamma"][None, :, None, None]
        b = self.params["beta"][None, :, None, None]
        if not self.training:
            xhat = (x - self.running_mean[None, :, None, None]) / np.sqrt(
                self.running_var[None, :, None, None] + self.eps)
            return g * xhat + b
        mean = x.mean((0, 2, 3))
        var = x.var((0, 2, 3))
        m = x.size / x.shape[1]
        self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
        # PyTorch keeps the *unbiased* variance in the running estimate
        self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var * m / max(m - 1, 1)
        self.inv_std = 1.0 / np.sqrt(var + self.eps)[None, :, None, None]
        self.xhat = (x - mean[None, :, None, None]) * self.inv_std
        return g * self.xhat + b

    def backward(self, dout):
        xhat, inv_std = self.xhat, self.inv_std
        m = dout.size / dout.shape[1]
        self.grads = {"gamma": (dout * xhat).sum((0, 2, 3)), "beta": dout.sum((0, 2, 3))}
        dxhat = dout * self.params["gamma"][None, :, None, None]
        # d/dx of (x - mean)/std, with mean and std themselves functions of x
        return inv_std / m * (m * dxhat - dxhat.sum((0, 2, 3), keepdims=True)
                              - xhat * (dxhat * xhat).sum((0, 2, 3), keepdims=True))


class GlobalAvgPool(Module):
    def forward(self, x):
        self.x_shape = x.shape
        return x.mean((2, 3))

    def backward(self, dout):
        n, c, h, w = self.x_shape
        return np.broadcast_to(dout[:, :, None, None] / (h * w), self.x_shape).copy()


class Sequential(Module):
    def __init__(self, *layers: Module):
        super().__init__()
        self.layers = list(layers)

    def children(self):
        return self.layers

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def backward(self, dout):
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout


class Residual(Module):
    """y = relu(body(x) + shortcut(x)). In the backward pass the addition copies
    the gradient to both branches, so the identity path delivers dL/dy to x
    unchanged however badly the body's gradients behave — the reason ResNets
    with 100+ layers still train."""

    def __init__(self, body: Module, shortcut: Module | None = None):
        super().__init__()
        self.body, self.shortcut, self.relu = body, shortcut, ReLU()

    def children(self):
        return [self.body, self.relu] + ([self.shortcut] if self.shortcut else [])

    def forward(self, x):
        sc = self.shortcut(x) if self.shortcut else x
        return self.relu(self.body(x) + sc)

    def backward(self, dout):
        d = self.relu.backward(dout)
        dx = self.body.backward(d)
        return dx + (self.shortcut.backward(d) if self.shortcut else d)


def softmax_cross_entropy(logits: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    """Mean cross-entropy and its gradient w.r.t. the logits: (softmax - onehot) / N.
    Subtracting the row max first keeps exp() from overflowing."""
    z = logits - logits.max(1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(1, keepdims=True))
    n = len(y)
    loss = -logp[np.arange(n), y].mean()
    d = np.exp(logp)
    d[np.arange(n), y] -= 1
    return float(loss), d / n


class SGD:
    """SGD with momentum and classic L2 weight decay (added to the gradient)."""

    def __init__(self, params: list[tuple[Module, str]], lr: float = 0.05, momentum: float = 0.9,
                 weight_decay: float = 0.0):
        self.params, self.lr, self.momentum, self.wd = params, lr, momentum, weight_decay
        self.v = [np.zeros_like(m.params[k]) for m, k in params]

    def step(self) -> None:
        for (m, k), v in zip(self.params, self.v, strict=True):
            g = m.grads[k] + self.wd * m.params[k]
            v *= self.momentum
            v += g
            m.params[k] -= self.lr * v
