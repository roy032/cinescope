"""Week 4 in one run: autograd engine, gradient checks, and a NumPy CNN that learns.

    python p1_cnn/week4_backprop.py            # ~10 s on a laptop CPU
    python p1_cnn/week4_backprop.py --epochs 8 --out p1_cnn/results

Prints (1) a micrograd MLP learning XOR, (2) the gradient-check table for every
layer — against finite differences, and against PyTorch if it is installed —
and (3) a tiny NumPy ResNet trained on a synthetic shot-scale task, with its
confusion matrix. Writes results/week4.json.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cinescope.nn.autograd import MLP, Value  # noqa: E402
from cinescope.nn.gradcheck import check_module  # noqa: E402
from cinescope.nn.layers import (  # noqa: E402
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
from cinescope.nn.toy import CLASSES, make_framing_toy, tiny_resnet  # noqa: E402


def micrograd_xor(steps: int = 200) -> list[float]:
    pts = [(-1, -1, -1), (-1, 1, 1), (1, -1, 1), (1, 1, -1)]
    net = MLP(2, [8, 1], seed=1)
    hist = []
    for _ in range(steps):
        loss = sum(((net([x, y])[0] - t) ** 2 for x, y, t in pts), Value(0.0)) / len(pts)
        loss.backward()
        for p in net.parameters():
            p.data -= 0.1 * p.grad
        hist.append(loss.data)
    return hist


def gradcheck_table() -> dict[str, float]:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(2, 3, 8, 8))
    noties = rng.permutation(np.arange(x.size, dtype=float)).reshape(x.shape) / 50 - 3.01
    cases = {
        "Linear": (Linear(6, 4), rng.normal(size=(3, 6))),
        "Conv2d 3x3 s1 p1": (Conv2d(3, 4, 3, padding=1), x),
        "Conv2d 3x3 s2 p1": (Conv2d(3, 4, 3, stride=2, padding=1), x),
        "Conv2d 1x1 s2": (Conv2d(3, 4, 1, stride=2), x),
        "ReLU": (ReLU(), noties),
        "MaxPool2d 2": (MaxPool2d(2), noties),
        "BatchNorm2d": (BatchNorm2d(3), x),
        "GlobalAvgPool": (GlobalAvgPool(), x),
        "Residual block": (Residual(Sequential(Conv2d(3, 3, 3, padding=1, bias=False), BatchNorm2d(3), ReLU(),
                                               Conv2d(3, 3, 3, padding=1, bias=False), BatchNorm2d(3))), x),
    }
    return {name: max(check_module(m, inp).values()) for name, (m, inp) in cases.items()}


def pytorch_check() -> dict[str, float] | None:
    try:
        import torch
    except ImportError:
        return None
    rng = np.random.default_rng(1)
    x = rng.normal(size=(2, 3, 9, 9))
    conv = Conv2d(3, 4, 3, stride=2, padding=1)
    t = torch.nn.Conv2d(3, 4, 3, stride=2, padding=1).double()
    with torch.no_grad():
        t.weight.copy_(torch.from_numpy(conv.params["W"]))
        t.bias.copy_(torch.from_numpy(conv.params["b"]))
    tx = torch.from_numpy(x).requires_grad_()
    out = conv(x)
    tout = t(tx)
    r = rng.normal(size=out.shape)
    dx = conv.backward(r)
    tout.backward(torch.from_numpy(r))
    return {"conv forward": float(np.abs(out - tout.detach().numpy()).max()),
            "conv dx": float(np.abs(dx - tx.grad.numpy()).max()),
            "conv dW": float(np.abs(conv.grads["W"] - t.weight.grad.numpy()).max())}


def train_toy(epochs: int, n_train: int = 1500, lr: float = 0.1, bs: int = 32) -> dict:
    xtr, ytr = make_framing_toy(n_train, seed=0)
    xte, yte = make_framing_toy(500, seed=1)
    net = tiny_resnet(seed=0)
    opt = SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    steps = epochs * math.ceil(n_train / bs)
    rng, step, log = np.random.default_rng(0), 0, []
    for ep in range(epochs):
        losses = []
        net.train()
        for b in np.array_split(rng.permutation(n_train), math.ceil(n_train / bs)):
            opt.lr = lr * 0.5 * (1 + math.cos(math.pi * step / steps))     # cosine schedule
            step += 1
            loss, d = softmax_cross_entropy(net(xtr[b]), ytr[b])
            net.backward(d)
            opt.step()
            losses.append(loss)
        net.eval()
        pred = net(xte).argmax(1)
        log.append({"epoch": ep + 1, "loss": float(np.mean(losses)), "test_acc": float((pred == yte).mean())})
        print(f"  epoch {ep + 1}: loss {log[-1]['loss']:.3f}  test acc {log[-1]['test_acc']:.3f}")
    cm = np.zeros((3, 3), int)
    np.add.at(cm, (yte, pred), 1)
    return {"log": log, "confusion": cm.tolist(), "classes": list(CLASSES)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--out", default=str(Path(__file__).parent / "results"))
    args = ap.parse_args()

    t0 = time.time()
    hist = micrograd_xor()
    print(f"[1] micrograd MLP on XOR: loss {hist[0]:.3f} -> {hist[-1]:.4f}")

    print("[2] gradient check (max relative error; < 1e-6 means the backward pass is right)")
    table = gradcheck_table()
    for name, err in table.items():
        print(f"  {name:<18} {err:.1e}  {'ok' if err < 1e-6 else 'WRONG'}")
    torch_diff = pytorch_check()
    if torch_diff is None:
        print("  (PyTorch not installed: skipping the PyTorch comparison)")
    else:
        for name, err in torch_diff.items():
            print(f"  vs PyTorch {name:<12} max abs diff {err:.1e}")

    print(f"[3] NumPy tiny ResNet on synthetic shot scale ({args.epochs} epochs)")
    res = train_toy(args.epochs)
    print("  confusion (rows = true, cols = predicted):", *CLASSES)
    for c, row in zip(CLASSES, res["confusion"], strict=True):
        print(f"  {c:>7} {row}")
    print(f"done in {time.time() - t0:.0f} s")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "week4.json").write_text(json.dumps(
        {"xor_loss": [hist[0], hist[-1]], "gradcheck": table, "vs_pytorch": torch_diff, "toy": res},
        indent=2))


if __name__ == "__main__":
    main()
