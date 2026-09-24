"""Numerical gradient checking by central differences.

    df/dx_i ≈ (f(x + h e_i) - f(x - h e_i)) / 2h         error O(h²)

Compared with the analytic gradient by the relative error
max|a - n| / max(|a|, |n|). In float64 with h = 1e-5, a correct backward
pass typically gives < 1e-7; a wrong one gives > 1e-2. ReLU and max-pool kinks
can produce isolated outliers if a probe crosses a kink, so inputs are chosen
away from ties.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .layers import Module


def numeric_grad(f: Callable[[], float], x: np.ndarray, h: float = 1e-5) -> np.ndarray:
    """Gradient of the scalar f() w.r.t. the array x, which f reads in place."""
    g = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    for _ in it:
        i = it.multi_index
        old = x[i]
        x[i] = old + h
        fp = f()
        x[i] = old - h
        fm = f()
        x[i] = old
        g[i] = (fp - fm) / (2 * h)
    return g


def rel_error(a: np.ndarray, b: np.ndarray) -> float:
    """max |a - b| scaled by the larger gradient magnitude. Array-wise rather than
    element-wise, so an entry whose true gradient is exactly zero (a conv bias
    followed by BatchNorm, which cancels it) does not divide noise by noise;
    the 1e-3 floor turns the comparison absolute for near-zero gradients."""
    scale = max(float(np.abs(a).max(initial=0)), float(np.abs(b).max(initial=0)), 1e-3)
    return float(np.abs(a - b).max(initial=0)) / scale


def check_module(module: Module, x: np.ndarray, seed: int = 1234, h: float = 1e-5) -> dict[str, float]:
    """Relative error of dL/dx and of every parameter gradient, for the loss
    L = sum(module(x) * R) with a fixed random R (so every output matters)."""
    x = x.astype(np.float64).copy()
    out = module(x)
    r = np.random.default_rng(seed).normal(size=out.shape)

    def loss() -> float:
        return float((module(x) * r).sum())

    module(x)
    dx = module.backward(r)
    analytic = {"x": dx, **{f"{id(m)}.{k}": m.grads[k].copy() for m, k in module.parameters()}}
    errors = {"x": rel_error(dx, numeric_grad(loss, x, h))}
    for m, k in module.parameters():
        errors[f"{type(m).__name__}.{k}"] = rel_error(analytic[f"{id(m)}.{k}"],
                                                      numeric_grad(loss, m.params[k], h))
    return errors
