"""A scalar reverse-mode automatic differentiation engine (micrograd-style).

Every `Value` remembers the values it was computed from and a closure that
pushes its gradient back to them (the local chain rule). `backward()` orders the
graph topologically and runs those closures from the output back to the inputs.

Two details that are easy to get wrong:
  * gradients *accumulate* (`+=`): a value used twice (x * x) receives a
    contribution from each use;
  * the order must be a topological sort, not a plain depth-first walk from the
    output: a node's closure may only run once its gradient is complete.
"""
from __future__ import annotations

import math
import random
from collections.abc import Callable, Iterable


class Value:
    __slots__ = ("data", "grad", "_backward", "_prev", "op", "label")

    def __init__(self, data: float, _children: Iterable[Value] = (), op: str = "", label: str = ""):
        self.data = float(data)
        self.grad = 0.0
        self._backward: Callable[[], None] = lambda: None
        self._prev = tuple(_children)
        self.op = op
        self.label = label

    # --- arithmetic -----------------------------------------------------------------
    def __add__(self, other: Value | float) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other: Value | float) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, k: float) -> Value:
        if isinstance(k, Value):
            raise TypeError("only constant exponents are supported; use exp/log for x**y")
        out = Value(self.data ** k, (self,), f"**{k}")

        def _backward():
            self.grad += k * self.data ** (k - 1) * out.grad
        out._backward = _backward
        return out

    def exp(self) -> Value:
        e = math.exp(self.data)
        out = Value(e, (self,), "exp")

        def _backward():
            self.grad += e * out.grad
        out._backward = _backward
        return out

    def log(self) -> Value:
        out = Value(math.log(self.data), (self,), "log")

        def _backward():
            self.grad += out.grad / self.data
        out._backward = _backward
        return out

    def tanh(self) -> Value:
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad += (1 - t * t) * out.grad
        out._backward = _backward
        return out

    def relu(self) -> Value:
        out = Value(self.data if self.data > 0 else 0.0, (self,), "relu")

        def _backward():
            self.grad += (self.data > 0) * out.grad
        out._backward = _backward
        return out

    def __neg__(self) -> Value:
        return self * -1

    def __sub__(self, other: Value | float) -> Value:
        return self + (-other)

    def __truediv__(self, other: Value | float) -> Value:
        return self * (other ** -1 if isinstance(other, Value) else 1.0 / other)

    def __radd__(self, other: float) -> Value:
        return self + other

    def __rmul__(self, other: float) -> Value:
        return self * other

    def __rsub__(self, other: float) -> Value:
        return Value(other) - self

    def __rtruediv__(self, other: float) -> Value:
        return Value(other) * self ** -1

    # --- backprop -------------------------------------------------------------------
    def topo(self) -> list[Value]:
        order: list[Value] = []
        seen: set[int] = set()
        stack: list[tuple[Value, bool]] = [(self, False)]
        while stack:                       # iterative: deep graphs would overflow recursion
            v, done = stack.pop()
            if done:
                order.append(v)
                continue
            if id(v) in seen:
                continue
            seen.add(id(v))
            stack.append((v, True))
            stack.extend((c, False) for c in v._prev if id(c) not in seen)
        return order

    def backward(self) -> None:
        order = self.topo()
        for v in order:
            v.grad = 0.0
        self.grad = 1.0
        for v in reversed(order):
            v._backward()

    def __repr__(self) -> str:
        return f"Value(data={self.data:.6g}, grad={self.grad:.6g})"


# --- a tiny MLP on top, to train something end to end ------------------------------
class Neuron:
    def __init__(self, n_in: int, act: str = "tanh", rng: random.Random | None = None):
        rng = rng or random.Random(0)
        scale = 1 / math.sqrt(n_in)
        self.w = [Value(rng.uniform(-scale, scale)) for _ in range(n_in)]
        self.b = Value(0.0)
        self.act = act

    def __call__(self, x: list[Value | float]) -> Value:
        s = sum((wi * xi for wi, xi in zip(self.w, x, strict=True)), self.b)
        return {"tanh": s.tanh, "relu": s.relu, "linear": lambda: s}[self.act]()

    def parameters(self) -> list[Value]:
        return [*self.w, self.b]


class MLP:
    def __init__(self, n_in: int, sizes: list[int], seed: int = 0):
        rng = random.Random(seed)
        dims = [n_in, *sizes]
        self.layers = [[Neuron(dims[i], "linear" if i == len(sizes) - 1 else "tanh", rng)
                        for _ in range(dims[i + 1])] for i in range(len(sizes))]

    def __call__(self, x: list[Value | float]) -> list[Value]:
        for layer in self.layers:
            x = [n(x) for n in layer]
        return x

    def parameters(self) -> list[Value]:
        return [p for layer in self.layers for n in layer for p in n.parameters()]
