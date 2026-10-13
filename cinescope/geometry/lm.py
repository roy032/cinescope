"""Levenberg-Marquardt for small dense non-linear least squares, in NumPy.

Minimise ½‖r(x)‖². Gauss-Newton linearises r(x + δ) ≈ r + J δ and solves the
normal equations JᵀJ δ = -Jᵀr; it converges fast near the minimum but can
overshoot far from it. Levenberg's fix adds a damping term,

    (JᵀJ + λ D) δ = -Jᵀr,       D = diag(JᵀJ)   (Marquardt's scaling)

Large λ gives a short gradient-descent step, small λ gives Gauss-Newton. After
each step compare the actual cost decrease with the decrease the linear model
predicted (the gain ratio ρ): if ρ > 0 accept and shrink λ, otherwise reject
and grow λ. The update rule is Nielsen's (1999), which Madsen, Nielsen &
Tingleff's lecture notes recommend.

Calibration uses this solver; bundle adjustment, with thousands of unknowns,
uses scipy's sparse trust-region solver on the residuals and sparse Jacobian
written in `ba.py`.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass
class LMResult:
    x: np.ndarray
    cost: float              # ½‖r‖² at x
    iterations: int
    converged: bool


def numeric_jacobian(fun: Callable[[np.ndarray], np.ndarray], x: np.ndarray,
                     r0: np.ndarray | None = None, eps: float = 1e-6) -> np.ndarray:
    """Forward differences, one column per parameter."""
    r0 = fun(x) if r0 is None else r0
    jac = np.empty((len(r0), len(x)))
    for i in range(len(x)):
        h = eps * max(1.0, abs(x[i]))
        xp = x.copy()
        xp[i] += h
        jac[:, i] = (fun(xp) - r0) / h
    return jac


def levenberg_marquardt(fun: Callable[[np.ndarray], np.ndarray], x0: np.ndarray,
                        jac: Callable[[np.ndarray], np.ndarray] | None = None,
                        max_iter: int = 100, tol: float = 1e-10) -> LMResult:
    x = np.asarray(x0, float).copy()
    jac = jac or (lambda z: numeric_jacobian(fun, z))
    r = fun(x)
    cost = 0.5 * r @ r
    j = jac(x)
    a, g = j.T @ j, j.T @ r
    lam = 1e-3 * np.max(np.diag(a))
    nu = 2.0
    for it in range(1, max_iter + 1):
        d = np.diag(np.maximum(np.diag(a), 1e-12))
        try:
            step = np.linalg.solve(a + lam * d, -g)
        except np.linalg.LinAlgError:
            lam *= nu
            nu *= 2
            continue
        if np.linalg.norm(step) < tol * (np.linalg.norm(x) + tol):
            return LMResult(x, cost, it, True)
        x_new = x + step
        r_new = fun(x_new)
        cost_new = 0.5 * r_new @ r_new
        predicted = -(step @ g + 0.5 * step @ a @ step)    # decrease the linear model promises
        rho = (cost - cost_new) / predicted if predicted > 0 else -1.0
        if rho > 0:
            converged = (cost - cost_new) < tol * max(cost, 1e-30)
            x, r, cost = x_new, r_new, cost_new
            j = jac(x)
            a, g = j.T @ j, j.T @ r
            lam *= max(1 / 3, 1 - (2 * rho - 1) ** 3)
            nu = 2.0
            if converged or np.max(np.abs(g)) < tol:
                return LMResult(x, cost, it, True)
        else:
            lam *= nu
            nu *= 2
    return LMResult(x, cost, max_iter, False)
