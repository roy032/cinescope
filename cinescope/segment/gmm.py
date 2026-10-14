"""Full-covariance Gaussian mixture colour models, as GrabCut uses them.

GrabCut (Rother et al., 2004) models foreground and background colours with
one K=5 component GMM each. It does not run EM to convergence; each outer
iteration assigns every pixel to its single most likely component and re-fits
weights, means and covariances from those hard assignments (as OpenCV's
implementation does). That is cheap, and the graph cut that follows changes
the training sets anyway.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..palette.kmeans import kmeans

_LOG2PI = np.log(2 * np.pi)


@dataclass
class GMM:
    weights: np.ndarray   # (K,)
    means: np.ndarray     # (K, D)
    covs: np.ndarray      # (K, D, D)

    @classmethod
    def fit(cls, x: np.ndarray, k: int = 5, labels: np.ndarray | None = None,
            reg: float = 1.0, seed: int = 0, max_init: int = 4000) -> GMM:
        """Fit from hard component labels; without labels, k-means gives them.
        `reg` is added to every covariance diagonal so flat regions (one
        colour, as compressed video is full of) do not give singular matrices."""
        x = np.asarray(x, dtype=np.float64)
        d = x.shape[1]
        k = max(1, min(k, len(x)))
        if labels is None:
            if len(x) > max_init:
                sub = x[np.random.default_rng(seed).choice(len(x), max_init, replace=False)]
            else:
                sub = x
            centres = kmeans(sub, k, n_init=1, max_iter=20, seed=seed).centres
            labels = np.argmin(((x[:, None, :] - centres[None]) ** 2).sum(-1), 1)
        weights, means, covs = [], [], []
        for j in range(k):
            xj = x[labels == j]
            if len(xj) < 2:
                continue
            mu = xj.mean(0)
            c = np.cov(xj, rowvar=False, bias=True).reshape(d, d) + reg * np.eye(d)
            weights.append(len(xj))
            means.append(mu)
            covs.append(c)
        if not weights:          # degenerate: one broad component
            mu = x.mean(0)
            return cls(np.ones(1), mu[None], (np.eye(d) * (x.var() + reg))[None])
        w = np.asarray(weights, np.float64)
        return cls(w / w.sum(), np.asarray(means), np.asarray(covs))

    def component_loglik(self, x: np.ndarray) -> np.ndarray:
        """(N, K): log(weight_k * N(x | mean_k, cov_k))."""
        x = np.asarray(x, dtype=np.float64)
        out = np.empty((len(x), len(self.weights)))
        for j, (w, mu, c) in enumerate(zip(self.weights, self.means, self.covs, strict=True)):
            l_chol = np.linalg.cholesky(c)
            z = np.linalg.inv(l_chol) @ (x - mu).T              # whitened residuals (D x D inverse)
            logdet = 2 * np.log(np.diag(l_chol)).sum()
            out[:, j] = np.log(w) - 0.5 * ((z * z).sum(0) + logdet + x.shape[1] * _LOG2PI)
        return out

    def loglik(self, x: np.ndarray) -> np.ndarray:
        """(N,): log p(x) under the mixture (log-sum-exp over components)."""
        c = self.component_loglik(x)
        m = c.max(1, keepdims=True)
        return (m + np.log(np.exp(c - m).sum(1, keepdims=True)))[:, 0]

    def refit(self, x: np.ndarray, reg: float = 1.0) -> GMM:
        """One GrabCut step: hard-assign x to its likeliest component, re-fit."""
        labels = self.component_loglik(x).argmax(1)
        return GMM.fit(x, k=len(self.weights), labels=labels, reg=reg)
