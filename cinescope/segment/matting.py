"""Closed-form matting (Levin, Lischinski & Weiss, 2008), from scratch.

A binary mask says each pixel is person or not; at hair, motion blur and
defocused edges the truth is a mix, I = alpha*F + (1 - alpha)*B. Matting
recovers alpha. Closed-form matting assumes that inside every small window w
(3x3 here) F and B each lie on a line in colour space — the "colour line"
model. Then alpha is an affine function of colour in the window,
alpha_i ≈ a·I_i + b, and eliminating a and b analytically leaves a quadratic
cost alpha^T L alpha with the *matting Laplacian*

    L_ij = sum over windows k containing i and j of
           delta_ij - (1/|w|) (1 + (I_i - mu_k)^T (Sigma_k + eps/|w| E)^-1 (I_j - mu_k))

(mu_k, Sigma_k: mean and covariance of the window's colours). A trimap fixes
alpha = 1 on sure foreground and 0 on sure background; the unknown band
minimises alpha^T L alpha subject to that — one sparse, symmetric positive
definite linear system, L_uu alpha_u = -L_uk alpha_k, solved here by
conjugate gradients.

Trimap from a mask: erode it for sure-foreground, dilate for sure-background,
the band between is unknown.

Foreground colours: a cut-out pasted on a new background needs F, not I
(I still contains the old background at the edges). `estimate_foreground`
uses blur-fusion (Forte & Pitié, 2021): local weighted means of foreground
and background colours, then one correction so that alpha*F + (1-alpha)*B
reproduces I.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy import ndimage as ndi
from scipy.sparse.linalg import cg


def trimap_from_mask(mask: np.ndarray, radius: int | None = None) -> np.ndarray:
    """(H, W) bool -> float trimap: 1 sure fg, 0 sure bg, 0.5 unknown.
    radius defaults to 3% of the mask's height, clipped to 3..15 px."""
    if radius is None:
        ys = np.nonzero(mask.any(1))[0]
        hgt = (ys.max() - ys.min() + 1) if len(ys) else 0
        radius = int(np.clip(round(0.03 * hgt), 3, 15))
    st = np.ones((3, 3), bool)
    fg = ndi.binary_erosion(mask, st, radius)
    bg = ~ndi.binary_dilation(mask, st, radius)
    tri = np.full(mask.shape, 0.5)
    tri[fg] = 1.0
    tri[bg] = 0.0
    return tri


def matting_laplacian(img: np.ndarray, active: np.ndarray | None = None, eps: float = 1e-7,
                      r: int = 1) -> sp.csr_matrix:
    """Sparse (N, N) matting Laplacian of an (H, W, 3) float image in 0..1.
    Only windows centred on `active` pixels contribute (all interior windows
    if None) — windows far from the unknown band do not change the solution."""
    h, w, c = img.shape
    n = h * w
    win = (2 * r + 1) ** 2
    idx = np.arange(n).reshape(h, w)
    centres = np.ones((h, w), bool)
    centres[:r], centres[-r:], centres[:, :r], centres[:, -r:] = False, False, False, False
    if active is not None:
        centres &= active
    cy, cx = np.nonzero(centres)
    offs = [(dy, dx) for dy in range(-r, r + 1) for dx in range(-r, r + 1)]
    widx = np.stack([idx[cy + dy, cx + dx] for dy, dx in offs], 1)        # (K, win)
    flat = img.reshape(-1, c)
    wcol = flat[widx]                                                    # (K, win, 3)
    mu = wcol.mean(1, keepdims=True)
    d = wcol - mu
    cov = np.einsum("kwi,kwj->kij", d, d) / win + (eps / win) * np.eye(c)
    inv = np.linalg.inv(cov)
    # (K, win, win): 1/|w| (1 + d_i^T inv d_j)
    vals = (1.0 + np.einsum("kwi,kij,kvj->kwv", d, inv, d)) / win
    rows = np.repeat(widx, win, axis=1).ravel()
    cols = np.tile(widx, (1, win)).ravel()
    # L = D - W where W_ij = sum_k vals; D_ii = sum_j W_ij (each window's rows sum to 1)
    wmat = sp.coo_matrix((vals.ravel(), (rows, cols)), shape=(n, n)).tocsr()
    deg = np.asarray(wmat.sum(1)).ravel()
    return (sp.diags(deg) - wmat).tocsr()


def closed_form_alpha(img: np.ndarray, trimap: np.ndarray, eps: float = 1e-7,
                      tol: float = 1e-7, maxiter: int = 4000) -> np.ndarray:
    """img (H, W, 3) uint8 or float 0..1; trimap (H, W) 0 / 0.5 / 1 -> alpha (H, W)."""
    img = np.asarray(img, np.float64)
    if img.max() > 1.5:
        img = img / 255.0
    known_fg = trimap >= 0.99
    known_bg = trimap <= 0.01
    unknown = ~(known_fg | known_bg)
    alpha = known_fg.astype(np.float64)
    if not unknown.any():
        return alpha
    # windows touching the unknown band (a window centred within r of it)
    active = ndi.binary_dilation(unknown, np.ones((3, 3), bool), 1)
    lap = matting_laplacian(img, active, eps)
    u = unknown.ravel()
    k = ~u
    l_uu = lap[u][:, u]
    rhs = -lap[u][:, k] @ alpha.ravel()[k]
    # Jacobi preconditioner: L_uu is SPD with a strong diagonal
    diag = l_uu.diagonal()
    diag[diag <= 0] = 1.0
    precond = sp.diags(1.0 / diag)
    x0 = np.full(int(u.sum()), 0.5)
    sol, info = cg(l_uu, rhs, x0=x0, rtol=tol, maxiter=maxiter, M=precond)
    out = alpha.ravel().copy()
    out[u] = sol
    return np.clip(out, 0, 1).reshape(trimap.shape)


def _box_blur(x: np.ndarray, r: int) -> np.ndarray:
    size = 2 * r + 1
    if x.ndim == 3:
        return ndi.uniform_filter(x, size=(size, size, 1), mode="nearest")
    return ndi.uniform_filter(x, size=size, mode="nearest")


def estimate_foreground(img: np.ndarray, alpha: np.ndarray, radii=(90, 6)) -> np.ndarray:
    """Blur-fusion foreground estimate; img uint8 or 0..1 -> F float 0..1."""
    img = np.asarray(img, np.float64)
    if img.max() > 1.5:
        img = img / 255.0
    a = alpha[..., None]
    f, b = img, img
    for r in radii:
        ba = _box_blur(alpha, r)[..., None]
        bf = _box_blur(f * a, r) / (ba + 1e-5)
        bb = _box_blur(b * (1 - a), r) / ((1 - ba) + 1e-5)
        f = np.clip(bf + a * (img - a * bf - (1 - a) * bb), 0, 1)
        b = bb
    return f


def sad(alpha: np.ndarray, truth: np.ndarray, region: np.ndarray | None = None) -> float:
    """Sum of absolute differences, in thousands (the alphamatting.com convention)."""
    d = np.abs(alpha - truth)
    return float((d if region is None else d[region]).sum() / 1000.0)


def mse(alpha: np.ndarray, truth: np.ndarray, region: np.ndarray | None = None) -> float:
    d = (alpha - truth) ** 2
    return float(d.mean() if region is None else d[region].mean())
