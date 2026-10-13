"""Bundle adjustment: refine every camera and every 3D point together so that
the reprojected points land on the observed ones (Triggs et al., 2000).

Unknowns: per camera an axis-angle rotation and a translation (6), per point a
position (3), and optionally one focal length shared by all frames (a single
shot has one lens; a zoom would break this). Residuals: for each observation
(camera i sees point j at u_ij), the 2-vector π(K, R_i X_j + t_i) - u_ij.

The Jacobian is sparse: an observation's two rows touch only its camera's six
columns, its point's three, and the focal length. With 30 cameras and 3,000
points it has ~20,000 x 2 rows by 9,000 columns, ~0.1% non-zero. Its blocks,
written out analytically:

    x_c = R X + t,   p = (x_c / z_c, y_c / z_c),   u = f p + c
    ∂u/∂x_c = f [[1/z, 0, -x/z²], [0, 1/z, -y/z²]]
    ∂x_c/∂X = R,   ∂x_c/∂t = I,   ∂x_c/∂r = -R [X]ₓ J_r(r)   (J_r: SO(3) right Jacobian)
    ∂u/∂f = p

Two solvers share these blocks:

  "schur" (default, ours) — Levenberg-Marquardt on the normal equations
      [U  W ] [δc]   [-g_c]
      [Wᵀ V ] [δp] = [-g_p]
  V is block-diagonal (one 3x3 block per point: points do not interact), so
  points are eliminated cheaply: (U - W V⁻¹ Wᵀ) δc = -g_c + W V⁻¹ g_p — the
  Schur complement, a small dense system in the camera unknowns only — then
  each δp_j = V_j⁻¹(-g_pj - W_jᵀ δc). This is the trick that makes BA with
  millions of points feasible (Ceres, g2o, COLMAP all do it). A Huber loss is
  applied by iteratively reweighting observations.

  "scipy" — the same residuals and sparse Jacobian handed to
  scipy.optimize.least_squares (trust-region reflective, LSMR inner solver).
  Its inexact inner solves crawl along the gauge freedom (global scale), so it
  needs many more iterations to reach the same error: kept for comparison.

The first camera is held fixed (it defines the world frame); the global scale
stays free and the damping pins it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .camera import log_rotation, right_jacobian, rodrigues_batch


@dataclass
class BAResult:
    rots: np.ndarray            # (C, 3, 3)
    ts: np.ndarray              # (C, 3)
    points: np.ndarray          # (P, 3)
    focal: float
    initial_rmse: float         # mean reprojection error (px) before (name kept short: mean L2, not RMS)
    final_rmse: float           # and after
    nfev: int


def reprojection(rots, ts, points, focal, centre, cam_idx, pt_idx) -> np.ndarray:
    xc = np.einsum("nij,nj->ni", rots[cam_idx], points[pt_idx]) + ts[cam_idx]
    return focal * xc[:, :2] / xc[:, 2:3] + centre


class Problem:
    """Parameter packing, residuals and analytic Jacobian blocks."""

    def __init__(self, rots, ts, points, focal, centre, cam_idx, pt_idx, obs, fixed=(0,),
                 refine_focal=False):
        self.rots, self.ts = np.asarray(rots, float), np.asarray(ts, float)
        self.n_pts = len(points)
        self.centre = np.asarray(centre, float)
        self.cam_idx, self.pt_idx, self.obs = np.asarray(cam_idx), np.asarray(pt_idx), np.asarray(obs, float)
        self.focal0, self.refine_focal = float(focal), refine_focal
        n_cam = len(self.rots)
        self.free = np.array([c not in fixed for c in range(n_cam)])
        self.slot = np.full(n_cam, -1)
        self.slot[self.free] = np.arange(self.free.sum())
        self.n_free = int(self.free.sum())
        self.n_c = 6 * self.n_free + int(refine_focal)      # camera-side unknowns
        cams = [np.r_[log_rotation(self.rots[c]), self.ts[c]] for c in np.flatnonzero(self.free)]
        self.x0 = np.concatenate([np.concatenate(cams) if cams else np.zeros(0),
                                  np.asarray(points, float).ravel()]
                                 + ([np.array([focal])] if refine_focal else []))

    def split(self, x):
        cams = x[:6 * self.n_free].reshape(self.n_free, 6)
        pts = x[6 * self.n_free:6 * self.n_free + 3 * self.n_pts].reshape(self.n_pts, 3)
        f = x[-1] if self.refine_focal else self.focal0
        r_all, t_all = self.rots.copy(), self.ts.copy()
        if self.n_free:
            r_all[self.free] = rodrigues_batch(cams[:, :3])
            t_all[self.free] = cams[:, 3:]
        return cams, r_all, t_all, pts, float(f)

    def residuals(self, x) -> np.ndarray:
        _, r_all, t_all, pts, f = self.split(x)
        return reprojection(r_all, t_all, pts, f, self.centre, self.cam_idx, self.pt_idx) - self.obs

    def blocks(self, x):
        """(J_cam (n,2,6) — zero rows for fixed cameras, J_pt (n,2,3), J_f (n,2))."""
        cams, r_all, t_all, pts, f = self.split(x)
        rr = r_all[self.cam_idx]
        xw = pts[self.pt_idx]
        xc = np.einsum("nij,nj->ni", rr, xw) + t_all[self.cam_idx]
        z = xc[:, 2]
        n = len(z)
        du = np.zeros((n, 2, 3))
        du[:, 0, 0] = du[:, 1, 1] = f / z
        du[:, 0, 2] = -f * xc[:, 0] / z ** 2
        du[:, 1, 2] = -f * xc[:, 1] / z ** 2
        j_pt = du @ rr
        j_cam = np.zeros((n, 2, 6))
        slot = self.slot[self.cam_idx]
        o = np.flatnonzero(slot >= 0)
        if len(o):
            jr_all = np.stack([right_jacobian(v) for v in cams[:, :3]])
            sk = np.zeros((len(o), 3, 3))
            xo = xw[o]
            sk[:, 0, 1], sk[:, 0, 2], sk[:, 1, 2] = -xo[:, 2], xo[:, 1], -xo[:, 0]
            sk[:, 1, 0], sk[:, 2, 0], sk[:, 2, 1] = xo[:, 2], -xo[:, 1], xo[:, 0]
            j_cam[o, :, :3] = -du[o] @ rr[o] @ sk @ jr_all[slot[o]]
            j_cam[o, :, 3:] = du[o]
        j_f = xc[:, :2] / z[:, None]
        return j_cam, j_pt, j_f

    def sparse_jacobian(self, x):
        from scipy.sparse import csr_matrix

        j_cam, j_pt, j_f = self.blocks(x)
        n = len(j_pt)
        r_idx = np.arange(2 * n).reshape(n, 2)
        pc = 6 * self.n_free + 3 * self.pt_idx[:, None] + np.arange(3)[None]
        rows = [np.repeat(r_idx[:, :, None], 3, 2).ravel()]
        cols = [np.repeat(pc[:, None, :], 2, 1).ravel()]
        vals = [j_pt.ravel()]
        slot = self.slot[self.cam_idx]
        o = np.flatnonzero(slot >= 0)
        if len(o):
            cc = 6 * slot[o][:, None] + np.arange(6)[None]
            rows.append(np.repeat(r_idx[o][:, :, None], 6, 2).ravel())
            cols.append(np.repeat(cc[:, None, :], 2, 1).ravel())
            vals.append(j_cam[o].ravel())
        if self.refine_focal:
            rows.append(r_idx.ravel())
            cols.append(np.full(2 * n, len(self.x0) - 1))
            vals.append(j_f.ravel())
        return csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(2 * n, len(self.x0)))


def _huber_weights(r: np.ndarray, delta: float) -> np.ndarray:
    e = np.linalg.norm(r, axis=1)
    return np.where(e <= delta, 1.0, delta / np.maximum(e, 1e-12))


def _huber_cost(r: np.ndarray, delta: float) -> float:
    e = np.linalg.norm(r, axis=1)
    return float(np.where(e <= delta, 0.5 * e ** 2, delta * (e - 0.5 * delta)).sum())


def _schur_lm(pb: Problem, huber_px: float, max_iter: int) -> tuple[np.ndarray, int]:
    x = pb.x0.copy()
    n_pts, n_c, nf6 = pb.n_pts, pb.n_c, 6 * pb.n_free
    slot = pb.slot[pb.cam_idx]
    has = slot >= 0
    r = pb.residuals(x)
    cost = _huber_cost(r, huber_px)
    lam, nu, evals = 1e-4, 2.0, 1
    for _ in range(max_iter):
        w = _huber_weights(r, huber_px)[:, None, None]
        j_cam, j_pt, j_f = pb.blocks(x)
        # camera-side Jacobian per observation, (n, 2, n_c) would be wasteful; keep blocks.
        jc = j_cam
        if pb.refine_focal:
            jc = np.concatenate([j_cam, j_f[:, :, None]], 2)        # (n, 2, 7)
        jtw_c = np.transpose(jc, (0, 2, 1)) * w.transpose(0, 2, 1)            # (n, k, 2)
        u_blk = jtw_c @ jc                                                     # (n, k, k)
        g_c_blk = (jtw_c @ r[:, :, None])[:, :, 0]                             # (n, k)
        w_blk = jtw_c @ j_pt                                                   # (n, k, 3)
        v_blk = np.transpose(j_pt, (0, 2, 1)) @ (j_pt * w)                     # (n, 3, 3)
        g_p_blk = (np.transpose(j_pt, (0, 2, 1)) @ (w * r[:, :, None]))[:, :, 0]

        # Accumulate per-observation blocks into U (dense, cameras), V (3x3 per
        # point) and W (cameras x 3 per point) with scatter-adds.
        hs, sl, pt = np.flatnonzero(has), slot[has], pb.pt_idx[has]
        u6 = np.zeros((pb.n_free, 6, 6))
        np.add.at(u6, sl, u_blk[hs, :6, :6])
        u_mat = np.zeros((n_c, n_c))
        for c in range(pb.n_free):
            u_mat[6 * c:6 * c + 6, 6 * c:6 * c + 6] = u6[c]
        g6 = np.zeros((pb.n_free, 6))
        np.add.at(g6, sl, g_c_blk[hs, :6])
        g_c = np.zeros(n_c)
        g_c[:nf6] = g6.ravel()
        w6 = np.zeros((n_pts, pb.n_free, 6, 3))
        np.add.at(w6, (pt, sl), w_blk[hs, :6])
        wm = np.zeros((n_pts, n_c, 3))                                         # W_j, dense per point
        wm[:, :nf6] = w6.reshape(n_pts, nf6, 3)
        if pb.refine_focal:
            uf = np.zeros((pb.n_free, 6))
            np.add.at(uf, sl, u_blk[hs, :6, 6])
            u_mat[:nf6, -1] = u_mat[-1, :nf6] = uf.ravel()
            u_mat[-1, -1] = u_blk[:, 6, 6].sum()
            g_c[-1] = g_c_blk[:, 6].sum()
            np.add.at(wm[:, -1], pb.pt_idx, w_blk[:, 6])
        v = np.zeros((n_pts, 3, 3))
        np.add.at(v, pb.pt_idx, v_blk)
        g_p = np.zeros((n_pts, 3))
        np.add.at(g_p, pb.pt_idx, g_p_blk)

        while True:
            ud = u_mat + lam * np.diag(np.maximum(np.diag(u_mat), 1e-9))
            vd = v + lam * np.einsum("ij,nj->nij", np.eye(3), np.maximum(np.diagonal(v, 0, 1, 2), 1e-9))
            vinv = np.linalg.inv(vd)
            wv = wm @ vinv                                                     # (P, n_c, 3)
            s = ud - wv.transpose(1, 0, 2).reshape(n_c, -1) @ wm.transpose(1, 0, 2).reshape(n_c, -1).T
            rhs = -g_c + np.einsum("pik,pk->i", wv, g_p)
            try:
                dc = np.linalg.solve(s, rhs) if n_c else np.zeros(0)
            except np.linalg.LinAlgError:
                lam *= nu
                nu *= 2
                continue
            dp = np.einsum("pij,pj->pi", vinv, -g_p - np.einsum("pki,k->pi", wm, dc))
            step = np.concatenate([dc[:nf6], dp.ravel(), dc[nf6:]])
            x_new = x + step
            r_new = pb.residuals(x_new)
            evals += 1
            cost_new = _huber_cost(r_new, huber_px) if np.all(np.isfinite(r_new)) else np.inf
            if cost_new < cost:
                done = (cost - cost_new) < 1e-9 * cost
                x, r, cost = x_new, r_new, cost_new
                lam = max(lam / 3, 1e-9)
                nu = 2.0
                break
            lam *= nu
            nu *= 2
            if lam > 1e10:
                return x, evals
        if done:
            break
    return x, evals


def bundle_adjust(rots: np.ndarray, ts: np.ndarray, points: np.ndarray, focal: float,
                  centre: tuple[float, float], cam_idx: np.ndarray, pt_idx: np.ndarray,
                  obs: np.ndarray, fixed: tuple[int, ...] = (0,), refine_focal: bool = False,
                  huber_px: float = 2.0, max_iter: int = 50, solver: str = "schur") -> BAResult:
    pb = Problem(rots, ts, points, focal, centre, cam_idx, pt_idx, obs, fixed, refine_focal)
    r0 = pb.residuals(pb.x0)
    if solver == "schur":
        x, nfev = _schur_lm(pb, huber_px, max_iter)
    elif solver == "scipy":
        from scipy.optimize import least_squares

        res = least_squares(lambda z: pb.residuals(z).ravel(), pb.x0, jac=pb.sparse_jacobian,
                            method="trf", tr_solver="lsmr", loss="huber", f_scale=huber_px,
                            x_scale="jac", max_nfev=max_iter)
        x, nfev = res.x, int(res.nfev)
    else:
        raise ValueError(solver)
    _, r_all, t_all, pts, f = pb.split(x)
    r1 = pb.residuals(x)
    return BAResult(r_all, t_all, pts, f, float(np.linalg.norm(r0, axis=1).mean()),
                    float(np.linalg.norm(r1, axis=1).mean()), nfev)
