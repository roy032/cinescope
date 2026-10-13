"""3D Gaussian splatting (Kerbl et al., 2023), from scratch in PyTorch, on the CPU.

The scene is a cloud of N anisotropic 3D Gaussians, each with a mean μ, a
covariance Σ = R S Sᵀ Rᵀ (rotation from a unit quaternion, per-axis scales S
stored as logs so they stay positive), an opacity o (stored as a logit) and an
RGB colour (stored as a logit; view-dependent spherical harmonics are left out
— degree 0 only). Rendering a camera (K, R_c, t_c):

1. Project. x_c = R_c μ + t_c. The image of a 3D Gaussian under a perspective
   camera is not exactly Gaussian; EWA splatting (Zwicker et al., 2001)
   linearises the projection at the mean, with Jacobian
       J = [[fx/z, 0, -fx x/z²], [0, fy/z, -fy y/z²]],
   so the 2D covariance is Σ' = J R_c Σ R_cᵀ Jᵀ (+ 0.3 px² on the diagonal, a
   low-pass filter so no splat is thinner than a pixel).
2. Sort by depth, front to back.
3. Composite. For each pixel, walking the sorted splats:
       α_i = o_i · exp(-½ dᵀ Σ'_i⁻¹ d),  d = pixel - projected mean
       C = Σ_i c_i α_i T_i,   T_i = Π_{j<i} (1 - α_j)
   the same front-to-back "over" operator as volume rendering in NeRF, but
   over a sorted list of splats instead of samples along a ray.
4. Tiles. The image is cut into 16x16-pixel tiles; each splat is listed only
   in the tiles its 3σ ellipse touches, so a pixel only visits nearby splats.

Every step is a differentiable torch operation, so the photometric loss
against the training frames back-propagates to every Gaussian parameter;
Adam does the rest. The official implementation does the same with CUDA
kernels (and gsplat likewise) — neither runs without a GPU, hence this one.

Adaptive density control (simplified from the paper): every few hundred
steps, Gaussians whose projected means receive large gradients (the
optimiser keeps trying to move them: under-reconstructed regions) are
cloned if small and split in two if large; nearly transparent ones are
pruned.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

TILE = 16


def quat_to_rot(q: torch.Tensor) -> torch.Tensor:
    q = q / q.norm(dim=-1, keepdim=True)
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1).reshape(q.shape[:-1] + (3, 3))


@dataclass
class Camera:
    k: np.ndarray
    rot: np.ndarray
    t: np.ndarray
    width: int
    height: int


class Gaussians(torch.nn.Module):
    def __init__(self, means: np.ndarray, colors: np.ndarray, scales: np.ndarray, opacity: float = 0.1):
        super().__init__()
        n = len(means)
        f = torch.float32
        self.means = torch.nn.Parameter(torch.as_tensor(means, dtype=f))
        self.log_scales = torch.nn.Parameter(torch.log(torch.as_tensor(scales, dtype=f)).reshape(n, 1).repeat(1, 3)
                                             if np.ndim(scales) == 1 else torch.log(torch.as_tensor(scales, dtype=f)))
        quats = torch.zeros(n, 4)
        quats[:, 0] = 1
        self.quats = torch.nn.Parameter(quats)
        self.opacity_logit = torch.nn.Parameter(torch.full((n,), float(np.log(opacity / (1 - opacity)))))
        c = torch.as_tensor(np.clip(colors, 0.01, 0.99), dtype=f)
        self.color_logit = torch.nn.Parameter(torch.log(c / (1 - c)))

    def __len__(self) -> int:
        return self.means.shape[0]

    def covariances(self) -> torch.Tensor:
        r = quat_to_rot(self.quats)
        s = torch.exp(self.log_scales)
        m = r * s[:, None, :]
        return m @ m.transpose(1, 2)


def project(g: Gaussians, cam: Camera, near: float = 0.05):
    """Projected means (N, 2), 2D covariance inverses (N, 3: a, b, c), depths,
    3σ radii in pixels, and a validity mask."""
    dt = g.means.dtype                          # float32 normally; float64 for gradient checks
    rot = torch.as_tensor(cam.rot, dtype=dt)
    t = torch.as_tensor(cam.t, dtype=dt)
    fx, fy, cx, cy = (float(v) for v in (cam.k[0, 0], cam.k[1, 1], cam.k[0, 2], cam.k[1, 2]))
    xc = g.means @ rot.T + t
    z = xc[:, 2]
    valid = z > near
    zs = torch.where(valid, z, torch.ones_like(z))
    u = fx * xc[:, 0] / zs + cx
    v = fy * xc[:, 1] / zs + cy
    j = torch.zeros(len(g), 2, 3, dtype=dt)
    j[:, 0, 0] = fx / zs
    j[:, 0, 2] = -fx * xc[:, 0] / zs ** 2
    j[:, 1, 1] = fy / zs
    j[:, 1, 2] = -fy * xc[:, 1] / zs ** 2
    jw = j @ rot
    cov2 = jw @ g.covariances() @ jw.transpose(1, 2)
    a = cov2[:, 0, 0] + 0.3
    b = cov2[:, 0, 1]
    c = cov2[:, 1, 1] + 0.3
    det = a * c - b * b
    valid = valid & (det > 1e-12)
    det = torch.where(valid, det, torch.ones_like(det))
    conic = torch.stack([c / det, -b / det, a / det], -1)
    mid = 0.5 * (a + c)
    lam = mid + torch.sqrt(torch.clamp(mid * mid - det, min=0.1))
    radius = torch.ceil(3 * torch.sqrt(lam))
    valid = valid & (u + radius > 0) & (u - radius < cam.width) & (v + radius > 0) & (v - radius < cam.height)
    return torch.stack([u, v], -1), conic, z, radius, valid


def _tile_lists(means2d, radius, depth, valid, width, height):
    """(order of (gaussian, tile) pairs sorted by tile then depth, tile ids, gaussian ids)."""
    with torch.no_grad():
        tx = (width + TILE - 1) // TILE
        ty = (height + TILE - 1) // TILE
        idx = torch.nonzero(valid).squeeze(1)
        u, v, r = means2d[idx, 0], means2d[idx, 1], radius[idx]
        x0 = torch.clamp(((u - r) / TILE).floor(), 0, tx - 1).long()
        x1 = torch.clamp(((u + r) / TILE).floor(), 0, tx - 1).long()
        y0 = torch.clamp(((v - r) / TILE).floor(), 0, ty - 1).long()
        y1 = torch.clamp(((v + r) / TILE).floor(), 0, ty - 1).long()
        nx, ny = x1 - x0 + 1, y1 - y0 + 1
        cnt = nx * ny
        gid = torch.repeat_interleave(idx, cnt)
        start = torch.repeat_interleave(torch.cumsum(cnt, 0) - cnt, cnt)
        local = torch.arange(int(cnt.sum())) - start
        nxr = torch.repeat_interleave(nx, cnt)
        tile = (torch.repeat_interleave(y0, cnt) + local // nxr) * tx + torch.repeat_interleave(x0, cnt) + local % nxr
        rank = torch.empty_like(depth, dtype=torch.long)
        rank[torch.argsort(depth)] = torch.arange(len(depth))
        key = tile * len(depth) + rank[gid]
        order = torch.argsort(key)
        return tile[order], gid[order], tx, ty


def render(g: Gaussians, cam: Camera, background=(0.0, 0.0, 0.0), return_alpha: bool = False,
           means2d_grad_hook=None, budget: int = 4096):
    """(H, W, 3) image in [0, 1] (and the accumulated opacity, if asked).
    `means2d_grad_hook` receives ∂L/∂(projected means) during backward —
    what densification needs. `budget` caps tiles x splats per batch (memory)."""
    w, h = cam.width, cam.height
    means2d, conic, depth, radius, valid = project(g, cam)
    if means2d_grad_hook is not None and means2d.requires_grad:
        means2d.register_hook(means2d_grad_hook)
    tile, gid, tx, ty = _tile_lists(means2d, radius, depth.detach(), valid, w, h)
    opac = torch.sigmoid(g.opacity_logit)
    col = torch.sigmoid(g.color_logit)
    dt = g.means.dtype
    bg = torch.as_tensor(background, dtype=dt)
    n_tiles, npx = tx * ty, TILE * TILE
    oy, ox = torch.meshgrid(torch.arange(TILE, dtype=dt), torch.arange(TILE, dtype=dt), indexing="ij")
    ty_all, tx_all = torch.div(torch.arange(n_tiles), tx, rounding_mode="floor"), torch.arange(n_tiles) % tx
    px_all = ox.reshape(1, -1) + (tx_all * TILE)[:, None] + 0.5                 # (tiles, P)
    py_all = oy.reshape(1, -1) + (ty_all * TILE)[:, None] + 0.5
    bounds = torch.searchsorted(tile, torch.arange(n_tiles + 1)) if len(tile) else torch.zeros(n_tiles + 1, dtype=torch.long)
    counts = bounds[1:] - bounds[:-1]
    # Tiles are processed in batches of similar splat counts, padded to the
    # batch's longest list: one set of tensor ops per batch instead of per tile.
    busy = torch.nonzero(counts > 0).squeeze(1)
    busy = busy[torch.argsort(counts[busy])]
    done_idx, done_rgb, done_acc = [], [], []
    i = 0
    while i < len(busy):
        j = i + 1
        while j < len(busy) and (j + 1 - i) * int(counts[busy[j]]) <= budget:
            j += 1
        chunk = busy[i:j]
        i = j
        kmax = int(counts[chunk].max())
        pos = torch.arange(kmax)[None]
        live = pos < counts[chunk][:, None]                                     # (T, K)
        gidx = gid[torch.clamp(bounds[chunk][:, None] + pos, max=len(gid) - 1)]
        mu = means2d[gidx]                                                      # (T, K, 2)
        dx = px_all[chunk][:, None, :] - mu[..., 0:1]                          # (T, K, P)
        dy = py_all[chunk][:, None, :] - mu[..., 1:2]
        cn = conic[gidx]
        power = -0.5 * (cn[..., 0:1] * dx * dx + cn[..., 2:3] * dy * dy) - cn[..., 1:2] * dx * dy
        alpha = torch.clamp(opac[gidx][..., None] * torch.exp(torch.clamp(power, min=-12.0, max=0.0)), max=0.99)
        alpha = alpha * ((alpha > 1.0 / 255) & live[..., None]).to(dt)
        # T_i = Π_{j<i} (1 - α_j), as exp of an exclusive cumulative sum of logs:
        # same value as cumprod, but its backward pass is a cheap reversed cumsum
        log_t = torch.cumsum(torch.log1p(-alpha), 1)
        trans = torch.exp(torch.cat([torch.zeros_like(log_t[:, :1]), log_t[:, :-1]], 1))
        wgt = alpha * trans                                                     # (T, K, P)
        rgb = torch.einsum("tkp,tkc->tpc", wgt, col[gidx])
        a_acc = wgt.sum(1)
        done_idx.append(chunk)
        done_rgb.append(rgb + (1 - a_acc)[..., None] * bg)
        done_acc.append(a_acc)
    tiles_rgb = bg.expand(n_tiles, npx, 3).clone()
    tiles_acc = torch.zeros(n_tiles, npx, dtype=dt)
    if done_idx:
        idx = torch.cat(done_idx)
        tiles_rgb = tiles_rgb.index_copy(0, idx, torch.cat(done_rgb))
        tiles_acc = tiles_acc.index_copy(0, idx, torch.cat(done_acc))
    out = tiles_rgb.reshape(ty, tx, TILE, TILE, 3).permute(0, 2, 1, 3, 4).reshape(ty * TILE, tx * TILE, 3)
    acc = tiles_acc.reshape(ty, tx, TILE, TILE).permute(0, 2, 1, 3).reshape(ty * TILE, tx * TILE)
    out, acc = out[:h, :w], acc[:h, :w]
    return (out, acc) if return_alpha else out


def render_dense_reference(g: Gaussians, cam: Camera, background=(0.0, 0.0, 0.0)) -> torch.Tensor:
    """Same image without tiles (every pixel visits every splat): O(N·P), only
    for testing that the tiled renderer is exact."""
    means2d, conic, depth, _, valid = project(g, cam)
    ids = torch.nonzero(valid).squeeze(1)
    ids = ids[torch.argsort(depth[ids])]
    ys, xs = torch.meshgrid(torch.arange(cam.height, dtype=torch.float32),
                            torch.arange(cam.width, dtype=torch.float32), indexing="ij")
    px, py = xs.reshape(-1) + 0.5, ys.reshape(-1) + 0.5
    dx = px[None] - means2d[ids, 0:1]
    dy = py[None] - means2d[ids, 1:2]
    cn = conic[ids]
    power = -0.5 * (cn[:, 0:1] * dx * dx + cn[:, 2:3] * dy * dy) - cn[:, 1:2] * dx * dy
    alpha = torch.clamp(torch.sigmoid(g.opacity_logit)[ids, None] * torch.exp(torch.clamp(power, min=-12.0, max=0.0)), max=0.99)
    # the tiled renderer drops splats outside their 3σ tiles; mimic by the same 1/255 cut
    alpha = alpha * (alpha > 1.0 / 255).float()
    trans = torch.cumprod(torch.cat([torch.ones(1, alpha.shape[1]), 1 - alpha[:-1]], 0), 0)
    wgt = alpha * trans
    rgb = wgt.T @ torch.sigmoid(g.color_logit)[ids] + (1 - wgt.sum(0))[:, None] * torch.as_tensor(background)
    return rgb.reshape(cam.height, cam.width, 3)


# ------------------------------------------------------------------ training
def ssim(a: torch.Tensor, b: torch.Tensor, win: int = 7) -> torch.Tensor:
    """Mean structural similarity of two (H, W, 3) images (box window)."""
    x, y = a.permute(2, 0, 1)[None], b.permute(2, 0, 1)[None]
    pool = torch.nn.functional.avg_pool2d
    mx, my = pool(x, win, 1), pool(y, win, 1)
    vx = pool(x * x, win, 1) - mx ** 2
    vy = pool(y * y, win, 1) - my ** 2
    cxy = pool(x * y, win, 1) - mx * my
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    s = ((2 * mx * my + c1) * (2 * cxy + c2)) / ((mx ** 2 + my ** 2 + c1) * (vx + vy + c2))
    return s.mean()


def psnr(a: torch.Tensor | np.ndarray, b: torch.Tensor | np.ndarray) -> float:
    a = torch.as_tensor(a, dtype=torch.float32)
    b = torch.as_tensor(b, dtype=torch.float32)
    mse = torch.mean((a - b) ** 2).item()
    return float(10 * np.log10(1.0 / max(mse, 1e-12)))


def init_scales(points: np.ndarray, k: int = 3) -> np.ndarray:
    """Initial isotropic scale: mean distance to the k nearest neighbours (as in the paper)."""
    from scipy.spatial import cKDTree

    d, _ = cKDTree(points).query(points, k=k + 1)
    return np.clip(d[:, 1:].mean(1), 1e-4, None)


@dataclass
class TrainConfig:
    iters: int = 1500
    lr_means: float = 1.6e-4          # x scene extent
    lr_scales: float = 5e-3
    lr_quats: float = 1e-3
    lr_opacity: float = 5e-2
    lr_color: float = 1e-2
    ssim_weight: float = 0.2
    densify_every: int = 100
    densify_from: int = 200
    densify_until: int = 1000
    grad_thresh: float = 0.0002       # mean |∂L/∂(projected mean)| in normalised image units
    max_gaussians: int = 30000
    prune_opacity: float = 0.005
    seed: int = 0


def _make_optim(g: Gaussians, cfg: TrainConfig, extent: float):
    return torch.optim.Adam([
        {"params": [g.means], "lr": cfg.lr_means * extent},
        {"params": [g.log_scales], "lr": cfg.lr_scales},
        {"params": [g.quats], "lr": cfg.lr_quats},
        {"params": [g.opacity_logit], "lr": cfg.lr_opacity},
        {"params": [g.color_logit], "lr": cfg.lr_color},
    ], eps=1e-15)


def _densify(g: Gaussians, grad_acc, grad_cnt, cfg: TrainConfig, extent: float, rng) -> Gaussians:
    with torch.no_grad():
        avg = grad_acc / torch.clamp(grad_cnt, min=1)
        big = torch.exp(g.log_scales).max(1).values > 0.01 * extent
        want = avg > cfg.grad_thresh
        room = cfg.max_gaussians - len(g)
        clone = torch.nonzero(want & ~big).squeeze(1)[: max(room // 2, 0)]
        split = torch.nonzero(want & big).squeeze(1)[: max(room // 4, 0)]
        keep = torch.sigmoid(g.opacity_logit) > cfg.prune_opacity
        keep[split] = False                      # a split Gaussian is replaced by its two children
        means = [g.means[keep]]
        ls = [g.log_scales[keep]]
        qs = [g.quats[keep]]
        ops = [g.opacity_logit[keep]]
        cols = [g.color_logit[keep]]
        if len(clone):
            means.append(g.means[clone])
            ls.append(g.log_scales[clone])
            qs.append(g.quats[clone])
            ops.append(g.opacity_logit[clone])
            cols.append(g.color_logit[clone])
        if len(split):
            for _ in range(2):
                std = torch.exp(g.log_scales[split])
                offs = torch.as_tensor(rng.normal(size=std.shape), dtype=torch.float32) * std
                offs = (quat_to_rot(g.quats[split]) @ offs[..., None])[..., 0]
                means.append(g.means[split] + offs)
                ls.append(g.log_scales[split] - np.log(1.6))
                qs.append(g.quats[split])
                ops.append(g.opacity_logit[split])
                cols.append(g.color_logit[split])
        new = Gaussians.__new__(Gaussians)
        torch.nn.Module.__init__(new)
        new.means = torch.nn.Parameter(torch.cat(means))
        new.log_scales = torch.nn.Parameter(torch.cat(ls))
        new.quats = torch.nn.Parameter(torch.cat(qs))
        new.opacity_logit = torch.nn.Parameter(torch.cat(ops))
        new.color_logit = torch.nn.Parameter(torch.cat(cols))
    return new


def train(g: Gaussians, cams: list[Camera], images: list[np.ndarray], cfg: TrainConfig | None = None,
          log_every: int = 100, background=(0.0, 0.0, 0.0), callback=None) -> tuple[Gaussians, list[dict]]:
    """Optimise the Gaussians to reproduce `images` (uint8 (H, W, 3)) from `cams`."""
    cfg = cfg or TrainConfig()
    rng = np.random.default_rng(cfg.seed)
    targets = [torch.as_tensor(im, dtype=torch.float32) / 255.0 for im in images]
    centres = np.stack([-c.rot.T @ c.t for c in cams])
    pts = g.means.detach().numpy()
    extent = float(np.linalg.norm(pts - centres.mean(0), axis=1).max()) * 1.1
    opt = _make_optim(g, cfg, extent)
    grad_acc = torch.zeros(len(g))
    grad_cnt = torch.zeros(len(g))
    log = []
    for it in range(1, cfg.iters + 1):
        i = int(rng.integers(len(cams)))
        cam = cams[i]
        # keep the projected means' gradient (in pixels) to drive densification
        means2d_hook = {}

        def hook(grad, store=means2d_hook):
            store["g"] = grad

        img = render(g, cam, background, means2d_grad_hook=hook)
        l1 = torch.abs(img - targets[i]).mean()
        loss = (1 - cfg.ssim_weight) * l1 + cfg.ssim_weight * (1 - ssim(img, targets[i]))
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            g.quats.data = g.quats.data / g.quats.data.norm(dim=1, keepdim=True)
            if "g" in means2d_hook:
                gr = means2d_hook["g"]
                # gradient w.r.t. normalised image coordinates ([-1, 1], as in the paper)
                norm = torch.sqrt((gr[:, 0] * cam.width) ** 2 + (gr[:, 1] * cam.height) ** 2) / 2
                seen = gr.abs().sum(1) > 0
                grad_acc[seen] += norm[seen]
                grad_cnt[seen] += 1
        if cfg.densify_from <= it <= cfg.densify_until and it % cfg.densify_every == 0:
            g = _densify(g, grad_acc, grad_cnt, cfg, extent, rng)
            opt = _make_optim(g, cfg, extent)
            grad_acc = torch.zeros(len(g))
            grad_cnt = torch.zeros(len(g))
        if it % log_every == 0 or it == 1:
            log.append({"iter": it, "loss": round(float(loss.detach()), 5), "psnr": round(psnr(img.detach(), targets[i]), 2),
                        "gaussians": len(g)})
            if callback:
                callback(log[-1])
    return g, log


# ------------------------------------------------------------------ baseline
def render_points(points: np.ndarray, colors: np.ndarray, cam: Camera, radius: int = 1,
                  background=(0, 0, 0)) -> np.ndarray:
    """The simple baseline: every point a (2r+1)² square of its colour, z-buffered."""
    xc = points @ cam.rot.T + cam.t
    ok = xc[:, 2] > 0.05
    xc, col = xc[ok], colors[ok]
    u = np.rint(cam.k[0, 0] * xc[:, 0] / xc[:, 2] + cam.k[0, 2] - 0.5).astype(int)
    v = np.rint(cam.k[1, 1] * xc[:, 1] / xc[:, 2] + cam.k[1, 2] - 0.5).astype(int)
    out = np.zeros((cam.height, cam.width, 3), np.float64) + np.asarray(background, float)
    zbuf = np.full((cam.height, cam.width), np.inf)
    order = np.argsort(-xc[:, 2])            # far first, near overwrite
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            uu, vv = u[order] + dx, v[order] + dy
            m = (uu >= 0) & (uu < cam.width) & (vv >= 0) & (vv < cam.height)
            z = xc[order, 2][m]
            uu, vv, cc = uu[m], vv[m], col[order][m]
            closer = z < zbuf[vv, uu]
            out[vv[closer], uu[closer]] = cc[closer]
            zbuf[vv[closer], uu[closer]] = z[closer]
    return out
