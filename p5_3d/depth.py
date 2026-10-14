"""Week 16: depth maps — SfM points (sparse), our plane-sweep stereo (dense),
MiDaS small (monocular network) — and how well they agree.

    python p5_3d/depth.py --synthetic
    python p5_3d/depth.py trailers/goldeneye.mp4 --shot 18
    # -> p5_3d/results/depth_<name>.json, outputs/<clip>/depth_shot<N>.jpg

On the synthetic set everything is scored against the true depth. On a film
shot there is no true depth, so the SfM points serve as the reference: they
are triangulated from many views and bundle-adjusted, i.e. the most trusted
depth available. Agreement is scale-invariant, because SfM depth is known only
up to scale and MiDaS's only up to scale and shift:

  si-log     scale-invariant log RMSE (Eigen et al., 2014); 0 = same up to scale
  abs_rel    mean |d - d_ref| / d_ref after the best single scale (median ratio)
  spearman   rank correlation — is the near-to-far ordering the same?

For MiDaS the scale and shift of its inverse depth are first fitted to the
reference points (least squares in inverse depth), as in the MiDaS paper's
evaluation; two parameters fitted on hundreds of points.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cinescope.geometry.metrics import (  # noqa: E402
    abs_rel,
    delta_accuracy,
    fit_inverse_depth_affine,
    median_scale,
    scale_invariant_log_rmse,
    spearman,
)
from cinescope.geometry.mvs import plane_sweep  # noqa: E402
from cinescope.geometry.shot import load_shot, reconstruct, to_gray  # noqa: E402

RESULTS = Path(__file__).parent / "results"


def scores(pred: np.ndarray, ref: np.ndarray) -> dict:
    ok = np.isfinite(pred) & np.isfinite(ref) & (pred > 0) & (ref > 0)
    p, r = pred[ok], ref[ok]
    if len(p) < 5:
        return {"n": int(len(p))}
    return {"n": int(len(p)), "si_log": round(scale_invariant_log_rmse(p, r), 4),
            "abs_rel": round(abs_rel(p, r), 4), "delta1.25": round(delta_accuracy(p, r), 4),
            "spearman": round(spearman(p, r), 4), "scale": round(median_scale(p, r), 4)}


def midas_depth(inv: np.ndarray, ref_inv_at: np.ndarray, ref_depth: np.ndarray) -> np.ndarray:
    """MiDaS inverse depth -> depth, with scale/shift fitted on reference points."""
    a, b = fit_inverse_depth_affine(ref_inv_at, ref_depth)
    inv_fit = a * inv + b
    return np.where(inv_fit > 1e-6, 1.0 / np.maximum(inv_fit, 1e-6), np.nan)


def colour_depth(d: np.ndarray, near: float, far: float) -> np.ndarray:
    """Inverse-depth heat map (near = bright), nan = black."""
    import matplotlib

    v = (1 / np.where(np.isfinite(d), d, np.inf) - 1 / far) / (1 / near - 1 / far)
    rgb = (matplotlib.colormaps["magma"](np.clip(v, 0, 1))[..., :3] * 255).astype(np.uint8)
    rgb[~np.isfinite(d)] = 0
    return rgb


def neighbours(frames: list[int], ref: int, n: int) -> list[int]:
    others = sorted((f for f in frames if f != ref), key=lambda f: abs(f - ref))
    return sorted(others[:n])


def synthetic(n_src: int = 4) -> dict:
    from cinescope.geometry.midas import MiDaS
    from cinescope.geometry.synthetic import dolly_shot

    out = {}
    midas = MiDaS()
    for move in ("dolly", "orbit"):
        s = dolly_shot(n=9, width=240, height=180, move=move)
        g = to_gray(s.frames)
        ref, srcs = 4, [0, 2, 6, 8]
        gt = s.depth[ref]
        near, far = np.percentile(gt, 1) * 0.7, np.percentile(gt, 99) * 1.5
        r = plane_sweep(g[ref], [g[i] for i in srcs[:n_src]], s.k, (s.rots[ref], s.ts[ref]),
                        [(s.rots[i], s.ts[i]) for i in srcs[:n_src]], near, far)
        inv = midas(s.frames[ref])
        md = midas_depth(inv.ravel(), inv.ravel(), gt.ravel()).reshape(gt.shape)   # oracle affine fit
        out[move] = {"mvs_confident": scores(r.depth, gt), "mvs_all": scores(r.raw_depth, gt),
                     "mvs_coverage": round(float(r.confident.mean()), 4), "midas": scores(md, gt)}
        print(move, json.dumps(out[move]))
        Image.fromarray(np.concatenate([s.frames[ref], colour_depth(gt, near, far), colour_depth(r.depth, near, far),
                                        colour_depth(md, near, far)], 1)).save(f"outputs/p5/depth_synth_{move}.jpg")
    return out


def clip(video: str, shot: int, width: int, stride: int, max_frames: int, n_src: int) -> dict:
    from cinescope.geometry.midas import MiDaS

    sf = load_shot(video, shot, width=width, stride=stride, max_frames=max_frames)
    rec, tracks = reconstruct(sf)
    print("\n".join(rec.log[-2:]))
    if not rec.ok:
        return {"ok": False, "log": rec.log}
    kfs = rec.keyframes
    ref = kfs[len(kfs) // 2]
    srcs = neighbours(kfs, ref, n_src)
    idx = {f: i for i, f in enumerate(sf.ids)}
    gray = to_gray(sf.frames)
    rot, t = rec.poses[ref]
    # SfM points seen in the reference keyframe
    tids, xy = tracks.in_frame(ref)
    keep = np.array([tt in rec.points for tt in tids], bool)
    pts = np.stack([rec.points[tt] for tt in tids[keep]])
    z = pts @ rot[2] + t[2]
    uv = xy[keep]
    front = z > 0
    z, uv = z[front], uv[front]
    near, far = np.percentile(z, 2) * 0.6, np.percentile(z, 98) * 2.0
    r = plane_sweep(gray[idx[ref]], [gray[idx[f]] for f in srcs], rec.k, rec.poses[ref],
                    [rec.poses[f] for f in srcs], near, far)
    ui = np.clip(np.rint(uv).astype(int), 0, [gray.shape[2] - 1, gray.shape[1] - 1])
    mvs_at = r.depth[ui[:, 1], ui[:, 0]]
    inv = MiDaS()(sf.frames[idx[ref]])
    inv_at = inv[ui[:, 1], ui[:, 0]]
    md = midas_depth(inv, inv_at, z)
    md_at = md[ui[:, 1], ui[:, 0]]
    both = r.confident & np.isfinite(md)
    res = {"ok": True, "reference_frame": ref, "source_frames": srcs, "sfm_points_in_ref": int(len(z)),
           "mvs_vs_sfm": scores(mvs_at, z), "mvs_coverage": round(float(r.confident.mean()), 4),
           "midas_vs_sfm": scores(md_at, z), "midas_vs_sfm_rank_only": round(spearman(inv_at, -z), 4),
           "mvs_vs_midas_dense": scores(r.depth[both], md[both]), "sfm": rec.summary()}
    print(json.dumps(res, indent=1))
    out_dir = Path("outputs") / Path(video).stem
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = sf.frames[idx[ref]].copy()
    for (u, v), zz in zip(ui, z, strict=True):               # SfM points coloured by depth
        c = colour_depth(np.array([[zz]]), near, far)[0, 0]
        frame[max(0, v - 1):v + 2, max(0, u - 1):u + 2] = c
    Image.fromarray(np.concatenate([frame, colour_depth(r.depth, near, far), colour_depth(md, near, far)], 1)
                    ).save(out_dir / f"depth_shot{shot}.jpg", quality=88)
    np.savez_compressed(out_dir / f"depth_shot{shot}.npz", mvs=r.depth.astype(np.float32),
                        midas_inv=inv.astype(np.float32), frame=ref)
    return res


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", nargs="?")
    ap.add_argument("--shot", type=int, nargs="+", default=[0])
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--width", type=int, default=480)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--max-frames", type=int, default=60)
    ap.add_argument("--sources", type=int, default=4)
    args = ap.parse_args(argv)
    RESULTS.mkdir(parents=True, exist_ok=True)
    Path("outputs/p5").mkdir(parents=True, exist_ok=True)
    if args.synthetic:
        (RESULTS / "depth_synthetic.json").write_text(json.dumps(synthetic(args.sources), indent=1), encoding="utf-8")
    if args.video:
        res = {f"{Path(args.video).stem}#{s}": clip(args.video, s, args.width, args.stride, args.max_frames,
                                                    args.sources) for s in args.shot}
        (RESULTS / f"depth_{Path(args.video).stem}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
