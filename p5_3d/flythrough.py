"""Week 17: turn a moving shot into a set you can fly through — 3D Gaussian
splatting, from scratch in PyTorch on the CPU.

    python p5_3d/flythrough.py --synthetic
    python p5_3d/flythrough.py trailers/goldeneye.mp4 --shot 18
    # -> p5_3d/results/flythrough_<name>.json, outputs/<clip>/flythrough_shot<N>.mp4 (+ .jpg strip)

Pipeline: our SfM gives camera poses and sparse points; our plane-sweep stereo
on the middle keyframe adds a dense coloured point cloud; both initialise the
Gaussians, which are optimised (L1 + 0.2 D-SSIM, Adam, densification) to
reproduce the shot's frames. Every 4th frame is held out and never seen in
training; PSNR on those frames is compared with the simple baseline of
rendering the initial point cloud directly (z-buffered square points, size
chosen on the training frames).

--synthetic does the same on the ray-cast room with the true poses, so the
renderer and optimiser are scored in isolation, and the fly-through views are
also compared with true renders of the room from the same novel cameras.

The official 3DGS code and gsplat need CUDA; this runs anywhere, slowly: a
few thousand Gaussians at ~200 px wide train at a few iterations per second.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cinescope.geometry import splat  # noqa: E402
from cinescope.geometry.camera import camera_centre, log_rotation, rodrigues  # noqa: E402
from cinescope.geometry.mvs import plane_sweep  # noqa: E402
from cinescope.geometry.shot import to_gray  # noqa: E402

RESULTS = Path(__file__).parent / "results"


def backproject(depth: np.ndarray, rgb: np.ndarray, k: np.ndarray, rot: np.ndarray, t: np.ndarray,
                step: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """World points and colours from a depth map (nan = skip), every `step` pixels."""
    h, w = depth.shape
    ys, xs = np.mgrid[step // 2:h:step, step // 2:w:step]
    z = depth[ys, xs]
    ok = np.isfinite(z) & (z > 0)
    ys, xs, z = ys[ok], xs[ok], z[ok]
    xc = np.stack([(xs + 0.5 - k[0, 2]) / k[0, 0] * z, (ys + 0.5 - k[1, 2]) / k[1, 1] * z, z], 1)
    return (xc - t) @ rot, rgb[ys, xs] / 255.0


def novel_path(poses: list[tuple[np.ndarray, np.ndarray]], n: int = 48, sway: float = 0.6,
               push: float = 0.3) -> list[tuple[np.ndarray, np.ndarray]]:
    """A smooth camera path through the shot's span: interpolate first -> last
    pose (rotation via axis-angle), sway sideways and push in, beyond what the
    real camera did, so the views are genuinely new."""
    (r0, t0), (r1, t1) = poses[0], poses[-1]
    c0, c1 = camera_centre(r0, t0), camera_centre(r1, t1)
    span = max(np.linalg.norm(c1 - c0), 1e-3)
    d = log_rotation(r1 @ r0.T)
    out = []
    for i in range(n):
        a = i / (n - 1)
        rot = rodrigues(a * d) @ r0
        c = (1 - a) * c0 + a * c1
        right, fwd = rot[0], rot[2]
        c = c + right * sway * span * np.sin(2 * np.pi * a) + fwd * push * span * np.sin(np.pi * a)
        out.append((rot, -rot @ c))
    return out


def best_point_radius(points, colors, cams, images) -> int:
    scores = {}
    for r in (0, 1, 2, 3):
        scores[r] = np.mean([splat.psnr(splat.render_points(points, colors, c, r), im / 255.0)
                             for c, im in zip(cams, images, strict=True)])
    return max(scores, key=scores.get)


def run(frames: np.ndarray, ids: list[int], k: np.ndarray, poses: dict[int, tuple], sparse: np.ndarray | None,
        ref: int, srcs: list[int], near: float, far: float, iters: int, out_dir: Path, name: str,
        truth_render=None) -> dict:
    torch.set_num_threads(1)          # a shared CPU: extra threads only spin
    h, w = frames.shape[1:3]
    idx = {f: i for i, f in enumerate(ids)}
    with_pose = [f for f in ids if f in poses]
    test = with_pose[2::4]
    train = [f for f in with_pose if f not in test]
    cam = {f: splat.Camera(k, *poses[f], w, h) for f in with_pose}
    gray = to_gray(frames)
    t0 = time.perf_counter()
    mvs = plane_sweep(gray[idx[ref]], [gray[idx[f]] for f in srcs], k, poses[ref], [poses[f] for f in srcs],
                      near, far, n_planes=48)
    pts, cols = backproject(mvs.depth, frames[idx[ref]], k, *poses[ref])
    if sparse is not None and len(sparse):
        # colour the SfM points from the reference frame where they project inside it
        uv = (sparse @ poses[ref][0].T + poses[ref][1])
        ok = uv[:, 2] > 0
        u = np.clip((k[0, 0] * uv[ok, 0] / uv[ok, 2] + k[0, 2]).astype(int), 0, w - 1)
        v = np.clip((k[1, 1] * uv[ok, 1] / uv[ok, 2] + k[1, 2]).astype(int), 0, h - 1)
        pts = np.concatenate([pts, sparse[ok]])
        cols = np.concatenate([cols, frames[idx[ref]][v, u] / 255.0])
    t_mvs = time.perf_counter() - t0
    print(f"{name}: {len(pts)} initial points ({int(mvs.confident.sum())} confident MVS pixels), "
          f"{len(train)} training / {len(test)} held-out frames")

    radius = best_point_radius(pts, cols, [cam[f] for f in train], [frames[idx[f]] for f in train])
    g = splat.Gaussians(pts, cols, splat.init_scales(pts))
    t0 = time.perf_counter()
    g, log = splat.train(g, [cam[f] for f in train], [frames[idx[f]] for f in train],
                         splat.TrainConfig(iters=iters, densify_until=int(0.7 * iters), max_gaussians=10000),
                         log_every=50, callback=lambda r: print("  ", r, flush=True))
    t_train = time.perf_counter() - t0
    per_frame = []
    with torch.no_grad():
        for f in test:
            gt = frames[idx[f]] / 255.0
            per_frame.append({"frame": int(f), "splat_psnr": round(splat.psnr(splat.render(g, cam[f]), gt), 2),
                              "points_psnr": round(splat.psnr(splat.render_points(pts, cols, cam[f], radius), gt), 2)})
        train_psnr = float(np.mean([splat.psnr(splat.render(g, cam[f]), frames[idx[f]] / 255.0) for f in train]))
        path = novel_path([poses[f] for f in with_pose])
        views = [(splat.render(g, splat.Camera(k, r, t, w, h)).numpy() * 255).clip(0, 255).astype(np.uint8)
                 for r, t in path]
    res = {"initial_points": int(len(pts)), "final_gaussians": len(g), "iters": iters,
           "train_frames": len(train), "test_frames": len(test), "point_radius": radius,
           "heldout_psnr_splat": round(float(np.mean([p["splat_psnr"] for p in per_frame])), 2),
           "heldout_psnr_points": round(float(np.mean([p["points_psnr"] for p in per_frame])), 2),
           "train_psnr_splat": round(train_psnr, 2), "per_frame": per_frame,
           "seconds_mvs": round(t_mvs, 1), "seconds_train": round(t_train, 1),
           "s_per_iter": round(t_train / iters, 3), "log": log}
    if truth_render is not None:
        truth = [truth_render(r, t) for r, t in path]
        res["novel_view_psnr_vs_truth"] = round(float(np.mean([splat.psnr(v / 255.0, tr / 255.0)
                                                               for v, tr in zip(views, truth, strict=True)])), 2)
        res["novel_view_psnr_points_vs_truth"] = round(float(np.mean([
            splat.psnr(splat.render_points(pts, cols, splat.Camera(k, r, t, w, h), radius), tr / 255.0)
            for (r, t), tr in zip(path, truth, strict=True)])), 2)
    out_dir.mkdir(parents=True, exist_ok=True)
    from cinescope.io.video import write_video

    write_video(out_dir / f"flythrough_{name}.mp4", np.stack(views), fps=24)
    # a strip for the README: held-out frame | splat | points, and four fly-through views
    f = test[len(test) // 2]
    with torch.no_grad():
        sp = (splat.render(g, cam[f]).numpy() * 255).clip(0, 255).astype(np.uint8)
    pt = (splat.render_points(pts, cols, cam[f], radius) * 255).clip(0, 255).astype(np.uint8)
    row1 = np.concatenate([frames[idx[f]], sp, pt], 1)
    row2 = np.concatenate([views[i] for i in np.linspace(0, len(views) - 1, 3).astype(int)], 1)
    Image.fromarray(np.concatenate([row1, row2], 0)).save(out_dir / f"flythrough_{name}.jpg", quality=85)
    print(json.dumps({k2: v for k2, v in res.items() if k2 not in ("log", "per_frame")}))
    return res


def synthetic(iters: int) -> dict:
    from cinescope.geometry.synthetic import Room, dolly_shot, render

    s = dolly_shot(n=24, width=160, height=120, move="dolly")
    ids = list(range(24))
    poses = {i: (s.rots[i], s.ts[i]) for i in ids}
    room = Room()
    return run(s.frames, ids, s.k, poses, None, ref=12, srcs=[4, 8, 16, 20], near=2.0, far=15.0, iters=iters,
               out_dir=Path("outputs/p5"), name="synthetic",
               truth_render=lambda r, t: render(room, s.k, r, t, 160, 120)[0])


def clip(video: str, shot: int, iters: int, width: int, sfm_width: int, stride: int, max_frames: int) -> dict:
    from cinescope.geometry.shot import load_shot, reconstruct

    sf = load_shot(video, shot, width=sfm_width, stride=stride, max_frames=max_frames)
    rec, _ = reconstruct(sf)
    print("\n".join(rec.log[-2:]))
    if not rec.ok:
        return {"ok": False, "log": rec.log}
    s = width / sfm_width
    k = rec.k.copy()
    k[:2] *= s
    small = np.stack([np.asarray(Image.fromarray(f).resize((width, round(f.shape[0] * s)), Image.BILINEAR))
                      for f in sf.frames])
    kfs = rec.keyframes
    ref = kfs[len(kfs) // 2]
    srcs = sorted(sorted((f for f in kfs if f != ref), key=lambda f: abs(f - ref))[:4])
    pts = np.stack(list(rec.points.values()))
    z = pts @ rec.poses[ref][0][2] + rec.poses[ref][1][2]
    z = z[z > 0]
    near, far = np.percentile(z, 2) * 0.6, np.percentile(z, 98) * 2.0
    res = run(small, sf.ids, k, rec.poses, pts, ref, srcs, near, far, iters,
              Path("outputs") / Path(video).stem, f"shot{shot}")
    res["sfm"] = rec.summary()
    return res


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", nargs="?")
    ap.add_argument("--shot", type=int, default=0)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--iters", type=int, default=800)
    ap.add_argument("--width", type=int, default=240, help="training resolution")
    ap.add_argument("--sfm-width", type=int, default=480)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--max-frames", type=int, default=60)
    args = ap.parse_args(argv)
    RESULTS.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    if args.synthetic:
        res = synthetic(args.iters)
        (RESULTS / "flythrough_synthetic.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    if args.video:
        res = clip(args.video, args.shot, args.iters, args.width, args.sfm_width, args.stride, args.max_frames)
        (RESULTS / f"flythrough_{Path(args.video).stem}_shot{args.shot}.json").write_text(
            json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
