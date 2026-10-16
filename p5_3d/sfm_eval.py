"""Week 15: two-view geometry and structure from motion, ours vs OpenCV vs COLMAP.

    python p5_3d/sfm_eval.py --synthetic                 # ground-truth scenes
    python p5_3d/sfm_eval.py trailers/goldeneye.mp4 trailers/testvideo.mp4
    # -> p5_3d/results/sfm_synthetic.json, p5_3d/results/sfm_clips.json

Shots are chosen by rule, not by hand: every shot of at least --min-seconds
that Phase 0's camera classifier did not call "static" (a static camera has
no baseline, so there is nothing to reconstruct).

Per shot, on the same LK tracks:
  ours        our solvers + our bundle adjustment
  ours-noBA   our solvers, no bundle adjustment
  opencv      findEssentialMat / recoverPose / triangulatePoints / solvePnPRansac,
              no bundle adjustment (OpenCV's main modules have none)
and, from the frames themselves, COLMAP (pycolmap, SIFT + Ceres) if installed.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cinescope.geometry import twoview  # noqa: E402
from cinescope.geometry.camera import camera_centre, look_at, rotation_angle_deg  # noqa: E402
from cinescope.geometry.metrics import (  # noqa: E402
    trajectory_errors,
    translation_direction_error_deg,
)
from cinescope.geometry.sfm import OurSolvers, SfMParams, run_sfm  # noqa: E402
from cinescope.geometry.shot import (  # noqa: E402
    default_k,
    find_shots,
    load_shot,
    normalise_contrast,
    to_gray,
)
from cinescope.geometry.synthetic import dolly_shot  # noqa: E402
from cinescope.geometry.tracks import track_shot  # noqa: E402

RESULTS = Path(__file__).parent / "results"


def _opencv():
    try:
        from cinescope.geometry.baselines import OpenCVSolvers

        return OpenCVSolvers()
    except ImportError:
        return None


def variants(focal_known: bool):
    out = {"ours": SfMParams(refine_focal=not focal_known),
           "ours-noBA": SfMParams(use_ba=False)}
    cv = _opencv()
    if cv is not None:
        out["opencv"] = SfMParams(use_ba=False, solvers=cv)
    return out


# ---------------------------------------------------------------- two-view on synthetic matches
def two_view_benchmark(trials: int = 20, n: int = 200, noise_px: float = 0.5, outliers: float = 0.3,
                       focal: float = 500.0, seed: int = 0) -> dict:
    """Relative pose from noisy matches with outliers: rotation and translation-
    direction error (deg), ours (8-point RANSAC, then LM on the Sampson error)
    vs OpenCV (5-point RANSAC + recoverPose)."""
    rng = np.random.default_rng(seed)
    cv = _opencv()
    errs = {"ours-8pt": [], "ours-8pt+LM": [], "opencv": []}
    ours = OurSolvers()
    for _ in range(trials):
        x = rng.uniform([-3, -2, 5], [3, 2, 12], (n, 3))
        r1, t1 = np.eye(3), np.zeros(3)
        eye = rng.uniform([-1, -0.3, -0.5], [1, 0.3, 0.5])
        r2, t2 = look_at(eye, np.array([0, 0, 8.0]) + rng.normal(0, 0.3, 3))
        pts = []
        for rot, t in ((r1, t1), (r2, t2)):
            xc = x @ rot.T + t
            pts.append(xc[:, :2] / xc[:, 2:3] + rng.normal(0, noise_px / focal, (n, 2)))
        bad = rng.random(n) < outliers
        pts[1][bad] = rng.uniform(-0.4, 0.4, (bad.sum(), 2))
        t_gt = t2 - r2 @ t1          # camera 1 is at the origin
        e, inl = ours.essential(pts[0], pts[1], focal, 1.0)
        rot, t, _ = twoview.recover_pose(e, pts[0][inl], pts[1][inl])
        errs["ours-8pt"].append((rotation_angle_deg(rot, r2), translation_direction_error_deg(t, t_gt)))
        rot, t, _ = ours.recover_pose(e, pts[0][inl], pts[1][inl])
        errs["ours-8pt+LM"].append((rotation_angle_deg(rot, r2), translation_direction_error_deg(t, t_gt)))
        if cv is not None:
            e, inl = cv.essential(pts[0], pts[1], focal, 1.0)
            rot, t, _ = cv.recover_pose(e, pts[0][inl], pts[1][inl])
            errs["opencv"].append((rotation_angle_deg(rot, r2), translation_direction_error_deg(t, t_gt)))
    return {k: {"rot_err_deg_median": float(np.median([a for a, _ in v])),
                "t_dir_err_deg_median": float(np.median([b for _, b in v]))} for k, v in errs.items() if v}


# ---------------------------------------------------------------- synthetic shots
def synthetic(width: int = 240, frames: int = 30) -> dict:
    out = {"two_view": two_view_benchmark()}
    print("two-view (200 matches, 0.5 px noise, 30% outliers):", out["two_view"])
    for move in ("dolly", "push", "orbit", "pan"):
        s = dolly_shot(n=frames, width=width, height=width * 3 // 4, move=move)
        gray = to_gray(s.frames)
        t0 = time.perf_counter()
        tracks = track_shot(gray, max_corners=500)
        t_track = time.perf_counter() - t0
        row = {"frames": frames, "tracks": len(tracks.obs), "track_seconds": round(t_track, 1)}
        for name, p in variants(focal_known=True).items():
            t0 = time.perf_counter()
            rec = run_sfm(copy.deepcopy(tracks), s.k, width, p)
            r = {**rec.summary(), "seconds": round(time.perf_counter() - t0, 1)}
            if rec.ok and len(rec.poses) >= 3:
                fr = sorted(rec.poses)
                gt_c = np.stack([camera_centre(s.rots[f], s.ts[f]) for f in fr])
                r.update(trajectory_errors(rec.centres(fr), gt_c, np.stack([rec.poses[f][0] for f in fr]), s.rots[fr]))
            row[name] = r
        # unknown focal: start from the size prior (1.2 x width), let BA refine it
        prior_k = default_k(width, width * 3 // 4)
        rec = run_sfm(copy.deepcopy(tracks), prior_k, width, SfMParams(refine_focal=True))
        row["ours-focal-from-prior"] = {**rec.summary(), "gt_focal": float(s.k[0, 0]),
                                        "prior_focal": float(prior_k[0, 0])}
        out[move] = row
        print(move, json.dumps({k: (v if not isinstance(v, dict) else
                                    {kk: v.get(kk) for kk in ("ok", "keyframes", "points", "mean_reproj_px",
                                                              "rot_err_deg_mean", "ate_rmse_rel_path", "focal")})
                                for k, v in row.items()}))
    return out


# ---------------------------------------------------------------- clips
def pick_shots(video: str, min_seconds: float) -> list[dict]:
    shots = find_shots(video)
    return [s for s in shots if (s["t1"] - s["t0"]) >= min_seconds
            and s.get("camera", {}).get("label") not in ("static", None)]


def clips(videos: list[str], width: int, stride: int, max_frames: int, min_seconds: float,
          colmap: bool) -> dict:
    out = {}
    for video in videos:
        for s in pick_shots(video, min_seconds):
            key = f"{Path(video).stem}#{s['index']}"
            t0 = time.perf_counter()
            sf = load_shot(video, s["index"], width=width, stride=stride, max_frames=max_frames)
            gray = normalise_contrast(to_gray(sf.frames))
            tracks = track_shot(gray, sf.ids, max_corners=800)
            h, w = sf.frames.shape[1:3]
            row = {"camera_label": s["camera"]["label"], "frames": len(sf.ids), "first": sf.ids[0], "last": sf.ids[-1],
                   "tracks": len(tracks.obs), "track_seconds": round(time.perf_counter() - t0, 1)}
            k = default_k(w, h)
            for name, p in variants(focal_known=False).items():
                t1 = time.perf_counter()
                rec = run_sfm(copy.deepcopy(tracks), k, w, p)
                row[name] = {**rec.summary(), "seconds": round(time.perf_counter() - t1, 1)}
            if colmap:
                try:
                    from cinescope.geometry.baselines import run_pycolmap

                    t1 = time.perf_counter()
                    c = run_pycolmap(sf.frames, sf.ids, Path("outputs/p5/colmap") / key.replace("#", "_"))
                    c.pop("poses", None)
                    row["colmap"] = {**c, "seconds": round(time.perf_counter() - t1, 1)}
                except ImportError:
                    pass
            out[key] = row
            print(key, json.dumps({k: (v if not isinstance(v, dict) else
                                       {kk: v.get(kk) for kk in ("ok", "keyframes", "registered", "points",
                                                                 "mean_reproj_px", "focal")})
                                   for k, v in row.items()}), flush=True)
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("videos", nargs="*")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--width", type=int, default=480)
    ap.add_argument("--stride", type=int, default=2, help="track every n-th frame")
    ap.add_argument("--max-frames", type=int, default=60)
    ap.add_argument("--min-seconds", type=float, default=1.5)
    ap.add_argument("--no-colmap", action="store_true")
    args = ap.parse_args(argv)
    RESULTS.mkdir(parents=True, exist_ok=True)
    if args.synthetic:
        res = synthetic()
        (RESULTS / "sfm_synthetic.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    if args.videos:
        res = clips(args.videos, args.width, args.stride, args.max_frames, args.min_seconds, not args.no_colmap)
        (RESULTS / "sfm_clips.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
        ok = {k: v for k, v in res.items() if v["ours"]["ok"]}
        print(f"\n{len(ok)}/{len(res)} shots reconstructed by ours")
        for name in ("ours", "ours-noBA", "opencv", "colmap"):
            vals = [v[name]["mean_reproj_px"] for v in res.values() if name in v and v[name].get("ok")]
            if vals:
                print(f"{name:10s} mean reprojection error {np.mean(vals):.3f} px over {len(vals)} shots")


if __name__ == "__main__":
    main()
