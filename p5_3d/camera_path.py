"""Camera path export for any clip and shot.

    python p5_3d/camera_path.py trailers/goldeneye.mp4 --shot 18
    python p5_3d/camera_path.py clip.mp4 --shot 3 --focal 900      # if the focal length (px at --width) is known
    # -> outputs/<clip>/camera_path_shot<N>.json and camera_path_shot<N>.png

Shots come from Phase 0's detector (cached in outputs/<clip>/shots.json). If
the camera translates, the path is full 6-DoF from structure from motion (up
to an unknown scale: units are "scene units", not metres); if it only
rotates/zooms, the rotation-only path from homographies is exported instead
and "mode" says which. The JSON stores per frame: R (world->camera), t,
camera centre, pan/tilt/roll in degrees relative to the first frame, and,
for SfM, the sparse point cloud — enough to drive a 3D package's camera.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cinescope.geometry.campath import (  # noqa: E402
    path_from_reconstruction,
    path_from_rotations,
    rotation_only_path,
)
from cinescope.geometry.shot import default_k, load_shot, reconstruct  # noqa: E402


def plot(path: list[dict], points: np.ndarray | None, fps: float, title: str, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frames = np.array([p["frame"] for p in path])
    t = (frames - frames[0]) / fps
    c = np.array([p["centre"] for p in path])
    kf = np.array([p["keyframe"] for p in path])
    fig, ax = plt.subplots(1, 4, figsize=(16, 4))

    def draw_path(a, i, j, sgn, arrows):
        a.plot(c[:, i], sgn * c[:, j], "-", c="C0", lw=1.2)
        a.scatter(c[kf, i], sgn * c[kf, j], c="C1", s=14, zorder=3, label="keyframes")
        a.scatter(c[:1, i], sgn * c[:1, j], c="C2", s=36, zorder=4, label="start")
        span = np.ptp(c[:, [i, j]], 0).max()
        for p in path[:: max(1, len(path) // 10)] if arrows and span > 1e-9 else []:
            d = np.array(p["R"]).T @ np.array([0, 0, 1.0])      # viewing direction
            ctr = np.array(p["centre"])
            a.annotate("", (ctr[i] + 0.15 * span * d[i], sgn * (ctr[j] + 0.15 * span * d[j])), (ctr[i], sgn * ctr[j]),
                       arrowprops={"arrowstyle": "->", "color": "C3", "lw": 0.8})

    draw_path(ax[0], 0, 2, 1, True)
    ax[0].set_title("camera path, top view (x, z)")
    ax[0].set_aspect("equal", adjustable="datalim")
    ax[0].legend(fontsize=7)
    if points is not None and len(points):
        # the bulk of the scene: far, poorly triangulated points (small parallax
        # angle, long depth tail) would otherwise set the axes' scale
        dist = np.linalg.norm(points - c.mean(0), axis=1)
        keep = dist <= 3 * np.median(dist)
        ax[1].scatter(points[keep, 0], points[keep, 2], s=3, c="0.55",
                      label=f"SfM points within 3x median distance ({int(keep.sum())}/{len(points)})")
        draw_path(ax[1], 0, 2, 1, False)
        ax[1].set_title("scene, top view (x, z)")
        ax[1].set_aspect("equal", adjustable="datalim")
        ax[1].legend(fontsize=7)
    else:
        ax[1].text(0.5, 0.5, "rotation only:\nno parallax, no 3D points", ha="center", va="center")
        ax[1].set_axis_off()
    for key, col in (("pan_deg", "C0"), ("tilt_deg", "C1"), ("roll_deg", "C2")):
        ax[2].plot(t, [p[key] for p in path], c=col, label=key.replace("_deg", ""))
    ax[2].set_xlabel("seconds")
    ax[2].set_ylabel("degrees from first frame")
    ax[2].legend(fontsize=8)
    ax[2].set_title("orientation")
    for k, (name, col) in enumerate((("x (right)", "C0"), ("y (down)", "C1"), ("z (forward)", "C2"))):
        ax[3].plot(t, c[:, k] - c[0, k], c=col, label=name)
    if "zoom" in path[0]:
        ax3b = ax[3].twinx()
        ax3b.plot(t, [p["zoom"] for p in path], "k--", lw=1, label="zoom")
        ax3b.set_ylabel("zoom factor")
    ax[3].set_xlabel("seconds")
    ax[3].set_title("camera position (start = 0)")
    ax[3].legend(fontsize=8)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=80)
    plt.close(fig)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--shot", type=int, required=True)
    ap.add_argument("--width", type=int, default=480)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--max-frames", type=int, default=120)
    ap.add_argument("--focal", type=float, help="focal length in pixels at --width (default: estimated)")
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args(argv)

    t0 = time.perf_counter()
    sf = load_shot(args.video, args.shot, width=args.width, stride=args.stride, max_frames=args.max_frames)
    rec, tracks = reconstruct(sf, focal=args.focal)
    h, w = sf.frames.shape[1:3]
    if rec.ok:
        mode = "sfm"
        path = path_from_reconstruction(rec)
        points = np.stack(list(rec.points.values()))
        k = rec.k
    else:
        mode = "rotation-only"
        k = default_k(w, h, args.focal)
        path = path_from_rotations(rotation_only_path(tracks, k))
        points = None
    result = {
        "video": str(args.video), "shot": args.shot, "first_frame": sf.ids[0], "last_frame": sf.ids[-1],
        "fps": sf.fps, "stride": args.stride, "image_size": [w, h], "mode": mode,
        "intrinsics": {"fx": float(k[0, 0]), "fy": float(k[1, 1]), "cx": float(k[0, 2]), "cy": float(k[1, 2]),
                       "focal_source": "given" if args.focal else ("bundle adjustment" if mode == "sfm" else "size prior")},
        "convention": "x_cam = R X + t; x right, y down, z forward; units arbitrary (monocular scale)",
        "sfm": rec.summary(), "sfm_log": rec.log,
        "frames": path,
        "points": None if points is None else np.round(points, 5).tolist(),
        "seconds": round(time.perf_counter() - t0, 1),
    }
    out = Path(args.out) / Path(args.video).stem
    out.mkdir(parents=True, exist_ok=True)
    (out / f"camera_path_shot{args.shot}.json").write_text(json.dumps(result), encoding="utf-8")
    pans = [p["pan_deg"] for p in path]
    tilts = [p["tilt_deg"] for p in path]
    title = (f"{Path(args.video).name} shot {args.shot} ({mode}): {len(path)} frames, "
             f"pan {min(pans):+.1f}..{max(pans):+.1f} deg, tilt {min(tilts):+.1f}..{max(tilts):+.1f} deg")
    if mode == "sfm":
        title += f", {len(points)} points, reproj {rec.mean_reproj_px:.2f} px"
    plot(path, points, sf.fps, title, out / f"camera_path_shot{args.shot}.png")
    print(title)
    print(f"-> {out / f'camera_path_shot{args.shot}.json'}")


if __name__ == "__main__":
    main()
