"""Week 14: calibrate a camera from chessboard photos (Zhang's method, ours vs
OpenCV), and test the self-calibration fallback that film clips need.

    python p5_3d/calibrate.py data/calib/left*.jpg --board 9x6
    # -> p5_3d/results/calibration.json, outputs/p5/undistort.jpg

Corner detection is OpenCV's (findChessboardCorners + cornerSubPix) for both
methods, so the comparison is of the calibration itself.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cinescope.geometry.calib import calibrate, chessboard_points  # noqa: E402
from cinescope.geometry.selfcalib import (  # noqa: E402
    detect_segments,
    estimate_focal,
    focal_from_vps,
    focal_prior,
)


def grid_segments(corners: np.ndarray, cols: int, rows: int) -> tuple[np.ndarray, np.ndarray]:
    """End-to-end segments along each board row and each board column."""
    g = corners.reshape(rows, cols, 2)
    r = np.concatenate([g[:, 0], g[:, -1]], 1)
    c = np.concatenate([g[0], g[-1]], 1)
    return r, c


def vp_of(segs: np.ndarray) -> np.ndarray:
    p1 = np.c_[segs[:, :2], np.ones(len(segs))]
    p2 = np.c_[segs[:, 2:], np.ones(len(segs))]
    ln = np.cross(p1, p2)
    ln /= np.linalg.norm(ln[:, :2], axis=1, keepdims=True)
    return np.linalg.svd(ln)[2][-1]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="+")
    ap.add_argument("--board", default="9x6", help="inner corners, cols x rows")
    ap.add_argument("--out", default=str(Path(__file__).parent / "results" / "calibration.json"))
    args = ap.parse_args(argv)
    import cv2

    cols, rows = (int(v) for v in args.board.lower().split("x"))
    files = sorted(f for pat in args.images for f in glob.glob(pat))
    obj = chessboard_points(cols, rows)
    obj_pts, img_pts, used = [], [], []
    for f in files:
        g = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        ok, c = cv2.findChessboardCorners(g, (cols, rows))
        if not ok:
            print(f"no board found: {f}")
            continue
        c = cv2.cornerSubPix(g, c, (11, 11), (-1, -1),
                             (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3))
        obj_pts.append(obj)
        img_pts.append(c.reshape(-1, 2).astype(np.float64))
        used.append(Path(f).name)
    size = (g.shape[1], g.shape[0])
    print(f"{len(used)} views, {size[0]}x{size[1]}")

    t = time.perf_counter()
    ours = calibrate(obj_pts, img_pts, size)
    t_ours = time.perf_counter() - t
    o3 = [np.c_[o, np.zeros(len(o))].astype(np.float32) for o in obj_pts]
    i32 = [i.astype(np.float32) for i in img_pts]
    t = time.perf_counter()
    rms_same, k_same, d_same, _, _ = cv2.calibrateCamera(
        o3, i32, size, None, None, flags=cv2.CALIB_FIX_K3 | cv2.CALIB_ZERO_TANGENT_DIST)
    t_cv = time.perf_counter() - t
    rms_full, k_full, d_full, _, _ = cv2.calibrateCamera(o3, i32, size, None, None)

    def kd(k, d):
        return {"fx": float(k[0, 0]), "fy": float(k[1, 1]), "cx": float(k[0, 2]), "cy": float(k[1, 2]),
                "dist": [float(v) for v in np.ravel(d)]}

    # --- self-calibration checks on the same photos: the focal length the
    # fallback would guess, against the calibrated one.
    fx = ours.k[0, 0]
    vp_board, vp_lines = [], []
    for f, c in zip(used, img_pts, strict=True):
        r, cc = grid_segments(c, cols, rows)
        fb = focal_from_vps(vp_of(r), vp_of(cc), (size[0] / 2, size[1] / 2))
        vp_board.append(fb)
        g = cv2.imread(str(Path(files[0]).parent / f), cv2.IMREAD_GRAYSCALE)
        est = estimate_focal(detect_segments(g), size[0], size[1])
        vp_lines.append(est.f if est.method == "vanishing-points" else None)

    def err_stats(vals):
        v = np.array([x for x in vals if x is not None])
        if not len(v):
            return {"n": 0}
        e = np.abs(v - fx) / fx
        return {"n": int(len(v)), "median_f": float(np.median(v)), "median_rel_err": float(np.median(e)),
                "per_view": [None if x is None else round(float(x), 1) for x in vals]}

    prior = focal_prior(*size)
    result = {
        "views": used, "image_size": list(size),
        "ours": {**ours.to_dict(), "closed_form": kd(ours.closed_form_k, [0, 0]), "lm_iterations": ours.iterations,
                 "seconds": round(t_ours, 3)},
        "opencv_same_model": {**kd(k_same, d_same), "rms_px": float(rms_same), "seconds": round(t_cv, 3)},
        "opencv_full_model": {**kd(k_full, d_full), "rms_px": float(rms_full)},
        "self_calibration": {
            "calibrated_fx": float(fx),
            "size_prior": {"f": prior, "rel_err": abs(prior - fx) / fx},
            "vp_from_board_grid": err_stats(vp_board),
            "vp_from_hough_lines": err_stats(vp_lines),
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=1), encoding="utf-8")

    print(f"{'':22s}{'fx':>9s}{'fy':>9s}{'cx':>9s}{'cy':>9s}{'k1':>9s}{'k2':>9s}{'RMS px':>9s}")
    rowfmt = "{:22s}{:9.2f}{:9.2f}{:9.2f}{:9.2f}{:9.4f}{:9.4f}{:9.4f}"
    print(rowfmt.format("ours (closed form)", *np.diag(ours.closed_form_k)[:2], *ours.closed_form_k[:2, 2],
                        0, 0, float("nan")))
    print(rowfmt.format("ours (LM)", ours.k[0, 0], ours.k[1, 1], ours.k[0, 2], ours.k[1, 2], *ours.dist, ours.rms))
    print(rowfmt.format("cv2 (k1,k2)", k_same[0, 0], k_same[1, 1], k_same[0, 2], k_same[1, 2],
                        d_same.ravel()[0], d_same.ravel()[1], rms_same))
    print(rowfmt.format("cv2 (k1-3, p1-2)", k_full[0, 0], k_full[1, 1], k_full[0, 2], k_full[1, 2],
                        d_full.ravel()[0], d_full.ravel()[1], rms_full))
    sc = result["self_calibration"]
    print(f"self-calibration: prior f={prior:.0f} ({sc['size_prior']['rel_err']:.0%} off); "
          f"VP board grid median f={sc['vp_from_board_grid'].get('median_f', float('nan')):.0f} "
          f"({sc['vp_from_board_grid'].get('median_rel_err', float('nan')):.1%}); "
          f"VP Hough lines: {sc['vp_from_hough_lines']['n']}/{len(used)} views, "
          f"median f={sc['vp_from_hough_lines'].get('median_f', float('nan')):.0f}")

    # undistorted example, for the README
    out_dir = Path("outputs/p5")
    out_dir.mkdir(parents=True, exist_ok=True)
    img = cv2.imread(str(Path(files[0]).parent / used[0]))
    und = cv2.undistort(img, ours.k, np.r_[ours.dist, 0, 0, 0])
    cv2.imwrite(str(out_dir / "undistort.jpg"), np.concatenate([img, und], 1))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
