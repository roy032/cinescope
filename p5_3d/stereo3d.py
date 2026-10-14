"""Week 16: convert 2D footage to stereo 3D by depth-image-based rendering.

    python p5_3d/stereo3d.py trailers/goldeneye.mp4 --start 66 --seconds 4
    python p5_3d/stereo3d.py --synthetic          # scored against a true stereo pair
    # -> outputs/<clip>/stereo_anaglyph.mp4, stereo_sbs.mp4, stereo_still.jpg

Depth per frame from MiDaS small (one network pass, no camera motion needed —
which is why a conversion pipeline uses monocular depth rather than SfM: most
shots barely move). The depth levels (far, near, screen plane) are smoothed
over time with an exponential moving average, otherwise the depth budget
pumps as objects enter and leave frame.

--synthetic renders the room from two cameras a known baseline apart, then
synthesises the right view from the left view and its true depth, and scores
it (PSNR) with three hole-filling choices and with no warping at all.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cinescope.geometry.dibr import (  # noqa: E402
    anaglyph,
    depth_range,
    disparity_from_inverse_depth,
    render_view,
    side_by_side,
    stereo_pair,
)

RESULTS = Path(__file__).parent / "results"


def psnr(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    d = (a.astype(np.float64) - b.astype(np.float64)) ** 2
    mse = d[mask].mean() if mask is not None else d.mean()
    return float(10 * np.log10(255.0 ** 2 / max(mse, 1e-12)))


def synthetic(baseline: float = 0.15) -> dict:
    from cinescope.geometry.camera import intrinsics, look_at
    from cinescope.geometry.synthetic import Room, render

    room = Room()
    w, h = 320, 240
    k = intrinsics(0.9 * w, w / 2, h / 2)
    rot, t = look_at([0.0, -0.2, -0.5], [0.0, 0.2, 6.0])
    left, depth = render(room, k, rot, t, w, h)
    right_true, _ = render(room, k, rot, t - np.array([baseline, 0, 0]), w, h)   # camera moved +x
    disp = k[0, 0] * baseline / depth
    border = np.zeros((h, w), bool)
    border[:, :w - int(disp.max()) - 2] = True        # ignore the right-edge strip no left pixel reaches
    out = {"baseline": baseline, "max_disparity_px": round(float(disp.max()), 2),
           "no_warp_psnr": round(psnr(left, right_true, border), 2)}
    for side in ("background", "foreground"):
        view, holes = render_view(left, disp, -1.0, side)
        out[f"dibr_{side}_fill_psnr"] = round(psnr(view, right_true, border), 2)
        out[f"dibr_{side}_fill_psnr_in_holes"] = round(psnr(view, right_true, holes & border), 2)
        out["hole_fraction"] = round(float(holes[border].mean()), 4)
    print(json.dumps(out))
    Image.fromarray(np.concatenate([left, render_view(left, disp, -1.0)[0], right_true], 1)).save(
        "outputs/p5/dibr_synth.png")
    return out


def convert(video: str, start: float, seconds: float, width: int, max_disp: float, out_dir: Path,
            smooth: float = 0.8) -> dict:
    from cinescope.geometry.midas import MiDaS
    from cinescope.io.video import VideoReader, write_video

    midas = MiDaS()
    ana, sbs = [], []
    levels = None
    holes, t_depth, t_dibr = [], 0.0, 0.0
    with VideoReader(video, width=width) as vr:
        fps = vr.info.fps
        for f in vr.frames(start_s=start, end_s=start + seconds):
            t0 = time.perf_counter()
            inv = midas(f.rgb)
            t1 = time.perf_counter()
            lv = np.array(depth_range(inv))
            levels = lv if levels is None else smooth * levels + (1 - smooth) * lv
            disp = disparity_from_inverse_depth(inv, f.rgb.shape[1], max_disp, levels=tuple(levels))
            left, right, hole = stereo_pair(f.rgb, disp)
            t_depth += t1 - t0
            t_dibr += time.perf_counter() - t1
            holes.append(hole)
            ana.append(anaglyph(left, right))
            sbs.append(side_by_side(left, right, half=False))
    out_dir.mkdir(parents=True, exist_ok=True)
    write_video(out_dir / "stereo_anaglyph.mp4", np.stack(ana), fps)
    write_video(out_dir / "stereo_sbs.mp4", np.stack(sbs), fps)
    mid = len(ana) // 2
    Image.fromarray(np.concatenate([sbs[mid], np.concatenate([ana[mid], ana[mid]], 1)], 0)).save(
        out_dir / "stereo_still.jpg", quality=85)
    n = len(ana)
    res = {"frames": n, "fps": fps, "width": width, "max_disparity_pct": max_disp,
           "mean_hole_fraction": round(float(np.mean(holes)), 4),
           "ms_per_frame_depth": round(1000 * t_depth / n, 1), "ms_per_frame_dibr": round(1000 * t_dibr / n, 1)}
    print(json.dumps(res))
    return res


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", nargs="?")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--width", type=int, default=480)
    ap.add_argument("--max-disp", type=float, default=2.5, help="nearest-point disparity, %% of width")
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args(argv)
    RESULTS.mkdir(parents=True, exist_ok=True)
    Path("outputs/p5").mkdir(parents=True, exist_ok=True)
    res = {}
    if args.synthetic:
        res["synthetic"] = synthetic()
    if args.video:
        res[Path(args.video).stem] = convert(args.video, args.start, args.seconds, args.width, args.max_disp,
                                             Path("outputs") / Path(args.video).stem)
    if res:
        path = RESULTS / "stereo3d.json"
        old = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        path.write_text(json.dumps({**old, **res}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
