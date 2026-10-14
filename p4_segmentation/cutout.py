"""Cut an actor out of a frame: box -> mask -> matte -> RGBA PNG.

    python p4_segmentation/cutout.py trailers/goldeneye.mp4 --t 12.5 --box 400 60 700 400
    python p4_segmentation/cutout.py trailers/vtest.avi --t 20            # largest person, auto
    python p4_segmentation/cutout.py clip.mp4 --t 3 --method sam2        # SAM 2 mask instead of ours
    # -> outputs/<clip>/cutout_<t>.png (RGBA) and cutout_<t>_preview.jpg

Default pipeline is all from scratch: GrabCut (own max-flow) for the mask,
closed-form matting on a trimap from that mask for soft edges, blur-fusion
foreground colours. The box comes from --box or, if omitted, from the
largest person YOLOv8n-seg finds.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cinescope.segment.pipeline import (  # noqa: E402
    SEGMENTERS,
    frame_at,
    get_segmenter,
    largest_person,
    rgba_cutout,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clip")
    ap.add_argument("--t", type=float, default=0.0, help="time in seconds")
    ap.add_argument("--box", type=float, nargs=4, metavar=("X0", "Y0", "X1", "Y1"))
    ap.add_argument("--method", default="grabcut", choices=SEGMENTERS)
    ap.add_argument("--no-matting", action="store_true", help="hard mask edges")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--out", default=None, help="output PNG path")
    args = ap.parse_args()
    from cinescope.segment.models import set_threads

    set_threads(args.threads)
    rgb, t = frame_at(args.clip, args.t)
    if args.box:
        box = np.asarray(args.box, np.float64)
    else:
        from cinescope.segment.models import PersonDetector

        box = largest_person(rgb, PersonDetector(threads=args.threads))
        if box is None:
            sys.exit(f"no person found at t={t:.2f} s; pass --box x0 y0 x1 y1")
        print("largest person box:", np.round(box).astype(int).tolist())
    seg = get_segmenter(args.method, args.threads)
    t0 = time.perf_counter()
    mask = seg(rgb, box[None])[0]
    t1 = time.perf_counter()
    if not mask.any():
        sys.exit("empty mask; try another --method or a tighter --box")
    rgba, (x0, y0) = rgba_cutout(rgb, mask, matting=not args.no_matting)
    t2 = time.perf_counter()
    print(f"mask ({args.method}): {1000 * (t1 - t0):.0f} ms, {int(mask.sum())} px; "
          f"matting: {1000 * (t2 - t1):.0f} ms; cut-out {rgba.shape[1]}x{rgba.shape[0]} at ({x0}, {y0})")

    from PIL import Image

    out = Path(args.out) if args.out else Path("outputs") / Path(args.clip).stem / f"cutout_{t:.2f}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba, "RGBA").save(out)
    # preview: the cut-out over a checkerboard, next to the original crop
    h, w = rgba.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    checker = np.where(((yy // 12 + xx // 12) % 2)[..., None] == 0, 200, 120).astype(np.float64)
    a = rgba[..., 3:4] / 255.0
    over = (a * rgba[..., :3] + (1 - a) * checker).astype(np.uint8)
    prev = np.concatenate([rgb[y0:y0 + h, x0:x0 + w], over], 1)
    Image.fromarray(prev).save(out.with_name(out.stem + "_preview.jpg"), quality=85)
    print("->", out)


if __name__ == "__main__":
    main()
