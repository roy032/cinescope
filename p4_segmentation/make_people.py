"""Sample frames with people from clips and store YOLOv8n-seg boxes + masks.

    python p4_segmentation/make_people.py --role train trailers/classroom.mp4 trailers/one-by-one-person-detection.mp4 ...
    python p4_segmentation/make_people.py --role test  trailers/goldeneye.mp4 trailers/vtest.avi trailers/people-detection.mp4
    # -> outputs/p4data/people_<role>.npz

`train` sets feed the U-Net (distilled masks and composite cut-outs) and pick
every hyper-parameter; `test` sets are only ever scored. Frames wider than
--max-width are downscaled (960 px is plenty for masks and keeps CPU time sane).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cinescope.io.video import VideoReader  # noqa: E402
from cinescope.segment.data import sample_people, save_people  # noqa: E402
from cinescope.segment.models import PersonDetector, set_threads  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clips", nargs="+")
    ap.add_argument("--role", required=True, help="name of the set, e.g. train / test")
    ap.add_argument("--every", type=float, default=2.0, help="seconds between sampled frames")
    ap.add_argument("--max-width", type=int, default=960)
    ap.add_argument("--max-frames", type=int, default=None, help="per clip")
    ap.add_argument("--keep-empty", action="store_true", help="keep frames with no usable person")
    ap.add_argument("--out", default="outputs/p4data")
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args()
    set_threads(args.threads)
    det = PersonDetector(threads=args.threads)
    frames = []
    for clip in args.clips:
        with VideoReader(clip) as vr:
            w = vr.info.width
        t0 = time.time()
        got = sample_people(clip, det, every_s=args.every,
                            width=args.max_width if w > args.max_width else None,
                            max_frames=args.max_frames)
        if not args.keep_empty:
            got = [f for f in got if len(f.boxes)]
        n_people = sum(len(f.boxes) for f in got)
        print(f"{Path(clip).name}: {len(got)} frames, {n_people} people ({time.time() - t0:.0f} s)")
        frames += got
    out = Path(args.out) / f"people_{args.role}.npz"
    save_people(frames, out)
    print("->", out)


if __name__ == "__main__":
    main()
