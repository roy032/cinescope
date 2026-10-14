"""Remove an object (by default: the largest person) from a stretch of a clip.

    python p4_segmentation/remove.py trailers/vtest.avi --start 10 --end 16 --box 600 150 640 240
    python p4_segmentation/remove.py trailers/vtest.avi --start 10 --end 16          # largest person
    # -> outputs/<clip>/removed_<start>_<end>.mp4, removed_<start>_<end>.jpg (before / mask / after strip)

The object is tracked from --start to --end (best-IoU YOLOv8n-seg person each
frame, else constant velocity), segmented in every frame (--segmenter,
default our GrabCut), and filled from a background plate: the median of
pixels that other frames saw uncovered. --context seconds before and after
the range are tracked too and contribute background samples only. Pixels
never uncovered fall back to spatial inpainting (--spatial). A moving camera
is detected and handled by per-frame spatial inpainting instead.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cinescope.io.video import VideoReader, write_video  # noqa: E402
from cinescope.segment.pipeline import SEGMENTERS, get_segmenter, largest_person  # noqa: E402
from cinescope.segment.video_remove import remove_object, track_object  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clip")
    ap.add_argument("--start", type=float, required=True)
    ap.add_argument("--end", type=float, required=True)
    ap.add_argument("--box", type=float, nargs=4, metavar=("X0", "Y0", "X1", "Y1"),
                    help="object box in the first frame at --start (default: largest person)")
    ap.add_argument("--segmenter", default="grabcut", choices=SEGMENTERS)
    ap.add_argument("--spatial", default="telea", choices=("telea", "ns", "criminisi"))
    ap.add_argument("--context", type=float, default=4.0, help="seconds tracked before/after for background samples")
    ap.add_argument("--dilate", type=int, default=4, help="mask dilation, pixels")
    ap.add_argument("--max-width", type=int, default=960)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    from cinescope.segment.models import PersonDetector, set_threads

    set_threads(args.threads)
    with VideoReader(args.clip) as vr:
        fps, width = vr.info.fps, vr.info.width
    t_lo = max(0.0, args.start - args.context)
    with VideoReader(args.clip, width=args.max_width if width > args.max_width else None) as vr:
        clip_frames = list(vr.frames(start_s=t_lo, end_s=args.end + args.context))
    times = np.array([f.t for f in clip_frames])
    rgbs = [f.rgb for f in clip_frames]
    scale = rgbs[0].shape[1] / width
    i0 = int(np.searchsorted(times, args.start - 1e-6))
    i1 = int(np.searchsorted(times, args.end - 1e-6))
    if i1 <= i0:
        sys.exit("no frames in [start, end)")
    det = PersonDetector(threads=args.threads)
    if args.box:
        box0 = np.asarray(args.box, np.float64) * scale
    else:
        box0 = largest_person(rgbs[i0], det)
        if box0 is None:
            sys.exit(f"no person at t={args.start}; pass --box")
        print("largest person box:", np.round(box0 / scale).astype(int).tolist())
    seg = get_segmenter(args.segmenter, args.threads, detector=det)

    t0 = time.perf_counter()
    fwd = track_object(rgbs[i0:], box0, det, seg, dilate=args.dilate)
    back = track_object(rgbs[:i0 + 1][::-1], box0, det, seg, dilate=args.dilate) if i0 > 0 else None
    masks = (back.masks[::-1][:-1] if back else []) + fwd.masks
    others = (back.others[::-1][:-1] if back else []) + fwd.others
    matched = np.r_[back.matched[::-1][:-1] if back else [], fwd.matched]
    t1 = time.perf_counter()
    # context frames only give background samples: mask the object and every other person
    context = [(rgbs[i], masks[i] | others[i]) for i in range(len(rgbs)) if i < i0 or i >= i1]
    out, report = remove_object(rgbs[i0:i1], masks[i0:i1], spatial=args.spatial, context=context,
                                exclude=others[i0:i1])
    t2 = time.perf_counter()
    report.update(frames=i1 - i0, context_frames=len(context), segmenter=args.segmenter,
                  tracked_by_detection=round(float(matched.mean()), 3),
                  track_seg_s=round(t1 - t0, 1), fill_s=round(t2 - t1, 1),
                  ms_per_frame=round(1000 * (t2 - t0) / len(rgbs), 1))
    print(json.dumps(report))

    stem = Path(args.clip).stem
    out_path = Path(args.out) if args.out else Path("outputs") / stem / f"removed_{args.start:g}_{args.end:g}.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_video(out_path, np.stack(out), fps=fps)
    # before / mask / after strip at the middle frame
    k = (i1 - i0) // 2
    from PIL import Image

    before = rgbs[i0 + k]
    shown = before.copy()
    shown[masks[i0 + k]] = (0.5 * shown[masks[i0 + k]] + [127, 0, 0]).astype(np.uint8)
    Image.fromarray(np.concatenate([before, shown, out[k]], 1)).save(out_path.with_suffix(".jpg"), quality=85)
    out_path.with_suffix(".json").write_text(json.dumps(report, indent=1))
    print("->", out_path)


if __name__ == "__main__":
    main()
