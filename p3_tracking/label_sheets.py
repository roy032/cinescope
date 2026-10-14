"""Make labelling sheets for face identities (ground truth for Weeks 9-10).

    python p3_tracking/label_sheets.py trailers/goldeneye.mp4 --every 12

Samples every N-th frame, draws every face the detector proposes (score >=
0.3) with a number, and tiles the frames into numbered contact sheets
(outputs/label_<clip>/sheet_XX.jpg). It also writes a template
outputs/label_<clip>/<clip>.ids.json with one entry per proposed box and
"id": null. The labeller looks at the sheets and fills in each box's
identity (a name, the same name for the same person all through the clip),
"fp" for a box that is not a face, or "ignore" for a face too small or
blurred to identify; faces the detector missed are added by hand as
{"bbox": [...], "id": ...}. The sheets show **no tracker output**, so the
labels cannot inherit the tracker's mistakes.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cinescope.io.video import VideoReader  # noqa: E402
from p3_tracking.track_clip import load_or_detect  # noqa: E402

COLORS = [(255, 64, 64), (64, 200, 255), (255, 210, 0), (120, 255, 120), (255, 120, 255), (255, 160, 60)]


def font(size: int):
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", size)
    except OSError:
        return ImageFont.load_default()


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--every", type=int, default=12, help="label every N-th frame")
    ap.add_argument("--min-score", type=float, default=0.3)
    ap.add_argument("--thumb", type=int, default=480, help="thumbnail width")
    ap.add_argument("--cols", type=int, default=3)
    ap.add_argument("--rows", type=int, default=4)
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args(argv)

    stem = Path(args.video).stem
    dets = load_or_detect(args.video, Path(args.out) / stem)
    out = Path(args.out) / f"label_{stem}"
    out.mkdir(parents=True, exist_ok=True)
    f = font(16)
    entries, thumbs = [], []
    with VideoReader(args.video) as vr:
        for fr in vr.frames():
            k = fr.index
            if k % args.every:
                continue
            fd = dets.frames[k]
            keep = np.flatnonzero(fd.scores >= args.min_score)
            s = args.thumb / fr.rgb.shape[1]
            im = Image.fromarray(fr.rgb).resize((args.thumb, round(fr.rgb.shape[0] * s)), Image.BILINEAR)
            d = ImageDraw.Draw(im)
            boxes = []
            for n, j in enumerate(keep):
                b = fd.boxes[j]
                c = COLORS[n % len(COLORS)]
                d.rectangle([*(b[:2] * s), *(b[2:] * s)], outline=c, width=2)
                d.text((b[0] * s + 2, max(0, b[1] * s - 18)), str(n), fill=c, font=f)
                boxes.append({"n": n, "bbox": np.round(b, 1).tolist(), "score": round(float(fd.scores[j]), 3),
                              "id": None})
            d.rectangle([0, 0, 92, 20], fill=(0, 0, 0))
            d.text((3, 1), f"#{k}", fill=(255, 255, 255), font=f)
            thumbs.append(im)
            entries.append({"index": k, "t": round(fr.t, 3), "boxes": boxes})
    per = args.cols * args.rows
    w, h = thumbs[0].size if thumbs else (args.thumb, args.thumb)
    for s in range(0, len(thumbs), per):
        sheet = Image.new("RGB", (args.cols * w, args.rows * h), (30, 30, 30))
        for i, im in enumerate(thumbs[s:s + per]):
            sheet.paste(im, ((i % args.cols) * w, (i // args.cols) * h))
        sheet.save(out / f"sheet_{s // per:02d}.jpg", quality=85)
    tmpl = {"clip": stem, "video": Path(args.video).name, "every": args.every, "fps": dets.fps,
            "width": dets.width, "height": dets.height, "frames": entries}
    (out / f"{stem}.ids.json").write_text(json.dumps(tmpl, indent=1))
    print(f"{len(thumbs)} frames on {(len(thumbs) + per - 1) // per} sheets in {out}")


if __name__ == "__main__":
    main()
