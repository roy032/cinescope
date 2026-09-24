"""Thumbnail grid of a video — the first "does decoding work?" check.

    python -m cinescope.io.preview data/clips/trailer.mp4 --cols 6 --rows 5

Writes outputs/<name>_preview.jpg: frames sampled evenly across the clip, each
labelled with its timestamp.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .video import VideoReader


def contact_sheet(frames: list[np.ndarray], times: list[float], cols: int,
                  pad: int = 4, label: bool = True) -> Image.Image:
    h, w = frames[0].shape[:2]
    rows = int(np.ceil(len(frames) / cols))
    sheet = Image.new("RGB", (cols * (w + pad) + pad, rows * (h + pad) + pad), (16, 16, 16))
    draw = ImageDraw.Draw(sheet)
    for i, (rgb, t) in enumerate(zip(frames, times, strict=True)):
        x, y = pad + (i % cols) * (w + pad), pad + (i // cols) * (h + pad)
        sheet.paste(Image.fromarray(rgb), (x, y))
        if label:
            text = f"{int(t // 60)}:{t % 60:05.2f}"
            draw.rectangle([x, y, x + 7 * len(text) + 6, y + 14], fill=(0, 0, 0))
            draw.text((x + 3, y + 1), text, fill=(255, 255, 255))
    return sheet


def main(argv: list[str] | None = None) -> Path:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--cols", type=int, default=6)
    ap.add_argument("--rows", type=int, default=5)
    ap.add_argument("--width", type=int, default=240, help="thumbnail width in pixels")
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args(argv)

    with VideoReader(args.video, width=args.width) as vr:
        n = args.cols * args.rows
        duration = vr.info.duration_s or 1.0
        fps = n / duration
        frames, times = [], []
        for f in vr.frames(sample_fps=fps):
            frames.append(f.rgb)
            times.append(f.t)
            if len(frames) == n:
                break
        info = vr.info
    if not frames:
        raise SystemExit(f"no frames decoded from {args.video}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{Path(args.video).stem}_preview.jpg"
    contact_sheet(frames, times, args.cols).save(path, quality=88)
    print(f"{info.width}x{info.height} @ {info.fps:.3f} fps, {info.duration_s:.1f}s "
          f"({info.backend}) -> {path}")
    return path


if __name__ == "__main__":
    main()
