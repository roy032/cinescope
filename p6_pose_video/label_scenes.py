"""Render every shot of a clip (three frames each) for scene-type labelling by eye.

    python p6_pose_video/label_scenes.py goldeneye --out outputs/p6/label_scenes
    # look at outputs/p6/label_scenes/goldeneye_shots.jpg, write p6_pose_video/labels/goldeneye_scenes.json

The sheet shows each Phase 0 shot as one row: frames at 25%, 50% and 75% of
the shot, with the shot index and time span. Labels are written before any
classifier is run on the clip. Classes: see cinescope/video/scenes.py.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cinescope.video.clips import clip_name, resolve, shots  # noqa: E402
from cinescope.video.frames import frames_at  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clip")
    ap.add_argument("--width", type=int, default=240)
    ap.add_argument("--per-sheet", type=int, default=12)
    ap.add_argument("--out", default="outputs/p6/label_scenes")
    args = ap.parse_args(argv)
    path = resolve(args.clip)
    sh = shots(path)["shots"]
    times = [s["t0"] + f * (s["t1"] - s["t0"]) for s in sh for f in (0.25, 0.5, 0.75)]
    frames = frames_at(path, times, width=args.width)
    h = frames[0].shape[0]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for page in range(0, len(sh), args.per_sheet):
        rows = sh[page:page + args.per_sheet]
        sheet = Image.new("RGB", (110 + 3 * (args.width + 4), len(rows) * (h + 4) + 4), (20, 20, 20))
        dr = ImageDraw.Draw(sheet)
        for r, s in enumerate(rows):
            i = page + r
            y = 4 + r * (h + 4)
            dr.text((4, y + 4), f"shot {s['index']}", fill=(255, 255, 0))
            dr.text((4, y + 20), f"{s['t0']:.1f}-{s['t1']:.1f}s", fill=(255, 255, 255))
            for j in range(3):
                sheet.paste(Image.fromarray(frames[3 * i + j]), (110 + j * (args.width + 4), y))
        p = out / f"{clip_name(path)}_shots_{page // args.per_sheet}.jpg"
        sheet.save(p, quality=88)
        print(p)


if __name__ == "__main__":
    main()
