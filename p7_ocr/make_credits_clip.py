"""A synthetic credits sequence with known text: two static title cards over
frames of any clip, then rolling credits on black.

    python p7_ocr/make_credits_clip.py --bg trailers/Megamind.avi
    # -> outputs/p7/credits/credits_test.mp4 and p7_ocr/labels/credits_test.json

Names are random first-name/surname pairs (no real crew); roles are the usual
credit roles. Rolling speed is 60 px/s at 540 p, a typical end-credits pace.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from cinescope.io.video import VideoReader  # noqa: E402
from cinescope.ocr.synth import find_fonts, load_font  # noqa: E402

FIRST = ["Eleanor", "Arthur", "Margaret", "Henry", "Beatrice", "Thomas", "Clara", "Walter", "Edith",
         "Frederick", "Harriet", "Samuel", "Lucy", "Albert", "Florence", "George", "Violet", "Edmund",
         "Agnes", "Charles", "Mabel", "Oliver", "Ada", "Hugh", "Rosa", "Philip", "Ida", "Lionel"]
LAST = ["Whitcombe", "Ashford", "Pemberton", "Hale", "Granger", "Morley", "Fenwick", "Lockhart",
        "Thornbury", "Carrow", "Bramley", "Kingsley", "Sutton", "Rowe", "Halloway", "Marsh",
        "Ellery", "Dunmore", "Wickham", "Ainsworth", "Blakeley", "Corbett", "Prescott", "Tilney"]
ROLES = ["Directed by", "Produced by", "Written by", "Director of Photography", "Film Editor",
         "Music by", "Production Designer", "Costume Designer", "Sound Mixer", "Casting by",
         "Visual Effects Supervisor", "Script Supervisor", "Gaffer", "Key Grip", "Colourist"]


def font_named(part: str, size: int):
    fonts = [f for f in find_fonts() if part.lower() in Path(f).name.lower()]
    return load_font(fonts[0] if fonts else None, size)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bg", default=str(ROOT / "trailers" / "Megamind.avi"))
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--name", default="credits_test")
    ap.add_argument("--fps", type=float, default=24.0)
    args = ap.parse_args(argv)
    rng = random.Random(args.seed)
    W, H = 960, 540

    def name():
        return f"{rng.choice(FIRST)} {rng.choice(LAST)}"

    with VideoReader(args.bg, width=W) as vr:
        bgs = [f.rgb for f in vr.frames(sample_fps=0.25, start_s=1.5)][:4]
    bgs = [np.asarray(Image.fromarray(b).resize((W, H))) for b in bgs]

    title_font = font_named("LiberationSerif-Bold", 54)
    card_font = font_named("LiberationSans-Regular", 34)
    role_font = font_named("LiberationSans-Regular", 22)
    name_font = font_named("LiberationSans-Bold", 30)

    title = "THE SPECKLED BAND"
    director = name()
    cards = [([title], [title_font]), (["Directed by", director], [card_font, name_font])]
    roll = []                                      # (text, font, colour)
    for role in ROLES:
        roll.append((role, role_font, (190, 190, 190)))
        for _ in range(rng.choice([1, 1, 2])):
            roll.append((name(), name_font, (255, 255, 255)))
        roll.append(("", None, None))
    truth = [title, "Directed by", director] + [t for t, _, _ in roll if t]

    # the roll as one tall image
    gap = 46
    page_h = H + gap * len(roll) + H
    page = Image.new("RGB", (W, page_h), (0, 0, 0))
    d = ImageDraw.Draw(page)
    y = H
    for text, f, col in roll:
        if text:
            d.text((W / 2, y), text, font=f, fill=col, anchor="mm")
        y += gap
    page = np.asarray(page)

    out = ROOT / "outputs" / "p7" / "credits" / f"{args.name}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    import av

    speed = 60.0                                   # px / s
    card_s = 3.0
    roll_s = (page_h - H) / speed
    n = int((2 * card_s + roll_s) * args.fps)
    with av.open(str(out), "w") as c:
        s = c.add_stream("libx264", rate=Fraction(args.fps).limit_denominator(1001))
        s.width, s.height, s.pix_fmt = W, H, "yuv420p"
        s.options = {"crf": "23"}
        for i in range(n):
            t = i / args.fps
            if t < 2 * card_s:
                k = int(t // card_s)
                img = Image.fromarray(bgs[k % len(bgs)])
                dd = ImageDraw.Draw(img)
                lines, fonts = cards[k]
                for j, (text, f) in enumerate(zip(lines, fonts, strict=True)):
                    dd.text((W / 2, H / 2 + (j - (len(lines) - 1) / 2) * 50), text, font=f,
                            fill=(255, 255, 255), anchor="mm", stroke_width=2, stroke_fill=(0, 0, 0))
                frame = np.asarray(img)
            else:
                off = int((t - 2 * card_s) * speed)
                frame = page[off:off + H]
                if frame.shape[0] < H:
                    frame = np.concatenate([frame, np.zeros((H - frame.shape[0], W, 3), np.uint8)])
            for p in s.encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(frame), format="rgb24")):
                c.mux(p)
        for p in s.encode():
            c.mux(p)
    lab = ROOT / "p7_ocr" / "labels" / f"{args.name}.json"
    lab.write_text(json.dumps({"video": f"outputs/p7/credits/{args.name}.mp4", "background": Path(args.bg).name,
                               "seed": args.seed, "lines": truth}, indent=1))
    print(f"{len(truth)} lines, {n / args.fps:.1f} s -> {out}, {lab}")


if __name__ == "__main__":
    main()
