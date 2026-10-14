"""Synthetic subtitle-style text lines for training a recogniser.

There is no labelled corpus of subtitle crops, but subtitles are rendered
text, so we can render them ourselves with known labels. What makes a crop
look like a real burned-in subtitle, and is randomised here:

  * fonts — every TrueType/OpenType font found on the machine, minus a
    held-out list (so a test can use a font the model never saw);
  * colour — white (most common) or yellow, sometimes light grey;
  * a dark outline (stroke 1-3 px) and/or a drop shadow, or neither;
  * the background — a crop of a real video frame, at the scale the text
    would sit on it, optionally with a translucent dark box behind the text;
  * the camera chain — blur, down/up-sampling, JPEG compression, noise;
  * the crop — around the ink (as a text detector boxes it), plus a random
    margin on each side, because the detector's boxes are not pixel-exact
    and its padding varies.

Everything works without system fonts (PIL's built-in font), which keeps the
tests independent of the machine.
"""
from __future__ import annotations

import io
import os
import random
import re
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from .charset import CHARS, normalize

FONT_DIRS = ["/usr/share/fonts", "/usr/local/share/fonts", str(Path.home() / ".fonts"),
             "C:/Windows/Fonts", "/Library/Fonts", "/System/Library/Fonts"]
# fonts that cannot draw Latin text or that no subtitle uses
_SKIP = re.compile(r"emoji|unifont|japanese|symbol|wingding|webdings|opens___|dingbat|cursor|"
                   r"\.pfb$|\.pcf", re.I)


def find_fonts(exclude: Sequence[str] = ()) -> list[str]:
    """All usable font files under the usual font folders, minus any whose
    file name contains one of `exclude` (case-insensitive)."""
    out = []
    for d in FONT_DIRS:
        if not os.path.isdir(d):
            continue
        for root, _, files in os.walk(d):
            for f in files:
                if f.lower().endswith((".ttf", ".otf")) and not _SKIP.search(f) \
                        and not any(x.lower() in f.lower() for x in exclude):
                    out.append(os.path.join(root, f))
    return sorted(out)


def load_font(path: str | None, size: int):
    if path is None:
        try:
            return ImageFont.load_default(size=size)       # Pillow >= 10.1: scalable default
        except TypeError:
            return ImageFont.load_default()
    return ImageFont.truetype(path, size)


# ---------------------------------------------------------------- text sources
def corpus_lines(text: str, rng: random.Random, n: int, max_chars: int = 44) -> list[str]:
    """n random snippets of a running text: word spans of 1..max_chars
    characters, mostly cut at word boundaries, a few in capitals (credits,
    signs) or with a dialogue dash in front."""
    words = normalize(text).split()
    out = []
    while len(out) < n:
        i = rng.randrange(len(words))
        target = rng.choice([rng.randint(1, 12), rng.randint(8, max_chars), rng.randint(20, max_chars)])
        span = []
        while i < len(words) and len(" ".join(span + [words[i]])) <= target:
            span.append(words[i])
            i += 1
        s = " ".join(span) or words[i - 1][:target]
        r = rng.random()
        if r < 0.08:
            s = s.upper()
        elif r < 0.13:
            s = "- " + s
        if s.strip():
            out.append(s[:max_chars])
    return out


def random_strings(rng: random.Random, n: int, vocab: Sequence[str]) -> list[str]:
    """Shuffled words, numbers and random characters: lines no language model
    can guess, so the network has to look at every glyph."""
    out = []
    for _ in range(n):
        r = rng.random()
        if r < 0.5 and vocab:
            words = [rng.choice(vocab) for _ in range(rng.randint(1, 7))]
            words = [w.capitalize() if rng.random() < 0.3 else w for w in words]
            s = " ".join(words)
            if rng.random() < 0.4:
                s += rng.choice(".,!?:;")
        elif r < 0.75:
            s = " ".join(str(rng.randint(0, 10 ** rng.randint(1, 5))) for _ in range(rng.randint(1, 3)))
            if rng.random() < 0.5:
                s = rng.choice(["Chapter", "Part", "Scene", "Year", "No.", "Room", "Track"]) + " " + s
        else:
            s = "".join(rng.choice(CHARS) for _ in range(rng.randint(1, 20))).strip()
        out.append(" ".join(s.split())[:44] or "a")
    return out


# ---------------------------------------------------------------- rendering
COLOURS = [((255, 255, 255), 0.6), ((255, 235, 40), 0.25), ((210, 210, 210), 0.1), ((120, 230, 255), 0.05)]


def _pick_colour(rng: random.Random):
    r, acc = rng.random(), 0.0
    for c, p in COLOURS:
        acc += p
        if r < acc:
            jitter = [max(0, min(255, v + rng.randint(-15, 15))) for v in c]
            return tuple(jitter)
    return COLOURS[0][0]


def background_crop(frames: Sequence[np.ndarray] | None, w: int, h: int, rng: random.Random) -> Image.Image:
    """A w×h patch from one of `frames` (RGB arrays), taken from a random place
    at a random scale; flat grey noise if no frames are given."""
    if not frames:
        g = rng.randint(0, 200)
        arr = np.clip(np.random.default_rng(rng.randrange(1 << 30)).normal(g, rng.uniform(0, 30), (h, w, 3)),
                      0, 255).astype(np.uint8)
        return Image.fromarray(arr)
    f = frames[rng.randrange(len(frames))]
    fh, fw = f.shape[:2]
    scale = rng.uniform(0.5, 1.5)
    cw, ch = max(4, min(fw, int(w * scale))), max(4, min(fh, int(h * scale)))
    x0, y0 = rng.randint(0, fw - cw), rng.randint(0, fh - ch)
    patch = Image.fromarray(f[y0:y0 + ch, x0:x0 + cw]).resize((w, h), Image.BILINEAR)
    if rng.random() < 0.5:
        patch = patch.transpose(Image.FLIP_LEFT_RIGHT)
    return patch


def render_line(text: str, rng: random.Random, fonts: Sequence[str | None] = (None,),
                frames: Sequence[np.ndarray] | None = None, out_h: int = 32,
                font: str | None = "random", degrade: bool = True) -> Image.Image:
    """Render one subtitle-style line and return the cropped grey image of
    height out_h (width keeps the aspect ratio)."""
    font_path = rng.choice(list(fonts)) if font == "random" else font
    size = rng.randint(22, 46)
    pil_font = load_font(font_path, size)
    stroke = rng.choice([0, 1, 2, 2, 3, 3]) if size > 26 else rng.choice([0, 1, 2])
    shadow = rng.random() < 0.35
    probe = ImageDraw.Draw(Image.new("L", (1, 1)))
    x0, y0, x1, y1 = probe.textbbox((0, 0), text, font=pil_font, stroke_width=stroke)
    tw, th = max(1, x1 - x0), max(1, y1 - y0)
    ascent, descent = (pil_font.getmetrics() if hasattr(pil_font, "getmetrics") else (th, 0))
    line_h = ascent + descent + 2 * stroke
    margin = size                                               # room for crop jitter
    w, h = tw + 2 * margin, line_h + 2 * margin
    img = background_crop(frames, w, h, rng).convert("RGB")
    if rng.random() < 0.15:                                     # translucent subtitle box
        box = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        ImageDraw.Draw(box).rectangle([margin - 6, margin - 4, margin + tw + 6, margin + line_h + 4],
                                      fill=(0, 0, 0, rng.randint(120, 230)))
        img = Image.alpha_composite(img.convert("RGBA"), box).convert("RGB")
    draw = ImageDraw.Draw(img)
    org = (margin - x0, margin + stroke)
    fill = _pick_colour(rng)
    if shadow:
        d = rng.randint(1, 4)
        draw.text((org[0] + d, org[1] + d), text, font=pil_font, fill=(0, 0, 0),
                  stroke_width=stroke, stroke_fill=(0, 0, 0))
    edge = tuple(rng.randint(0, 40) for _ in range(3))
    draw.text(org, text, font=pil_font, fill=fill, stroke_width=stroke, stroke_fill=edge)

    # crop around the *ink* (what a text detector boxes), with random margins
    # covering the detector's padding (0.18 × height) and its errors either way
    ix0, iy0, ix1, iy1 = draw.textbbox(org, text, font=pil_font, stroke_width=stroke)
    ih = max(1, iy1 - iy0)
    top = max(0, int(iy0 - rng.uniform(0.0, 0.4) * ih))
    bottom = min(h, int(iy1 + rng.uniform(0.0, 0.4) * ih) + 1)
    left = max(0, int(ix0 - rng.uniform(0.0, 0.8) * ih))
    right = min(w, int(ix1 + rng.uniform(0.0, 0.8) * ih) + 1)
    img = img.crop((left, top, right, bottom))
    if degrade:
        img = _degrade(img, rng)
    g = img.convert("L")
    new_w = max(out_h // 2, round(g.width * out_h / g.height))
    return g.resize((new_w, out_h), Image.BILINEAR)


def _degrade(img: Image.Image, rng: random.Random) -> Image.Image:
    if rng.random() < 0.5:                                      # low-resolution source
        f = rng.uniform(0.4, 0.9)
        small = img.resize((max(1, int(img.width * f)), max(1, int(img.height * f))), Image.BILINEAR)
        img = small.resize(img.size, Image.BILINEAR)
    if rng.random() < 0.5:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.2, 1.3)))
    if rng.random() < 0.7:                                      # video / JPEG compression
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=rng.randint(15, 90))
        img = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
    if rng.random() < 0.3:
        arr = np.asarray(img).astype(np.float32)
        arr = arr * rng.uniform(0.6, 1.1) + rng.uniform(-25, 25)
        arr += np.random.default_rng(rng.randrange(1 << 30)).normal(0, rng.uniform(1, 8), arr.shape)
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    return img


def frames_from_clips(paths: Sequence[str], per_clip: int = 40, width: int = 640,
                      seed: int = 0) -> list[np.ndarray]:
    """Evenly spaced frames from each clip, as background material."""
    from ..io.video import VideoReader

    out = []
    for p in paths:
        with VideoReader(p, width=width) as vr:
            dur = vr.info.duration_s or 60.0
            got = [f.rgb for f in vr.frames(sample_fps=per_clip / max(dur, 1e-3))]
        out.extend(got[:per_clip])
    return out
