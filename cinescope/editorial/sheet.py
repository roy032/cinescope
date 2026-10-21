"""A printable storyboard of a report: every shot's frame with its timecode,
length, framing, camera move and the characters in it, as PNG pages or one PDF.

This is the contact sheet an editor or a script supervisor marks up: shot
numbers match the EDL, FCPXML and CSV exports, so a note on paper finds its
event in the cut.
"""
from __future__ import annotations

from pathlib import Path

from .exports import cut_list, rate_of
from .pacing import pacing
from .timecode import to_timecode

INK = (24, 24, 28)
DIM = (110, 110, 118)
PAPER = (250, 250, 247)
RULE = (214, 214, 210)
ACCENT = (196, 62, 44)


def _font(size: int):
    from PIL import ImageFont

    for name in ("DejaVuSans.ttf", "Arial.ttf", "arial.ttf", "Helvetica.ttc"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)            # Pillow >= 10.1
    except TypeError:
        return ImageFont.load_default()


def _fit(draw, text: str, font, width: int) -> str:
    if draw.textlength(text, font=font) <= width:
        return text
    while text and draw.textlength(text + "…", font=font) > width:
        text = text[:-1]
    return text + "…"


def _frame_at(video: str | None, t: float, width: int):
    """The frame at t decoded from the source video, or None (no video, or past its end)."""
    if not video or not Path(video).is_file():
        return None
    try:
        from PIL import Image

        from ..io.video import VideoReader

        with VideoReader(video, width=width) as vr:
            f = next(iter(vr.frames(start_s=t)), None)
        return None if f is None else Image.fromarray(f.rgb)
    except Exception:  # noqa: BLE001 - a missing frame leaves a blank cell, not a failed sheet
        return None


def contact_sheet(report: dict, root: str | Path, cols: int = 4, rows: int = 4, page_w: int = 2339,
                  margin: int = 80, video: str | None = None) -> list:
    """Pages (PIL images, A4 landscape at 200 dpi by default) of `rows` x `cols`
    shots each. `root` is outputs/<clip>/, which the report's thumbnail paths
    are relative to. A shot without a thumbnail (the 'frames' stage did not run)
    gets its middle frame decoded from `video` (default: the report's
    video_path), or a blank frame if that cannot be read."""
    from PIL import Image, ImageDraw

    video = video or report.get("video_path")

    root = Path(root)
    rate = rate_of(report)
    df = rate.drop_frame_possible
    info = report["info"]
    shots = cut_list(report)
    page_h = int(page_w / 1.4142)
    aspect = info["height"] / max(info["width"], 1)
    gap = 28
    cell_w = (page_w - 2 * margin - (cols - 1) * gap) // cols
    img_h = int(cell_w * aspect)
    head_h = 150
    cell_h = (page_h - 2 * margin - head_h - (rows - 1) * gap) // rows
    img_h = min(img_h, cell_h - 78)
    img_w = int(img_h / aspect)
    f_title, f_head, f_tc, f_txt = _font(46), _font(24), _font(24), _font(21)
    p = pacing(report)
    per_page = cols * rows
    n_pages = max(1, -(-len(shots) // per_page))
    pages = []
    for pg in range(n_pages):
        im = Image.new("RGB", (page_w, page_h), PAPER)
        d = ImageDraw.Draw(im)
        d.text((margin, margin), report.get("clip", "clip"), font=f_title, fill=INK)
        facts = [f"{len(shots)} shots", f"{info['duration_s']:.1f} s", f"{info['width']}x{info['height']}",
                 f"{rate.fps:.3f} fps"]
        if p:
            facts += [f"average shot {p['asl_s']:.1f} s", f"median {p['median_s']:.1f} s",
                      f"{p['cuts_per_min']:.1f} cuts/min"]
        d.text((margin, margin + 62), "   ·   ".join(facts), font=f_head, fill=DIM)
        d.text((page_w - margin, margin + 14), f"page {pg + 1} of {n_pages}", font=f_head, fill=DIM, anchor="ra")
        y_rule = margin + head_h - 30
        d.line([(margin, y_rule), (page_w - margin, y_rule)], fill=RULE, width=3)
        for k, s in enumerate(shots[pg * per_page:(pg + 1) * per_page]):
            r, c = divmod(k, cols)
            x = margin + c * (cell_w + gap)
            y = margin + head_h + r * (cell_h + gap)
            box = (x, y, x + img_w, y + img_h)
            th = root / s["thumb"] if s["thumb"] else None
            pic = None
            if th is not None and th.is_file():
                with Image.open(th) as t:
                    pic = t.convert("RGB")
            else:
                pic = _frame_at(video, (s["t0"] + s["t1"]) / 2, img_w)
            if pic is not None:
                im.paste(pic.resize((img_w, img_h), Image.LANCZOS), box[:2])
            else:
                d.rectangle(box, fill=(60, 60, 64))
            d.rectangle(box, outline=RULE, width=2)
            tc = to_timecode(s["f0"], rate, df)
            d.text((x, y + img_h + 10), f"#{s['index']}", font=f_tc, fill=ACCENT)
            d.text((x + 70, y + img_h + 10), tc, font=f_tc, fill=INK)
            d.text((x + img_w, y + img_h + 10), f"{s['t1'] - s['t0']:.1f} s", font=f_tc, fill=DIM, anchor="ra")
            line = " · ".join(v for v in (s["framing_name"] or s["framing"], s["camera"], s["scene"]) if v)
            if s["characters"]:
                line = f"{line} · {', '.join(s['characters'])}" if line else ", ".join(s["characters"])
            d.text((x, y + img_h + 42), _fit(d, line or "—", f_txt, img_w), font=f_txt, fill=DIM)
        pages.append(im)
    return pages


def save_sheet(report: dict, root: str | Path, dest: str | Path, **kw) -> list[Path]:
    """dest.pdf -> one multi-page PDF; dest.png -> dest.png, dest-2.png, ... one per page."""
    dest = Path(dest)
    pages = contact_sheet(report, root, **kw)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.suffix.lower() == ".pdf":
        pages[0].save(dest, "PDF", resolution=200.0, save_all=True, append_images=pages[1:])
        return [dest]
    out = []
    for i, pg in enumerate(pages):
        p = dest if i == 0 else dest.with_name(f"{dest.stem}-{i + 1}{dest.suffix}")
        pg.save(p)
        out.append(p)
    return out
