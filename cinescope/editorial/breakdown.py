"""A shareable breakdown of a report: one self-contained HTML file.

Everything a producer, a director of photography or a film class would want on
one page: the facts, the movie barcode, the pace curve, the look and sound lanes
(when those stages ran), the characters, and every shot with its frame,
timecode and labels. Images are embedded as data URIs and the charts are inline
SVG, so the file opens anywhere, offline, and can be mailed or uploaded as is.
Shot numbers and timecodes match the EDL / FCPXML / CSV exports.

    python -m cinescope.editorial outputs/<clip>/report.json -f html
"""
from __future__ import annotations

import base64
import html
import io
from pathlib import Path

from .exports import cut_list, rate_of
from .pacing import pacing
from .timecode import to_timecode

W = 1000                       # SVG user units across a timeline


def _e(v) -> str:
    return html.escape("" if v is None else str(v), quote=True)


def _img_uri(path: Path, width: int, quality: int = 72) -> str | None:
    """A JPEG data URI of the image at `path`, at most `width` px wide; None if unreadable."""
    try:
        from PIL import Image

        with Image.open(path) as im:
            im = im.convert("RGB")
            if im.width > width:
                im = im.resize((width, max(1, round(im.height * width / im.width))))
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=quality, optimize=True)
    except (OSError, ValueError):
        return None
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _clock(t: float) -> str:
    t = max(0.0, float(t))
    h, m, s = int(t // 3600), int(t % 3600 // 60), t % 60
    return f"{h}:{m:02d}:{s:04.1f}" if h else f"{m}:{s:04.1f}"


def _x(t: float, dur: float) -> float:
    return W * t / max(dur, 1e-6)


def _lane(title: str, svg_body: str, h: int, note: str = "") -> str:
    return (f'<div class="lane"><div class="lt">{_e(title)}</div>'
            f'<svg viewBox="0 0 {W} {h}" preserveAspectRatio="none" style="height:{h}px" role="img" '
            f'aria-label="{_e(title)}">{svg_body}</svg>'
            + (f'<div class="ln">{_e(note)}</div>' if note else "") + "</div>")


def _barcode(rep: dict, dur: float) -> str:
    cols = (rep.get("barcode") or {}).get("colors") or []
    if not cols:
        return ""
    w = W / len(cols)
    body = "".join(f'<rect x="{i * w:.2f}" y="0" width="{w + 0.6:.2f}" height="40" fill="rgb({c[0]},{c[1]},{c[2]})"/>'
                   for i, c in enumerate(cols))
    return _lane("Movie barcode", body, 40, "the mean colour of every second")


def _pace(pc: dict | None, dur: float) -> str:
    if not pc:
        return ""
    t, r = pc["curve"]["t"], pc["curve"]["cuts_per_min"]
    top = max(max(r), 1e-6)
    pts = " ".join(f"{_x(a, dur):.1f},{58 - 54 * b / top:.1f}" for a, b in zip(t, r, strict=True))
    body = (f'<polygon points="0,60 {pts} {W},60" class="area"/>'
            f'<polyline points="{pts}" class="curve"/>')
    return _lane("Pace", body, 60, f"cuts per minute over a {pc['window_s']:.0f} s window; "
                                   f"fastest {pc['fastest']['cuts_per_min']:.0f} at {_clock(pc['fastest']['t'])}")


def _warm(w: float) -> str:
    x = max(-1.0, min(1.0, (w or 0.0) / 20))
    a, b = (128, 128, 128), ((64, 156, 214) if x < 0 else (235, 150, 50))
    r, g, bb = (round(p + (q - p) * abs(x)) for p, q in zip(a, b, strict=True))
    return f"rgb({r},{g},{bb})"


def _look(rep: dict, dur: float) -> str:
    shots = [s for s in rep.get("shots") or [] if s.get("look")]
    if not shots:
        return ""
    def cell(sh: dict, fill: str) -> str:
        x0, x1 = _x(sh["t0"], dur), _x(sh["t1"], dur)
        return f'<rect x="{x0:.1f}" y="0" width="{max(x1 - x0, 0.8):.1f}" height="22" fill="{fill}"/>'

    def grey(y: float) -> str:
        v = round(255 * max(0.0, min(1.0, y)))
        return f"rgb({v},{v},{v})"

    exp = "".join(cell(sh, grey(sh["look"]["brightness"])) for sh in shots)
    tmp = "".join(cell(sh, _warm(sh["look"]["warmth"])) for sh in shots)
    return _lane("Exposure", exp, 22) + _lane("Colour temperature", tmp, 22, "blue cool, grey neutral, amber warm")


def _loudness(snd: dict | None, dur: float) -> str:
    if not snd or not snd.get("audio") or not (snd.get("envelope") or {}).get("lufs"):
        return ""
    env = snd["envelope"]
    lo, hi = -60.0, max(-10.0, max(env["lufs"]))
    pts = " ".join(f"{_x(env['t0'] + i * env['dt'], dur):.1f},{60 - 58 * (max(v, lo) - lo) / (hi - lo):.1f}"
                   for i, v in enumerate(env["lufs"]))
    bands = "".join(f'<rect x="{_x(e["t0"], dur):.1f}" y="0" width="{max(_x(e["t1"], dur) - _x(e["t0"], dur), 2):.1f}" '
                    f'height="60" class="{_e(e["label"])}"/>' for e in snd.get("timeline", []))
    x0, x1 = _x(env["t0"], dur), _x(env["t0"] + (len(env["lufs"]) - 1) * env["dt"], dur)
    return _lane("Loudness", bands + f'<polygon points="{x0:.1f},60 {pts} {x1:.1f},60" class="area"/>', 60,
                 "momentary loudness, LUFS; red: peaks, blue: silences")


def _story(st: dict | None, dur: float) -> str:
    if not st or not (st.get("intensity") or {}).get("value"):
        return ""
    iv = st["intensity"]
    pts = " ".join(f"{_x(iv['t0'] + i * iv['dt'], dur):.1f},{58 - 54 * v:.1f}" for i, v in enumerate(iv["value"]))
    x0, x1 = _x(iv["t0"], dur), _x(iv["t0"] + (len(iv["value"]) - 1) * iv["dt"], dur)
    acts = "".join(f'<rect x="{_x(a["t0"], dur):.1f}" y="0" width="{max(_x(a["t1"], dur) - _x(a["t0"], dur), 1):.1f}" '
                   f'height="60" class="act{a["act"]}"/>' for a in st.get("acts", []))
    scenes = "".join(f'<line x1="{_x(s["t0"], dur):.1f}" x2="{_x(s["t0"], dur):.1f}" y1="0" y2="8" class="scene"/>'
                     for s in st.get("scenes", [])[1:])
    beats = "".join(f'<line x1="{_x(b["t"], dur):.1f}" x2="{_x(b["t"], dur):.1f}" y1="0" y2="60" class="beat">'
                    f'<title>{_e(b["label"])} {_clock(b["t"])}</title></line>' for b in st.get("beats", []))
    sm = st.get("summary") or {}
    note = (f"intensity from {', '.join(sm.get('signals', []))}; shaded: acts 1, 2, 3; lines: "
            + ", ".join(f"{b['label']} {_clock(b['t'])}" for b in st.get("beats", [])))
    return _lane("Story", acts + f'<polygon points="{x0:.1f},60 {pts} {x1:.1f},60" class="area"/>'
                 f'<polyline points="{pts}" class="curve"/>' + scenes + beats, 60, note)


def _facts(rep: dict, pc: dict | None) -> list[tuple[str, str]]:
    i = rep["info"]
    fps = f"{i['fps']:.3f}".rstrip("0").rstrip(".")
    f = [("Length", _clock(i["duration_s"])), ("Frame", f"{i['width']}×{i['height']} @ {fps} fps")]
    if pc:
        f += [("Shots", str(pc["n_shots"])), ("Average shot", f"{pc['asl_s']:.1f} s"),
              ("Median shot", f"{pc['median_s']:.1f} s"), ("Cuts per minute", f"{pc['cuts_per_min']:.1f}")]
    lk = ((rep.get("extra") or {}).get("look") or {}).get("summary") or {}
    if lk:
        f += [("Aspect ratio", lk.get("aspect_name", "")), ("Lighting", lk["key"]), ("Colour", lk["temperature"])]
    ed = ((rep.get("extra") or {}).get("editing") or {}).get("summary") or {}
    if ed:
        f += [("Camera set-ups", str(ed["setups"])), ("Shot/reverse-shot", str(ed["exchanges"]))]
    sd = ((rep.get("extra") or {}).get("sound") or {}).get("summary") or {}
    if sd.get("integrated_lufs") is not None:
        f += [("Loudness", f"{sd['integrated_lufs']} LUFS"), ("Loudness range", f"{sd['lra_lu']} LU")]
        if sd.get("speech_share") is not None:
            f.append(("Speech", f"{round(100 * sd['speech_share'])}%"))
    sy = ((rep.get("extra") or {}).get("story") or {}).get("summary") or {}
    if sy:
        f += [("Scenes", str(sy["scenes"])), ("Average scene", f"{sy['mean_scene_s']:.0f} s"),
              ("Arc", sy["arc"])]
    cp = ((rep.get("extra") or {}).get("composition") or {}).get("summary") or {}
    if cp:
        f += [("On the thirds", f"{round(100 * cp['thirds_share'])}%"),
              ("Centred", f"{round(100 * cp['share']['centre'])}%"),
              ("Shallow focus", f"{round(100 * cp['shallow_share'])}%")]
    ch = (rep.get("characters") or {}).get("list")
    if ch:
        f.append(("Characters", str(len(ch))))
    return f


def _characters(rep: dict, root: Path) -> str:
    chars = (rep.get("characters") or {}).get("list") or []
    if not chars:
        return ""
    cards = []
    for c in sorted(chars, key=lambda c: -c.get("screen_time_s", 0)):
        uri = _img_uri(root / c["thumb"], 96) if c.get("thumb") else None
        img = f'<img src="{uri}" alt="">' if uri else '<div class="noimg"></div>'
        cards.append(f'<div class="char">{img}<div><b>{_e(c.get("name") or c["id"])}</b><br>'
                     f'<span>{c.get("screen_time_s", 0):.0f} s on screen · {c.get("speaking_time_s", 0):.0f} s talking</span>'
                     f'</div></div>')
    return f'<h2>Characters</h2><div class="chars">{"".join(cards)}</div>'


def _shots(rep: dict, root: Path, thumb_width: int) -> str:
    rate = rate_of(rep)
    by_index = {s["index"]: s for s in rep.get("shots") or []}
    cells = []
    for row in cut_list(rep):
        s = by_index.get(row["index"], {})
        uri = _img_uri(root / row["thumb"], thumb_width) if row.get("thumb") else None
        img = f'<img src="{uri}" alt="shot {row["index"]}" loading="lazy">' if uri else '<div class="noimg"></div>'
        tags = [row["framing_name"] or row["framing"], row["camera"], row["scene"]]
        lk = s.get("look") or {}
        if lk:
            tags += [lk.get("key") if lk.get("key") != "normal" else None, lk.get("temperature")
                     if lk.get("temperature") != "neutral" else None]
        cp = s.get("composition") or {}
        if cp:
            tags += [cp.get("placement"), "shallow focus" if cp.get("shallow") else None]
        ed = s.get("editing") or {}
        if ed.get("recurs"):
            tags.append(f"set-up {ed['setup'] + 1}")
        tag = " · ".join(_e(t) for t in tags if t)
        who = ", ".join(row["characters"])
        cells.append(f'<figure>{img}<figcaption><b>#{row["index"]}</b> <span class="tc">'
                     f'{to_timecode(row["f0"], rate)}</span> <span class="d">{row["t1"] - row["t0"]:.1f} s</span>'
                     + (f'<br><span class="tg">{tag}</span>' if tag else "")
                     + (f'<br><span class="who">{_e(who)}</span>' if who else "") + "</figcaption></figure>")
    return f'<h2>Shots</h2><div class="grid">{"".join(cells)}</div>'


CSS = """
:root{--bg:#fbfbf8;--fg:#1c1c20;--dim:#6b6b73;--card:#fff;--line:#e2e2dc;--accent:#c43e2c;--lane:#ecece6}
@media (prefers-color-scheme:dark){:root{--bg:#121315;--fg:#e8e8e8;--dim:#9a9ea6;--card:#1b1d20;--line:#2c3036;
--accent:#e8a33d;--lane:#24272c}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
font:15px/1.45 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
main{max-width:1180px;margin:0 auto;padding:28px 16px 48px}
h1{font-size:30px;margin:0 0 2px;letter-spacing:-.01em}h2{font-size:18px;margin:34px 0 10px}
.sub{color:var(--dim);margin:0 0 18px}.desc{font-size:17px;margin:6px 0 18px}
.facts{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:8px}
.fact{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:8px 10px}
.fact b{display:block;font-size:16px}.fact span{color:var(--dim);font-size:12px}
.lane{margin:12px 0}.lane .lt{font-size:13px;font-weight:600;color:var(--dim);margin-bottom:4px}
.lane svg{width:100%;display:block;background:var(--lane);border-radius:4px}
.lane .ln{font-size:12px;color:var(--dim);margin-top:3px}
.area{fill:color-mix(in srgb,var(--accent) 40%,transparent)}.curve{fill:none;stroke:var(--accent);stroke-width:1.5;
vector-effect:non-scaling-stroke}.peak{fill:#e05545;opacity:.25}.silence{fill:#4f7fbf;opacity:.25}
.act1,.act3{fill:var(--fg);opacity:.04}.act2{fill:var(--fg);opacity:.09}.scene{stroke:var(--fg);stroke-width:1;opacity:.5;
vector-effect:non-scaling-stroke}.beat{stroke:#e05545;stroke-width:1.5;stroke-dasharray:3 2;vector-effect:non-scaling-stroke}
.chars{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:8px}
.char{display:flex;gap:10px;align-items:center;background:var(--card);border:1px solid var(--line);border-radius:8px;
padding:6px}.char img,.char .noimg{width:48px;height:48px;border-radius:6px;object-fit:cover;background:var(--lane)}
.char span{color:var(--dim);font-size:12px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(170px,1fr));gap:10px}
figure{margin:0;background:var(--card);border:1px solid var(--line);border-radius:8px;overflow:hidden}
figure img,figure .noimg{width:100%;display:block;aspect-ratio:16/9;object-fit:cover;background:var(--lane)}
figcaption{padding:6px 8px;font-size:12.5px}.tc{font-variant-numeric:tabular-nums;color:var(--dim)}
.d{float:right;color:var(--dim)}.tg{color:var(--accent)}.who{color:var(--dim)}
footer{color:var(--dim);font-size:12px;margin-top:36px;border-top:1px solid var(--line);padding-top:10px}
@media print{body{background:#fff;color:#000}figure,.fact{break-inside:avoid}}
"""


def breakdown_html(rep: dict, root: str | Path, thumb_width: int = 200) -> str:
    """The page for `rep` (a report.json dict); `root` is outputs/<clip>/, where its
    assets (shot thumbnails, character faces) live."""
    from .. import __version__

    root = Path(root)
    dur = float(rep["info"]["duration_s"]) or 1.0
    pc = rep.get("pacing") or pacing(rep)
    extra = rep.get("extra") or {}
    look = (extra.get("look") or {}).get("summary") or {}
    facts = "".join(f'<div class="fact"><b>{_e(v)}</b><span>{_e(k)}</span></div>' for k, v in _facts(rep, pc))
    comp = (extra.get("composition") or {}).get("summary") or {}
    desc = " ".join(d for d in (look.get("description"), comp.get("description")) if d)
    desc = f'<p class="desc">{_e(desc)}</p>' if desc else ""
    charts = _barcode(rep, dur) + _pace(pc, dur) + _look(rep, dur) + _loudness(extra.get("sound"), dur)
    charts += _story(extra.get("story"), dur)
    title = f"{rep['clip']}: CineScope breakdown"
    return (
        "<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
        f"<title>{_e(title)}</title><style>{CSS}</style></head><body><main>"
        f"<h1>{_e(rep['clip'])}</h1>"
        f"<p class=\"sub\">CineScope breakdown of {_e(rep.get('video', rep['clip']))} · "
        f"generated {_e(rep.get('generated', ''))}</p>"
        f"{desc}<div class=\"facts\">{facts}</div>"
        + (f"<h2>Over time</h2>{charts}" if charts else "")
        + _characters(rep, root) + (_shots(rep, root, thumb_width) if rep.get("shots") else "")
        + f"<footer>Made with CineScope {__version__} · every number computed from the pixels and the "
          "soundtrack · shot numbers and timecodes match the EDL, FCPXML and CSV exports</footer>"
        "</main></body></html>\n"
    )


def save_breakdown(rep: dict, root: str | Path, dest: str | Path) -> Path:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(breakdown_html(rep, root), encoding="utf-8", newline="\n")
    return dest
