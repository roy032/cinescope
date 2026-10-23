"""Many clips at once: finding videos, the cache on disk, and a library page.

    videos(["trailers/", "extra.mkv"])     the video files named or inside folders, sorted
    scan("outputs")                         one entry per analysed clip: facts + disk use
    library_html(scan("outputs"))           outputs/library.html: every clip on one page
    clear("outputs", "clip", ["faces"])     delete a clip's cached results (all, or some stages)

The library page is self-contained like the breakdown (cinescope.editorial.breakdown):
each clip is a card with its movie barcode and its key facts (length, shots, pace,
look, loudness, tempo, story arc, characters), a table under it sorts by any
column, and each card links to that clip's own breakdown page.
"""
from __future__ import annotations

import html
import json
import shutil
import time
from pathlib import Path

VIDEO_EXT = (".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v")


def videos(paths: list[str | Path], recursive: bool = True) -> list[Path]:
    """The video files among `paths`: files as given, folders searched for VIDEO_EXT.
    Duplicates removed, order kept (folders sorted inside)."""
    out: list[Path] = []
    seen: set[Path] = set()
    for p in map(Path, paths):
        if p.is_dir():
            it = p.rglob("*") if recursive else p.iterdir()
            found = sorted(q for q in it if q.is_file() and q.suffix.lower() in VIDEO_EXT)
        elif p.exists():
            found = [p]
        else:
            raise FileNotFoundError(p)
        for q in found:
            r = q.resolve()
            if r not in seen:
                seen.add(r)
                out.append(q)
    return out


def clip_names(files: list[Path], taken: set[str] | None = None) -> list[str]:
    """One output name per file: its stem, or, when an earlier file already has that
    stem (Season 1/ep01.mp4, Season 2/ep01.mp4), the stem plus its folder's name, so
    no clip's results overwrite another's. The first file keeps the plain stem, as it
    always did, so its cached results stay where they are. Numbered if still taken."""
    used = set(taken or ())
    names: list[str] = []
    for f in files:
        n = f.stem
        if n in used:
            n = base = f"{f.stem}-{f.parent.name or 'root'}"
            k = 2
            while n in used:
                n, k = f"{base}-{k}", k + 1
        used.add(n)
        names.append(n)
    return names


def _size(p: Path) -> int:
    if p.is_file():
        return p.stat().st_size
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def disk_use(clip_dir: Path) -> dict:
    """Bytes per stage (stages/<name>), plus assets, exports and the rest."""
    stages = {}
    sd = clip_dir / "stages"
    if sd.is_dir():
        for d in sorted(sd.iterdir()):
            if d.is_dir():
                stages[d.name] = _size(d)
    other = {n: _size(clip_dir / n) for n in ("assets", "exports") if (clip_dir / n).exists()}
    total = _size(clip_dir)
    return {"total": total, "stages": stages, **other}


def _get(d: dict | None, *keys, default=None):
    for k in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(k)
    return default if d is None else d


def facts(rep: dict) -> dict:
    """The comparable numbers of one report (None where a stage did not run)."""
    ex = rep.get("extra") or {}
    pc = rep.get("pacing") or {}
    return {
        "duration_s": _get(rep, "info", "duration_s"),
        "resolution": f"{_get(rep, 'info', 'width')}×{_get(rep, 'info', 'height')}",
        "shots": len(rep.get("shots") or []) or None,
        "asl_s": pc.get("asl_s"),
        "cuts_per_min": pc.get("cuts_per_min"),
        "aspect": _get(ex, "look", "summary", "aspect_name"),
        "key": _get(ex, "look", "summary", "key"),
        "temperature": _get(ex, "look", "summary", "temperature"),
        "lufs": _get(ex, "sound", "summary", "integrated_lufs"),
        "bpm": _get(ex, "rhythm", "summary", "bpm"),
        "scenes": _get(ex, "story", "summary", "scenes"),
        "arc": _get(ex, "story", "summary", "arc"),
        "characters": len(_get(rep, "characters", "list", default=[])) or None,
        "highlights_s": _get(ex, "highlights", "summary", "duration_s"),
    }


def scan(out: str | Path) -> list[dict]:
    """Every outputs/<clip>/ with a report.json, newest first."""
    out = Path(out)
    entries = []
    if not out.is_dir():
        return entries
    for rp in out.glob("*/report.json"):
        try:
            rep = json.loads(rp.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        d = rp.parent
        vp = rep.get("video_path")
        entries.append({
            "clip": rep.get("clip", d.name), "dir": d, "video": rep.get("video"),
            "video_found": bool(vp) and Path(vp).exists(), "generated": rep.get("generated", ""),
            "stages": rep.get("stages", []), "errors": sorted(rep.get("errors") or {}),
            "facts": facts(rep), "disk": disk_use(d), "report": rep,
        })
    return sorted(entries, key=lambda e: e["generated"], reverse=True)


def clear(out: str | Path, clip: str, stages: list[str] | None = None) -> list[Path]:
    """Delete a clip's cached results: everything under outputs/<clip>/ (stages None)
    or only stages/<name>/ for the given stages, plus report.json (it describes
    them). The next run recomputes what is missing. Returns what was removed."""
    d = Path(out) / clip
    if not d.is_dir() or not (d / "report.json").exists():
        raise FileNotFoundError(f"no analysed clip {clip!r} in {out}")
    removed = []
    if not stages:
        shutil.rmtree(d)
        return [d]
    for n in stages:
        sd = d / "stages" / n
        if sd.is_dir():
            shutil.rmtree(sd)
            removed.append(sd)
    if removed:
        (d / "report.json").unlink()
        removed.append(d / "report.json")
    return removed


def human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


def cache_table(entries: list[dict]) -> str:
    """Plain-text disk use: one line per clip, its biggest stages after it."""
    if not entries:
        return "no analysed clips"
    lines = []
    for e in sorted(entries, key=lambda e: -e["disk"]["total"]):
        dk = e["disk"]
        flag = "" if e["video_found"] else "  (video not found)"
        lines.append(f"{e['clip']:<32} {human_bytes(dk['total']):>9}{flag}")
        big = sorted(dk["stages"].items(), key=lambda kv: -kv[1])[:4]
        if big:
            lines.append("    " + ", ".join(f"{n} {human_bytes(b)}" for n, b in big if b))
    lines.append(f"{'total':<32} {human_bytes(sum(e['disk']['total'] for e in entries)):>9}")
    return "\n".join(lines)


# ------------------------------------------------------------------------------ the page
def _e(v) -> str:
    return html.escape("" if v is None else str(v), quote=True)


def _clock(t) -> str:
    if t is None:
        return "–"
    t = float(t)
    h, m, s = int(t // 3600), int(t % 3600 // 60), int(round(t % 60))
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


COLUMNS = [  # key, header, format
    ("duration_s", "Length", _clock),
    ("shots", "Shots", lambda v: v),
    ("asl_s", "Avg shot", lambda v: f"{v:.1f} s"),
    ("cuts_per_min", "Cuts/min", lambda v: f"{v:.1f}"),
    ("aspect", "Aspect", lambda v: v),
    ("key", "Lighting", lambda v: v),
    ("temperature", "Colour", lambda v: v),
    ("lufs", "Loudness", lambda v: f"{v} LUFS"),
    ("bpm", "Tempo", lambda v: f"{v:.0f} BPM"),
    ("scenes", "Scenes", lambda v: v),
    ("arc", "Arc", lambda v: v),
    ("characters", "Characters", lambda v: v),
    ("highlights_s", "Highlights", _clock),
]


def _cell(v, fmt) -> str:
    return "–" if v is None else _e(fmt(v))


CSS = """
:root{color-scheme:light dark;--bg:#fff;--fg:#1d1d1f;--dim:#6b6b70;--card:#f6f6f7;--line:#e2e2e5;--accent:#c4661f}
@media (prefers-color-scheme:dark){:root{--bg:#141416;--fg:#ececee;--dim:#9a9aa1;--card:#1d1d20;--line:#2c2c31;--accent:#f0a35e}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.45 system-ui,sans-serif}
main{max-width:1200px;margin:0 auto;padding:24px 16px 48px}h1{margin:0 0 4px;font-size:26px}
.sub{color:var(--dim);margin:0 0 20px}
.cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(270px,1fr));gap:12px;margin-bottom:28px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;overflow:hidden;color:inherit;
text-decoration:none;display:flex;flex-direction:column}.card:hover{border-color:var(--accent)}
.card .bc{height:46px;width:100%;object-fit:fill;display:block;background:var(--line)}
.card .body{padding:8px 10px}.card h2{font-size:16px;margin:0 0 4px;overflow-wrap:anywhere}
.card dl{display:grid;grid-template-columns:auto 1fr;gap:1px 10px;margin:0;font-size:13px}
.card dt{color:var(--dim)}.card dd{margin:0}.err{color:#d0443a;font-size:12.5px;margin-top:4px}
.table{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:13.5px}
th,td{padding:5px 8px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}
th{cursor:pointer;user-select:none;color:var(--dim);font-weight:600}th[aria-sort]{color:var(--fg)}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}a{color:var(--accent)}
footer{color:var(--dim);font-size:12px;margin-top:30px}
"""

SORT_JS = """
document.querySelectorAll('th').forEach((th,i)=>th.addEventListener('click',()=>{
const tb=th.closest('table').tBodies[0],asc=th.getAttribute('aria-sort')!=='ascending';
th.closest('tr').querySelectorAll('th').forEach(h=>h.removeAttribute('aria-sort'));
th.setAttribute('aria-sort',asc?'ascending':'descending');
const key=r=>{const c=r.cells[i],v=c.dataset.v??c.textContent;const n=parseFloat(v);return isNaN(n)?v.toLowerCase():n};
[...tb.rows].sort((a,b)=>{const x=key(a),y=key(b);return (x>y?1:x<y?-1:0)*(asc?1:-1)}).forEach(r=>tb.appendChild(r));
}));
"""


def library_html(entries: list[dict], link=None) -> str:
    """The library page. link(clip_dir_name) -> the href of a clip's breakdown; default:
    relative to outputs/ (where library.html is written)."""
    from ..editorial.breakdown import _img_uri

    cards, rows = [], []
    for e in entries:
        f, d, rep = e["facts"], Path(e["dir"]), e["report"]
        href = _e(link(d.name) if link else f"{d.name}/exports/{d.name}_breakdown.html")
        bc = (rep.get("barcode") or {}).get("image")
        uri = _img_uri(d / bc, 400) if bc else None
        img = f'<img class="bc" src="{uri}" alt="">' if uri else '<div class="bc"></div>'
        dl = "".join(f"<dt>{_e(h)}</dt><dd>{_cell(f.get(k), fmt)}</dd>" for k, h, fmt in COLUMNS[:8]
                     if f.get(k) is not None)
        err = f'<div class="err">failed: {_e(", ".join(e["errors"]))}</div>' if e["errors"] else ""
        cards.append(f'<a class="card" href="{href}">{img}<div class="body"><h2>{_e(e["clip"])}</h2>'
                     f"<dl>{dl}</dl>{err}</div></a>")
        cells = "".join(
            f'<td class="{"num" if isinstance(f.get(k), (int, float)) else ""}" '
            f'data-v="{_e(f.get(k) if f.get(k) is not None else "")}">{_cell(f.get(k), fmt)}</td>'
            for k, _, fmt in COLUMNS)
        rows.append(f'<tr><td><a href="{href}">{_e(e["clip"])}</a></td>{cells}'
                    f'<td class="num" data-v="{e["disk"]["total"]}">{human_bytes(e["disk"]["total"])}</td></tr>')
    head = "<th>Clip</th>" + "".join(f"<th>{_e(h)}</th>" for _, h, _ in COLUMNS) + '<th class="num">On disk</th>'
    total = sum(float(e["facts"].get("duration_s") or 0) for e in entries)
    return (
        "<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
        f"<title>CineScope library</title><style>{CSS}</style></head><body><main>"
        "<h1>CineScope library</h1>"
        f'<p class="sub">{len(entries)} analysed clip{"s" if len(entries) != 1 else ""}, '
        f"{_clock(total)} of video · generated {time.strftime('%Y-%m-%d %H:%M')} · "
        "click a clip for its breakdown, a column to sort</p>"
        + (f'<div class="cards">{"".join(cards)}</div>'
           f'<div class="table"><table><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'
           if entries else "<p>No analysed clips yet: python -m cinescope.pipeline &lt;video or folder&gt;</p>")
        + "<footer>Made with CineScope · every number computed from the pixels and the soundtrack</footer>"
        f"</main><script>{SORT_JS}</script></body></html>\n"
    )


def save_library(out: str | Path, breakdowns: bool = True) -> Path:
    """Write outputs/library.html (and each clip's breakdown page when missing or older
    than its report, so the cards' links work)."""
    out = Path(out)
    entries = scan(out)
    if breakdowns:
        from ..editorial.breakdown import save_breakdown

        for e in entries:
            d = Path(e["dir"])
            dest = d / "exports" / f"{d.name}_breakdown.html"
            stale = not dest.exists() or dest.stat().st_mtime < (d / "report.json").stat().st_mtime
            if stale and e["report"].get("shots"):
                save_breakdown(e["report"], d, dest)
    dest = out / "library.html"
    dest.write_text(library_html(entries), encoding="utf-8")
    return dest
