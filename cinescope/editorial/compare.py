"""Compare two films: their style side by side, as numbers and as one HTML page.

Every metric comes from the two reports alone (nothing is re-analysed), so any
two clips CineScope has processed can be compared, whichever stages ran on each:
a metric only one of them has is shown with a dash for the other.

    python -m cinescope.editorial.compare outputs/a/report.json outputs/b/report.json [-o page.html] [--json]

Curves over time (pace, story intensity) are drawn against the share of the
running time, so a 90-minute film and a 2-minute trailer line up start to end.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .breakdown import CSS, W, _e
from .pacing import pacing


def _get(d: dict | None, *path, default=None):
    for p in path:
        if not isinstance(d, dict) or p not in d:
            return default
        d = d[p]
    return d


# (group, label, unit, getter); a getter returns a number, a string or None
def _metrics() -> list[tuple[str, str, str, callable]]:
    pc = lambda k: (lambda r, p: p and p.get(k))          # noqa: E731 - a table of small lambdas
    ex = lambda *path: (lambda r, p: _get(r.get("extra"), *path))  # noqa: E731
    return [
        ("Picture", "Length", "s", lambda r, p: _get(r, "info", "duration_s")),
        ("Picture", "Aspect ratio", "", ex("look", "summary", "aspect_name")),
        ("Editing", "Shots", "", pc("n_shots")),
        ("Editing", "Average shot", "s", pc("asl_s")),
        ("Editing", "Median shot", "s", pc("median_s")),
        ("Editing", "Cuts per minute", "", pc("cuts_per_min")),
        ("Editing", "Camera set-ups", "", ex("editing", "summary", "setups")),
        ("Editing", "Shot/reverse-shot runs", "", ex("editing", "summary", "exchanges")),
        ("Editing", "Montages", "", ex("editing", "summary", "montages")),
        ("Editing", "Long takes", "", ex("editing", "summary", "long_takes")),
        ("Look", "Lighting", "", ex("look", "summary", "key")),
        ("Look", "Colour", "", ex("look", "summary", "temperature")),
        ("Look", "Brightness", "", ex("look", "summary", "brightness")),
        ("Look", "Contrast", "", ex("look", "summary", "contrast")),
        ("Look", "Saturation", "", ex("look", "summary", "saturation")),
        ("Look", "Warmth", "", ex("look", "summary", "warmth")),
        ("Look", "Teal and orange", "", ex("look", "summary", "teal_orange")),
        ("Look", "Symmetry", "", ex("look", "summary", "symmetry")),
        ("Composition", "On the thirds", "%", lambda r, p: _pct(_get(r.get("extra"), "composition", "summary",
                                                                      "thirds_share"))),
        ("Composition", "Centred", "%", lambda r, p: _pct(_get(r.get("extra"), "composition", "summary", "share",
                                                                "centre"))),
        ("Composition", "Shallow focus", "%", lambda r, p: _pct(_get(r.get("extra"), "composition", "summary",
                                                                      "shallow_share"))),
        ("Sound", "Loudness", "LUFS", ex("sound", "summary", "integrated_lufs")),
        ("Sound", "Loudness range", "LU", ex("sound", "summary", "lra_lu")),
        ("Sound", "Speech", "%", lambda r, p: _pct(_get(r.get("extra"), "sound", "summary", "speech_share"))),
        ("Sound", "Onsets per minute", "", ex("sound", "summary", "onsets_per_min")),
        ("Sound", "Tempo", "BPM", ex("rhythm", "summary", "bpm")),
        ("Sound", "Music with a beat", "%", lambda r, p: _pct(_get(r.get("extra"), "rhythm", "summary",
                                                                    "rhythmic_share"))),
        ("Sound", "Cuts on the beat", "%", lambda r, p: _pct(_get(r.get("extra"), "rhythm", "summary",
                                                                   "cuts_on_beat", "on_beat"))),
        ("Story", "Scenes", "", ex("story", "summary", "scenes")),
        ("Story", "Average scene", "s", ex("story", "summary", "mean_scene_s")),
        ("Story", "Arc", "", ex("story", "summary", "arc")),
        ("People", "Characters", "", lambda r, p: len(_get(r, "characters", "list", default=[])) or None),
    ]


# counts that grow with the running time: shown, but left out of the similarity and
# the headline (a feature has more shots than a trailer whatever its style)
EXTENSIVE = {"Length", "Shots", "Camera set-ups", "Shot/reverse-shot runs", "Montages", "Long takes", "Scenes",
             "Average scene", "Characters"}


def _pct(v):
    return None if v is None else round(100 * float(v), 1)


def _num(v) -> float | None:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def profile(rep: dict) -> dict:
    """{label: value} for every metric the report has."""
    p = rep.get("pacing") or pacing(rep)
    out = {}
    for _, label, _, get in _metrics():
        try:
            v = get(rep, p)
        except (KeyError, TypeError, ValueError):
            v = None
        if v is not None:
            out[label] = v
    return out


def _resample(t: list[float], v: list[float], dur: float, n: int = 100) -> list[float] | None:
    if not t or not v or dur <= 0:
        return None
    x = np.linspace(0, 1, n)
    return np.round(np.interp(x, np.asarray(t, float) / dur, np.asarray(v, float)), 3).tolist()


def curves(rep: dict, n: int = 100) -> dict:
    """Pace and story intensity against the share of the running time (0..1, n points)."""
    dur = float(_get(rep, "info", "duration_s", default=0.0) or 0.0)
    out = {}
    pc = rep.get("pacing") or pacing(rep)
    if pc:
        out["pace"] = _resample(pc["curve"]["t"], pc["curve"]["cuts_per_min"], dur, n)
    iv = _get(rep.get("extra"), "story", "intensity")
    if iv and iv.get("value"):
        t = [iv["t0"] + i * iv["dt"] for i in range(len(iv["value"]))]
        out["intensity"] = _resample(t, iv["value"], dur, n)
    return {k: v for k, v in out.items() if v}


def compare(a: dict, b: dict) -> dict:
    """Both profiles row by row, the largest differences in words, and a 0..1 style
    similarity: the mean over shared metrics (not the EXTENSIVE counts) of
    1 - |a - b| / max(|a|, |b|) for numbers, and 1 or 0 for labels."""
    pa, pb = profile(a), profile(b)
    rows, sims, diffs = [], [], []
    for group, label, unit, _ in _metrics():
        va, vb = pa.get(label), pb.get(label)
        if va is None and vb is None:
            continue
        na, nb = _num(va), _num(vb)
        row = {"group": group, "metric": label, "unit": unit, "a": va, "b": vb, "delta": None, "ratio": None}
        if na is not None and nb is not None:
            row["delta"] = round(nb - na, 3)
            top = max(abs(na), abs(nb))
            if top > 1e-9 and label not in EXTENSIVE:
                rel = min(1.0, abs(nb - na) / top)
                sims.append(1.0 - rel)
                diffs.append((rel, label, na, nb, unit))
            if min(abs(na), abs(nb)) > 1e-9 and na * nb > 0:
                row["ratio"] = round(nb / na, 3)
        elif va is not None and vb is not None and label not in EXTENSIVE:
            sims.append(1.0 if va == vb else 0.0)
            if va != vb:
                diffs.append((0.5, label, va, vb, unit))
        rows.append(row)
    diffs.sort(key=lambda d: -d[0])
    na, nb = a.get("clip", "A"), b.get("clip", "B")
    headline = []
    for _, label, va, vb, unit in diffs[:3]:
        if _num(va) is not None and _num(vb) is not None:
            more = nb if vb > va else na
            headline.append(f"{label}: {_fmt(va, unit)} vs {_fmt(vb, unit)} (higher in {more})")
        else:
            headline.append(f"{label}: {va} vs {vb}")
    return {"a": na, "b": nb, "rows": rows, "similarity": round(float(np.mean(sims)), 3) if sims else None,
            "headline": headline, "curves": {"a": curves(a), "b": curves(b)}}


def _fmt(v, unit: str = "") -> str:
    if v is None:
        return "–"
    if isinstance(v, float):
        s = f"{v:.3f}".rstrip("0").rstrip(".") if abs(v) < 10 else f"{v:.1f}".rstrip("0").rstrip(".")
    else:
        s = str(v)
    return f"{s} {unit}".strip() if unit and unit != "%" else f"{s}%" if unit == "%" else s


def _overlay(title: str, ca: list[float] | None, cb: list[float] | None, note: str) -> str:
    if not ca and not cb:
        return ""
    top = max(max(ca or [0]), max(cb or [0]), 1e-9)
    body = ""
    for c, cls in ((ca, "ca"), (cb, "cb")):
        if c:
            pts = " ".join(f"{W * i / (len(c) - 1):.1f},{58 - 54 * v / top:.1f}" for i, v in enumerate(c))
            body += f'<polyline points="{pts}" class="{cls}"/>'
    return (f'<div class="lane"><div class="lt">{_e(title)}</div><svg viewBox="0 0 {W} 60" '
            f'preserveAspectRatio="none" style="height:60px" role="img" aria-label="{_e(title)}">{body}</svg>'
            f'<div class="ln">{_e(note)}</div></div>')


def _barcode_row(rep: dict) -> str:
    cols = _get(rep, "barcode", "colors") or []
    if not cols:
        return '<div class="noimg" style="height:28px"></div>'
    w = W / len(cols)
    body = "".join(f'<rect x="{i * w:.2f}" y="0" width="{w + 0.6:.2f}" height="28" fill="rgb({c[0]},{c[1]},{c[2]})"/>'
                   for i, c in enumerate(cols))
    return f'<svg viewBox="0 0 {W} 28" preserveAspectRatio="none" style="height:28px;width:100%">{body}</svg>'


EXTRA_CSS = """
table.cmp{border-collapse:collapse;width:100%;margin:8px 0 18px;font-size:14px}
table.cmp th,table.cmp td{padding:5px 8px;border-bottom:1px solid var(--line);text-align:left}
table.cmp td.n,table.cmp th.n{text-align:right;font-variant-numeric:tabular-nums}table.cmp tr.g td{font-weight:600;
color:var(--dim);padding-top:14px;border-bottom:none}.ca{fill:none;stroke:var(--accent);stroke-width:1.8;
vector-effect:non-scaling-stroke}.cb{fill:none;stroke:#3d7bd9;stroke-width:1.8;stroke-dasharray:5 3;
vector-effect:non-scaling-stroke}.key{display:inline-block;width:18px;height:3px;vertical-align:middle;margin:0 6px}
.bc{margin:6px 0 14px}.bc b{display:block;font-size:13px;color:var(--dim);margin-bottom:3px}
"""


def compare_html(a: dict, b: dict) -> str:
    """One self-contained page comparing two reports."""
    from .. import __version__

    c = compare(a, b)
    rows, group = [], None
    for r in c["rows"]:
        if r["group"] != group:
            group = r["group"]
            rows.append(f'<tr class="g"><td colspan="4">{_e(group)}</td></tr>')
        d = "" if r["delta"] is None else ("+" if r["delta"] > 0 else "") + _fmt(r["delta"], r["unit"])
        rows.append(f'<tr><td>{_e(r["metric"])}</td><td class="n">{_e(_fmt(r["a"], r["unit"]))}</td>'
                    f'<td class="n">{_e(_fmt(r["b"], r["unit"]))}</td><td class="n">{_e(d)}</td></tr>')
    ca, cb = c["curves"]["a"], c["curves"]["b"]
    legend = (f'<p class="sub"><span class="key" style="background:var(--accent)"></span>{_e(c["a"])}'
              f'<span class="key" style="background:#3d7bd9"></span>{_e(c["b"])} · curves run start to end of each '
              "clip, whatever its length</p>")
    charts = (_overlay("Pace", ca.get("pace"), cb.get("pace"), "cuts per minute against the share of running time")
              + _overlay("Story intensity", ca.get("intensity"), cb.get("intensity"),
                         "cutting, loudness and motion combined, 0..1"))
    sim = "" if c["similarity"] is None else f'<div class="fact"><b>{round(100 * c["similarity"])}%</b>' \
                                             "<span>Style similarity</span></div>"
    head = "".join(f"<li>{_e(h)}</li>" for h in c["headline"])
    title = f"{c['a']} vs {c['b']}: CineScope comparison"
    return (
        "<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
        f"<title>{_e(title)}</title><style>{CSS}{EXTRA_CSS}</style></head><body><main>"
        f"<h1>{_e(c['a'])} <span style=\"color:var(--dim)\">vs</span> {_e(c['b'])}</h1>"
        f"<p class=\"sub\">CineScope comparison · two reports, side by side</p>"
        f"<div class=\"facts\">{sim}</div>"
        + (f"<h2>Biggest differences</h2><ul>{head}</ul>" if head else "")
        + f'<h2>Colour over time</h2><div class="bc"><b>{_e(c["a"])}</b>{_barcode_row(a)}</div>'
          f'<div class="bc"><b>{_e(c["b"])}</b>{_barcode_row(b)}</div>'
        + (f"<h2>Over time</h2>{legend}{charts}" if charts else "")
        + f'<h2>Side by side</h2><table class="cmp"><thead><tr><th></th><th class="n">{_e(c["a"])}</th>'
          f'<th class="n">{_e(c["b"])}</th><th class="n">difference</th></tr></thead><tbody>{"".join(rows)}'
          "</tbody></table>"
        + f"<footer>Made with CineScope {__version__} · every number computed from the pixels and the "
          "soundtrack of each clip</footer></main></body></html>\n"
    )


def _load(path: str) -> dict:
    p = Path(path)
    if p.is_dir():
        p = p / "report.json"
    if not p.is_file():
        raise FileNotFoundError(f"no report at {p}; run python -m cinescope.pipeline <video> first")
    return json.loads(p.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m cinescope.editorial.compare",
                                 description="Compare the style of two clips CineScope has analysed.")
    ap.add_argument("a", help="outputs/<clip>/report.json (or its folder)")
    ap.add_argument("b", help="the other report")
    ap.add_argument("-o", "--out", help="write the HTML comparison here")
    ap.add_argument("--json", action="store_true", help="print the comparison as JSON")
    a = ap.parse_args(argv)
    try:
        ra, rb = _load(a.a), _load(a.b)
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        return 2
    c = compare(ra, rb)
    if a.json:
        print(json.dumps(c, indent=1))
    else:
        if c["similarity"] is not None:
            print(f"{c['a']} vs {c['b']}: style similarity {round(100 * c['similarity'])}%")
        for h in c["headline"]:
            print(f"  {h}")
        for r in c["rows"]:
            print(f"  {r['group']:<12} {r['metric']:<24} {_fmt(r['a'], r['unit']):>14} {_fmt(r['b'], r['unit']):>14}")
    if a.out:
        dest = Path(a.out)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(compare_html(ra, rb), encoding="utf-8", newline="\n")
        print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
