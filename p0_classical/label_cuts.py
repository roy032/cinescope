"""Build a click-to-label page for shot boundaries.

    python p0_classical/label_cuts.py trailers/Ted_Lasso_900s.mkv

Writes outputs/label_<clip>/index.html. Open it in a browser: every frame of
the clip is shown, 12 frames (half a second at 24 fps) per row. Click the FIRST frame of each new shot.

  click                 mark / unmark a hard cut before this frame
  shift+click twice     mark a gradual transition (first and last blended frame)
  right-click           remove the gradual transition under the cursor
  "Download labels"     saves <clip>.labels.json — move it to p0_classical/labels/

Nothing from the detector is shown on the page: suggestions would bias the
labels towards what the detector already finds, and the test set would then
measure agreement with ourselves instead of accuracy. Progress is kept in the
browser's local storage, so you can close the tab and continue later.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cinescope.io.video import VideoReader  # noqa: E402

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Label cuts — {name}</title>
<style>
 body {{ font: 14px system-ui, sans-serif; background:#111; color:#ddd; margin:0; }}
 header {{ position:sticky; top:0; background:#1b1b1b; padding:10px 16px; z-index:2;
          border-bottom:1px solid #333; display:flex; gap:16px; align-items:center; flex-wrap:wrap; }}
 button {{ background:#2d6a4f; color:#fff; border:0; padding:7px 12px; border-radius:6px; cursor:pointer; }}
 .row {{ position:relative; margin:2px 16px 2px 72px; width:{row_w}px; height:{th}px; }}
 .row img {{ display:block; width:{row_w}px; height:{th}px; cursor:crosshair; }}
 .t {{ position:absolute; left:-62px; top:{ty}px; color:#888; font-size:12px; width:56px; text-align:right; }}
 .mark {{ position:absolute; top:0; width:3px; height:{th}px; background:#ff3b30; pointer-events:none; }}
 .grad {{ position:absolute; top:0; height:{th}px; background:rgba(255,196,0,.35);
          border-left:3px solid #ffc400; border-right:3px solid #ffc400; pointer-events:none; }}
 .pending {{ position:absolute; top:0; width:3px; height:{th}px; background:#ffc400; pointer-events:none; }}
 kbd {{ background:#333; padding:1px 5px; border-radius:3px; }}
</style></head><body>
<header>
 <b>{name}</b> · {n} frames @ {fps:.3f} fps ·
 <span>click = cut before this frame · <kbd>shift</kbd>+click start, then end = gradual</span>
 <span id="count"></span>
 <button id="dl">Download labels</button>
 <button id="clear" style="background:#6a2d2d">Clear all</button>
</header>
<div id="rows"></div>
<script>
const META = {meta};
const KEY = "cinescope-labels-" + META.video;
let st = JSON.parse(localStorage.getItem(KEY) || '{{"cuts":[],"gradual":[],"pending":null}}');
const rowsEl = document.getElementById("rows");
for (let r = 0; r < META.n_rows; r++) {{
  const row = document.createElement("div"); row.className = "row"; row.dataset.r = r;
  const t = r * META.per_row / META.fps;
  row.innerHTML = `<span class="t">${{Math.floor(t/60)}}:${{(t%60).toFixed(1).padStart(4,"0")}}</span>` +
                  `<img loading="lazy" src="rows/${{String(r).padStart(4,"0")}}.jpg">`;
  row.querySelector("img").addEventListener("click", ev => {{
    const col = Math.floor(ev.offsetX / META.tw);
    const f = r * META.per_row + col;
    if (f >= META.n_frames) return;
    if (ev.shiftKey) {{
      if (st.pending === null) st.pending = f;
      else {{ const a = Math.min(st.pending, f), b = Math.max(st.pending, f);
             st.gradual.push([a, b]); st.pending = null; }}
    }} else {{
      const i = st.cuts.indexOf(f);
      if (i >= 0) st.cuts.splice(i, 1); else st.cuts.push(f);
    }}
    save(); draw();
  }});
  row.addEventListener("contextmenu", ev => {{        // right-click removes a gradual span
    ev.preventDefault();
    const f = r * META.per_row + Math.floor(ev.offsetX / META.tw);
    st.gradual = st.gradual.filter(([a, b]) => f < a || f > b); save(); draw();
  }});
  rowsEl.appendChild(row);
}}
function place(el, f) {{ return [Math.floor(f / META.per_row), (f % META.per_row) * META.tw]; }}
function draw() {{
  document.querySelectorAll(".mark,.grad,.pending").forEach(e => e.remove());
  const rows = rowsEl.children;
  for (const f of st.cuts) {{ const [r, x] = place(null, f); const m = document.createElement("div");
    m.className = "mark"; m.style.left = (x - 1) + "px"; rows[r].appendChild(m); }}
  for (const [a, b] of st.gradual) {{
    for (let r = Math.floor(a / META.per_row); r <= Math.floor(b / META.per_row); r++) {{
      const lo = Math.max(a, r * META.per_row), hi = Math.min(b, (r + 1) * META.per_row - 1);
      const g = document.createElement("div"); g.className = "grad";
      g.style.left = ((lo % META.per_row) * META.tw) + "px";
      g.style.width = ((hi - lo + 1) * META.tw - 6) + "px"; rows[r].appendChild(g); }}
  }}
  if (st.pending !== null) {{ const [r, x] = place(null, st.pending); const p = document.createElement("div");
    p.className = "pending"; p.style.left = x + "px"; rows[r].appendChild(p); }}
  document.getElementById("count").textContent =
    `${{st.cuts.length}} cuts · ${{st.gradual.length}} gradual`;
}}
function save() {{ localStorage.setItem(KEY, JSON.stringify(st)); }}
document.getElementById("dl").onclick = () => {{
  const transitions = st.cuts.map(f => ({{kind: "cut", frame: f, t: +(META.times[f] ?? f / META.fps).toFixed(3)}}))
    .concat(st.gradual.map(([a, b]) => ({{kind: "dissolve", start: a, end: b,
      t: +(META.times[Math.floor((a + b) / 2)] ?? a / META.fps).toFixed(3)}})))
    .sort((x, y) => (x.frame ?? x.start) - (y.frame ?? y.start));
  const out = {{video: META.video, fps: META.fps, n_frames: META.n_frames,
               labeller: "human", transitions}};
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([JSON.stringify(out, null, 1)], {{type: "application/json"}}));
  a.download = META.stem + ".labels.json"; a.click();
}};
document.getElementById("clear").onclick = () => {{
  if (confirm("Remove every label for this clip?")) {{ st = {{cuts: [], gradual: [], pending: null}}; save(); draw(); }}
}};
draw();
</script></body></html>
"""


def main(argv: list[str] | None = None) -> Path:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--thumb-width", type=int, default=96)
    ap.add_argument("--per-row", type=int, default=12,
                    help="frames per row (12 = half a second at 24 fps; fits a laptop screen)")
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args(argv)

    video = Path(args.video)
    out = Path(args.out) / f"label_{video.stem}"
    (out / "rows").mkdir(parents=True, exist_ok=True)
    times: list[float] = []
    with VideoReader(video, width=args.thumb_width) as vr:
        fps = vr.info.fps
        per_row = max(1, args.per_row)
        row: list[np.ndarray] = []
        n_rows = 0
        th = tw = 0
        for f in vr.frames():
            th, tw = f.rgb.shape[:2]
            row.append(f.rgb)
            times.append(round(f.t, 3))
            if len(row) == per_row:
                Image.fromarray(np.concatenate(row, 1)).save(out / "rows" / f"{n_rows:04d}.jpg", quality=80)
                n_rows += 1
                row = []
        if row:
            pad = [np.zeros_like(row[0])] * (per_row - len(row))
            Image.fromarray(np.concatenate(row + pad, 1)).save(out / "rows" / f"{n_rows:04d}.jpg", quality=80)
            n_rows += 1
    meta = {"video": video.name, "stem": video.stem, "fps": fps, "n_frames": len(times),
            "per_row": per_row, "n_rows": n_rows, "tw": tw, "th": th, "times": times}
    html = PAGE.format(name=video.name, n=len(times), fps=fps, row_w=per_row * tw, th=th,
                       ty=max(0, th // 2 - 8), meta=json.dumps(meta))
    (out / "index.html").write_text(html, encoding="utf-8")
    print(f"{len(times)} frames in {n_rows} rows -> open {out / 'index.html'}")
    return out / "index.html"


if __name__ == "__main__":
    main()
