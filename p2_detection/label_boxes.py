"""Build a draw-boxes labelling page for frames sampled from a clip.

    python p2_detection/label_boxes.py trailers/Ted_Lasso_900s.mkv --start 120 --seconds 60 --fps 0.5 --tag dark

Writes outputs/boxes_<clip>_<tag>/ with frames/*.jpg and index.html. Open the
page, draw a box around every face (and every prop you want to evaluate),
then "Download labels" and move the JSON into p2_detection/labels/.

  drag                  draw a box of the current class
  1-9                   choose the class (the list is shown at the top)
  click a box           select it;  Delete / Backspace removes it
  n / p  (or → / ←)     next / previous frame
  i                     toggle "ignore" on the selected box — faces too small,
                        too blurred or too occluded to judge. Ignored boxes are
                        like COCO crowd regions: detecting them neither helps
                        nor hurts, missing them costs nothing.

As with the cut labels, no detector output is shown, so the labels are
independent of both detectors being compared. Work is saved in the browser's
local storage as you go.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cinescope.io.video import VideoReader  # noqa: E402

PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Label boxes — __NAME__</title>
<style>
 body { font: 14px system-ui, sans-serif; background:#111; color:#ddd; margin:0; }
 header { background:#1b1b1b; padding:10px 16px; border-bottom:1px solid #333;
          display:flex; gap:14px; align-items:center; flex-wrap:wrap; }
 button { background:#2d6a4f; color:#fff; border:0; padding:7px 12px; border-radius:6px; cursor:pointer; }
 #wrap { padding:12px 16px; }
 canvas { max-width:100%; cursor:crosshair; background:#000; }
 .cls { padding:2px 8px; border-radius:4px; border:1px solid #444; }
 .cls.on { outline:2px solid #fff; }
 kbd { background:#333; padding:1px 5px; border-radius:3px; }
</style></head><body>
<header>
 <b>__NAME__</b> <span id="pos"></span>
 <span id="classes"></span>
 <span><kbd>n</kbd>/<kbd>p</kbd> frame · <kbd>1-9</kbd> class · <kbd>Del</kbd> remove · <kbd>i</kbd> ignore</span>
 <span id="count"></span>
 <button id="dl">Download labels</button>
</header>
<div id="wrap"><canvas id="c"></canvas></div>
<script>
const META = __META__;
const KEY = "cinescope-boxes-" + META.name;
const COLORS = ["#ff3b30","#34c759","#0a84ff","#ffd60a","#bf5af2","#ff9f0a","#64d2ff","#ff375f","#30d158"];
let st;
try { st = JSON.parse(localStorage.getItem(KEY)) || null; } catch (e) { st = null; }
if (!st || !st.boxes) st = {boxes: META.frames.map(() => [])};
let cur = 0, cls = 0, sel = -1, drag = null;
const cv = document.getElementById("c"), ctx = cv.getContext("2d");
const img = new Image();
function save() { try { localStorage.setItem(KEY, JSON.stringify(st)); } catch (e) {} }
function classesBar() {
  document.getElementById("classes").innerHTML = META.classes.map((c, i) =>
    `<span class="cls ${i === cls ? "on" : ""}" style="color:${COLORS[i % 9]}">${i + 1} ${c}</span>`).join(" ");
}
function load(i) {
  cur = Math.max(0, Math.min(META.frames.length - 1, i)); sel = -1;
  img.onload = () => { cv.width = img.naturalWidth; cv.height = img.naturalHeight; draw(); };
  img.src = "frames/" + META.frames[cur].file;
  document.getElementById("pos").textContent =
    `frame ${cur + 1}/${META.frames.length} · t=${META.frames[cur].t.toFixed(1)}s`;
}
function draw() {
  ctx.drawImage(img, 0, 0);
  const lw = Math.max(2, cv.width / 500);
  st.boxes[cur].forEach((b, k) => {
    const col = COLORS[b[4] % 9];
    ctx.lineWidth = k === sel ? lw * 2 : lw;
    ctx.setLineDash(b[5] ? [lw * 4, lw * 3] : []);
    ctx.strokeStyle = col; ctx.strokeRect(b[0], b[1], b[2], b[3]);
  });
  ctx.setLineDash([]);
  if (drag && drag.w) { ctx.strokeStyle = "#fff"; ctx.lineWidth = lw; ctx.strokeRect(drag.x, drag.y, drag.w, drag.h); }
  const total = st.boxes.reduce((a, b) => a + b.length, 0);
  const done = st.boxes.filter(b => b.length).length;
  document.getElementById("count").textContent = `${total} boxes · ${done} frames with boxes`;
}
function pt(e) {
  const r = cv.getBoundingClientRect();
  return [(e.clientX - r.left) * cv.width / r.width, (e.clientY - r.top) * cv.height / r.height];
}
cv.addEventListener("mousedown", e => { const [x, y] = pt(e); drag = {x0: x, y0: y, x, y, w: 0, h: 0}; });
cv.addEventListener("mousemove", e => {
  if (!drag) return;
  const [x, y] = pt(e);
  drag.x = Math.min(drag.x0, x); drag.y = Math.min(drag.y0, y);
  drag.w = Math.abs(x - drag.x0); drag.h = Math.abs(y - drag.y0); draw();
});
cv.addEventListener("mouseup", e => {
  if (!drag) return;
  const minSide = Math.max(3, cv.width / 400);
  if (drag.w >= minSide && drag.h >= minSide) {
    st.boxes[cur].push([Math.round(drag.x), Math.round(drag.y), Math.round(drag.w), Math.round(drag.h), cls, 0]);
    sel = st.boxes[cur].length - 1; save();
  } else {                                   // a click: select the smallest box under the cursor
    const [x, y] = pt(e); let best = -1, area = Infinity;
    st.boxes[cur].forEach((b, k) => {
      if (x >= b[0] && x <= b[0] + b[2] && y >= b[1] && y <= b[1] + b[3] && b[2] * b[3] < area) { best = k; area = b[2] * b[3]; }
    });
    sel = best;
  }
  drag = null; draw();
});
document.addEventListener("keydown", e => {
  if (e.key === "n" || e.key === "ArrowRight") load(cur + 1);
  else if (e.key === "p" || e.key === "ArrowLeft") load(cur - 1);
  else if ((e.key === "Delete" || e.key === "Backspace") && sel >= 0) { st.boxes[cur].splice(sel, 1); sel = -1; save(); draw(); e.preventDefault(); }
  else if (e.key === "i" && sel >= 0) { st.boxes[cur][sel][5] = st.boxes[cur][sel][5] ? 0 : 1; save(); draw(); }
  else if (/^[1-9]$/.test(e.key) && +e.key <= META.classes.length) { cls = +e.key - 1; classesBar(); }
});
document.getElementById("dl").onclick = () => {
  const out = {clip: META.video, name: META.name, classes: META.classes,
    frames: META.frames.map((f, i) => ({...f, boxes: st.boxes[i].map(b =>
      ({bbox: b.slice(0, 4), label: META.classes[b[4]], ignore: b[5] ? 1 : 0}))}))};
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([JSON.stringify(out, null, 1)], {type: "application/json"}));
  a.download = META.name + ".boxes.json"; a.click();
};
classesBar(); load(0);
</script></body></html>
"""


def main(argv: list[str] | None = None) -> Path:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--start", type=float, default=0.0, help="seconds")
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--fps", type=float, default=0.5, help="frames per second to sample")
    ap.add_argument("--width", type=int, default=1280, help="frames are saved at this width")
    ap.add_argument("--classes", default="face,person,car,cell phone,knife,bottle",
                    help="comma-separated; the first is the default")
    ap.add_argument("--tag", default="", help="e.g. dark / profile / crowd")
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args(argv)

    name = Path(args.video).stem + (f"_{args.tag}" if args.tag else "")
    out = Path(args.out) / f"boxes_{name}"
    (out / "frames").mkdir(parents=True, exist_ok=True)
    frames = []
    with VideoReader(args.video, width=args.width) as vr:
        for f in vr.frames(sample_fps=args.fps, start_s=args.start, end_s=args.start + args.seconds):
            file = f"{len(frames):04d}.jpg"
            Image.fromarray(f.rgb).save(out / "frames" / file, quality=92)
            frames.append({"file": file, "t": round(f.t, 3), "index": f.index,
                           "width": f.rgb.shape[1], "height": f.rgb.shape[0]})
    meta = {"video": str(args.video), "name": name, "classes": [c.strip() for c in args.classes.split(",")],
            "frames": frames}
    page = PAGE.replace("__NAME__", name).replace("__META__", json.dumps(meta).replace("</", "<\\/"))
    (out / "index.html").write_text(page, encoding="utf-8")
    print(f"{len(frames)} frames -> {out / 'index.html'}")
    return out


if __name__ == "__main__":
    main()
