"""Build a keyboard labelling page for the shot-scale test set.

    python p1_cnn/label_scales.py            # 300 random shots from the test split
    python p1_cnn/label_scales.py --shots data/public_shots/shots.csv --set public   # the public-clip set

Writes outputs/label_scales/index.html. Open it in a browser; for each shot
press 1-5 (ECS, CS, MS, FS, LS), 0 to skip a shot that has no clear subject
(black, text, a blur), Backspace to go back. "Download labels" saves
<set>_test.labels.json (own_test.labels.json by default); p1_cnn/eval_own.py
picks it up from Downloads or from p1_cnn/labels/.

The page never shows what the teacher or the network predicted: the test labels
must be independent of the systems they score. Progress is kept in the
browser's local storage.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHOTS = ROOT / "data" / "own_shots" / "shots.csv"

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Label shot scale</title>
<style>
 body { font: 15px system-ui, sans-serif; background:#111; color:#ddd; margin:0; }
 header { background:#1b1b1b; padding:10px 16px; display:flex; gap:16px; align-items:center; flex-wrap:wrap;
          border-bottom:1px solid #333; }
 button { background:#2d6a4f; color:#fff; border:0; padding:7px 12px; border-radius:6px; cursor:pointer; }
 main { display:flex; gap:20px; padding:16px; flex-wrap:wrap; }
 #big { max-width:min(960px, 100%); display:block; }
 .small img { width:220px; display:block; margin-bottom:6px; opacity:.85; }
 .keys { display:grid; grid-template-columns: 40px 1fr; gap:6px 10px; max-width:520px; margin:12px 16px; }
 .k { background:#333; border-radius:4px; text-align:center; padding:2px 0; font-weight:600; }
 .cur { color:#8fd694; font-weight:600; }
</style></head><body>
<header>
 <b>Shot scale</b> <span id="pos"></span> <span id="cur" class="cur"></span>
 <button onclick="download()">Download labels</button>
 <span style="opacity:.7">1-5 label &middot; 0 skip &middot; Backspace back &middot; &larr; &rarr; move</span>
</header>
<div class="keys">
 <span class="k">1</span><span><b>ECS</b> extreme close-up: part of a face (eyes, mouth) or a small detail fills the frame</span>
 <span class="k">2</span><span><b>CS</b> close-up: head, perhaps shoulders</span>
 <span class="k">3</span><span><b>MS</b> medium: from about the waist or chest up</span>
 <span class="k">4</span><span><b>FS</b> full: the whole body, head to feet, just fits</span>
 <span class="k">5</span><span><b>LS</b> long: people are small in a large space, or a wide view with nobody</span>
 <span class="k">0</span><span>skip: no clear subject (black, titles, heavy blur)</span>
</div>
<main><img id="big"><div class="small" id="small"></div></main>
<script>
const ITEMS = __ITEMS__;
const NAMES = ["ECS", "CS", "MS", "FS", "LS"];
const KEY = "cinescope-scale-labels-__SET__";
let st = JSON.parse(localStorage.getItem(KEY) || '{"labels":{},"i":0}');
function save() { localStorage.setItem(KEY, JSON.stringify(st)); }
function show() {
  st.i = Math.max(0, Math.min(ITEMS.length - 1, st.i));
  const it = ITEMS[st.i];
  document.getElementById("big").src = it.frames[Math.floor(it.frames.length / 2)];
  document.getElementById("small").innerHTML = it.frames.map(f => `<img src="${f}">`).join("");
  const done = Object.keys(st.labels).length;
  document.getElementById("pos").textContent = `shot ${st.i + 1} / ${ITEMS.length} (${done} labelled)`;
  document.getElementById("cur").textContent = st.labels[it.key] ? "current: " + st.labels[it.key] : "";
  save();
}
document.addEventListener("keydown", e => {
  const it = ITEMS[st.i];
  if (e.key >= "1" && e.key <= "5") { st.labels[it.key] = NAMES[+e.key - 1]; st.i++; }
  else if (e.key === "0") { st.labels[it.key] = "skip"; st.i++; }
  else if (e.key === "Backspace" || e.key === "ArrowLeft") { st.i--; e.preventDefault(); }
  else if (e.key === "ArrowRight") { st.i++; }
  else return;
  show();
});
function download() {
  const out = {labeller: "human", classes: NAMES, n: ITEMS.length, labels: st.labels};
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([JSON.stringify(out, null, 1)], {type: "application/json"}));
  a.download = "__SET___test.labels.json"; a.click();
}
show();
</script></body></html>
"""


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--shots", default=str(SHOTS), help="shots.csv from build_own_shots.py")
    ap.add_argument("--set", default="own", help="name of the shot set (labels file <set>_test.labels.json)")
    ap.add_argument("--split", default="test", help="which split to label; 'all' for every shot (cross-validated sets)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "outputs" / "label_scales"))
    args = ap.parse_args(argv)

    shots = Path(args.shots)
    with shots.open(encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if args.split in ("all", r["split"])]
    random.Random(args.seed).shuffle(rows)
    rows = rows[:args.n]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rel = Path(os.path.relpath(shots.parent.resolve(), out.resolve()))
    items = [{"key": f"{r['episode']}/{r['shot']}",
              "frames": [(rel / p).as_posix() for p in r["frames"].split(";")]} for r in rows]
    page = PAGE.replace("__ITEMS__", json.dumps(items)).replace("__SET__", args.set)
    (out / "index.html").write_text(page, encoding="utf-8")
    print(f"{len(items)} {args.split} shots -> {out / 'index.html'}")


if __name__ == "__main__":
    main()
