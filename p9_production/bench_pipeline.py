"""The "after" measurement: every feature through cinescope.pipeline, and the 2-hour extrapolation.

    python p9_production/bench_pipeline.py trailers/goldeneye.mp4 --backend onnx --workers 2
    python p9_production/bench_pipeline.py --table          # before vs after -> results/film_2h.json

A run recomputes every stage (--force all) into outputs/bench_pipeline/<tag>/ and
writes p9_production/results/pipeline_<clip>_<tag>.json: per-stage seconds,
the shared decode, the wall clock, and the machine's load average before and
after (the 4 CPUs were shared with other jobs; see the README).

--table joins every tools_<clip>.json (bench_tools.py: each phase's own tool, one
after the other) with the pipeline runs of the same clip and extrapolates both
to a 2-hour film:

    T(2 h) = F + (7200 / D) * (W - F)

D is the clip's length in seconds, W the measured wall-clock seconds and F the
part that does not grow with the film: the camera-path stage (Phase 5), which
reconstructs one shot whatever the length. Everything else is per frame, per
shot or per face track, all of which grow in proportion to running time on
average. Model loading and ONNX export (seconds, once) are left in W, so the
extrapolation slightly overestimates both columns.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RES = ROOT / "p9_production" / "results"

# which pipeline stages do the work of which phase tool
FEATURES = {
    "p0": ("Shots, palettes, camera moves (Phase 0)", ["shots", "palette", "camera"]),
    "p1": ("Framing per shot (Phase 1)", ["framing"]),
    "p3": ("Faces, who is talking, characters (Phase 3)", ["faces", "speaking", "characters"]),
    "p5": ("Camera path of one shot (Phase 5)", ["campath"]),
    "p6": ("People, poses, action events, scene types (Phase 6)", ["pose", "scenes"]),
    "p7": ("Subtitles and on-screen text (Phase 7)", ["subtitles"]),
    "p8": ("Search index (Phase 8)", ["search"]),
}
FIXED_TOOLS, FIXED_STAGES = ("p5",), ("campath",)


def _load() -> list[float]:
    try:
        return [round(x, 2) for x in os.getloadavg()]
    except OSError:
        return []


def bench(args) -> dict:
    import torch

    from cinescope.pipeline import Options, run
    from cinescope.pipeline.report import summary

    torch.set_num_threads(args.threads)
    tag = args.tag or f"{args.backend}-w{args.workers}"
    clip = Path(args.video).stem
    load0 = _load()
    t0 = time.perf_counter()
    rep = run(args.video, args.stages, ROOT / "outputs" / "bench_pipeline" / tag, {},
              Options(backend=args.backend, threads=args.threads, workers=args.workers, force=("all",)))
    wall = time.perf_counter() - t0
    print(summary(rep))
    t = rep["timing"]
    out = {"video": rep["video"], "duration_s": rep["info"]["duration_s"], "backend": args.backend,
           "workers": args.workers, "threads": args.threads, "wall_s": round(wall, 2),
           "decode_s": t["decode_s"], "frames_decoded": t["frames_decoded"],
           "stages": {n: s["total_s"] for n, s in t["stages"].items()},
           "frame_s": {n: s.get("frame_s", 0.0) for n, s in t["stages"].items()},
           "loadavg_before": load0, "loadavg_after": _load(), "cpu_count": t["cpu_count"],
           "when": time.strftime("%Y-%m-%d %H:%M")}
    RES.mkdir(parents=True, exist_ok=True)
    (RES / f"pipeline_{clip}_{tag}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return out


def two_hours(wall: float, fixed: float, dur: float) -> float:
    return fixed + 7200.0 / dur * (wall - fixed)


def table() -> dict:
    rows = []
    for tf in sorted(RES.glob("tools_*.json")):
        clip = tf.stem[len("tools_"):]
        tools = json.loads(tf.read_text())
        dur = tools["duration_s"]
        before = two_hours(tools["total_s"], sum(tools["seconds"].get(k, 0) for k in FIXED_TOOLS), dur)
        for pf in sorted(RES.glob(f"pipeline_{clip}_*.json")):
            p = json.loads(pf.read_text())
            fixed = sum(p["stages"].get(k, 0) for k in FIXED_STAGES)
            per = {}
            for k, (label, names) in FEATURES.items():
                b = tools["seconds"].get(k)
                a = sum(p["stages"].get(n, 0) for n in names)
                per[k] = {"feature": label, "before_s": b, "after_s": round(a, 1),
                          "before_x_realtime": round(dur / b, 3) if b else None,
                          "after_x_realtime": round(dur / a, 3) if a else None}
            rows.append({"clip": clip, "duration_s": dur, "run": pf.stem, "backend": p["backend"],
                         "workers": p["workers"], "before_wall_s": tools["total_s"], "after_wall_s": p["wall_s"],
                         "decode_s": p["decode_s"], "per_feature": per,
                         "before_2h_h": round(before / 3600, 1), "after_2h_h": round(two_hours(p["wall_s"], fixed, dur) / 3600, 1),
                         "speedup": round(tools["total_s"] / p["wall_s"], 2)})
    doc = {"formula": "T(2h) = F + (7200 / D) * (W - F); F = camera path of one shot (fixed per film)",
           "rows": rows}
    (RES / "film_2h.json").write_text(json.dumps(doc, indent=1))
    for r in rows:
        print(f"\n### {r['clip']} ({r['duration_s']:.0f} s) — {r['run']}\n")
        print("| Feature | before: phase tools (s) | after: pipeline (s) | speed-up |")
        print("|---|---|---|---|")
        for v in r["per_feature"].values():
            sp = f"{v['before_s'] / v['after_s']:.1f}x" if v["before_s"] and v["after_s"] else "–"
            print(f"| {v['feature']} | {v['before_s']} | {v['after_s']} | {sp} |")
        print(f"| shared decode (pipeline only) | – | {r['decode_s']} | |")
        print(f"| **wall clock** | **{r['before_wall_s']}** | **{r['after_wall_s']}** | **{r['speedup']}x** |")
        print(f"| **2-hour film (extrapolated)** | **{r['before_2h_h']} h** | **{r['after_2h_h']} h** | |")
    return doc


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", nargs="?")
    ap.add_argument("--backend", default="onnx", choices=["torch", "onnx", "tensorrt", "cuda"])
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--stages", nargs="+")
    ap.add_argument("--tag")
    ap.add_argument("--table", action="store_true")
    args = ap.parse_args(argv)
    if args.video:
        bench(args)
    if args.table or not args.video:
        table()


if __name__ == "__main__":
    main()
