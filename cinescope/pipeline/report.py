"""One report per clip: outputs/<clip>/report.json, what the web app reads.

The stages' own results stay in stages/<name>/result.json; the report merges
the per-shot ones into one shot list (framing, camera move, palette and a
thumbnail per shot), copies the timelines, and records the run's timing with
its throughput (seconds of video per wall-clock second), and the editing
pace (average and median shot length, a cuts-per-minute curve). Asset paths
are relative to outputs/<clip>/.

Cross-stage links are made here because they are cheap and change whenever
any of the stages they join changes: the search index's documents get the
subtitle cues and the characters on screen during them (the hooks Phase 8
left for Phases 3 and 7), so chat answers can cite dialogue and people.
"""
from __future__ import annotations

import time
from pathlib import Path

from .core import Context


def _shot_thumbs(ctx: Context, shots: list[dict], width: int = 192) -> list[str | None]:
    if "frames" not in ctx.results:
        return [None] * len(shots)
    from .stages import FrameStore, _load_rgb, _resize_w, _save_jpeg

    store = FrameStore(ctx)
    d = ctx.out / "assets" / "shots"
    d.mkdir(parents=True, exist_ok=True)
    out = []
    for s in shots:
        if not len(store.idx):
            out.append(None)
            continue
        pos = store.nearest((s["start"] + s["end"]) // 2, s["start"], s["end"])
        name = f"assets/shots/{s['index']:05d}.jpg"
        p = ctx.out / name
        if not p.exists():
            _save_jpeg(p, _resize_w(_load_rgb(store.root / store.files[pos]), width), 82)
        out.append(name)
    return out


def _link_search(ctx: Context) -> None:
    if "search" not in ctx.results:
        return
    from ..search.shotindex import ShotIndex, attach_characters, attach_subtitles

    root = ctx.out / "stages" / "search" / ctx.results["search"]["index"]
    if not (root / "docs.json").exists():
        return
    ix = ShotIndex.load(root)
    if "subtitles" in ctx.results:
        attach_subtitles(ix.docs, ctx.results["subtitles"]["cues"])
    if "characters" in ctx.results:
        app = [{"name": c["id"], "t0": a, "t1": b} for c in ctx.results["characters"]["characters"]
               for a, b in c["on_screen"]]
        attach_characters(ix.docs, app)
    ix.save_docs()


def assemble(ctx: Context, names: list[str], run_info: dict) -> dict:
    R = ctx.results
    info = ctx.info
    rep: dict = {
        "clip": ctx.clip, "video": Path(ctx.video).name, "video_path": str(Path(ctx.video).resolve()),
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "info": {"width": info.width, "height": info.height, "fps": round(float(info.fps), 4),
                 "duration_s": round(ctx.duration, 3)},
        "stages": names, "keys": {n: ctx.keys[n] for n in names},
    }
    if "shots" in R:
        shots = [dict(s) for s in R["shots"]["shots"]]
        for i, th in enumerate(_shot_thumbs(ctx, shots)):
            shots[i]["thumb"] = th
        for name, field, key in (("camera", "camera", "moves"), ("palette", "palette", "palettes"),
                                 ("framing", "framing", "shots")):
            if name in R and R[name].get(key):
                for s, v in zip(shots, R[name][key], strict=False):
                    s[field] = v
        rep["shots"] = shots
        rep["transitions"] = R["shots"]["transitions"]
    if "palette" in R:
        rep["barcode"] = {"image": f"stages/palette/{R['palette']['barcode']}" if R["palette"]["barcode"] else None,
                          "fps": R["palette"]["barcode_fps"], "colors": R["palette"]["barcode_colors"]}
    if "framing" in R:
        rep["framing_method"] = R["framing"].get("method")
    if "characters" in R:
        c = R["characters"]
        rep["characters"] = {
            "embedder": c.get("embedder"), "asd": c.get("asd"), "n_tracks": c.get("n_tracks"),
            "list": [{**ch, "thumb": f"stages/characters/{ch['thumb']}"} for ch in c["characters"]]}
    if "subtitles" in R:
        s = R["subtitles"]
        rep["subtitles"] = {"reader": s["reader"], "srt": f"stages/subtitles/{s['srt']}", "cues": s["cues"],
                            "credits": s["credits"]}
    if "search" in R:
        rep["search"] = {**R["search"], "index": f"stages/search/{R['search']['index']}"}
    if "campath" in R and R["campath"].get("file"):
        rep["camera_path"] = {**R["campath"], "file": f"stages/campath/{R['campath']['file']}"}
    # stages the report does not know by name (Phase 4's cutouts, Phase 6's pose and
    # scene types, ...) follow two conventions and appear without changes here:
    #   result["per_shot"]: a list, one JSON value per shot -> shots[i][<stage name>]
    #   result["timeline"]: [{"t0", "t1", "label", ...}]     -> report["timelines"][<stage name>]
    # and the whole result is kept under report["extra"][<stage name>].
    known = {"shots", "palette", "camera", "frames", "faces", "speaking", "characters", "subtitles", "search",
             "framing", "campath"}
    for n in names:
        if n in known or n not in R:
            continue
        r = R[n]
        rep.setdefault("extra", {})[n] = r
        if isinstance(r, dict) and isinstance(r.get("per_shot"), list) and "shots" in rep:
            for s, v in zip(rep["shots"], r["per_shot"], strict=False):
                s[n] = v
        if isinstance(r, dict) and isinstance(r.get("timeline"), list):
            rep.setdefault("timelines", {})[n] = r["timeline"]
    _link_search(ctx)
    if "shots" in rep:
        from ..editorial.pacing import pacing

        rep["pacing"] = pacing(rep)

    dur = ctx.duration
    stages_t = run_info["stages"]
    per = {}
    for n, t in stages_t.items():
        tot = t.get("total_s", 0.0) or 0.0
        per[n] = {**t, "video_s_per_wall_s": round(dur / tot, 3) if tot > 0 else None}
    wall = run_info["wall_s"]
    rep["timing"] = {**run_info, "stages": per,
                     "video_s_per_wall_s": round(dur / wall, 4) if wall > 0 else None,
                     "extrapolated_2h_film_s": round(7200.0 * wall / dur, 0) if dur > 0 else None,
                     "cpu_count": len(_cpus())}
    return rep


def _cpus() -> list[int]:
    try:
        import os

        return sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        import os

        return list(range(os.cpu_count() or 1))


def summary(rep: dict) -> str:
    """A few human lines about a report (the CLI prints this)."""
    lines = [f"{rep['clip']}: {rep['info']['duration_s']:.1f} s of video, "
             f"{rep['info']['width']}x{rep['info']['height']} @ {rep['info']['fps']:.2f} fps"]
    if "shots" in rep:
        moves: dict[str, int] = {}
        for s in rep["shots"]:
            lab = (s.get("camera") or {}).get("label")
            if lab:
                moves[lab] = moves.get(lab, 0) + 1
        lines.append(f"  {len(rep['shots'])} shots" + (f", camera moves {moves}" if moves else ""))
    if "characters" in rep:
        lines.append(f"  {len(rep['characters']['list'])} characters ({rep['characters']['asd']} speaker detection)")
    if "subtitles" in rep:
        lines.append(f"  {len(rep['subtitles']['cues'])} subtitle cues, "
                     f"{len(rep['subtitles']['credits'])} other text lines")
    if "search" in rep:
        lines.append(f"  search index: {rep['search']['n_docs']} segments")
    t = rep["timing"]
    lines.append(f"  wall {t['wall_s']:.1f} s ({t['video_s_per_wall_s']} s of video per second; "
                 f"a 2-hour film at this rate: {t['extrapolated_2h_film_s'] / 3600:.2f} h)")
    for n, st in t["stages"].items():
        tag = "cached" if st.get("cached") else f"{st.get('total_s', 0):.1f} s"
        lines.append(f"    {n:<11} {tag}")
    return "\n".join(lines)
