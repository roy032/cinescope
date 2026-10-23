"""Run CineScope on clips: python -m cinescope.pipeline CLIP_OR_FOLDER ... [options]

    python -m cinescope.pipeline trailers/goldeneye.mp4
    python -m cinescope.pipeline trailers/ more/clip.mkv          # every video in a folder, then a library page
    python -m cinescope.pipeline clip.mp4 --stages shots palette camera
    python -m cinescope.pipeline clip.mp4 --backend onnx --workers 2 --threads 2
    python -m cinescope.pipeline clip.mp4 --set search.encoder=openclip --set faces.max_side=640
    python -m cinescope.pipeline --list
    python -m cinescope.pipeline --library                         # rebuild outputs/library.html
    python -m cinescope.pipeline --cache                           # disk use per clip and stage
    python -m cinescope.pipeline --clear CLIP [--clear-stages faces search]

-> outputs/<clip>/report.json (+ stages/<name>/ with each stage's result and assets)

Stages already computed with the same clip, version and parameters are loaded
from the cache; --force NAME (or --force all) recomputes. An interrupted run
resumes its decoding pass from the last checkpoint.

A stage that fails is reported (and the stages that need it skipped) while
the rest finish; the exit status is then 1. --strict stops at the first error.

With several clips (or a folder) each clip runs in turn; a file that cannot be
read is reported and the batch goes on. At the end a table sums up the batch
and outputs/library.html shows every analysed clip on one page, each linked to
its breakdown (cinescope.pipeline.library).
"""
from __future__ import annotations

import argparse
import json
import os


def _value(s: str):
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return s


def main(argv: list[str] | None = None) -> dict | None:
    from . import core
    from . import stages as _builtin  # noqa: F401
    from .report import summary

    ap = argparse.ArgumentParser(prog="python -m cinescope.pipeline", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clips", nargs="*", metavar="clip", help="video files and/or folders of videos")
    ap.add_argument("--stages", nargs="+", help=f"default: all ({', '.join(core.default_stages())})")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--backend", default="torch", choices=["torch", "onnx", "tensorrt", "cuda"])
    ap.add_argument("--threads", type=int, default=int(os.environ.get("OMP_NUM_THREADS", 2) or 2))
    ap.add_argument("--workers", type=int, default=1, help=">1: stages in parallel threads/processes")
    ap.add_argument("--force", nargs="*", default=[], help="stages to recompute (or 'all')")
    ap.add_argument("--set", action="append", default=[], metavar="STAGE.PARAM=VALUE",
                    help="stage parameter (JSON value), e.g. faces.max_side=640")
    ap.add_argument("--max-seconds", type=float, help="only the first N seconds")
    ap.add_argument("--checkpoint-s", type=float, default=60.0)
    ap.add_argument("--strict", action="store_true", help="stop at the first stage that fails")
    ap.add_argument("--list", action="store_true", help="list the registered stages")
    ap.add_argument("--library", action="store_true", help="only (re)build outputs/library.html")
    ap.add_argument("--no-library", action="store_true", help="do not write library.html after a batch")
    ap.add_argument("--cache", action="store_true", help="show the disk space each clip's results use")
    ap.add_argument("--clear", metavar="CLIP", help="delete a clip's cached results")
    ap.add_argument("--clear-stages", nargs="+", metavar="STAGE", help="with --clear: only these stages")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    from . import library

    if args.cache:
        print(library.cache_table(library.scan(args.out)))
        return None
    if args.clear:
        for p in library.clear(args.out, args.clear, args.clear_stages):
            print(f"removed {p}")
        return None
    if args.library:
        print(f"-> {library.save_library(args.out)}")
        return None
    if args.list or not args.clips:
        for name, cls in core.STAGES.items():
            req = ", ".join(cls.requires) or "-"
            kind = "frame" if cls.streams else "post"
            print(f"{name:<11} v{cls.version}  {kind:<5}  requires {req:<28} params {cls.defaults}")
        return None
    params: dict[str, dict] = {}
    for s in args.set:
        k, _, v = s.partition("=")
        stage, _, p = k.partition(".")
        params.setdefault(stage, {})[p] = _value(v)
    opts = core.Options(backend=args.backend, threads=args.threads, workers=args.workers,
                        force=tuple(args.force), checkpoint_s=args.checkpoint_s, max_seconds=args.max_seconds,
                        strict=args.strict)
    try:                                          # the classical stages (shots, palette ...) run without PyTorch
        import torch
    except ImportError:
        pass
    else:
        torch.set_num_threads(args.threads)
    files = library.videos(args.clips)
    if not files:
        raise SystemExit(f"no videos in {args.clips} (looked for {', '.join(library.VIDEO_EXT)})")
    log = None if args.quiet else print
    names = library.clip_names(files)
    if len(files) == 1:
        rep = core.run(files[0], args.stages, args.out, params, opts, log=log)
        print(summary(rep))
        print(f"-> {os.path.join(args.out, rep['clip'], 'report.json')}")
        return rep
    reports, failed = [], {}
    for k, (f, name) in enumerate(zip(files, names, strict=True), 1):
        print(f"[{k}/{len(files)}] {f}" + (f"  -> {name}" if name != f.stem else ""))
        try:
            reports.append(core.run(f, args.stages, args.out, params, opts, log=log, name=name))
        except core.Interrupted:
            raise
        except Exception as e:                   # noqa: BLE001 - one bad file does not stop the batch
            if args.strict:
                raise
            failed[str(f)] = f"{type(e).__name__}: {e}"
            print(f"  failed: {failed[str(f)]}")
    print(batch_table(reports, failed))
    if not args.no_library:
        print(f"-> {library.save_library(args.out)}")
    return {"batch": True, "reports": reports, "failed": failed,
            "errors": failed or {r["clip"]: r["errors"] for r in reports if r.get("errors")}}


def batch_table(reports: list[dict], failed: dict[str, str]) -> str:
    """One line per clip of a batch: length, shots, wall time, speed, failed stages."""
    lines = [f"{'clip':<30} {'length':>8} {'shots':>6} {'wall':>8} {'speed':>7}  status"]
    for r in reports:
        t = r.get("timing") or {}
        st = "ok" if not r.get("errors") else "failed: " + ", ".join(r["errors"])
        lines.append(f"{r['clip'][:30]:<30} {r['info']['duration_s']:>7.1f}s {len(r.get('shots') or []):>6} "
                     f"{t.get('wall_s', 0):>7.1f}s {t.get('video_s_per_wall_s') or 0:>6.1f}x  {st}")
    for f, err in failed.items():
        lines.append(f"{os.path.basename(f)[:30]:<30} {'':>8} {'':>6} {'':>8} {'':>7}  not analysed: {err}")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys

    rep = main()
    sys.exit(1 if rep and rep.get("errors") else 0)
