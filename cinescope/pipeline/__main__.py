"""Run CineScope on a clip: python -m cinescope.pipeline CLIP [options]

    python -m cinescope.pipeline trailers/goldeneye.mp4
    python -m cinescope.pipeline clip.mp4 --stages shots palette camera
    python -m cinescope.pipeline clip.mp4 --backend onnx --workers 2 --threads 2
    python -m cinescope.pipeline clip.mp4 --set search.encoder=openclip --set faces.max_side=640
    python -m cinescope.pipeline --list

-> outputs/<clip>/report.json (+ stages/<name>/ with each stage's result and assets)

Stages already computed with the same clip, version and parameters are loaded
from the cache; --force NAME (or --force all) recomputes. An interrupted run
resumes its decoding pass from the last checkpoint.
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
    ap.add_argument("clip", nargs="?")
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
    ap.add_argument("--list", action="store_true", help="list the registered stages")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    if args.list or not args.clip:
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
                        force=tuple(args.force), checkpoint_s=args.checkpoint_s, max_seconds=args.max_seconds)
    try:                                          # the classical stages (shots, palette ...) run without PyTorch
        import torch
    except ImportError:
        pass
    else:
        torch.set_num_threads(args.threads)
    rep = core.run(args.clip, args.stages, args.out, params, opts, log=None if args.quiet else print)
    print(summary(rep))
    print(f"-> {os.path.join(args.out, rep['clip'], 'report.json')}")
    return rep


if __name__ == "__main__":
    main()
