"""Finding clips by name and caching their Phase 0 shots.

Scripts take clip paths, but training sets are long lists of clips, so a bare
name ("goldeneye") is looked up in `$CINESCOPE_CLIPS` (a path list), then
`<repo>/trailers` and `<repo>/data/clips`.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
EXTS = (".mp4", ".avi", ".mkv", ".mov", ".webm")


def search_dirs() -> list[Path]:
    env = os.environ.get("CINESCOPE_CLIPS", "")
    dirs = [Path(p) for p in env.split(os.pathsep) if p]
    return dirs + [REPO / "trailers", REPO / "data" / "clips"]


def resolve(clip: str | Path) -> Path:
    p = Path(clip)
    if p.exists():
        return p
    for d in search_dirs():
        for ext in ("", *EXTS):
            q = d / f"{clip}{ext}"
            if q.exists():
                return q
    raise FileNotFoundError(f"clip {clip!r} not found (searched {[str(d) for d in search_dirs()]})")


def clip_name(path: str | Path) -> str:
    return Path(path).stem


def shots(path: str | Path, cache_dir: str | Path = "outputs/shots") -> dict:
    """Phase 0 shots of a clip (cached as JSON): {"fps", "n_frames", "shots": [{start, end, t0, t1}, ...]}."""
    path = resolve(path)
    cache = Path(cache_dir) / f"{clip_name(path)}.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    from ..shots.pipeline import analyse

    a = analyse(path, with_palettes=False)
    d = a.to_json()
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(d), encoding="utf-8")
    return d


def windows(shot_list: list[dict], seconds: float = 2.0, min_seconds: float = 0.5) -> list[tuple[int, float, float]]:
    """Split shots into windows of about `seconds` (long static shots become
    several samples, a short shot stays one): [(shot index, t0, t1)]."""
    out = []
    for i, s in enumerate(shot_list):
        dur = s["t1"] - s["t0"]
        if dur < min_seconds:
            continue
        n = max(1, int(round(dur / seconds)))
        edges = np.linspace(s["t0"], s["t1"], n + 1)
        out += [(i, float(a), float(b)) for a, b in zip(edges[:-1], edges[1:], strict=True)]
    return out
