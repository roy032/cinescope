"""The pipeline engine: a registry of stages, one shared decoding pass, a cache.

A *stage* is one feature of CineScope (shots, faces, subtitles, search ...).
Every stage has the same small interface, so adding one (Phase 4's cutouts,
Phase 6's scene types) is a class with a decorator:

    @register
    class Cutouts(Stage):
        name = "cutouts"
        version = "1"                       # bump when the output changes
        requires = ("shots",)               # results it reads, via ctx.results
        streams = {"rgb": Stream(width=640, fps=2)}   # frames it wants, if any
        defaults = {"model": "unet"}        # parameters (part of the cache key)

        def begin(self, ctx): self.masks = []
        def frame(self, ctx, views): self.masks.append(segment(views["rgb"].rgb))
        def end(self, ctx) -> dict: return {"n": len(self.masks)}     # JSON-able

Two kinds of stage, one interface:

  frame stages  declare `streams`. They see every frame they asked for during
                the single decoding pass, at the width and rate they asked for.
                The decoder resizes each frame once per distinct width, so two
                stages asking for 160 px share the resize as well as the decode.
  post stages   no streams: they run after the pass, on other stages' results
                (and on assets those stages left on disk, e.g. the stored key
                frames or face crops).

`end()` always runs in dependency order, so a frame stage may still read an
earlier stage's result there (faces need the shot cuts to reset the tracker).

Cache. Each stage's result lives in outputs/<clip>/stages/<name>/result.json,
next to any assets the stage wrote there, with its *key*: a hash of the clip's
content, the stage name, version and parameters, and the keys of the stages
it requires (so changing the shot detector invalidates everything built on
its shots). A run loads every stage whose key matches and computes only the
rest; if no frame stage is left, the video is not decoded at all.

Resume. The decoding pass is the long part (a film is hours of frames), so it
saves a checkpoint every `checkpoint_s` seconds of video: the stages' state
(everything except models, see Stage.transient) and the position. After an
interruption the next run restores the states, seeks to the position and
carries on; finished stages are cached anyway.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..io.video import Frame, VideoReader, _resize


@dataclass(frozen=True)
class Stream:
    width: int | None = None        # resize to this width (aspect kept); None = full resolution
    fps: float | None = None        # sample rate; None = every frame


class Stage:
    name: str = ""
    version: str = "1"
    requires: tuple[str, ...] = ()
    streams: dict[str, Stream] = {}
    defaults: dict = {}
    uses_backend: bool = False      # result depends on the model backend (torch / onnx / tensorrt)
    transient: tuple[str, ...] = ()  # attributes left out of decode checkpoints (models, sessions)

    def __init__(self, params: dict | None = None):
        unknown = set(params or {}) - set(self.defaults)
        if unknown:
            raise ValueError(f"stage {self.name!r}: unknown parameters {sorted(unknown)}")
        self.params = {**self.defaults, **(params or {})}

    # -- the interface
    def setup(self, ctx: Context) -> None:
        """Load models. Called before begin(), and again after a checkpoint restore."""

    def begin(self, ctx: Context) -> None:
        """Initialise accumulators (frame stages)."""

    def frame(self, ctx: Context, views: dict[str, Frame]) -> None:
        """One decoded frame; views[stream name] is present when this frame is in that stream."""

    def end(self, ctx: Context) -> dict:
        raise NotImplementedError

    # -- checkpointing
    def state(self) -> dict:
        skip = set(self.transient) | {"params"}
        return {k: v for k, v in self.__dict__.items() if k not in skip}

    def restore(self, state: dict) -> None:
        self.__dict__.update(state)


STAGES: dict[str, type[Stage]] = {}


def register(cls: type[Stage]) -> type[Stage]:
    if not cls.name:
        raise ValueError("a stage needs a name")
    STAGES[cls.name] = cls
    return cls


@dataclass
class Options:
    backend: str = "torch"          # torch | onnx | tensorrt (model inference for stages that use models)
    threads: int = 2                # intra-op threads for torch / onnxruntime
    workers: int = 1                # >1: frame stages run in their own threads; post stages may use processes
    checkpoint_s: float = 60.0      # decode checkpoint period, seconds of video
    force: tuple[str, ...] = ()     # stages to recompute even if cached ("all" for every one)
    onnx_dir: str | None = None     # where exported .onnx files are cached (default outputs/onnx)
    max_seconds: float | None = None  # only the first N seconds of the clip (quick looks, tests)


@dataclass
class Context:
    video: str
    clip: str
    info: object                    # VideoInfo
    out: Path                       # outputs/<clip>
    options: Options
    results: dict = field(default_factory=dict)
    progress: Callable[[str, float, str], None] | None = None
    log: Callable[[str], None] | None = print
    keys: dict = field(default_factory=dict)

    def stage_dir(self, name: str) -> Path:
        d = self.out / "stages" / name
        d.mkdir(parents=True, exist_ok=True)
        return d

    def report(self, stage: str, fraction: float, message: str = "") -> None:
        if self.progress is not None:
            self.progress(stage, float(min(max(fraction, 0.0), 1.0)), message)

    def say(self, msg: str) -> None:
        if self.log is not None:
            self.log(msg)

    @property
    def duration(self) -> float:
        d = float(self.info.duration_s)
        return min(d, self.options.max_seconds) if self.options.max_seconds else d


# --------------------------------------------------------------------------- keys and cache
def clip_hash(path: str | Path, chunk: int = 1 << 20) -> str:
    """Content hash of a video that does not read the whole file: size + first and
    last MiB. Renaming or moving the file keeps its cache; re-encoding it does not.
    (Hashing all of a 4 GB film would take longer than some stages.)"""
    p = Path(path)
    size = p.stat().st_size
    h = hashlib.sha1(str(size).encode())
    with open(p, "rb") as f:
        h.update(f.read(chunk))
        if size > chunk:
            f.seek(max(chunk, size - chunk))
            h.update(f.read(chunk))
    return h.hexdigest()[:16]


def stage_key(chash: str, stage: Stage, dep_keys: list[str], options: Options) -> str:
    blob = json.dumps([chash, stage.name, stage.version, stage.params, dep_keys,
                       options.backend if stage.uses_backend else None, options.max_seconds],
                      sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:16]


def _write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=_json_default), encoding="utf-8")
    os.replace(tmp, path)               # atomic: an interrupted write never leaves half a file


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def load_cached(ctx: Context, name: str, key: str) -> dict | None:
    p = ctx.out / "stages" / name / "result.json"
    if not p.exists():
        return None
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return doc if doc.get("key") == key else None


# --------------------------------------------------------------------------- planning
def resolve(names: list[str]) -> list[str]:
    """Requested stages plus everything they require, in dependency order."""
    order: list[str] = []
    seen: set[str] = set()

    def visit(n: str, path: tuple[str, ...]) -> None:
        if n in path:
            raise ValueError(f"dependency cycle: {' -> '.join(path + (n,))}")
        if n in seen:
            return
        if n not in STAGES:
            raise KeyError(f"unknown stage {n!r}; registered: {sorted(STAGES)}")
        for d in STAGES[n].requires:
            visit(d, path + (n,))
        seen.add(n)
        order.append(n)

    for n in names:
        visit(n, ())
    return order


# --------------------------------------------------------------------------- the decoding pass
class _Sampler:
    """VideoReader.frames' sample_fps rule, one stream at a time: the first
    frame at or after each 1/fps tick."""

    def __init__(self, fps: float | None, start: float = 0.0):
        self.step = 1.0 / fps if fps else None
        self.next_t = start

    def take(self, t: float) -> bool:
        if self.step is None:
            return True
        if t + 1e-6 < self.next_t:
            return False
        self.next_t += self.step * (int((t - self.next_t) / self.step) + 1)
        return True


class _StageThread:
    """Runs one frame stage's frame() calls in its own thread, in frame order.
    NumPy, OpenCV, PIL and onnxruntime release the GIL, so stages overlap."""

    def __init__(self, stage: Stage, ctx: Context, timing: dict):
        self.stage, self.ctx, self.timing = stage, ctx, timing
        self.q: queue.Queue = queue.Queue(maxsize=32)
        self.error: BaseException | None = None
        self.t = threading.Thread(target=self._loop, daemon=True, name=f"stage-{stage.name}")
        self.t.start()

    def _loop(self) -> None:
        while True:
            item = self.q.get()
            try:
                if item is None:
                    return
                if self.error is None:
                    t0 = time.perf_counter()
                    self.stage.frame(self.ctx, item)
                    self.timing[self.stage.name] += time.perf_counter() - t0
            except BaseException as e:      # noqa: BLE001 - re-raised in the main thread
                self.error = e
            finally:
                self.q.task_done()

    def put(self, views) -> None:
        if self.error is not None:
            raise self.error
        self.q.put(views)

    def drain(self) -> None:
        self.q.join()
        if self.error is not None:
            raise self.error

    def close(self) -> None:
        self.q.put(None)
        self.t.join()


class Interrupted(RuntimeError):
    """Raised by a stop hook (tests, the web app's cancel button)."""


def decode_pass(ctx: Context, stages: list[Stage], keys: dict[str, str],
                stop: Callable[[int, float], bool] | None = None) -> dict:
    """Decode the clip once and feed every frame stage. Returns timing (s)."""
    ckpt_path = ctx.out / "stages" / "_decode" / "checkpoint.pkl"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    want = {s.name: keys[s.name] for s in stages}
    widths = sorted({st.width or 0 for s in stages for st in s.streams.values()})
    samplers = {(s.name, k): _Sampler(st.fps) for s in stages for k, st in s.streams.items()}
    timing = {s.name: 0.0 for s in stages}
    timing["_decode"] = 0.0
    start_s, n_done = 0.0, 0
    if ckpt_path.exists():
        try:
            with open(ckpt_path, "rb") as f:
                ck = pickle.load(f)
        except (OSError, pickle.UnpicklingError, EOFError):
            ck = None
        if ck and ck.get("keys") == want:
            for s in stages:
                s.restore(ck["states"][s.name])
                s.setup(ctx)
            for k, nt in ck["samplers"].items():
                samplers[k].next_t = nt
            start_s, n_done = ck["next_t"], ck["n_done"]
            timing.update(ck.get("timing", {}))
            ctx.say(f"  resuming the decoding pass at {start_s:.1f} s")
    if n_done == 0:
        for s in stages:
            s.setup(ctx)
            s.begin(ctx)

    threads = [_StageThread(s, ctx, timing) for s in stages] if ctx.options.workers > 1 else None

    def checkpoint(next_t: float) -> None:
        if threads:
            for th in threads:
                th.drain()
        tmp = ckpt_path.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            pickle.dump({"keys": want, "states": {s.name: s.state() for s in stages},
                         "samplers": {k: v.next_t for k, v in samplers.items()},
                         "next_t": next_t, "n_done": n, "timing": dict(timing)}, f,
                        protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, ckpt_path)

    dur = max(ctx.duration, 1e-6)
    next_ck = start_s + ctx.options.checkpoint_s
    n = n_done
    t_frame = start_s
    end_s = ctx.options.max_seconds
    try:
        with VideoReader(ctx.video) as vr:
            t0 = time.perf_counter()
            for f in vr.frames(start_s=start_s, end_s=end_s):
                if f.t >= next_ck:
                    timing["_decode"] += time.perf_counter() - t0
                    checkpoint(f.t)
                    next_ck = f.t + ctx.options.checkpoint_s
                    t0 = time.perf_counter()
                if stop is not None and stop(n, f.t):
                    raise Interrupted(f"stopped at frame {n} ({f.t:.2f} s)")
                resized = {w: (f.rgb if w == 0 else _resize(f.rgb, w)) for w in widths}
                timing["_decode"] += time.perf_counter() - t0
                for s in stages:
                    views = {}
                    for k, st in s.streams.items():
                        if samplers[(s.name, k)].take(f.t):
                            views[k] = Frame(f.index, f.t, resized[st.width or 0])
                    if not views:
                        continue
                    if threads:
                        threads[stages.index(s)].put(views)
                    else:
                        t1 = time.perf_counter()
                        s.frame(ctx, views)
                        timing[s.name] += time.perf_counter() - t1
                n += 1
                t_frame = f.t
                if n % 50 == 0:
                    ctx.report("decode", t_frame / dur, f"{t_frame:.0f} / {dur:.0f} s")
                t0 = time.perf_counter()
        if threads:
            for th in threads:
                th.drain()
    finally:
        if threads:
            for th in threads:
                th.close()
    ctx.report("decode", 1.0, f"{n} frames")
    timing["_frames"] = n - n_done          # decoded by this run (a resumed run starts at n_done)
    timing["_resumed_at_s"] = start_s
    return timing


def remove_checkpoint(ctx: Context) -> None:
    p = ctx.out / "stages" / "_decode" / "checkpoint.pkl"
    if p.exists():
        p.unlink()


# --------------------------------------------------------------------------- the runner
def run(video: str | Path, stages: list[str] | None = None, out: str | Path = "outputs",
        params: dict[str, dict] | None = None, options: Options | None = None,
        progress: Callable[[str, float, str], None] | None = None, log: Callable[[str], None] | None = print,
        stop: Callable[[int, float], bool] | None = None) -> dict:
    """Run the stages on one clip; returns the report (also written to outputs/<clip>/report.json)."""
    from . import stages as _builtin  # noqa: F401  (registers the built-in stages)
    from .report import assemble

    video = str(video)
    options = options or Options()
    params = params or {}
    t_start = time.perf_counter()
    with VideoReader(video) as vr:
        info = vr.info
    clip = Path(video).stem
    ctx = Context(video, clip, info, Path(out) / clip, options, progress=progress, log=log)
    ctx.out.mkdir(parents=True, exist_ok=True)
    names = resolve(stages or default_stages())
    objs = {n: STAGES[n](params.get(n)) for n in names}
    chash = clip_hash(video)
    for n in names:
        ctx.keys[n] = stage_key(chash, objs[n], [ctx.keys[d] for d in objs[n].requires], options)

    timing: dict[str, dict] = {}
    todo: list[str] = []
    for n in names:
        forced = "all" in options.force or n in options.force
        doc = None if forced else load_cached(ctx, n, ctx.keys[n])
        if doc is not None:
            ctx.results[n] = doc["result"]
            timing[n] = {**doc.get("timing", {}), "cached": True}
        else:
            todo.append(n)
    ctx.say(f"{clip}: {len(names)} stages, cached {[n for n in names if n not in todo]}, to run {todo}")

    frame_stages = [objs[n] for n in todo if objs[n].streams]
    decode_t: dict = {}
    if frame_stages:
        decode_t = decode_pass(ctx, frame_stages, ctx.keys, stop=stop)
    for n in todo:
        s = objs[n]
        if not s.streams:
            s.setup(ctx)
            s.begin(ctx)
        ctx.report(n, 0.0, "running")
        t0 = time.perf_counter()
        result = s.end(ctx)
        t_end = time.perf_counter() - t0
        frame_s = round(decode_t.get(n, 0.0), 3)
        tm = {"frame_s": frame_s, "end_s": round(t_end, 3), "total_s": round(frame_s + t_end, 3), "cached": False}
        ctx.results[n] = result
        timing[n] = tm
        _write_json(ctx.stage_dir(n) / "result.json",
                    {"key": ctx.keys[n], "name": n, "version": s.version, "params": s.params,
                     "backend": options.backend if s.uses_backend else None, "timing": tm, "result": result})
        ctx.report(n, 1.0, f"{t_end:.1f} s")
        ctx.say(f"  {n}: {tm['total_s']:.1f} s")
    remove_checkpoint(ctx)

    wall = time.perf_counter() - t_start
    run_info = {"wall_s": round(wall, 3), "decode_s": round(decode_t.get("_decode", 0.0), 3),
                "frames_decoded": decode_t.get("_frames", 0), "resumed_at_s": decode_t.get("_resumed_at_s", 0.0),
                "stages": timing,
                "backend": options.backend, "workers": options.workers, "threads": options.threads}
    report = assemble(ctx, names, run_info)
    _write_json(ctx.out / "report.json", report)
    ctx.report("done", 1.0, f"{wall:.1f} s")
    return report


def default_stages() -> list[str]:
    from . import stages as _builtin  # noqa: F401

    return [n for n, cls in STAGES.items() if getattr(cls, "default", True)]
