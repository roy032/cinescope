"""The built-in stages: each wraps one phase's code behind the Stage interface.

    stage       phase  kind    reads          writes (outputs/<clip>/stages/<name>/)
    shots       0      frame   160 px, all    shot list, 96 px thumbnails of every frame (disk)
    frames      8/1    frame   full, 2 fps    640 px JPEG key frames (+ the first frame after
                                              every likely cut, so short shots have one)
    palette     0      post    shots          per-shot palettes, movie barcode
    camera      0      post    shots          camera move per shot
    faces       3      frame   full, all      detections, tracks, face crops (Phase 3 format)
    speaking    3      post    faces          per-track speaking scores (Light-ASD or ours)
    characters  3      post    faces,speaking character timeline with talking intervals
    subtitles   7      frame   full, 8+4 fps  subtitle cues, other on-screen text
    search      8      post    shots,frames   Phase 8 shot index (ShotIndex.load works on it)
    framing     1      post    shots,frames,  framing label per shot: the Phase 1 ResNet-18 if a
                               search,faces   checkpoint exists, else zero-shot CLIP + face size
    campath     5      post    shots,camera   camera path of one shot (SfM or rotation only)
    pose        6      frame   960 px, 4 fps  people, poses, action events per shot (YOLOv8n-pose)
    scenes      6      post    search         scene type per shot (CLIP prompts on the key-frame vectors)
    cutouts     4      post    frames         opt-in: RGBA cut-out of the largest person per shot
    look        -      post    shots          exposure key, contrast, grade and symmetry per shot
    editing     -      post    shots          camera set-ups, shot/reverse-shot, montages, long takes

Model stages honour Options.backend (torch / onnx / tensorrt) through
cinescope.pipeline.accel. Heavy per-shot loops (palettes, camera moves) use a
process pool when Options.workers > 1: each shot is independent and the
thumbnails are read from the memmap in each worker, so nothing large is
pickled.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from .core import Context, Stage, Stream, register
from .store import DiskArray

REPO = Path(__file__).resolve().parents[2]


def _save_jpeg(path: Path, rgb: np.ndarray, quality: int = 88) -> None:
    import cv2

    cv2.imwrite(str(path), np.ascontiguousarray(rgb[:, :, ::-1]), [cv2.IMWRITE_JPEG_QUALITY, quality])


def _load_rgb(path: Path) -> np.ndarray:
    import cv2

    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return np.ascontiguousarray(bgr[:, :, ::-1])


def _resize_w(rgb: np.ndarray, width: int) -> np.ndarray:
    import cv2

    h, w = rgb.shape[:2]
    if w <= width:
        return rgb
    return cv2.resize(rgb, (width, max(1, round(h * width / w))), interpolation=cv2.INTER_AREA)


def _pool_map(fn, items: list, workers: int) -> list:
    """map() over a process pool when workers > 1 (spawned, so no forked torch threads)."""
    if workers <= 1 or len(items) < 4:
        return [fn(x) for x in items]
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    chunk = max(1, len(items) // (workers * 4))
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as ex:
        return list(ex.map(fn, items, chunksize=chunk))


# =========================================================================== Phase 0
@register
class Shots(Stage):
    name = "shots"
    version = "1"
    streams = {"small": Stream(width=160)}
    defaults = {"thumb_width": 96}

    def begin(self, ctx: Context) -> None:
        d = ctx.stage_dir(self.name)
        self.times, self.hist, self.mean, self.std, self.edges = [], [], [], [], []
        self.thumbs = DiskArray(d / "thumbs.u8")

    def frame(self, ctx: Context, views) -> None:
        from ..imgproc.filters import to_gray
        from ..shots.features import _shrink, edge_density, hsv_histogram

        f = views["small"]
        gray = to_gray(f.rgb)
        self.times.append(f.t)
        self.hist.append(hsv_histogram(f.rgb).astype(np.float32))
        self.mean.append(float(gray.mean()))
        self.std.append(float(gray.std()))
        self.edges.append(edge_density(gray))
        self.thumbs.append(_shrink(f.rgb, self.params["thumb_width"]))

    def end(self, ctx: Context) -> dict:
        from ..shots import cuts
        from ..shots.features import FrameFeatures

        self.thumbs.close()
        feat = FrameFeatures(np.asarray(self.times), np.asarray(self.hist, np.float64), np.asarray(self.mean),
                             np.asarray(self.std), np.asarray(self.edges))
        transitions = cuts.detect_transitions(feat)
        shots = cuts.shots_from_transitions(transitions, len(feat), feat.times)
        d = ctx.stage_dir(self.name)
        np.save(d / "times.npy", feat.times)
        return {"fps": ctx.info.fps, "n_frames": len(feat), "width": ctx.info.width, "height": ctx.info.height,
                "transitions": [t.to_dict() for t in transitions], "shots": [s.to_dict() for s in shots],
                "thumbs": {"file": "thumbs.u8", "shape": list(self.thumbs.item_shape or (0, 0, 3)),
                           "n": len(self.thumbs)}}


def _thumbs(ctx: Context) -> np.ndarray:
    r = ctx.results["shots"]["thumbs"]
    path = ctx.out / "stages" / "shots" / r["file"]
    if r["n"] == 0:
        return np.zeros((0, *r["shape"]), np.uint8)
    return np.memmap(path, np.uint8, "r", shape=(r["n"], *r["shape"]))


def _palette_job(args) -> dict:
    from ..palette.kmeans import palette

    path, shape, n, a, b, step, k = args
    th = np.memmap(path, np.uint8, "r", shape=(n, *shape))
    frames = np.asarray(th[a:b + 1:step])
    p = palette(frames, k=min(k, max(1, frames.shape[0] * 4)))
    return {"rgb": p.rgb.tolist(), "weights": np.round(p.weights, 4).tolist()}


def _camera_job(args) -> dict:
    from ..motion.camera import classify_shot

    path, shape, n, a, b, fps = args
    th = np.memmap(path, np.uint8, "r", shape=(n, *shape))
    gray = np.asarray(th[a:b + 1]).astype(np.float64).mean(-1)
    return classify_shot(gray, fps).to_dict()


@register
class Palette(Stage):
    name = "palette"
    version = "1"
    requires = ("shots",)
    defaults = {"k": 5, "barcode_fps": 1.0}

    def end(self, ctx: Context) -> dict:
        from PIL import Image

        from ..palette.kmeans import barcode

        r = ctx.results["shots"]
        fps = r["fps"]
        path = str(ctx.out / "stages" / "shots" / r["thumbs"]["file"])
        shape, n = tuple(r["thumbs"]["shape"]), r["thumbs"]["n"]
        step = max(1, int(round(fps / 2)))                # 2 frames per second of shot, as Phase 0
        jobs = [(path, shape, n, s["start"], s["end"], step, self.params["k"]) for s in r["shots"]]
        pals = _pool_map(_palette_job, jobs, ctx.options.workers)
        th = _thumbs(ctx)
        cols = np.asarray(th[:: max(1, int(round(fps / self.params["barcode_fps"])))])
        bc = barcode(cols, height=1) if len(cols) else np.zeros((1, 0, 3), np.uint8)
        d = ctx.stage_dir(self.name)
        if bc.shape[1]:
            Image.fromarray(np.repeat(bc, 160, axis=0)).save(d / "barcode.png")
        return {"palettes": pals, "barcode": "barcode.png" if bc.shape[1] else None,
                "barcode_fps": self.params["barcode_fps"], "barcode_colors": bc[0].tolist()}


@register
class Camera(Stage):
    name = "camera"
    version = "1"
    requires = ("shots",)

    def end(self, ctx: Context) -> dict:
        r = ctx.results["shots"]
        path = str(ctx.out / "stages" / "shots" / r["thumbs"]["file"])
        shape, n = tuple(r["thumbs"]["shape"]), r["thumbs"]["n"]
        jobs = [(path, shape, n, s["start"], s["end"], r["fps"]) for s in r["shots"]]
        return {"moves": _pool_map(_camera_job, jobs, ctx.options.workers)}


# =========================================================================== key frames (Phases 1, 8)
@register
class Frames(Stage):
    """Key-frame store shared by every stage that looks at a few frames per shot.

    Those stages need frames chosen from the shot list, which exists only after
    the pass. Instead of a second decode, frames are kept as they stream past:
    one every 1/fps s, plus the first frame after every large change in a tiny
    grey version (a likely cut), so a 0.2 s shot still has a frame of its own.
    Stages then take the stored frame nearest to the one they would have picked.
    """
    name = "frames"
    version = "1"
    streams = {"full": Stream(None), "small": Stream(width=160)}
    defaults = {"fps": 2.0, "width": 640, "cut_diff": 12.0, "quality": 88}

    def begin(self, ctx: Context) -> None:
        (ctx.stage_dir(self.name) / "jpg").mkdir(exist_ok=True)
        self.saved: list[list] = []          # [frame index, t, file]
        self.prev = None
        self.next_t = 0.0

    def frame(self, ctx: Context, views) -> None:
        f, small = views["full"], views["small"].rgb
        tiny = small[::4, ::4].astype(np.float32).mean(-1)
        diff = float(np.abs(tiny - self.prev).mean()) if self.prev is not None else 1e9
        self.prev = tiny
        due = f.t + 1e-6 >= self.next_t
        if due:
            step = 1.0 / self.params["fps"]
            self.next_t += step * (int((f.t - self.next_t) / step) + 1)
        if due or diff > self.params["cut_diff"]:
            name = f"jpg/{f.index:07d}.jpg"
            _save_jpeg(ctx.stage_dir(self.name) / name, _resize_w(f.rgb, self.params["width"]),
                       self.params["quality"])
            self.saved.append([int(f.index), round(float(f.t), 4), name])

    def end(self, ctx: Context) -> dict:
        return {"frames": self.saved, "width": self.params["width"], "fps": self.params["fps"]}


class FrameStore:
    """Read side of the Frames stage."""

    def __init__(self, ctx: Context):
        r = ctx.results["frames"]
        self.root = ctx.out / "stages" / "frames"
        self.idx = np.array([f[0] for f in r["frames"]], np.int64)
        self.t = np.array([f[1] for f in r["frames"]])
        self.files = [f[2] for f in r["frames"]]

    def nearest(self, frame: int, lo: int | None = None, hi: int | None = None) -> int:
        """Position (in the store) of the stored frame nearest `frame`, inside [lo, hi] when possible."""
        if not len(self.idx):
            raise ValueError("no stored frames")
        cand = np.arange(len(self.idx))
        if lo is not None and hi is not None:
            inside = cand[(self.idx >= lo) & (self.idx <= hi)]
            if len(inside):
                cand = inside
        return int(cand[np.argmin(np.abs(self.idx[cand] - frame))])

    def load(self, pos: int) -> np.ndarray:
        return _load_rgb(self.root / self.files[pos])


# =========================================================================== Phase 3
@register
class Faces(Stage):
    """Face detection on every frame, with the crops Phase 3 needs cut on the spot.

    Phase 3 decodes twice: once to detect, once (after tracking) to cut crops
    from the finished tracks. Here every detection's crops are cut during the
    one pass and appended to disk: the 112 px aligned grey face (mouth motion),
    a colour one on every `rgb_every`-th frame (identity), and a grey patch
    around the face from which Light-ASD's crop is cut later along the
    *smoothed* track box (faces.align.asd_patch). end() then tracks (resetting
    at the shot cuts) and writes the kept tracks' crops in Phase 3's format.
    """
    name = "faces"
    version = "1"
    requires = ("shots",)
    streams = {"full": Stream(None)}
    defaults = {"detector": "yunet", "max_side": 960, "rgb_every": 3, "min_track_s": 0.4, "min_score": 0.6,
                "max_rgb": 24}
    transient = ("det",)

    def setup(self, ctx: Context) -> None:
        from ..faces.detectors import make_detector

        kw = {"max_side": self.params["max_side"]} if self.params["detector"] == "yunet" else {}
        self.det = make_detector(self.params["detector"], **kw)

    def begin(self, ctx: Context) -> None:
        d = ctx.stage_dir(self.name)
        self.times: list[float] = []
        self.boxes, self.scores, self.lms = [], [], []
        self.det_start: list[int] = []
        self.gray = DiskArray(d / "raw_gray.u8", (112, 112))
        self.patch = DiskArray(d / "raw_patch.u8", (168, 168))
        self.patch_geom: list[tuple] = []
        self.rgb = DiskArray(d / "raw_rgb.u8", (112, 112, 3))
        self.rgb_of: list[int] = []                 # per detection: row in raw_rgb or -1

    def frame(self, ctx: Context, views) -> None:
        import cv2

        from ..faces.align import align, asd_patch

        f = views["full"]
        fd = self.det(f.rgb)
        self.times.append(f.t)
        self.det_start.append(len(self.scores))
        if not len(fd):
            return
        gray = cv2.cvtColor(f.rgb, cv2.COLOR_RGB2GRAY)
        keep_rgb = (len(self.times) - 1) % self.params["rgb_every"] == 0
        for b, s, lm in zip(fd.boxes, fd.scores, fd.landmarks, strict=True):
            a = align(f.rgb, lm)
            self.gray.append(cv2.cvtColor(a, cv2.COLOR_RGB2GRAY))
            p, geom = asd_patch(gray, b)
            self.patch.append(p)
            self.patch_geom.append(geom)
            self.rgb_of.append(self.rgb.append(a) if keep_rgb else -1)
            self.boxes.append(np.asarray(b, np.float64))
            self.scores.append(float(s))
            self.lms.append(np.asarray(lm, np.float64))

    def end(self, ctx: Context) -> dict:
        from ..faces.align import asd_from_patch
        from ..faces.clip import Detections, TrackCrops, _medfilt, save_crops, track_clip
        from ..faces.detectors import FaceDets

        for a in (self.gray, self.patch, self.rgb):
            a.close()
        d = ctx.stage_dir(self.name)
        nd = len(self.scores)
        bounds = [*self.det_start, nd]
        boxes = np.asarray(self.boxes).reshape(-1, 4)
        scores = np.asarray(self.scores)
        lms = np.asarray(self.lms).reshape(-1, 5, 2)
        frames = [FaceDets(boxes[a:b], scores[a:b], lms[a:b]) for a, b in zip(bounds[:-1], bounds[1:], strict=True)]
        fps = ctx.info.fps
        dets = Detections(ctx.video, fps, ctx.info.width, ctx.info.height, np.asarray(self.times), frames)
        dets.save(d / "faces.npz")
        cuts = [s["start"] for s in ctx.results["shots"]["shots"]]
        tracks = track_clip(dets, cuts)
        p = self.params
        tracks = [t for t in tracks if len(t) >= p["min_track_s"] * fps and np.mean(t.scores) >= p["min_score"]]

        G, P, R = self.gray.array(), self.patch.array(), self.rgb.array()
        crops = []
        for t in tracks:
            rows = [bounds[f] + di for f, di in zip(t.frames, t.det_index, strict=True)]
            b = np.asarray(t.boxes)
            cx, cy = (b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2
            s = np.maximum(b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]) / 2
            cx, cy, s = _medfilt(cx), _medfilt(cy), _medfilt(s)
            smooth = np.c_[cx - s, cy - s, cx + s, cy + s]
            asd = np.stack([asd_from_patch(np.asarray(P[r]), self.patch_geom[r], smooth[k])
                            for k, r in enumerate(rows)])
            with_rgb = [k for k, r in enumerate(rows) if self.rgb_of[r] >= 0]
            if not with_rgb:
                with_rgb = [len(rows) // 2]
            pick = sorted({with_rgb[i] for i in np.linspace(0, len(with_rgb) - 1,
                                                             min(p["max_rgb"], len(with_rgb))).round().astype(int)})
            rgb = []
            for k in pick:
                r = rows[k]
                if self.rgb_of[r] >= 0:
                    rgb.append(np.asarray(R[self.rgb_of[r]]))
                else:                                     # no colour crop stored: grey as colour
                    rgb.append(np.repeat(np.asarray(G[r])[:, :, None], 3, axis=2))
            crops.append(TrackCrops(t, np.asarray(G[rows]), asd, np.asarray(rgb, np.uint8),
                                    [t.frames[k] for k in pick]))
        save_crops(d / "crops.npz", crops)
        del G, P, R
        for name in ("raw_gray.u8", "raw_patch.u8", "raw_rgb.u8"):      # the kept tracks' crops are saved
            with __import__("contextlib").suppress(OSError):
                os.remove(d / name)
        return {"n_frames": len(self.times), "n_detections": nd, "n_tracks": len(crops), "cuts": len(cuts),
                "detections": "faces.npz", "crops": "crops.npz", "detector": p["detector"]}


def _crops(ctx: Context):
    from ..faces.clip import load_crops

    return load_crops(ctx.out / "stages" / "faces" / ctx.results["faces"]["crops"])


def _times(ctx: Context) -> np.ndarray:
    return np.load(ctx.out / "stages" / "shots" / "times.npy")


@register
class Speaking(Stage):
    """Who is talking, per face track: Light-ASD (default when its weights are
    present) or the from-scratch audio-visual score of Phase 3."""
    name = "speaking"
    version = "1"
    requires = ("faces",)
    defaults = {"asd": "auto", "batched": True}
    uses_backend = True

    def end(self, ctx: Context) -> dict:
        from ..faces.detectors import weight_path
        from ..speaker.audio import load_audio, vad

        kind = self.params["asd"]
        if kind == "auto":
            kind = "lightasd" if weight_path("light_asd_ava.model").exists() else "ours"
        crops = _crops(ctx)
        times = _times(ctx)
        fps = ctx.info.fps
        audio = load_audio(ctx.video)
        scores: dict[int, np.ndarray] = {}
        if audio is None:
            scores = {c.track.id: np.zeros(len(c.track)) for c in crops}
        elif kind == "lightasd":
            from .accel import lightasd

            m = lightasd(ctx.options)
            for i, c in enumerate(crops):
                scores[c.track.id] = m.score(c.asd, times[np.asarray(c.track.frames)], audio,
                                             batched=self.params["batched"])
                ctx.report(self.name, (i + 1) / max(len(crops), 1), f"track {i + 1}/{len(crops)}")
        else:
            from ..speaker.active import score_track

            v = vad(audio)
            scores = {c.track.id: score_track(c.gray, c.track.frames, times, v, fps)["score"] for c in crops}
        d = ctx.stage_dir(self.name)
        np.savez_compressed(d / "scores.npz", **{str(k): np.asarray(v, np.float32) for k, v in scores.items()})
        return {"asd": kind, "threshold": 0.25 if kind == "ours" else 0.0, "audio": audio is not None,
                "scores": "scores.npz", "n_tracks": len(scores)}


@register
class Characters(Stage):
    name = "characters"
    version = "1"
    requires = ("faces", "speaking")
    defaults = {"embedder": "sface", "embed_ckpt": None}
    uses_backend = True

    def end(self, ctx: Context) -> dict:
        from ..faces.characters import timeline_from_crops

        crops = _crops(ctx)
        sp = ctx.results["speaking"]
        z = np.load(ctx.out / "stages" / "speaking" / sp["scores"])
        speak = {c.track.id: z[str(c.track.id)] for c in crops}
        emb = None
        embedder, ckpt = self.params["embedder"], self.params["embed_ckpt"]
        if embedder == "ours":
            from .accel import embednet

            ckpt = ckpt or str(REPO / "p3_tracking" / "checkpoints" / "embed_triplet.pt")
            emb = embednet(ckpt, ctx.options)
        cuts = [s["start"] for s in ctx.results["shots"]["shots"]]
        r = timeline_from_crops(ctx.video, _times(ctx), ctx.info.fps, cuts, crops, embedder=embedder,
                                embed_ckpt=ckpt, asd=sp["asd"], speak_threshold=sp["threshold"], emb=emb,
                                speak=speak)
        d = ctx.stage_dir(self.name)
        (d / "faces").mkdir(exist_ok=True)
        for c, th in zip(r["characters"], r.pop("_thumbs"), strict=True):
            c["thumb"] = f"faces/{c['id']}.jpg"
            _save_jpeg(d / c["thumb"], th, 90)
        for t in r["tracks"]:
            t.pop("speaking_score", None)               # per frame; kept in speaking/scores.npz
        r["video"] = Path(ctx.video).name
        return r


# =========================================================================== Phase 7
@register
class Subtitles(Stage):
    name = "subtitles"
    version = "1"
    streams = {"sub": Stream(None, fps=8.0), "credits": Stream(None, fps=4.0)}
    defaults = {"reader": "auto", "ckpt": None, "credits": True}
    uses_backend = True
    transient = ("rec",)

    def setup(self, ctx: Context) -> None:
        from .accel import ocr_reader

        self.rec, self.reader_name = ocr_reader(self.params["reader"], self.params["ckpt"], ctx.options)
        if hasattr(self, "sub"):                       # restored from a checkpoint: re-attach the reader
            self._attach()

    def _attach(self) -> None:
        from ..ocr.credits import CreditsReader
        from ..ocr.subtitles import SubtitleReader

        self.sub.reader = SubtitleReader(self.rec.read)
        self.cred.reader = CreditsReader(self.rec.read)

    def begin(self, ctx: Context) -> None:
        from ..ocr.credits import CreditsReader
        from ..ocr.subtitles import SubtitleReader

        self.sub = SubtitleReader(self.rec.read).session(1 / 8.0)
        self.cred = CreditsReader(self.rec.read).session()

    def frame(self, ctx: Context, views) -> None:
        if "sub" in views:
            self.sub.push(views["sub"].t, views["sub"].rgb)
        if self.params["credits"] and "credits" in views:
            self.cred.push(views["credits"].t, views["credits"].rgb)

    def state(self) -> dict:
        st = super().state()
        for k in ("sub", "cred"):
            if k in st:
                obj = st[k]
                st[k] = (type(obj), {kk: v for kk, v in obj.__dict__.items() if kk != "reader"})
        return st

    def restore(self, state: dict) -> None:
        state = dict(state)
        for k in ("sub", "cred"):
            if k in state:
                cls, dct = state[k]
                obj = cls.__new__(cls)
                obj.__dict__.update(dct)
                state[k] = obj
        super().restore(state)

    def end(self, ctx: Context) -> dict:
        from ..ocr.metrics import cer
        from ..ocr.srt import write_srt

        cues = self.sub.finish()
        lines = self.cred.finish() if self.params["credits"] else []
        subs = [ln for c in cues for ln in [c.flat, *c.text.split("\n")]]
        lines = [ln for ln in lines if not any(cer(s, ln["text"]) < 0.3 for s in subs)]
        write_srt(cues, ctx.stage_dir(self.name) / "subtitles.srt")
        return {"reader": self.reader_name, "srt": "subtitles.srt",
                "cues": [{"t0": round(c.start, 3), "t1": round(c.end, 3), "text": c.text} for c in cues],
                "credits": lines}


# =========================================================================== Phase 8
def _keyframes_for(store: FrameStore, start: int, end: int, n: int) -> list[int]:
    from ..search.shotindex import keyframe_indices

    out = []
    for fi in keyframe_indices(start, end, n):
        pos = store.nearest(fi, start, end)
        if pos not in out:
            out.append(pos)
    return out


@register
class Search(Stage):
    """Phase 8's index, built from the stored key frames instead of a new decode."""
    name = "search"
    version = "1"
    requires = ("shots", "frames")
    defaults = {"encoder": "openclip", "max_segment_s": 6.0, "keyframes": 3, "thumb_width": 320, "tags": True,
                "batch": 32}
    uses_backend = True

    def end(self, ctx: Context) -> dict:
        from PIL import Image

        from ..search.rag import TAG_VOCAB, zero_shot_tags
        from ..search.shotindex import IndexParams, ShotIndex, plan_segments
        from ..search.vecindex import normalize
        from .accel import clip_encoder

        p = self.params
        r = ctx.results["shots"]
        store = FrameStore(ctx)
        enc = clip_encoder(p["encoder"], ctx.options)
        root = ctx.stage_dir(self.name) / "index"
        (root / "frames").mkdir(parents=True, exist_ok=True)
        segs = plan_segments(r["shots"], p["max_segment_s"], r["fps"])
        docs, kf_pos, kf_owner = [], [], []
        for di, s in enumerate(segs):
            fr = []
            for j, pos in enumerate(_keyframes_for(store, s["start"], s["end"], p["keyframes"])):
                name = f"frames/{di:05d}_{j}.jpg"
                rgb = store.load(pos)
                th = Image.fromarray(rgb)
                th.resize((p["thumb_width"], max(1, round(rgb.shape[0] * p["thumb_width"] / rgb.shape[1])))) \
                  .save(root / name, quality=85)
                fr.append({"t": float(store.t[pos]), "file": name, "frame": int(store.idx[pos])})
                kf_pos.append(pos)
                kf_owner.append(di)
            docs.append({"id": di, "clip": ctx.clip, "video": ctx.video, "shot": s["shot"], "part": s["part"],
                         "n_parts": s["n_parts"], "t0": s["t0"], "t1": s["t1"], "frames": fr,
                         "tags": [], "subtitles": [], "characters": []})
        uniq = sorted(set(kf_pos))
        vec_of = {}
        for b in range(0, len(uniq), p["batch"]):
            chunk = uniq[b:b + p["batch"]]
            for pos, v in zip(chunk, enc.encode_images([store.load(q) for q in chunk]), strict=True):
                vec_of[pos] = v
            ctx.report(self.name, min(1.0, (b + len(chunk)) / max(len(uniq), 1)), "embedding key frames")
        dim = len(next(iter(vec_of.values()))) if vec_of else 512
        kf = np.asarray([vec_of[q] for q in kf_pos], np.float32).reshape(-1, dim)
        owner = np.asarray(kf_owner, np.int64)
        vec = np.zeros((len(docs), dim), np.float32)
        np.add.at(vec, owner, kf)
        vec = normalize(vec)
        if p["tags"] and len(docs):
            for d, tags in zip(docs, zero_shot_tags(vec, enc, TAG_VOCAB, 5), strict=True):
                d["tags"] = tags
        ip = IndexParams(max_segment_s=p["max_segment_s"], keyframes=p["keyframes"], thumb_width=p["thumb_width"])
        meta = {"encoder": enc.name, "video": ctx.video, "clip": ctx.clip, "fps": r["fps"],
                "n_shots": len(r["shots"]), "n_docs": len(docs), "n_keyframes": len(kf_pos),
                "params": ip.__dict__, "built_by": "cinescope.pipeline"}
        ShotIndex(docs, vec, meta, root, kf, owner).save(root)
        return {"index": "index", "encoder": enc.name, "spec": p["encoder"], "n_docs": len(docs), "n_keyframes": len(kf_pos),
                "n_encoded": len(uniq)}


# =========================================================================== Phase 1
@register
class Framing(Stage):
    """Shot framing (close-up ... long shot) per shot.

    With a Phase 1 checkpoint (`ckpt`, default p1_cnn/checkpoints/pretrained.pt)
    the ResNet-18 classifies three stored key frames per shot (25/50/75%) and
    averages them, as FramingClassifier does. Without one -- the checkpoint is
    trained on the owner's own videos and is not in the repo -- the label comes
    from Phase 1's automatic teacher: zero-shot CLIP prompts on the key-frame
    embeddings the search stage already computed (no extra network pass) fused
    with the largest face's height from the faces stage.
    """
    name = "framing"
    version = "1"
    requires = ("shots", "frames", "search", "faces")
    defaults = {"ckpt": None, "w_face": 1.0}
    uses_backend = True

    def end(self, ctx: Context) -> dict:
        ckpt = self.params["ckpt"] or str(REPO / "p1_cnn" / "checkpoints" / "pretrained.pt")
        if Path(ckpt).exists():
            return self._resnet(ctx, ckpt)
        return self._zero_shot(ctx)

    def _resnet(self, ctx: Context, ckpt: str) -> dict:
        from ..framing import SCALE_NAMES
        from .accel import framing_classifier

        clf = framing_classifier(ckpt, ctx.options)
        store = FrameStore(ctx)
        out = []
        for s in ctx.results["shots"]["shots"]:
            span = s["end"] - s["start"]
            pos = sorted({store.nearest(s["start"] + int(round(q * span)), s["start"], s["end"])
                          for q in (0.25, 0.5, 0.75)})
            p = clf.predict([store.load(q) for q in pos]).mean(0)
            k = int(p.argmax())
            out.append({"label": clf.classes[k], "name": SCALE_NAMES.get(clf.classes[k], clf.classes[k]),
                        "confidence": round(float(p[k]), 3),
                        "probs": {c: round(float(v), 3) for c, v in zip(clf.classes, p, strict=True)}})
        return {"method": "resnet18", "checkpoint": Path(ckpt).name, "shots": out}

    def _zero_shot(self, ctx: Context) -> dict:
        from ..faces.clip import Detections
        from ..framing import SCALE_NAMES, SCALES
        from ..framing.teacher import PROMPTS, class_scores, face_prior, face_ratio, fuse, softmax
        from ..search.shotindex import ShotIndex
        from .accel import clip_encoder

        ix = ShotIndex.load(ctx.out / "stages" / "search" / "index")
        spec = ctx.results["search"].get("spec", "openclip")
        if not spec.startswith("openclip"):            # the zero-shot prompts need CLIP's text tower
            return {"method": "none", "shots": []}
        enc = clip_encoder(spec, ctx.options)
        prompts = [(c, t) for c in SCALES for t in PROMPTS[c]]
        tvec = enc.encode_texts([t for _, t in prompts])
        dets = Detections.load(ctx.out / "stages" / "faces" / ctx.results["faces"]["detections"])
        h = ctx.info.height
        by_shot: dict[int, list[int]] = {}
        for i, owner in enumerate(ix.kf_owner):
            by_shot.setdefault(ix.docs[int(owner)]["shot"], []).append(i)
        frame_of = [f["frame"] for d in ix.docs for f in d["frames"]]
        out = []
        for s in ctx.results["shots"]["shots"]:
            rows = by_shot.get(s["index"], [])
            if not rows:
                out.append({"label": None, "name": None, "confidence": 0.0})
                continue
            logits = 100.0 * ix.kf_vectors[rows] @ tvec.T          # CLIP's logit scale
            p_text = softmax(class_scores(logits, [c for c, _ in prompts])).mean(0)
            ratios = [face_ratio(dets.frames[f].boxes, dets.frames[f].scores, h)
                      for f in (frame_of[i] for i in rows) if f < len(dets.frames)]
            ratios = [r for r in ratios if r is not None]
            p = fuse(p_text, face_prior(float(np.median(ratios))) if ratios else None, self.params["w_face"])
            k = int(np.argmax(p))
            out.append({"label": SCALES[k], "name": SCALE_NAMES[SCALES[k]], "confidence": round(float(p[k]), 3),
                        "probs": {c: round(float(v), 3) for c, v in zip(SCALES, p, strict=True)},
                        "face_ratio": round(float(np.median(ratios)), 3) if ratios else None})
        return {"method": "zero-shot CLIP + face size (Phase 1 teacher)", "shots": out}


# =========================================================================== Phase 5
@register
class CamPath(Stage):
    """Camera path of one shot (the longest, or `shot`): Phase 5's SfM, or the
    rotation-only path when the camera does not translate. It decodes its one
    shot itself (a seek, then ~120 frames): the shot is only known after the pass."""
    name = "campath"
    version = "1"
    requires = ("shots", "camera")
    defaults = {"shot": None, "width": 480, "stride": 2, "max_frames": 120, "max_points": 3000}

    def end(self, ctx: Context) -> dict:
        from ..geometry.campath import (
            path_from_reconstruction,
            path_from_rotations,
            rotation_only_path,
        )
        from ..geometry.shot import default_k, load_shot, reconstruct

        p = self.params
        shots = ctx.results["shots"]["shots"]
        if not shots:
            return {"mode": "none"}
        if p["shot"] is not None:
            s = shots[p["shot"]]
        else:
            moves = ctx.results["camera"]["moves"]
            moving = [x for x, m in zip(shots, moves, strict=True) if m["label"] not in ("static", "unknown")]
            s = max(moving or shots, key=lambda x: x["end"] - x["start"])
        if s["end"] - s["start"] < 8:
            return {"mode": "none", "shot": s["index"]}
        sf = load_shot(ctx.video, s["index"], width=p["width"], stride=p["stride"], max_frames=p["max_frames"],
                       start=s["start"], end=s["end"])
        rec, tracks = reconstruct(sf)
        h, w = sf.frames.shape[1:3]
        if rec.ok:
            mode, path = "sfm", path_from_reconstruction(rec)
            pts = np.stack(list(rec.points.values())) if rec.points else np.zeros((0, 3))
        else:
            mode, pts = "rotation-only", np.zeros((0, 3))
            path = path_from_rotations(rotation_only_path(tracks, default_k(w, h)))
        if len(pts) > p["max_points"]:
            pts = pts[np.random.default_rng(0).choice(len(pts), p["max_points"], replace=False)]
        fps = ctx.info.fps
        frames = [{"frame": int(q["frame"]), "t": round(q["frame"] / fps, 3),
                   "centre": np.round(q["centre"], 5).tolist(), "R": np.round(q["R"], 5).tolist(),
                   "pan_deg": round(q["pan_deg"], 3), "tilt_deg": round(q["tilt_deg"], 3),
                   "roll_deg": round(q["roll_deg"], 3)} for q in path]
        doc = {"mode": mode, "shot": s["index"], "t0": s["t0"], "t1": s["t1"], "image_size": [w, h],
               "frames": frames, "points": np.round(pts, 4).tolist()}
        (ctx.stage_dir(self.name) / "path.json").write_text(json.dumps(doc), encoding="utf-8")
        return {"mode": mode, "shot": s["index"], "n_frames": len(frames), "n_points": len(pts),
                "file": "path.json"}


# =========================================================================== Phase 6
@register
class Pose(Stage):
    """People, poses and action events per shot (Phase 6's breakdown, pose half).

    YOLOv8n-pose (or Phase 6's own SimpleBaseline on YOLO boxes, when its
    checkpoint exists) on frames sampled at `fps` during the one decoding
    pass, batched; end() tracks people within each shot and detects strikes,
    falls and runs (cinescope/pose/events.py)."""
    name = "pose"
    version = "1"
    requires = ("shots",)
    streams = {"pose": Stream(width=960, fps=4.0)}
    defaults = {"model": "auto", "ckpt": None, "conf": 0.4, "batch": 8}
    transient = ("det", "net", "pending")

    def setup(self, ctx: Context) -> None:
        from ..pose.yolo import YoloPose

        ck = Path(self.params["ckpt"] or REPO / "outputs" / "p6" / "pose_sb.pt")
        kind = self.params["model"]
        if kind == "auto":
            kind = "ours" if ck.exists() else "yolo"
        self.kind = kind
        self.det = YoloPose(pose=kind == "yolo", conf=self.params["conf"])
        self.net = None
        if kind == "ours":
            from ..pose.simplebaseline import load_checkpoint

            self.net, c = load_checkpoint(str(ck))
            self.input_hw = tuple(c.get("input_hw", (128, 96)))
        self.pending = []

    def begin(self, ctx: Context) -> None:
        self.t: list[float] = []
        self.frame_idx: list[int] = []
        self.people: list[dict] = []        # per sample: boxes (P,4), kpts (P,17,3), in full-frame pixels
        self.height = 1

    def _flush(self) -> None:
        if not self.pending:
            return
        frames = [f.rgb for f in self.pending]
        res = self.det(frames)
        if self.kind == "ours":
            from ..pose.simplebaseline import predict

            for f, p in zip(frames, res, strict=True):
                if len(p):
                    p.kpts = predict(self.net, f, p.boxes, self.input_hw)
        for f, p in zip(self.pending, res, strict=True):
            self.height = f.rgb.shape[0]
            self.t.append(float(f.t))
            self.frame_idx.append(int(f.index))
            self.people.append({"boxes": p.boxes, "kpts": p.kpts})
        self.pending = []

    def frame(self, ctx: Context, views) -> None:
        self.pending.append(views["pose"])
        if len(self.pending) >= self.params["batch"]:
            self._flush()

    def state(self) -> dict:
        self._flush()                       # a checkpoint holds no unprocessed frames
        return super().state()

    def end(self, ctx: Context) -> dict:
        from ..pose.events import EventParams, detect_events
        from ..pose.track import track_people

        self._flush()
        fi = np.asarray(self.frame_idx)
        per_shot, timeline = [], []
        for s in ctx.results["shots"]["shots"]:
            idx = np.flatnonzero((fi >= s["start"]) & (fi <= s["end"]))
            tracks = track_people([self.t[j] for j in idx], [self.people[j]["boxes"] for j in idx],
                                  [self.people[j]["kpts"] for j in idx])
            events = [e.to_dict() for e in detect_events(tracks, EventParams())]
            counts = [len(self.people[j]["boxes"]) for j in idx]
            heights = [float((b[3] - b[1]) / self.height) for j in idx for b in self.people[j]["boxes"]]
            per_shot.append({"max_people": int(max(counts)) if counts else 0,
                             "mean_people": round(float(np.mean(counts)), 2) if counts else 0.0,
                             "tracks": len(tracks),
                             "median_height_frac": round(float(np.median(heights)), 3) if heights else None,
                             "events": events})
            timeline += [{"t0": e["t0"], "t1": e["t1"], "label": e["type"]} for e in events]
        np.savez_compressed(ctx.stage_dir(self.name) / "people.npz", t=np.asarray(self.t),
                            frame=fi, n=np.asarray([len(p["boxes"]) for p in self.people]),
                            boxes=np.concatenate([p["boxes"] for p in self.people] or [np.zeros((0, 4))]),
                            kpts=np.concatenate([p["kpts"] for p in self.people] or [np.zeros((0, 17, 3))]))
        return {"model": self.kind, "fps": self.streams["pose"].fps, "n_samples": len(self.t),
                "people": "people.npz", "per_shot": per_shot, "timeline": sorted(timeline, key=lambda e: e["t0"])}


@register
class Scenes(Stage):
    """Scene type per shot (Phase 6's CLIP teacher: dialogue / action /
    establishing / crowd / text), from the key-frame vectors the search stage
    already computed: only the prompts go through the text tower."""
    name = "scenes"
    version = "1"
    requires = ("shots", "search")
    uses_backend = True

    def end(self, ctx: Context) -> dict:
        from ..search.shotindex import ShotIndex
        from ..video.scenes import CLASSES, PROMPTS, softmax
        from .accel import clip_encoder

        spec = ctx.results["search"].get("spec", "openclip")
        if not spec.startswith("openclip"):
            return {"method": "none", "per_shot": [], "timeline": []}
        ix = ShotIndex.load(ctx.out / "stages" / "search" / "index")
        enc = clip_encoder(spec, ctx.options)
        owner = [c for c in CLASSES for _ in PROMPTS[c]]
        tvec = enc.encode_texts([p for c in CLASSES for p in PROMPTS[c]])
        by_shot: dict[int, list[int]] = {}
        for i, o in enumerate(ix.kf_owner):
            by_shot.setdefault(ix.docs[int(o)]["shot"], []).append(i)
        per_shot, timeline = [], []
        for s in ctx.results["shots"]["shots"]:
            rows = by_shot.get(s["index"], [])
            if not rows:
                per_shot.append(None)
                continue
            lg = 100.0 * ix.kf_vectors[rows] @ tvec.T
            cls = np.stack([lg[:, [j for j, c in enumerate(owner) if c == k]].mean(1) for k in CLASSES], 1)
            p = softmax(cls).mean(0)
            k = int(p.argmax())
            per_shot.append({"type": CLASSES[k], "probs": {c: round(float(v), 3) for c, v in zip(CLASSES, p, strict=True)}})
            if timeline and timeline[-1]["label"] == CLASSES[k] and abs(timeline[-1]["t1"] - s["t0"]) < 0.1:
                timeline[-1]["t1"] = s["t1"]
            else:
                timeline.append({"t0": s["t0"], "t1": s["t1"], "label": CLASSES[k]})
        return {"method": "zero-shot CLIP prompts on key frames (Phase 6 teacher)", "per_shot": per_shot,
                "timeline": timeline}


# =========================================================================== Phase 4
@register
class Cutouts(Stage):
    """Phase 4: a matted RGBA cut-out of the largest person in the middle key
    frame of each shot. Opt-in (`--stages cutouts`): GrabCut with our own
    max-flow plus closed-form matting takes seconds per frame on a CPU. The web
    app also cuts out any frame on demand (/api/clips/{clip}/cutout)."""
    name = "cutouts"
    version = "1"
    requires = ("shots", "frames")
    defaults = {"method": "grabcut", "matting": True, "min_height": 0.2, "width": 640}
    default = False

    def end(self, ctx: Context) -> dict:
        from PIL import Image

        from ..segment.models import PersonDetector
        from ..segment.pipeline import get_segmenter, largest_person, rgba_cutout

        p = self.params
        store = FrameStore(ctx)
        det = PersonDetector(threads=ctx.options.threads)
        seg = get_segmenter(p["method"], ctx.options.threads, detector=det)
        d = ctx.stage_dir(self.name)
        shots = ctx.results["shots"]["shots"]
        per_shot = []
        for i, s in enumerate(shots):
            ctx.report(self.name, i / max(len(shots), 1), f"shot {i + 1}/{len(shots)}")
            if not len(store.idx):
                per_shot.append(None)
                continue
            pos = store.nearest((s["start"] + s["end"]) // 2, s["start"], s["end"])
            rgb = store.load(pos)
            box = largest_person(rgb, det)
            if box is None or (box[3] - box[1]) < p["min_height"] * rgb.shape[0]:
                per_shot.append(None)
                continue
            mask = seg(rgb, box[None])[0]
            if not mask.any():
                per_shot.append(None)
                continue
            rgba, (x0, y0) = rgba_cutout(rgb, mask, matting=p["matting"])
            name = f"shot_{s['index']:05d}.png"
            Image.fromarray(rgba, "RGBA").save(d / name)
            per_shot.append({"file": f"stages/cutouts/{name}", "t": float(store.t[pos]), "xy": [x0, y0],
                             "box": np.round(box, 1).tolist()})
        return {"method": p["method"], "per_shot": per_shot}


# =========================================================================== film language
def _shot_frames(th: np.ndarray, s: dict, n: int) -> np.ndarray:
    """Up to n thumbnails spread evenly over a shot."""
    a, b = int(s["start"]), int(min(s["end"], len(th) - 1))
    if b < a:
        return np.zeros((0, *th.shape[1:]), np.uint8)
    idx = np.unique(np.linspace(a, b, min(n, b - a + 1)).round().astype(int))
    return np.asarray(th[idx])


@register
class Look(Stage):
    """Lighting and grade per shot (cinescope.cinema.look): exposure key, contrast,
    saturation, colour temperature, teal-and-orange, symmetry; and the clip's
    overall look. Reads the shot stage's thumbnails, so it needs no decoding."""
    name = "look"
    version = "1"
    requires = ("shots",)
    defaults = {"frames_per_shot": 12}

    def end(self, ctx: Context) -> dict:
        from ..cinema.look import film_look, key_timeline, shot_look

        shots = ctx.results["shots"]["shots"]
        th = _thumbs(ctx)
        per_shot = []
        for s in shots:
            fr = _shot_frames(th, s, self.params["frames_per_shot"])
            per_shot.append(shot_look(fr) if len(fr) else None)
        return {"per_shot": per_shot, "summary": film_look(per_shot, [s["duration_s"] for s in shots]),
                "timeline": key_timeline(shots, per_shot)}


@register
class Editing(Stage):
    """Editing grammar (cinescope.cinema.editing): recurring camera set-ups, shot/reverse-shot exchanges, montages and long takes. Reads the shot
    stage's thumbnails, so it needs no decoding."""
    name = "editing"
    version = "1"
    requires = ("shots",)
    defaults = {"frames_per_shot": 5}

    def end(self, ctx: Context) -> dict:
        from ..cinema.editing import analyse, shot_descriptor

        shots = ctx.results["shots"]["shots"]
        th = _thumbs(ctx)
        descs = []
        for s in shots:
            # the middle of the shot: its first and last frames may be mid-dissolve
            n = s["end"] - s["start"] + 1
            mid = {**s, "start": s["start"] + n // 5, "end": s["end"] - n // 5}
            fr = _shot_frames(th, mid, self.params["frames_per_shot"])
            descs.append(shot_descriptor(fr if len(fr) else np.zeros((1, 6, 8, 3), np.uint8)))
        return analyse(shots, descs)
