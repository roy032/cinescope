"""Index any clip for plain-English search: shots -> key frames -> embeddings -> index on disk.

    from cinescope.search.encoders import get_encoder
    from cinescope.search.shotindex import build_clip_index, ShotIndex
    idx = build_clip_index("trailers/clip.mp4", get_encoder("openclip"), "outputs/clip/index")
    hits = ShotIndex.load("outputs/clip/index").search_text("a man walks away from an explosion", enc, k=5)

The unit of search is a *segment*: a Phase 0 shot (cinescope/shots/pipeline.py),
split into equal parts no longer than `max_segment_s` (6 s by default). Film shots
are mostly shorter than that, so a segment is usually exactly one shot; a 2-minute
static surveillance take becomes 24 segments, so "when does X happen?" still has
an answer finer than "somewhere in these two minutes".

Each segment gets `keyframes` frames (3, at 1/6, 1/2 and 5/6 of its span); its
vector is the renormalised mean of their embeddings. One vector per segment keeps
the ANN index simple; the key-frame vectors are saved too.

On disk (outputs/<clip>/index/):
    docs.json       one document per segment (schema below)
    vectors.npy     (n_docs, D) float32, L2-normalised
    keyframes.npy   (n_keyframes, D) float32 and keyframe_owner.npy
    frames/         the key frames as small JPEGs (for contact sheets and the chat)
    shots.json      the Phase 0 analysis the segments came from
    meta.json       encoder name, parameters, timings

Document schema — the hook points later phases fill:
    id, clip, video, shot, part, n_parts, t0, t1, frames: [{t, file}],
    tags:       [{tag, score}]      open_clip zero-shot tags (rag.TAG_VOCAB)
    subtitles:  [{t0, t1, text}]    Phase 7 OCR or an .srt file (attach_subtitles)
    characters: [{name, seconds}]   Phase 3 character timeline (attach_characters)
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .vecindex import FlatIndex, IndexSpec, normalize


@dataclass
class IndexParams:
    max_segment_s: float = 6.0
    keyframes: int = 3
    decode_width: int = 640      # key frames are decoded at this width (CLIP sees 224 px crops)
    thumb_width: int = 320
    n_tags: int = 5


def plan_segments(shots: list[dict], max_segment_s: float, fps: float) -> list[dict]:
    """Split each shot into ceil(duration / max_segment_s) equal parts (frame ranges inclusive)."""
    segs = []
    for s in shots:
        dur = s["t1"] - s["t0"]
        n = max(1, int(np.ceil(dur / max_segment_s - 1e-9)))
        edges = np.linspace(s["start"], s["end"] + 1, n + 1).round().astype(int)
        for p in range(n):
            a, b = int(edges[p]), int(edges[p + 1]) - 1
            if b < a:
                continue
            t0 = s["t0"] + (a - s["start"]) / fps
            t1 = s["t1"] if p == n - 1 else s["t0"] + (b + 1 - s["start"]) / fps
            segs.append({"shot": s["index"], "part": p, "n_parts": n, "start": a, "end": b,
                         "t0": round(float(t0), 3), "t1": round(float(t1), 3)})
    return segs


def keyframe_indices(start: int, end: int, n: int) -> list[int]:
    length = end - start + 1
    return sorted({start + min(length - 1, int((i + 0.5) * length / n)) for i in range(n)})


# --------------------------------------------------------------------------- hooks for later phases
def attach_subtitles(docs: list[dict], cues: list[dict]) -> None:
    """Give every document the subtitle cues ({t0, t1, text}) that overlap its span."""
    for d in docs:
        d["subtitles"] = [c for c in cues if c["t0"] < d["t1"] and c["t1"] > d["t0"]]


def attach_characters(docs: list[dict], appearances: list[dict]) -> None:
    """appearances: [{name, t0, t1}] (e.g. Phase 3 face tracks). Each document lists
    the characters on screen during it, with seconds of overlap, most first."""
    for d in docs:
        acc: dict[str, float] = {}
        for a in appearances:
            ov = min(d["t1"], a["t1"]) - max(d["t0"], a["t0"])
            if ov > 0:
                acc[a["name"]] = acc.get(a["name"], 0.0) + ov
        d["characters"] = [{"name": k, "seconds": round(v, 2)} for k, v in sorted(acc.items(), key=lambda kv: -kv[1])]


def parse_srt(text: str) -> list[dict]:
    """Minimal SubRip parser: [{t0, t1, text}]."""
    import re

    ts = re.compile(r"(\d+):(\d+):(\d+)[,.](\d+)\s*-->\s*(\d+):(\d+):(\d+)[,.](\d+)")
    cues = []
    for block in re.split(r"\n\s*\n", text.replace("\r\n", "\n").strip()):
        lines = block.strip().split("\n")
        for i, line in enumerate(lines):
            m = ts.search(line)
            if m:
                g = [int(x) for x in m.groups()]
                t0 = g[0] * 3600 + g[1] * 60 + g[2] + g[3] / 1000
                t1 = g[4] * 3600 + g[5] * 60 + g[6] + g[7] / 1000
                body = " ".join(x.strip() for x in lines[i + 1:] if x.strip())
                cues.append({"t0": t0, "t1": t1, "text": re.sub(r"<[^>]+>", "", body)})
                break
    return cues


# --------------------------------------------------------------------------- the index
class ShotIndex:
    def __init__(self, docs: list[dict], vectors: np.ndarray, meta: dict, root: Path | None = None,
                 kf_vectors: np.ndarray | None = None, kf_owner: np.ndarray | None = None) -> None:
        self.docs, self.vectors, self.meta, self.root = docs, vectors.astype(np.float32), meta, root
        self.kf_vectors, self.kf_owner = kf_vectors, kf_owner
        self._ann: dict[str, object] = {}

    def __len__(self) -> int:
        return len(self.docs)

    @property
    def encoder_name(self) -> str:
        return self.meta.get("encoder", "?")

    # -- persistence
    def save(self, root: str | Path) -> None:
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        (root / "docs.json").write_text(json.dumps(self.docs, indent=1), encoding="utf-8")
        (root / "meta.json").write_text(json.dumps(self.meta, indent=1), encoding="utf-8")
        np.save(root / "vectors.npy", self.vectors)
        if self.kf_vectors is not None and self.kf_owner is not None:
            np.save(root / "keyframes.npy", self.kf_vectors)
            np.save(root / "keyframe_owner.npy", self.kf_owner)
        self.root = root

    def save_docs(self) -> None:
        """Rewrite docs.json only (after attach_subtitles / attach_characters)."""
        assert self.root is not None
        (self.root / "docs.json").write_text(json.dumps(self.docs, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, root: str | Path) -> ShotIndex:
        root = Path(root)
        kf = root / "keyframes.npy"
        return cls(json.loads((root / "docs.json").read_text(encoding="utf-8")), np.load(root / "vectors.npy"),
                   json.loads((root / "meta.json").read_text(encoding="utf-8")), root,
                   np.load(kf) if kf.exists() else None,
                   np.load(root / "keyframe_owner.npy") if kf.exists() else None)

    @classmethod
    def merge(cls, indexes: list[ShotIndex]) -> ShotIndex:
        """One library index over several clips (all built with the same encoder)."""
        names = {i.encoder_name for i in indexes}
        if len(names) != 1:
            raise ValueError(f"indexes were built with different encoders: {names}")
        docs, vecs = [], []
        for ix in indexes:
            for d in ix.docs:
                docs.append({**d, "id": len(docs), "index_root": str(ix.root) if ix.root else None})
            vecs.append(ix.vectors)
        return cls(docs, np.concatenate(vecs), {"encoder": names.pop(), "merged": len(indexes)})

    # -- search
    def ann(self, spec: IndexSpec | None = None, train: np.ndarray | None = None):
        """The vector index over the documents (built once per spec). `train`: vectors
        to learn IVF/PQ codebooks from — a larger corpus than one clip; without it the
        documents themselves are used and the index shrinks its parameters to fit."""
        spec = spec or IndexSpec("flat")
        key = repr(spec)
        if key not in self._ann:
            if spec.kind != "flat":
                spec = fit_spec(spec, len(train) if train is not None else len(self.vectors))
            idx = spec.build(self.vectors.shape[1]) if spec.kind != "flat" else FlatIndex(self.vectors.shape[1])
            idx.train(train if train is not None else self.vectors)
            idx.add(self.vectors)
            self._ann[key] = idx
        return self._ann[key]

    def search_vectors(self, q: np.ndarray, k: int = 5, spec: IndexSpec | None = None,
                       train: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        return self.ann(spec, train).search(normalize(np.atleast_2d(q)), min(k, len(self.docs)))

    def search_text(self, query: str, encoder, k: int = 5, spec: IndexSpec | None = None,
                    train: np.ndarray | None = None) -> list[dict]:
        if encoder.name != self.encoder_name:
            raise ValueError(f"index built with {self.encoder_name}, query encoder is {encoder.name}")
        D, I = self.search_vectors(encoder.encode_texts([query]), k, spec, train)  # noqa: E741
        return [{**self.docs[int(i)], "score": round(float(s), 4), "rank": r + 1}
                for r, (s, i) in enumerate(zip(D[0], I[0], strict=True)) if i >= 0]

    def frame_path(self, doc: dict, j: int = 0) -> Path | None:
        root = Path(doc["index_root"]) if doc.get("index_root") else self.root
        if root is None or not doc.get("frames"):
            return None
        return root / doc["frames"][min(j, len(doc["frames"]) - 1)]["file"]


def fit_spec(spec: IndexSpec, n: int) -> IndexSpec:
    """Shrink IVF/PQ parameters for small collections: k-means needs more points than
    centroids (FAISS warns below 39 per centroid), so nlist <= n/39 and 2^nbits <= n/4."""
    nlist = max(1, min(spec.nlist, n // 39))
    nbits = int(max(1, min(spec.nbits, np.floor(np.log2(max(2, n // 4))))))
    return IndexSpec(spec.kind, nlist, min(spec.nprobe, nlist), spec.m, nbits, spec.refine, spec.coarse)


# --------------------------------------------------------------------------- building
def build_clip_index(video: str | Path, encoder, out_dir: str | Path, params: IndexParams | None = None,
                     tag_vocab: list[str] | None = None, analysis=None, log=print) -> ShotIndex:
    from PIL import Image

    from ..io.video import VideoReader
    from ..shots.pipeline import analyse

    p = params or IndexParams()
    out = Path(out_dir)
    (out / "frames").mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    res = analysis or analyse(video, with_palettes=False)
    res.save(out / "shots.json")
    shots = [s.to_dict() for s in res.shots]
    t_shots = time.perf_counter() - t0
    segs = plan_segments(shots, p.max_segment_s, res.fps)
    want: dict[int, list[tuple[int, int]]] = {}
    for di, s in enumerate(segs):
        for j, fi in enumerate(keyframe_indices(s["start"], s["end"], p.keyframes)):
            want.setdefault(fi, []).append((di, j))
    frames: dict[tuple[int, int], tuple[float, np.ndarray]] = {}
    last = max(want) if want else -1
    with VideoReader(video, width=p.decode_width) as vr:
        for f in vr.frames():
            if f.index in want:
                for key in want[f.index]:
                    frames[key] = (f.t, f.rgb)
            if f.index >= last:
                break
    t_decode = time.perf_counter() - t0 - t_shots

    clip = Path(video).stem
    docs, kf_rgb, kf_owner = [], [], []
    for di, s in enumerate(segs):
        fr = []
        for j in range(p.keyframes):
            if (di, j) not in frames:
                continue
            t, rgb = frames[(di, j)]
            name = f"frames/{di:05d}_{j}.jpg"
            th = Image.fromarray(rgb)
            th = th.resize((p.thumb_width, max(1, round(rgb.shape[0] * p.thumb_width / rgb.shape[1]))))
            th.save(out / name, quality=85)
            fr.append({"t": round(float(t), 3), "file": name})
            kf_rgb.append(rgb)
            kf_owner.append(di)
        docs.append({"id": di, "clip": clip, "video": str(video), "shot": s["shot"], "part": s["part"],
                     "n_parts": s["n_parts"], "t0": s["t0"], "t1": s["t1"], "frames": fr,
                     "tags": [], "subtitles": [], "characters": []})
    kf = encoder.encode_images(kf_rgb)
    owner = np.asarray(kf_owner, np.int64)
    vec = np.zeros((len(docs), kf.shape[1]), np.float32)
    np.add.at(vec, owner, kf)
    vec = normalize(vec)
    t_embed = time.perf_counter() - t0 - t_shots - t_decode

    if tag_vocab:
        from .rag import zero_shot_tags

        for d, tags in zip(docs, zero_shot_tags(vec, encoder, tag_vocab, p.n_tags), strict=True):
            d["tags"] = tags
    meta = {"encoder": encoder.name, "video": str(video), "clip": clip, "fps": res.fps,
            "n_shots": len(shots), "n_docs": len(docs), "n_keyframes": len(kf_rgb), "params": asdict(p),
            "timing_s": {"shots": round(t_shots, 2), "decode": round(t_decode, 2), "embed": round(t_embed, 2),
                         "total": round(time.perf_counter() - t0, 2)}}
    ix = ShotIndex(docs, vec, meta, out, kf.astype(np.float32), owner)
    ix.save(out)
    if log:
        log(f"{clip}: {len(shots)} shots -> {len(docs)} segments, {len(kf_rgb)} key frames, "
            f"{encoder.name}, {meta['timing_s']}")
    return ix


def index_dir(out_root: str | Path, video: str | Path, encoder_name: str) -> Path:
    """outputs/<clip>/index for open_clip; outputs/<clip>/index_<encoder> for others."""
    clip = Path(video).stem
    if encoder_name.startswith("openclip:ViT-B-32-quickgelu:vit_b_32-quickgelu-laion400m_e32"):
        return Path(out_root) / clip / "index"
    safe = "".join(ch if ch.isalnum() else "_" for ch in encoder_name)[:60]
    return Path(out_root) / clip / f"index_{safe}"
