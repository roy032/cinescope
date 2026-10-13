"""Score scene search on the hand-written queries in p8_search/labels/queries.json.

    python p8_search/eval_search.py --trailers trailers \
        --encoders openclip scratch:p8_search/checkpoints/dual_frames.pt \
        --train-frames outputs/p8_train_frames.npz

For every encoder x index type, two settings:
  per-clip   each query searches only its own clip's segments (what search.py does)
  library    each query searches the segments of all clips pooled together — harder:
             a "man in a navy t-shirt" must also beat every other clip's men
Index types: our exact search, our IVF-PQ, FAISS IndexFlatIP and FAISS IndexIVFPQ.
IVF/PQ codebooks are learned from the open_clip (or scratch) embeddings of the
*training* clips' frames (caption_frames.py), never from the evaluated clips — the
realistic setting of training codebooks once and adding new films to them. With
only ~2.4k training vectors: nlist=32, M=16 sub-quantisers x 8 bits, nprobe=8,
fixed before any run.

Reports Recall@1/5 and MRR on the test split; the dev clip (vtest) separately.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cinescope.search.encoders import get_encoder  # noqa: E402
from cinescope.search.metrics import first_relevant_rank, summarise  # noqa: E402
from cinescope.search.rag import TAG_VOCAB  # noqa: E402
from cinescope.search.shotindex import ShotIndex, build_clip_index, index_dir  # noqa: E402
from cinescope.search.vecindex import IndexSpec, normalize  # noqa: E402

IVFPQ = IndexSpec("ivfpq", nlist=32, nprobe=8, m=16, nbits=8)


class FaissIndex:
    """FAISS with the same interface as ours."""

    def __init__(self, kind: str, dim: int, spec: IndexSpec) -> None:
        import faiss

        self.faiss = faiss
        if kind == "faiss-flat":
            self.ix = faiss.IndexFlatIP(dim)
        else:
            self.q = faiss.IndexFlatIP(dim)
            self.ix = faiss.IndexIVFPQ(self.q, dim, spec.nlist, spec.m, spec.nbits, faiss.METRIC_INNER_PRODUCT)
            self.ix.nprobe = spec.nprobe

    def train(self, x: np.ndarray) -> None:
        if not self.ix.is_trained:
            self.ix.train(np.ascontiguousarray(x, np.float32))

    def add(self, x: np.ndarray) -> None:
        self.ix.add(np.ascontiguousarray(x, np.float32))

    def search(self, q: np.ndarray, k: int):
        return self.ix.search(np.ascontiguousarray(q, np.float32), k)


def build_ann(kind: str, vectors: np.ndarray, train: np.ndarray):
    dim = vectors.shape[1]
    if kind == "flat":
        ix = IndexSpec("flat").build(dim)
    elif kind == "ivfpq":
        ix = IVFPQ.build(dim)
    elif kind == "ivf":                      # diagnosis: probing loss only (exact vectors in the lists)
        ix = IndexSpec("ivf", nlist=IVFPQ.nlist, nprobe=IVFPQ.nprobe).build(dim)
    elif kind == "ivfpq-allprobe":           # diagnosis: quantisation loss only (every list probed)
        ix = IndexSpec("ivfpq", IVFPQ.nlist, IVFPQ.nlist, IVFPQ.m, IVFPQ.nbits).build(dim)
    elif kind == "ivfpq-rerank":             # IVF-PQ shortlist of 50, re-scored with the exact vectors
        ix = IndexSpec("ivfpq", IVFPQ.nlist, IVFPQ.nprobe, IVFPQ.m, IVFPQ.nbits, refine=50).build(dim)
    else:
        ix = FaissIndex(kind, dim, IVFPQ)
    ix.train(train)
    ix.add(vectors)
    return ix


def train_vectors(encoder, frames_npz: str | None) -> np.ndarray | None:
    if not frames_npz or not Path(frames_npz).exists():
        return None
    z = np.load(frames_npz, allow_pickle=False)
    if encoder.name == str(z["encoder"]):
        return z["emb"]
    # another encoder: embed the stored 64 px tiles with it
    return encoder.encode_images(list(z["images"]))


def random_baseline(labels: dict, clip_ix: dict, library: ShotIndex) -> dict:
    """Expected R@k / MRR of a random ranking, exactly: with n documents of which r are
    relevant, P(first relevant within top k) = 1 - C(n-r, k) / C(n, k). Small clips make
    per-clip R@5 easy (Megamind has 4 segments), which this row makes visible."""
    from math import comb

    def expect(n: int, r: int) -> dict:
        if r == 0:
            return {"R@1": 0.0, "R@5": 0.0, "MRR": 0.0}
        p = {f"R@{k}": 1 - comb(n - r, k) / comb(n, k) if k <= n else 1.0 for k in (1, 5)}
        # P(first relevant at rank j) = C(n-j, r-1) / C(n, r)
        p["MRR"] = sum(comb(n - j, r - 1) / comb(n, r) / j for j in range(1, n - r + 2))
        return p

    out: dict = {}
    for mode in ("clip", "library"):
        rows = []
        for name, c in labels["clips"].items():
            if c["split"] != "test":
                continue
            docs = clip_ix[name].docs if mode == "clip" else library.docs
            for q in c["queries"]:
                r = sum(1 for d in docs if first_relevant_rank([d], q["times"], clip=name) == 1)
                rows.append(expect(len(docs), r))
        out[mode] = {k: round(float(np.mean([x[k] for x in rows])), 4) for k in ("R@1", "R@5", "MRR")}
    return out


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", default=str(ROOT / "p8_search" / "labels" / "queries.json"))
    ap.add_argument("--trailers", default=str(ROOT / "trailers"), help="folder holding the labelled clips")
    ap.add_argument("--encoders", nargs="+", default=["openclip"])
    ap.add_argument("--train-frames", default="outputs/p8_train_frames.npz")
    ap.add_argument("--indexes", nargs="+", default=["flat", "ivfpq", "faiss-flat", "faiss-ivfpq", "ivf",
                                                     "ivfpq-allprobe", "ivfpq-rerank"])
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--results", default=str(ROOT / "p8_search" / "results" / "eval_search.json"))
    args = ap.parse_args(argv)

    labels = json.loads(Path(args.labels).read_text(encoding="utf-8"))
    try:
        import faiss  # noqa: F401
    except ImportError:
        args.indexes = [i for i in args.indexes if not i.startswith("faiss")]
    report: dict = {"note": "per-clip = query searches its own clip; library = all clips pooled", "runs": {}}
    for spec in args.encoders:
        enc = get_encoder(spec)
        clip_ix: dict[str, ShotIndex] = {}
        for name, c in labels["clips"].items():
            video = Path(args.trailers) / c["file"]
            d = index_dir(args.out, video, enc.name)
            clip_ix[name] = ShotIndex.load(d) if (d / "docs.json").exists() else \
                build_clip_index(video, enc, d, tag_vocab=TAG_VOCAB)
        library = ShotIndex.merge(list(clip_ix.values()))
        train = train_vectors(enc, args.train_frames)
        if train is None:
            print("no training frames: IVF-PQ codebooks are learned from the library itself")
            train = library.vectors
        train = normalize(train)
        queries = [(name, q) for name, c in labels["clips"].items() for q in c["queries"]]
        qv = enc.encode_texts([q["q"] for _, q in queries])
        for kind in args.indexes:
            t = time.perf_counter()
            per_split: dict[str, dict[str, list]] = {}
            detail = []
            lib_ann = build_ann(kind, library.vectors, train)
            per_ann = {n: build_ann(kind, ix.vectors, train) for n, ix in clip_ix.items()}
            for (name, q), v in zip(queries, qv, strict=True):
                split = labels["clips"][name]["split"]
                ix = clip_ix[name]
                _, I = per_ann[name].search(v[None], len(ix))  # noqa: E741
                r_clip = first_relevant_rank([ix.docs[i] for i in I[0] if i >= 0], q["times"])
                _, IL = lib_ann.search(v[None], len(library))
                r_lib = first_relevant_rank([library.docs[i] for i in IL[0] if i >= 0], q["times"], clip=name)
                s = per_split.setdefault(split, {"clip": [], "library": []})
                s["clip"].append(r_clip)
                s["library"].append(r_lib)
                detail.append({"clip": name, "split": split, "q": q["q"], "rank_clip": r_clip, "rank_library": r_lib,
                               "top_clip": [{"shot": ix.docs[i]["shot"], "t0": ix.docs[i]["t0"]} for i in I[0][:3] if i >= 0]})
            run = {split: {mode: summarise(r) for mode, r in modes.items()} for split, modes in per_split.items()}
            run["seconds"] = round(time.perf_counter() - t, 2)
            run["queries"] = detail
            report["runs"][f"{enc.name} | {kind}"] = run
            te = run.get("test", {})
            print(f"{enc.name:60s} {kind:12s} per-clip {te.get('clip')}  library {te.get('library')}", flush=True)
        report["docs"] = {n: len(ix) for n, ix in clip_ix.items()}
        report["train_vectors"] = int(len(train))
        report["random_ranking"] = random_baseline(labels, clip_ix, library)
        print(f"random ranking (expected): {report['random_ranking']}")
    Path(args.results).parent.mkdir(parents=True, exist_ok=True)
    Path(args.results).write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"-> {args.results}")
    return report


if __name__ == "__main__":
    main()
