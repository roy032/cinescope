"""Benchmark our vector indexes against FAISS: recall vs exact search, queries/second.

    python p8_search/bench_ann.py --real outputs/p8_train_frames.npz outputs/p8_eval_frames.npz
    python p8_search/bench_ann.py --synthetic 100000          # (1000000 on a machine with RAM to spare)

Datasets
  real-image   open_clip ViT-B/32 embeddings of frame tiles (caption_frames.py); 500 held-out
               tiles are the queries.
  real-text    the same database, queried with text embeddings (the 50 labelled queries + the
               tag vocabulary). Text and image embeddings occupy different regions of CLIP's
               space (the "modality gap"), so a text query sits far from every IVF centroid —
               worth measuring separately, since text queries are what search.py sends.
  synthetic    N unit vectors in 512-d from a mixture of 2,000 Gaussian clusters, 1,000 queries.

Ground truth is exact inner-product search. Recall@10 = |approx top-10 ∩ exact top-10| / 10;
1-R@1 = the true nearest neighbour is returned first. Both libraries get the same thread
count (OMP_NUM_THREADS, default 2) and the same nlist / M / nbits / nprobe.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cinescope.search.vecindex import (  # noqa: E402
    FlatIndex,
    IVFFlatIndex,
    IVFPQIndex,
    normalize,
    one_recall_at_k,
    recall_at_k,
)


def synthetic(n: int, nq: int, d: int = 512, clusters: int = 2000, spread: float = 0.8, seed: int = 0):
    """Cluster centre + noise of norm ~spread, renormalised: neighbours at cosine ~0.6-0.7."""
    rng = np.random.default_rng(seed)
    c = normalize(rng.standard_normal((clusters, d)).astype(np.float32))
    def draw(m):
        out = np.empty((m, d), np.float32)
        for s in range(0, m, 50000):
            k = min(50000, m - s)
            lab = rng.integers(clusters, size=k)
            out[s:s + k] = normalize(c[lab] + spread / np.sqrt(d) * rng.standard_normal((k, d)).astype(np.float32))
        return out
    return draw(n), draw(nq)


def faiss_index(kind: str, d: int, nlist: int, m: int, nbits: int):
    import faiss

    if kind == "flat":
        return faiss.IndexFlatIP(d), None
    q = faiss.IndexFlatIP(d)
    if kind == "ivf":
        return faiss.IndexIVFFlat(q, d, nlist, faiss.METRIC_INNER_PRODUCT), q
    return faiss.IndexIVFPQ(q, d, nlist, m, nbits, faiss.METRIC_INNER_PRODUCT), q


def run(name: str, xb: np.ndarray, xq: np.ndarray, nlist: int, m: int, nbits: int, nprobes: list[int],
        k: int = 10, log=print) -> list[dict]:
    d = xb.shape[1]
    rows = []
    exact = FlatIndex(d)
    exact.add(xb)
    t = time.perf_counter()
    _, gt = exact.search(xq, k)
    dt = time.perf_counter() - t
    rows.append({"dataset": name, "index": "Flat (ours, NumPy)", "nprobe": "-", "R@10": 1.0, "1-R@1": 1.0,
                 "qps": round(len(xq) / dt), "train_s": 0.0, "bytes/vec": exact.bytes_per_vector()})
    have_faiss = True
    try:
        import faiss

        faiss.omp_set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    except ImportError:
        have_faiss = False
    if have_faiss:
        fx, _ = faiss_index("flat", d, nlist, m, nbits)
        fx.add(xb)
        t = time.perf_counter()
        _, fi = fx.search(xq, k)
        rows.append({"dataset": name, "index": "FAISS IndexFlatIP", "nprobe": "-", "R@10": recall_at_k(fi, gt, k),
                     "1-R@1": one_recall_at_k(fi, gt, 1), "qps": round(len(xq) / (time.perf_counter() - t)),
                     "train_s": 0.0, "bytes/vec": 4 * d})
    builds = [("IVF-Flat", lambda: IVFFlatIndex(d, nlist), "ivf"),
              ("IVF-PQ", lambda: IVFPQIndex(d, nlist, m, nbits), "ivfpq"),
              ("IVF-PQ + re-rank 100", lambda: IVFPQIndex(d, nlist, m, nbits, refine=100), None)]
    for label, make, fkind in builds:
        ours = make()
        t = time.perf_counter()
        ours.train(xb)
        ours.add(xb)
        tr = time.perf_counter() - t
        fx = None
        if have_faiss and fkind:
            fx, _ = faiss_index(fkind, d, nlist, m, nbits)
            t = time.perf_counter()
            fx.train(xb)
            fx.add(xb)
            ftr = time.perf_counter() - t
        for npb in nprobes:
            ours.nprobe = npb
            if hasattr(ours, "ivf"):
                ours.ivf.nprobe = npb
            t = time.perf_counter()
            _, I = ours.search(xq, k)  # noqa: E741
            dt = time.perf_counter() - t
            rows.append({"dataset": name, "index": f"{label} (ours)", "nprobe": npb, "R@10": recall_at_k(I, gt, k),
                         "1-R@1": one_recall_at_k(I, gt, 1), "qps": round(len(xq) / dt), "train_s": round(tr, 1),
                         "bytes/vec": ours.bytes_per_vector()})
            if fx is not None:
                fx.nprobe = npb
                t = time.perf_counter()
                _, fi = fx.search(xq, k)
                dt = time.perf_counter() - t
                rows.append({"dataset": name, "index": f"{label} (FAISS)", "nprobe": npb,
                             "R@10": recall_at_k(fi, gt, k), "1-R@1": one_recall_at_k(fi, gt, 1),
                             "qps": round(len(xq) / dt), "train_s": round(ftr, 1),
                             "bytes/vec": (m * nbits / 8 + 8) if fkind == "ivfpq" else 4 * d + 8})
            log(" | ".join(str(v) for v in rows[-1].values()))
    return rows


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--real", nargs="*", default=[], help="npz files from caption_frames.py")
    ap.add_argument("--labels", default=str(ROOT / "p8_search" / "labels" / "queries.json"))
    ap.add_argument("--synthetic", type=int, default=0, help="number of synthetic vectors (0 = skip)")
    ap.add_argument("--m", type=int, default=32)
    ap.add_argument("--nbits", type=int, default=8)
    ap.add_argument("--out", default=str(ROOT / "p8_search" / "results" / "bench_ann.json"))
    args = ap.parse_args(argv)
    rows: list[dict] = []
    if args.real:
        emb = np.concatenate([np.load(f, allow_pickle=False)["emb"] for f in args.real]).astype(np.float32)
        rng = np.random.default_rng(0)
        perm = rng.permutation(len(emb))
        xq, xb = emb[perm[:500]], emb[perm[500:]]
        nlist = int(2 ** round(np.log2(max(16, np.sqrt(len(xb)) * 2))))
        print(f"real: {len(xb)} database vectors, nlist {nlist}")
        rows += run("real-image", xb, xq, nlist, args.m, args.nbits, [1, 4, 16])
        texts = []
        if Path(args.labels).exists():
            lab = json.loads(Path(args.labels).read_text(encoding="utf-8"))
            texts = [q["q"] for c in lab["clips"].values() for q in c["queries"]]
        from cinescope.search.encoders import get_encoder
        from cinescope.search.rag import TAG_VOCAB

        enc = get_encoder("openclip")
        tq = enc.encode_texts(texts + [f"a film still of {t}." for t in TAG_VOCAB])
        rows += run("real-text", xb, tq, nlist, args.m, args.nbits, [1, 4, 16])
    if args.synthetic:
        xb, xq = synthetic(args.synthetic, 1000)
        nlist = int(2 ** round(np.log2(np.sqrt(len(xb)) * 4)))
        print(f"synthetic: {len(xb)} vectors, nlist {nlist}")
        rows += run(f"synthetic-{len(xb) // 1000}k", xb, xq, nlist, args.m, args.nbits, [1, 8, 32])
    out = Path(args.out)
    old = json.loads(out.read_text()) if out.exists() else []
    keep = [r for r in old if r["dataset"] not in {x["dataset"] for x in rows}]
    out.write_text(json.dumps(keep + rows, indent=1), encoding="utf-8")
    print("| dataset | index | nprobe | R@10 | 1-R@1 | queries/s | train s | bytes/vec |")
    print("|---|---|---|---|---|---|---|---|")
    for r in keep + rows:
        print(f"| {r['dataset']} | {r['index']} | {r['nprobe']} | {r['R@10']:.3f} | {r['1-R@1']:.3f} | "
              f"{r['qps']} | {r['train_s']} | {r['bytes/vec']:.0f} |")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
