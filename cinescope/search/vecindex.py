"""Vector search from scratch: exact, IVF (inverted file) and IVF-PQ (product quantisation).

All three answer the same question — which stored vectors have the largest inner
product with the query? — and they all use inner product, because the vectors are
L2-normalised CLIP embeddings, for which inner product = cosine similarity.

Exact ("flat")
  One matrix product, Q @ X.T, then a partial sort (argpartition) for the top k.
  Cost O(N·D) per query: 1M vectors of 512 floats is 2 GB of memory traffic per
  query. Fine up to ~100k vectors; the yardstick every approximate index is scored
  against (recall@k = fraction of the exact top-k it returns).

IVF (Jégou et al., 2011; FAISS IndexIVFFlat)
  Train k-means with `nlist` centroids on a sample. Every vector is stored in the
  list of its nearest centroid. A query compares itself with the nlist centroids,
  then scans only the `nprobe` best lists: about N·nprobe/nlist vectors. The miss
  case is a true neighbour sitting in a list the query did not probe — nprobe is
  the speed/recall dial.

Product quantisation (Jégou, Douze & Schmid, 2011)
  Split each D-dim vector into M sub-vectors of D/M dims and run k-means with 256
  centroids in each sub-space. A vector is then stored as M one-byte codes (the
  nearest sub-centroid in each sub-space): 512 floats = 2048 bytes become 32 bytes.
  The *reconstruction* is the concatenation of the chosen sub-centroids.

  Asymmetric distance computation (ADC): the query stays exact. Because
  <q, x̂> = Σ_m <q_m, C_m[code_m]>, one table of M x 256 inner products per query
  (the LUT) turns scoring a stored vector into M table look-ups and adds — no
  decompression. ("Symmetric" would quantise the query too: cheaper tables, worse
  accuracy.)

IVF-PQ
  PQ applied to the *residual* x - c (vector minus its IVF centroid): residuals are
  smaller and more alike across lists than the raw vectors, so the same 256
  sub-centroids describe them more precisely. For inner product the residual trick
  stays cheap: <q, c + r̂> = <q, c> + Σ_m <q_m, C_m[code_m]>, and the second term's
  LUT does not depend on the list, so it is built once per query.
  Optional exact re-ranking (`refine`) re-scores the best candidates with the
  original vectors, if they are kept — FAISS's IndexRefineFlat.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..palette.kmeans import kmeans_pp_init


def normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), eps)


def topk(scores: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Row-wise top-k of a (Q, N) score matrix, best first: (values, indices).
    argpartition is O(N); only the k survivors are sorted."""
    n = scores.shape[1]
    k_ = min(k, n)
    if k_ == 0:
        return np.zeros((len(scores), 0), np.float32), np.zeros((len(scores), 0), np.int64)
    part = np.argpartition(-scores, k_ - 1, axis=1)[:, :k_] if k_ < n else np.tile(np.arange(n), (len(scores), 1))
    vals = np.take_along_axis(scores, part, 1)
    order = np.argsort(-vals, axis=1, kind="stable")
    return np.take_along_axis(vals, order, 1), np.take_along_axis(part, order, 1)


def _pad(vals: np.ndarray, ids: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Pad to k columns with (-inf, -1), the FAISS convention for "fewer than k found"."""
    if vals.shape[-1] >= k:
        return vals, ids
    pv = np.full(k, -np.inf, np.float32)
    pi = np.full(k, -1, np.int64)
    pv[:vals.shape[-1]] = vals
    pi[:ids.shape[-1]] = ids
    return pv, pi


# --------------------------------------------------------------------------- k-means
def kmeans_fast(x: np.ndarray, k: int, n_iter: int = 20, seed: int = 0, max_points_per_centroid: int = 256,
                spherical: bool = False) -> np.ndarray:
    """Lloyd's k-means in float32, vectorised for many centroids (cinescope/palette/kmeans.py
    loops over clusters in Python, which is fine for 5 palette colours, not for 256 x 32
    PQ codebooks). Same k-means++ initialisation. Like FAISS, trains on at most 256
    sampled vectors per centroid: centroids converge long before every point is seen.
    (A first version capped the sample at 16k points regardless of k: with 1,024 IVF
    lists that is 16 points per centroid, and IVF recall at nprobe=32 stalled at 0.96.)

    spherical=True renormalises centroids each step: for inner-product search every
    centroid then has norm 1, so "largest <x, c>" and "nearest c" pick the same list —
    without it, a large-norm centroid wins the inner product for far too many vectors
    and its list balloons."""
    x = np.asarray(x, np.float32)
    rng = np.random.default_rng(seed)
    max_points = max_points_per_centroid * k
    if len(x) > max_points:
        x = x[rng.choice(len(x), max_points, replace=False)]
    if len(x) < k:
        raise ValueError(f"need at least k={k} training points, got {len(x)}")
    # k-means++ on a sub-sample: its k sequential passes are the expensive part.
    init_src = x if len(x) <= 20 * k else x[rng.choice(len(x), 20 * k, replace=False)]
    c = kmeans_pp_init(init_src.astype(np.float64), k, rng).astype(np.float32)
    x2 = (x * x).sum(1)
    for _ in range(n_iter):
        d = x2[:, None] - 2 * x @ c.T + (c * c).sum(1)[None, :]
        lab = d.argmin(1)
        counts = np.bincount(lab, minlength=k).astype(np.float32)
        sums = np.zeros_like(c)
        np.add.at(sums, lab, x)
        empty = counts == 0
        c = np.where(empty[:, None], c, sums / np.maximum(counts, 1)[:, None])
        if empty.any():                       # re-seed empty clusters at the worst-served points
            worst = np.argsort(-d[np.arange(len(x)), lab])[:empty.sum()]
            c[empty] = x[worst]
        if spherical:
            c = normalize(c)
    return c.astype(np.float32)


def _assign(x: np.ndarray, c: np.ndarray, metric: str = "l2") -> np.ndarray:
    out = np.empty(len(x), np.int64)
    for s in range(0, len(x), 65536):
        xb = x[s:s + 65536]
        if metric == "ip":
            out[s:s + 65536] = (xb @ c.T).argmax(1)
        else:
            out[s:s + 65536] = ((c * c).sum(1)[None, :] - 2 * xb @ c.T).argmin(1)
    return out


# --------------------------------------------------------------------------- exact
class FlatIndex:
    """Exact inner-product search."""

    kind = "flat"

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.x = np.zeros((0, dim), np.float32)

    @property
    def ntotal(self) -> int:
        return len(self.x)

    def train(self, x: np.ndarray) -> None:      # nothing to learn; same interface as the others
        pass

    def add(self, x: np.ndarray) -> None:
        self.x = np.concatenate([self.x, np.asarray(x, np.float32)])

    def search(self, q: np.ndarray, k: int, chunk: int = 256) -> tuple[np.ndarray, np.ndarray]:
        q = np.atleast_2d(np.asarray(q, np.float32))
        D = np.full((len(q), k), -np.inf, np.float32)
        I = np.full((len(q), k), -1, np.int64)  # noqa: E741 — FAISS's names
        for s in range(0, len(q), chunk):
            v, i = topk(q[s:s + chunk] @ self.x.T, k)
            D[s:s + chunk, :v.shape[1]], I[s:s + chunk, :i.shape[1]] = v, i
        return D, I

    def bytes_per_vector(self) -> float:
        return 4.0 * self.dim

    def state(self) -> dict:
        return {"x": self.x}

    def load_state(self, s: dict) -> None:
        self.x = s["x"].astype(np.float32)


# --------------------------------------------------------------------------- inverted lists
class _InvertedLists:
    """Lists stored as one array sorted by list id + offsets (CSR), so a probe is a slice."""

    def __init__(self, nlist: int) -> None:
        self.nlist = nlist
        self.ids = np.zeros(0, np.int64)
        self.lists = np.zeros(0, np.int64)
        self.payload: np.ndarray | None = None       # vectors (IVF-Flat) or codes (IVF-PQ)
        self.offsets = np.zeros(nlist + 1, np.int64)

    def add(self, lists: np.ndarray, ids: np.ndarray, payload: np.ndarray) -> None:
        self.lists = np.concatenate([self.lists, lists])
        self.ids = np.concatenate([self.ids, ids])
        self.payload = payload if self.payload is None else np.concatenate([self.payload, payload])
        order = np.argsort(self.lists, kind="stable")
        self.lists, self.ids, self.payload = self.lists[order], self.ids[order], self.payload[order]
        self.offsets = np.concatenate([[0], np.cumsum(np.bincount(self.lists, minlength=self.nlist))])

    def candidates(self, probe: np.ndarray) -> np.ndarray:
        """Row positions of every vector in the probed lists."""
        parts = [np.arange(self.offsets[p], self.offsets[p + 1]) for p in probe]
        return np.concatenate(parts) if parts else np.zeros(0, np.int64)

    def sizes(self) -> np.ndarray:
        return np.diff(self.offsets)


class IVFFlatIndex:
    """Inverted file with exact vectors in the lists."""

    kind = "ivf"

    def __init__(self, dim: int, nlist: int = 64, nprobe: int = 8, seed: int = 0, coarse: str = "ip") -> None:
        self.dim, self.nlist, self.nprobe, self.seed, self.coarse = dim, nlist, nprobe, seed, coarse
        self.centroids: np.ndarray | None = None
        self.inv = _InvertedLists(nlist)
        self.ntotal = 0

    def train(self, x: np.ndarray) -> None:
        self.centroids = kmeans_fast(x, self.nlist, seed=self.seed, spherical=self.coarse == "ip")

    def _coarse(self, x: np.ndarray) -> np.ndarray:
        assert self.centroids is not None, "train() first"
        return _assign(x, self.centroids, self.coarse)

    def add(self, x: np.ndarray) -> None:
        x = np.asarray(x, np.float32)
        ids = np.arange(self.ntotal, self.ntotal + len(x))
        self.inv.add(self._coarse(x), ids, x)
        self.ntotal += len(x)

    def _probe(self, q: np.ndarray) -> np.ndarray:
        assert self.centroids is not None
        # coarse="ip": probe the lists with the largest <q, c>. For inner-product search this is
        # the right order — a stored score is <q, c> + <q, r> — and it is what FAISS does with
        # an IndexFlatIP quantiser. coarse="l2" probes the nearest centroids instead; k-means
        # centroids of unit vectors have different norms, so for a text query (far from
        # every image centroid: the modality gap) L2 and IP disagree a lot. bench_ann.py
        # measured it: text-query recall@10 at nprobe=1 was 0.20 with L2 probing.
        if self.coarse == "ip":
            d = -(q @ self.centroids.T)
        else:
            d = (self.centroids * self.centroids).sum(1)[None, :] - 2 * q @ self.centroids.T
        npb = min(self.nprobe, self.nlist)
        return np.argpartition(d, npb - 1, axis=1)[:, :npb]

    def search(self, q: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        q = np.atleast_2d(np.asarray(q, np.float32))
        probes = self._probe(q)
        D = np.empty((len(q), k), np.float32)
        I = np.empty((len(q), k), np.int64)  # noqa: E741
        for j in range(len(q)):
            rows = self.inv.candidates(probes[j])
            s = self.inv.payload[rows] @ q[j] if len(rows) else np.zeros(0, np.float32)
            v, i = topk(s[None, :], k)
            D[j], I[j] = _pad(v[0], self.inv.ids[rows][i[0]], k)
        return D, I

    def bytes_per_vector(self) -> float:
        return 4.0 * self.dim + 8

    def state(self) -> dict:
        return {"centroids": self.centroids, "ids": self.inv.ids, "lists": self.inv.lists,
                "payload": self.inv.payload, "params": np.array([self.nlist, self.nprobe, self.ntotal]),
                "coarse": np.array(self.coarse)}

    def load_state(self, s: dict) -> None:
        self.nlist, self.nprobe, self.ntotal = (int(v) for v in s["params"])
        self.coarse = str(s.get("coarse", "ip"))
        self.centroids = s["centroids"]
        self.inv = _InvertedLists(self.nlist)
        self.inv.add(s["lists"], s["ids"], s["payload"])


# --------------------------------------------------------------------------- product quantisation
class ProductQuantizer:
    def __init__(self, dim: int, m: int = 16, nbits: int = 8, seed: int = 0) -> None:
        if dim % m:
            raise ValueError(f"dim {dim} is not divisible by M={m}")
        self.dim, self.m, self.nbits, self.seed = dim, m, nbits, seed
        self.ksub = 2 ** nbits
        self.dsub = dim // m
        self.codebooks: np.ndarray | None = None      # (M, ksub, dsub)

    def _split(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, np.float32).reshape(len(x), self.m, self.dsub)

    def train(self, x: np.ndarray, n_iter: int = 20) -> None:
        xs = self._split(x)
        self.codebooks = np.stack([kmeans_fast(xs[:, j], self.ksub, n_iter=n_iter, seed=self.seed + j)
                                   for j in range(self.m)])

    def encode(self, x: np.ndarray) -> np.ndarray:
        assert self.codebooks is not None, "train() first"
        xs = self._split(x)
        dtype = np.uint8 if self.nbits <= 8 else np.uint16
        return np.stack([_assign(xs[:, j], self.codebooks[j], "l2") for j in range(self.m)], 1).astype(dtype)

    def decode(self, codes: np.ndarray) -> np.ndarray:
        assert self.codebooks is not None
        parts = [self.codebooks[j][codes[:, j]] for j in range(self.m)]
        return np.concatenate(parts, 1)

    def ip_tables(self, q: np.ndarray) -> np.ndarray:
        """ADC look-up tables: (Q, M, ksub) inner products of each query sub-vector with
        every sub-centroid."""
        assert self.codebooks is not None
        return np.einsum("qmd,mkd->qmk", self._split(q), self.codebooks)


class IVFPQIndex:
    """IVF + PQ-coded residuals, inner-product ADC; optional exact re-ranking."""

    kind = "ivfpq"

    def __init__(self, dim: int, nlist: int = 64, m: int = 16, nbits: int = 8, nprobe: int = 8,
                 refine: int = 0, seed: int = 0, coarse: str = "ip") -> None:
        self.dim, self.nlist, self.nprobe, self.refine, self.seed = dim, nlist, nprobe, refine, seed
        self.ivf = IVFFlatIndex(dim, nlist, nprobe, seed, coarse)
        self.pq = ProductQuantizer(dim, m, nbits, seed)
        self.inv = _InvertedLists(nlist)
        self.raw: np.ndarray | None = None            # kept only when refine > 0
        self.ntotal = 0

    def train(self, x: np.ndarray) -> None:
        x = np.asarray(x, np.float32)
        self.ivf.train(x)
        assert self.ivf.centroids is not None
        sample = x if len(x) <= 65536 else x[np.random.default_rng(self.seed).choice(len(x), 65536, replace=False)]
        lists = self.ivf._coarse(sample)
        self.pq.train(sample - self.ivf.centroids[lists])

    def add(self, x: np.ndarray) -> None:
        x = np.asarray(x, np.float32)
        assert self.ivf.centroids is not None, "train() first"
        lists = self.ivf._coarse(x)
        codes = self.pq.encode(x - self.ivf.centroids[lists])
        self.inv.add(lists, np.arange(self.ntotal, self.ntotal + len(x)), codes)
        if self.refine:
            self.raw = x if self.raw is None else np.concatenate([self.raw, x])
        self.ntotal += len(x)

    def search(self, q: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        q = np.atleast_2d(np.asarray(q, np.float32))
        self.ivf.nprobe = self.nprobe
        probes = self.ivf._probe(q)
        assert self.ivf.centroids is not None
        coarse = q @ self.ivf.centroids.T                                 # <q, c> per list
        luts = self.pq.ip_tables(q).reshape(len(q), -1)                   # (Q, M*ksub)
        offs = (np.arange(self.pq.m) * self.pq.ksub)[None, :]
        kk = max(k, self.refine)
        D = np.empty((len(q), k), np.float32)
        I = np.empty((len(q), k), np.int64)  # noqa: E741
        for j in range(len(q)):
            rows = self.inv.candidates(probes[j])
            if len(rows) == 0:
                D[j], I[j] = _pad(np.zeros(0, np.float32), np.zeros(0, np.int64), k)
                continue
            codes = self.inv.payload[rows].astype(np.int64)
            s = coarse[j, self.inv.lists[rows]] + luts[j][codes + offs].sum(1)
            v, i = topk(s[None, :], kk)
            ids = self.inv.ids[rows][i[0]]
            if self.refine and self.raw is not None:                     # exact re-rank
                v2, i2 = topk((self.raw[ids] @ q[j])[None, :], k)
                D[j], I[j] = _pad(v2[0], ids[i2[0]], k)
            else:
                D[j], I[j] = _pad(v[0][:k], ids[:k], k)
        return D, I

    def bytes_per_vector(self) -> float:
        return self.pq.m * self.pq.nbits / 8 + 8 + (4.0 * self.dim if self.refine else 0)

    def state(self) -> dict:
        s = {"centroids": self.ivf.centroids, "codebooks": self.pq.codebooks, "ids": self.inv.ids,
             "lists": self.inv.lists, "payload": self.inv.payload,
             "params": np.array([self.nlist, self.pq.m, self.pq.nbits, self.nprobe, self.refine, self.ntotal]),
             "coarse": np.array(self.ivf.coarse)}
        if self.raw is not None:
            s["raw"] = self.raw
        return s

    def load_state(self, s: dict) -> None:
        nlist, m, nbits, self.nprobe, self.refine, self.ntotal = (int(v) for v in s["params"])
        self.nlist = nlist
        self.ivf = IVFFlatIndex(self.dim, nlist, self.nprobe, self.seed, str(s.get("coarse", "ip")))
        self.ivf.centroids = s["centroids"]
        self.pq = ProductQuantizer(self.dim, m, nbits, self.seed)
        self.pq.codebooks = s["codebooks"]
        self.inv = _InvertedLists(nlist)
        self.inv.add(s["lists"], s["ids"], s["payload"])
        self.raw = s.get("raw")


# --------------------------------------------------------------------------- helpers
@dataclass
class IndexSpec:
    """How to build an index: kind in {"flat", "ivf", "ivfpq"} plus its parameters."""
    kind: str = "flat"
    nlist: int = 64
    nprobe: int = 8
    m: int = 16
    nbits: int = 8
    refine: int = 0
    coarse: str = "ip"

    def build(self, dim: int, seed: int = 0):
        if self.kind == "flat":
            return FlatIndex(dim)
        if self.kind == "ivf":
            return IVFFlatIndex(dim, self.nlist, self.nprobe, seed, self.coarse)
        if self.kind == "ivfpq":
            return IVFPQIndex(dim, self.nlist, self.m, self.nbits, self.nprobe, self.refine, seed, self.coarse)
        raise ValueError(f"unknown index kind {self.kind!r}")


def save_index(index, path: str | Path) -> None:
    np.savez(path, kind=np.array(index.kind), dim=np.array(index.dim),
             **{k: v for k, v in index.state().items() if v is not None})


def load_index(path: str | Path):
    with np.load(path, allow_pickle=False) as z:
        s = {k: z[k] for k in z.files}
    kind, dim = str(s.pop("kind")), int(s.pop("dim"))
    index = IndexSpec(kind).build(dim)
    index.load_state(s)
    return index


def recall_at_k(found: np.ndarray, truth: np.ndarray, k: int) -> float:
    """ANN recall: fraction of the exact top-k ids that appear in the returned top-k."""
    hits = [len(set(f[:k].tolist()) & set(t[:k].tolist())) for f, t in zip(found, truth, strict=True)]
    return float(np.mean(hits) / k)


def one_recall_at_k(found: np.ndarray, truth: np.ndarray, k: int) -> float:
    """FAISS's "1-recall@k": how often the true nearest neighbour is in the returned top-k."""
    return float(np.mean([t[0] in f[:k] for f, t in zip(found, truth, strict=True)]))
