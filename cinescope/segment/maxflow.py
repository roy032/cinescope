"""Max-flow / min-cut with the Boykov-Kolmogorov algorithm, in plain Python.

Why graph cuts: GrabCut (and most binary MRF segmentation) minimises an energy

    E(labels) = sum_p D_p(label_p) + sum_{p,q neighbours} V_pq * [label_p != label_q]

with D the data cost (how unlike the foreground/background colour model a pixel
is) and V >= 0 a smoothness cost (cutting between similar colours is expensive).
Build a graph with a source s (foreground) and a sink t (background), an edge
s->p of capacity D_p(bg), p->t of capacity D_p(fg), and p<->q of capacity V_pq.
Every s-t cut assigns each node to one side, and its capacity is exactly E of
that labelling — so the minimum cut is the global minimum of E. Max-flow equals
min-cut (Ford-Fulkerson), so we compute a maximum flow and read the cut off it:
the source side is everything still reachable from s in the residual graph.

Boykov-Kolmogorov (2004) is the algorithm vision uses for this. Generic
augmenting-path methods (Edmonds-Karp, Dinic) rebuild a search tree from
scratch for every path; BK keeps two search trees, S (grown from the source)
and T (grown from the sink), and *reuses* them:

  growth    active nodes take free neighbours into their tree along edges with
            residual capacity, until an S node touches a T node -> a path.
  augment   push the bottleneck along it. Saturated edges cut their child
            node off from its tree: that node becomes an "orphan".
  adoption  each orphan looks for a new parent in its own tree (one whose chain
            really reaches the terminal). Failing that it becomes free, and
            its children become orphans in turn.

Worst case is no better than other methods, but on grid-like vision graphs
with short paths it is several times faster in practice, which is why it is
the standard. Two tricks from the reference code are kept: terminal edges are
folded into one signed residual per node (`tr`), and adoption uses timestamps
and distance-to-terminal marks so the "does this chain reach the terminal?"
walk is short.

Pure Python lists, not NumPy: the algorithm visits one node and edge at a time,
and indexing a Python list is ~10x faster than indexing a NumPy array. It is
fast enough for superpixel graphs (~10^3 nodes) and narrow pixel bands
(~10^4 nodes), which is how `grabcut.py` uses it.
"""
from __future__ import annotations

from collections import deque

import numpy as np

_TERMINAL = -1   # parent "edge" of a node attached directly to its terminal
_ORPHAN = -2     # parent of a node that lost its parent
_FREE, _S, _T = 0, 1, 2
_INF_DIST = 1 << 30


class Graph:
    """A directed graph with source/sink capacities, for one max-flow run.

        g = Graph(n)
        g.add_edges(i, j, cap, rev_cap)      # arrays (or scalars): i->j and j->i
        g.add_tweights(i, cap_source, cap_sink)
        flow = g.maxflow()
        fg = g.source_side()                 # bool (n,): True = source (foreground)

    Edges are stored in pairs: edge 2k is i->j and 2k+1 is its sister j->i, so
    the reverse of edge e is e ^ 1.
    """

    def __init__(self, n: int) -> None:
        self.n = int(n)
        self._src: list[np.ndarray] = []
        self._dst: list[np.ndarray] = []
        self._cap: list[np.ndarray] = []
        self.tr = np.zeros(self.n)         # >0: residual s->p, <0: residual p->t
        self.flow0 = 0.0                   # flow that cancels when both terminals feed a node
        self.flow: float | None = None
        self._tree: list[int] | None = None

    def add_edges(self, i, j, cap, rev_cap=None) -> None:
        i = np.atleast_1d(np.asarray(i, dtype=np.int64))
        j = np.atleast_1d(np.asarray(j, dtype=np.int64))
        cap = np.broadcast_to(np.asarray(cap, dtype=np.float64), i.shape)
        rev = cap if rev_cap is None else np.broadcast_to(np.asarray(rev_cap, np.float64), i.shape)
        if np.any(cap < 0) or np.any(rev < 0):
            raise ValueError("capacities must be non-negative")
        if np.any(i == j):
            raise ValueError("self-loops are not allowed")
        # interleave i->j (cap) and j->i (rev) so sister edges are 2k, 2k+1
        self._src.append(np.stack([i, j], 1).ravel())
        self._dst.append(np.stack([j, i], 1).ravel())
        self._cap.append(np.stack([cap, rev], 1).ravel())

    def add_tweights(self, i, cap_source, cap_sink) -> None:
        """Add source->i and i->sink capacities. Only the difference matters for
        the cut; the common part is flow that goes straight through i."""
        i = np.atleast_1d(np.asarray(i, dtype=np.int64))
        cs = np.broadcast_to(np.asarray(cap_source, np.float64), i.shape)
        ct = np.broadcast_to(np.asarray(cap_sink, np.float64), i.shape)
        np.add.at(self.tr, i, cs - ct)
        self.flow0 += float(np.minimum(cs, ct).sum())

    # ------------------------------------------------------------------ solve
    def maxflow(self) -> float:
        n = self.n
        if self._src:
            src = np.concatenate(self._src)
            dst = np.concatenate(self._dst)
            cap_np = np.concatenate(self._cap)
        else:
            src = dst = np.zeros(0, np.int64)
            cap_np = np.zeros(0)
        # adjacency: for node p, edges first[p]:first[p+1] in `order` leave p
        order = np.argsort(src, kind="stable")
        counts = np.bincount(src, minlength=n)
        first = np.concatenate([[0], np.cumsum(counts)]).tolist()
        out_edges = order.tolist()
        head = dst.tolist()
        cap = cap_np.tolist()
        tr = self.tr.tolist()

        tree = [_FREE] * n
        parent = [_ORPHAN] * n
        ts = [0] * n           # time of the last "reaches terminal" check
        dist = [_INF_DIST] * n
        active: deque[int] = deque()
        is_active = [False] * n
        for p in range(n):
            if tr[p] > 0:
                tree[p], parent[p], dist[p], ts[p] = _S, _TERMINAL, 1, 0
            elif tr[p] < 0:
                tree[p], parent[p], dist[p], ts[p] = _T, _TERMINAL, 1, 0
            if tr[p] != 0:
                active.append(p)
                is_active[p] = True

        flow = 0.0
        time = 0
        orphans: deque[int] = deque()

        def find_path() -> int:
            """Grow S and T until they touch; return the bridging edge (from S to T) or -1."""
            while active:
                p = active[0]
                if tree[p] == _FREE:
                    active.popleft()
                    is_active[p] = False
                    continue
                tp = tree[p]
                for k in range(first[p], first[p + 1]):
                    e = out_edges[k]
                    # residual in the direction of the tree: S grows along p->q, T along q->p
                    if (cap[e] if tp == _S else cap[e ^ 1]) <= 0:
                        continue
                    q = head[e]
                    tq = tree[q]
                    if tq == _FREE:
                        tree[q] = tp
                        parent[q] = e ^ 1          # edge from q back to p
                        ts[q] = ts[p]
                        dist[q] = dist[p] + 1
                        if not is_active[q]:
                            active.append(q)
                            is_active[q] = True
                    elif tq != tp:
                        return e if tp == _S else e ^ 1
                    elif dist[q] > dist[p] and ts[q] <= ts[p]:
                        # same tree: keep the tree shallow (reference-code heuristic)
                        parent[q] = e ^ 1
                        ts[q] = ts[p]
                        dist[q] = dist[p] + 1
                active.popleft()
                is_active[p] = False
            return -1

        def augment(mid: int) -> float:
            # bottleneck: the bridge, the S chain back to s, the T chain on to t
            f = cap[mid]
            p = head[mid ^ 1]
            while parent[p] != _TERMINAL:
                e = parent[p]                      # p -> parent; flow runs parent -> p
                f = min(f, cap[e ^ 1])
                p = head[e]
            f = min(f, tr[p])
            q = head[mid]
            while parent[q] != _TERMINAL:
                e = parent[q]                      # q -> parent; flow runs q -> parent
                f = min(f, cap[e])
                q = head[e]
            f = min(f, -tr[q])
            # push
            cap[mid] -= f
            cap[mid ^ 1] += f
            p = head[mid ^ 1]
            while parent[p] != _TERMINAL:
                e = parent[p]
                cap[e] += f
                cap[e ^ 1] -= f
                if cap[e ^ 1] <= 0:
                    parent[p] = _ORPHAN
                    orphans.append(p)
                p = head[e]
            tr[p] -= f
            if tr[p] <= 0:
                parent[p] = _ORPHAN
                orphans.append(p)
            q = head[mid]
            while parent[q] != _TERMINAL:
                e = parent[q]
                cap[e ^ 1] += f
                cap[e] -= f
                if cap[e] <= 0:
                    parent[q] = _ORPHAN
                    orphans.append(q)
                q = head[e]
            tr[q] += f
            if tr[q] >= 0:
                parent[q] = _ORPHAN
                orphans.append(q)
            return f

        def adopt(p: int) -> None:
            tp = tree[p]
            best_e, best_d = -1, _INF_DIST
            for k in range(first[p], first[p + 1]):
                e = out_edges[k]
                # a parent q must be able to send flow to p in its tree's direction
                if (cap[e ^ 1] if tp == _S else cap[e]) <= 0:
                    continue
                q = head[e]
                if tree[q] != tp or parent[q] == _ORPHAN:
                    continue
                # walk q's chain to its terminal; stop early at a node checked this round
                d = 0
                r = q
                while True:
                    if ts[r] == time:
                        d += dist[r]
                        break
                    pr = parent[r]
                    d += 1
                    if pr == _TERMINAL:
                        ts[r] = time
                        dist[r] = 1
                        break
                    if pr == _ORPHAN:
                        d = _INF_DIST
                        break
                    r = head[pr]
                if d < _INF_DIST:
                    if d < best_d:
                        best_e, best_d = e, d
                    # mark the chain so later walks stop here
                    r = q
                    while ts[r] != time:
                        ts[r] = time
                        dist[r] = d
                        d -= 1
                        r = head[parent[r]]
            if best_e >= 0:
                parent[p] = best_e
                ts[p] = time
                dist[p] = best_d + 1
                return
            # no parent: p becomes free; its children become orphans, and any
            # neighbour that could grow into p again becomes active
            for k in range(first[p], first[p + 1]):
                e = out_edges[k]
                q = head[e]
                if tree[q] != tp:
                    continue
                pq = parent[q]
                if pq >= 0 and head[pq] == p:
                    parent[q] = _ORPHAN
                    orphans.append(q)
                if (cap[e ^ 1] if tp == _S else cap[e]) > 0 and not is_active[q]:
                    active.append(q)
                    is_active[q] = True
            tree[p] = _FREE

        while True:
            mid = find_path()
            if mid < 0:
                break
            time += 1
            flow += augment(mid)
            while orphans:
                adopt(orphans.popleft())

        self._tree = tree
        self.flow = flow + self.flow0
        return self.flow

    def source_side(self) -> np.ndarray:
        """True for nodes on the source side of the minimum cut (the S tree).
        Free nodes go to the sink side: they are not reachable from s."""
        if self._tree is None:
            raise RuntimeError("call maxflow() first")
        return np.asarray(self._tree, dtype=np.int8) == _S


def grid_edges(h: int, w: int, mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """4- and diagonal 8-neighbour pairs (i, j) of an h x w pixel grid, as flat
    indices, with i < j. `mask` (h, w) bool keeps only pairs inside it."""
    idx = np.arange(h * w).reshape(h, w)
    pairs = [
        (idx[:, :-1], idx[:, 1:]),     # right
        (idx[:-1, :], idx[1:, :]),     # down
        (idx[:-1, :-1], idx[1:, 1:]),  # down-right
        (idx[:-1, 1:], idx[1:, :-1]),  # down-left
    ]
    i = np.concatenate([a.ravel() for a, _ in pairs])
    j = np.concatenate([b.ravel() for _, b in pairs])
    if mask is not None:
        m = mask.ravel()
        keep = m[i] & m[j]
        i, j = i[keep], j[keep]
    return i, j
