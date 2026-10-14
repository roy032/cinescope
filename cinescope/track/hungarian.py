"""Optimal one-to-one assignment (the Hungarian algorithm), from scratch.

Tracking-by-detection is a matching problem: N existing tracks, M new
detections, a cost for every (track, detection) pair, and each track may take at
most one detection. Picking the cheapest pair, removing it and repeating
(greedy) is fast but can be wrong: one early cheap match can force two
expensive ones later. The Hungarian algorithm (Kuhn 1955, Munkres 1957) finds
the assignment whose *total* cost is minimal.

This is the shortest-augmenting-path form (as in Jonker-Volgenant and
`scipy.optimize.linear_sum_assignment`): rows are added one at a time; for each
new row, a Dijkstra-like search over reduced costs finds the cheapest way to
give it a column, possibly re-routing rows already assigned. Dual potentials
u (rows) and v (columns) keep every reduced cost c[i,j] - u[i] - v[j] >= 0,
which is what makes the greedy search inside Dijkstra correct. O(n^2 m).
"""
from __future__ import annotations

import numpy as np


def linear_sum_assignment(cost: np.ndarray, maximize: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """(row_ind, col_ind) of a minimum-cost matching; same contract as SciPy's.

    Rectangular matrices are fine: min(n, m) pairs are returned, rows sorted.
    Infinite entries mean "forbidden"; a ValueError is raised if no complete
    matching avoids them (SciPy does the same).
    """
    c = np.asarray(cost, dtype=np.float64)
    if c.ndim != 2:
        raise ValueError("cost must be 2-D")
    if maximize:
        c = -c
    if np.isnan(c).any() or np.isneginf(c).any():
        raise ValueError("cost contains NaN or -inf")
    transposed = c.shape[0] > c.shape[1]
    if transposed:
        c = c.T
    n, m = c.shape
    if n == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)

    # Forbidden pairs get a cost bigger than any feasible total, so they are
    # only used when nothing else works -- then we know there is no matching.
    finite = np.isfinite(c)
    big = (np.abs(c[finite]).sum() + 1.0) * (n + 1) if finite.any() else 1.0
    c = np.where(finite, c, big)

    # 1-based arrays with a virtual column 0, as in the classic formulation.
    u = np.zeros(n + 1)
    v = np.zeros(m + 1)
    p = np.zeros(m + 1, np.int64)        # p[j] = row assigned to column j (0 = free)
    way = np.zeros(m + 1, np.int64)      # previous column on the shortest path
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(m + 1, np.inf)
        used = np.zeros(m + 1, bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            # Relax every unused column from row i0 (vectorised inner loop).
            free = ~used[1:]
            cur = c[i0 - 1] - u[i0] - v[1:]
            better = free & (cur < minv[1:])
            minv[1:][better] = cur[better]
            way[1:][better] = j0
            cand = np.where(free, minv[1:], np.inf)
            j1 = int(np.argmin(cand)) + 1
            delta = cand[j1 - 1]
            # Shift potentials so the reduced cost of the chosen edge becomes 0.
            u[p[used]] += delta
            v[used] -= delta
            minv[1:][free] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        # Flip the augmenting path.
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1

    rows, cols = [], []
    for j in range(1, m + 1):
        if p[j]:
            rows.append(p[j] - 1)
            cols.append(j - 1)
    rows, cols = np.asarray(rows, np.int64), np.asarray(cols, np.int64)
    if (~finite[rows, cols]).any():
        raise ValueError("cost matrix is infeasible")
    if transposed:
        rows, cols = cols, rows
    order = np.argsort(rows)
    return rows[order], cols[order]


def match(cost: np.ndarray, max_cost: float) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Tracker-style assignment: optimal matching, then pairs costing more than
    max_cost are rejected. Returns (pairs, unmatched rows, unmatched cols).

    Gating *after* the optimum (not by setting gated pairs to inf first) is what
    SORT and ByteTrack do; with large gated costs the two agree.
    """
    cost = np.asarray(cost, dtype=np.float64)
    n, m = cost.shape
    if n == 0 or m == 0:
        return [], list(range(n)), list(range(m))
    gated = np.where(cost > max_cost, max_cost + 1.0, cost)   # keeps the problem feasible
    r, c = linear_sum_assignment(gated)
    pairs = [(int(i), int(j)) for i, j in zip(r, c, strict=True) if cost[i, j] <= max_cost]
    mr = {i for i, _ in pairs}
    mc = {j for _, j in pairs}
    return pairs, [i for i in range(n) if i not in mr], [j for j in range(m) if j not in mc]
