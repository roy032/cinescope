"""Incremental structure from motion over the keyframes of one shot.

The recipe is the one COLMAP and Bundler follow (Snavely et al., 2006;
Schönberger & Frahm, 2016), cut down to a single continuous shot:

1. Keyframes. Consecutive video frames barely move, and two views with no
   baseline cannot triangulate anything. A new keyframe starts when the
   median track has moved `kf_motion` of the frame width since the last one.
2. Initial pair. Over nearby keyframe pairs, estimate E with RANSAC and keep
   the pair with the most inliers *and* enough parallax (median triangulation
   angle). A pure pan or zoom has no parallax at all: SfM then reports failure
   instead of inventing depth (Phase 0's homography already describes it).
3. Grow. Register the keyframe sharing most tracks with existing 3D points by
   PnP + RANSAC; triangulate tracks newly seen by two or more registered
   keyframes (multi-view DLT), keeping points in front of every camera, with
   a wide enough angle and small reprojection error; run bundle adjustment
   every few registrations.
4. Finish. Global bundle adjustment (optionally refining the shared focal
   length), drop observations reprojecting worse than `filter_px`, adjust
   again. Every non-keyframe is then registered by PnP against the final
   points, giving a camera pose for every frame of the shot.

The headline metric is COLMAP's: mean reprojection error (px) over all
observations of the final model.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .ba import BAResult, bundle_adjust, reprojection
from .camera import camera_centre, pixels_to_normalised
from .pnp import ransac_pnp, reprojection_errors
from .tracks import Tracks
from .twoview import (
    ransac_essential,
    recover_pose,
    refine_relative_pose,
    triangulate,
    triangulation_angle,
)


@dataclass
class SfMParams:
    kf_motion: float = 0.02          # keyframe when median track motion > this x width
    kf_min_common: float = 0.5       # ... or fewer than this share of tracks survive
    init_max_gap: int = 4            # initial pair: keyframes at most this far apart
    min_init_angle: float = 1.0      # deg, median triangulation angle of the initial pair
    min_angle: float = 1.0           # deg, to keep a triangulated point
    ransac_px: float = 1.5
    pnp_px: float = 2.0
    tri_px: float = 2.0
    filter_px: float = 2.0
    ba_every: int = 3
    refine_focal: bool = False
    min_points: int = 30
    use_ba: bool = True              # False: no bundle adjustment (outlier filtering only)
    solvers: object = None           # None: ours; or baselines.OpenCVSolvers()


class OurSolvers:
    """The from-scratch geometric solvers behind one interface, so the same
    pipeline can run on OpenCV's (baselines.OpenCVSolvers) for comparison."""

    def essential(self, xn1, xn2, focal, thresh_px, seed=0):
        res = ransac_essential(xn1, xn2, focal, thresh_px, seed=seed)
        return res.e, res.inliers

    def recover_pose(self, e, xn1, xn2):
        rot, t, front = recover_pose(e, xn1, xn2)
        if front.sum() >= 8:                   # non-linear refinement on the points in front
            rot, t = refine_relative_pose(rot, t, xn1[front], xn2[front])
        return rot, t, front

    def pnp(self, x3, xn, focal, thresh_px, seed=0):
        res = ransac_pnp(x3, xn, focal, thresh_px, seed=seed)
        return res.rot, res.t, res.inliers

    def triangulate(self, pms, xs):
        return triangulate(pms, xs)


def _solvers(p):
    return p.solvers if p.solvers is not None else OurSolvers()


@dataclass
class Reconstruction:
    k: np.ndarray
    keyframes: list[int]                               # registered keyframes (frame ids)
    poses: dict[int, tuple[np.ndarray, np.ndarray]]    # frame -> (R, t); keyframes and, later, all frames
    points: dict[int, np.ndarray]                      # track id -> X
    obs: list[tuple[int, int, np.ndarray]]             # (frame, track, xy) used in the model
    mean_reproj_px: float = float("nan")
    median_reproj_px: float = float("nan")
    log: list[str] = field(default_factory=list)
    ok: bool = True

    def summary(self) -> dict:
        lens = {}
        for _f, t, _ in self.obs:
            lens[t] = lens.get(t, 0) + 1
        return {"ok": self.ok, "keyframes": len(self.keyframes), "points": len(self.points),
                "observations": len(self.obs), "mean_track_length": float(np.mean(list(lens.values()))) if lens else 0.0,
                "mean_reproj_px": round(self.mean_reproj_px, 4), "median_reproj_px": round(self.median_reproj_px, 4),
                "focal": round(float(self.k[0, 0]), 2), "frames_with_pose": len(self.poses)}

    def centres(self, frames: list[int] | None = None) -> np.ndarray:
        frames = sorted(self.poses) if frames is None else frames
        return np.stack([camera_centre(*self.poses[f]) for f in frames])


def select_keyframes(tracks: Tracks, width: float, p: SfMParams) -> list[int]:
    frames = tracks.frames
    kfs = [frames[0]]
    for f in frames[1:]:
        ids, a, b = tracks.common(kfs[-1], f)
        n_last = len(tracks.in_frame(kfs[-1])[0])
        if len(ids) < 8:
            kfs.append(f)
            continue
        motion = np.median(np.linalg.norm(b - a, axis=1))
        if motion > p.kf_motion * width or len(ids) < p.kf_min_common * n_last:
            kfs.append(f)
    if kfs[-1] != frames[-1]:
        ids, a, b = tracks.common(kfs[-1], frames[-1])
        if len(ids) and np.median(np.linalg.norm(b - a, axis=1)) > 0.5 * p.kf_motion * width:
            kfs.append(frames[-1])
    return kfs


def _initial_pair(tracks, kfs, k, p, log):
    best = None
    f = k[0, 0]
    for i in range(len(kfs)):
        for j in range(i + 1, min(len(kfs), i + 1 + p.init_max_gap)):
            ids, a, b = tracks.common(kfs[i], kfs[j])
            if len(ids) < max(p.min_points, 8):
                continue
            na, nb = pixels_to_normalised(a, k), pixels_to_normalised(b, k)
            sv = _solvers(p)
            e, inl = sv.essential(na, nb, f, p.ransac_px, seed=i * 31 + j)
            if e is None or inl.sum() < 8:
                continue
            rot, t, front = sv.recover_pose(e, na[inl], nb[inl])
            x = sv.triangulate([np.c_[np.eye(3), np.zeros(3)], np.c_[rot, t]], [na[inl], nb[inl]])
            ang = triangulation_angle(x, np.zeros(3), camera_centre(rot, t))[front]
            if len(ang) < p.min_points:
                continue
            med = float(np.median(ang))
            score = len(ang) * min(med, 3 * p.min_init_angle)
            log.append(f"init candidate {kfs[i]}-{kfs[j]}: {len(ang)} pts, median angle {med:.2f} deg")
            if med >= p.min_init_angle and (best is None or score > best[0]):
                best = (score, kfs[i], kfs[j], rot, t)
    return best


def _robust_triangulate(rec, k, o, fs, thresh, sv):
    """Multi-view DLT; while some view reprojects worse than `thresh`, drop the
    worst view and retry (a track that drifted for a few frames keeps its good part)."""
    fs = list(fs)
    while len(fs) >= 2:
        pms = [np.c_[rec.poses[fr][0], rec.poses[fr][1]] for fr in fs]
        xs = [pixels_to_normalised(o[fr][None], k) for fr in fs]
        x = sv.triangulate(pms, xs)[0]
        errs = np.array([reprojection_errors(*rec.poses[fr], x[None], xn, k[0, 0])[0]
                         for fr, xn in zip(fs, xs, strict=True)])            # inf = behind a camera
        if np.all(errs <= thresh):
            return x, fs
        fs.pop(int(np.argmax(errs)))
    return None, fs


def _triangulate_new(tracks, rec, k, p, frames_reg, candidates=None):
    """Triangulate tracks seen in >= 2 registered keyframes that have no point yet."""
    reg = set(frames_reg)
    added = 0
    ids = candidates if candidates is not None else tracks.obs.keys()
    for tid in ids:
        if tid in rec.points:
            continue
        o = tracks.obs[tid]
        fs = [fr for fr in o if fr in reg]
        if len(fs) < 2:
            continue
        x, kept = _robust_triangulate(rec, k, o, fs, p.tri_px, _solvers(p))
        if x is None:
            continue
        for fr in set(fs) - set(kept):           # views that disagree leave the track
            o.pop(fr)
        fs = kept
        cs = np.stack([camera_centre(*rec.poses[fr]) for fr in fs])
        # widest angle between any two of the rays (first vs farthest centre suffices)
        far = int(np.argmax(np.linalg.norm(cs - cs[0], axis=1)))
        if triangulation_angle(x[None], cs[0], cs[far])[0] < p.min_angle:
            continue
        rec.points[tid] = x
        added += 1
    return added


def _gather(tracks, rec, frames):
    cam_of = {fr: i for i, fr in enumerate(frames)}
    pt_ids = sorted(rec.points)
    pt_of = {t: i for i, t in enumerate(pt_ids)}
    ci, pi, uv = [], [], []
    for t in pt_ids:
        for fr, xy in tracks.obs[t].items():
            if fr in cam_of:
                ci.append(cam_of[fr])
                pi.append(pt_of[t])
                uv.append(xy)
    return pt_ids, np.asarray(ci), np.asarray(pi), np.asarray(uv)


def _ba(tracks, rec, k, p, frames, refine_focal=False, filter_px=None, max_iter=30):
    pt_ids, ci, pi, uv = _gather(tracks, rec, frames)
    if len(pt_ids) == 0:
        return k
    rots = np.stack([rec.poses[f][0] for f in frames])
    ts = np.stack([rec.poses[f][1] for f in frames])
    pts = np.stack([rec.points[t] for t in pt_ids])
    if p.use_ba:
        res = bundle_adjust(rots, ts, pts, k[0, 0], (k[0, 2], k[1, 2]), ci, pi, uv,
                            fixed=(0,), refine_focal=refine_focal, max_iter=max_iter)
    else:
        res = BAResult(rots, ts, pts, float(k[0, 0]), 0.0, 0.0, 0)
    for i, f in enumerate(frames):
        rec.poses[f] = (res.rots[i], res.ts[i])
    for i, t in enumerate(pt_ids):
        rec.points[t] = res.points[i]
    k = k.copy()
    k[0, 0] = k[1, 1] = res.focal
    if filter_px is not None:
        err = np.linalg.norm(reprojection(res.rots, res.ts, res.points, res.focal, k[:2, 2], ci, pi) - uv, axis=1)
        xc_z = np.einsum("nj,nj->n", res.rots[ci][:, 2], res.points[pi]) + res.ts[ci][:, 2]
        bad = (err > filter_px) | (xc_z <= 0)
        for c, q in zip(ci[bad], pi[bad], strict=True):
            tracks.obs[pt_ids[q]].pop(frames[c], None)
        # drop points left with fewer than two keyframe observations
        reg = set(frames)
        for t in pt_ids:
            if sum(fr in reg for fr in tracks.obs[t]) < 2:
                rec.points.pop(t, None)
    return k


def _error_stats(tracks, rec, k, frames):
    pt_ids, ci, pi, uv = _gather(tracks, rec, frames)
    if len(pt_ids) == 0:
        return float("nan"), float("nan"), []
    rots = np.stack([rec.poses[f][0] for f in frames])
    ts = np.stack([rec.poses[f][1] for f in frames])
    pts = np.stack([rec.points[t] for t in pt_ids])
    err = np.linalg.norm(reprojection(rots, ts, pts, k[0, 0], k[:2, 2], ci, pi) - uv, axis=1)
    obs = [(frames[c], pt_ids[q], xy) for c, q, xy in zip(ci, pi, uv, strict=True)]
    return float(err.mean()), float(np.median(err)), obs


def run_sfm(tracks: Tracks, k: np.ndarray, width: int, params: SfMParams | None = None,
            keyframes: list[int] | None = None, register_all: bool = True) -> Reconstruction:
    """Reconstruct one shot from its tracks. `tracks` is modified (outlier
    observations are removed), so pass a copy if you need the original."""
    p = params or SfMParams()
    k = np.asarray(k, float).copy()
    kfs = keyframes or select_keyframes(tracks, width, p)
    rec = Reconstruction(k, [], {}, {}, [])
    rec.log.append(f"{len(kfs)} keyframes: {kfs}")
    if len(kfs) < 2:
        rec.ok = False
        rec.log.append("fewer than two keyframes: the camera hardly moves")
        return rec
    init = _initial_pair(tracks, kfs, k, p, rec.log)
    if init is None:
        rec.ok = False
        rec.log.append("no keyframe pair with enough parallax (pan/zoom/static shot?)")
        return rec
    _, fa, fb, rot, t = init
    rec.poses[fa] = (np.eye(3), np.zeros(3))
    rec.poses[fb] = (rot, t)
    registered = [fa, fb]
    _triangulate_new(tracks, rec, k, p, registered)
    k = _ba(tracks, rec, k, p, sorted(registered), filter_px=p.filter_px)
    rec.log.append(f"initial pair {fa}-{fb}: {len(rec.points)} points")

    since_ba, blocked = 0, set()
    while True:
        todo = [f for f in kfs if f not in rec.poses and f not in blocked]
        if not todo:
            break
        lo, hi = kfs.index(min(registered)), kfs.index(max(registered))
        # grow outwards from the registered range, one neighbour on each side
        cands = [kfs[i] for i in (lo - 1, hi + 1) if 0 <= i < len(kfs) and kfs[i] in todo]
        cands += [f for f in todo if lo < kfs.index(f) < hi]          # gaps left inside the range
        if not cands:
            break
        best = None
        for f in cands:
            ids, xy = tracks.in_frame(f)
            m = np.array([t in rec.points for t in ids], bool)
            if best is None or m.sum() > best[1].sum():
                best = (f, m, ids, xy)
        f, m, ids, xy = best
        if m.sum() < 12:
            rec.log.append(f"keyframe {f}: only {int(m.sum())} 2D-3D matches, stopping this side")
            blocked.add(f)
            continue
        x3 = np.stack([rec.points[t] for t in ids[m]])
        xn = pixels_to_normalised(xy[m], k)
        rot, tv, inl = _solvers(p).pnp(x3, xn, k[0, 0], p.pnp_px, seed=len(registered))
        if rot is None or inl.sum() < 12:
            rec.log.append(f"keyframe {f}: PnP failed ({int(inl.sum())} inliers)")
            blocked.add(f)
            continue
        for t in ids[m][~inl]:                  # outlying observations do not join the model
            tracks.obs[int(t)].pop(f, None)
        rec.poses[f] = (rot, tv)
        registered.append(f)
        n_new = _triangulate_new(tracks, rec, k, p, registered, candidates=[int(t) for t in ids])
        rec.log.append(f"keyframe {f}: PnP {int(inl.sum())}/{int(m.sum())} inliers, +{n_new} points")
        since_ba += 1
        if since_ba >= p.ba_every:
            k = _ba(tracks, rec, k, p, sorted(registered), filter_px=p.filter_px)
            since_ba = 0

    frames = sorted(registered)
    k = _ba(tracks, rec, k, p, frames, refine_focal=p.refine_focal, filter_px=p.filter_px, max_iter=50)
    k = _ba(tracks, rec, k, p, frames, refine_focal=p.refine_focal, max_iter=50)
    rec.k = k
    rec.keyframes = frames
    rec.mean_reproj_px, rec.median_reproj_px, rec.obs = _error_stats(tracks, rec, k, frames)
    rec.log.append(f"final: {len(frames)} keyframes, {len(rec.points)} points, "
                   f"mean reprojection {rec.mean_reproj_px:.3f} px, focal {k[0, 0]:.1f}")
    if len(rec.points) < p.min_points:
        rec.ok = False
    if register_all:
        register_frames(tracks, rec, p)
    return rec


def register_frames(tracks: Tracks, rec: Reconstruction, p: SfMParams) -> None:
    """Pose of every non-keyframe: PnP + RANSAC against the fixed 3D points, then
    LM on the inliers (no bundle adjustment, so these poses are noisier)."""
    k = rec.k
    for f in tracks.frames:
        if f in rec.poses:
            continue
        ids, xy = tracks.in_frame(f)
        m = np.array([t in rec.points for t in ids], bool)
        if m.sum() < 12:
            continue
        x3 = np.stack([rec.points[t] for t in ids[m]])
        xn = pixels_to_normalised(xy[m], k)
        rot, t, inl = _solvers(p).pnp(x3, xn, k[0, 0], p.pnp_px)
        if rot is None or inl.sum() < 12:
            continue
        rec.poses[f] = (rot, t)
