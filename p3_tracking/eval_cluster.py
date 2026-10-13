"""Score character clustering against hand-labelled identities (Week 10).

    python p3_tracking/eval_cluster.py trailers/goldeneye.mp4 trailers/head-pose-face-detection-female-and-male.mp4 trailers/face-demographics-walking.mp4

For each clip: tracks (Week 9) -> one embedding per track (mean of its
aligned faces' embeddings) -> agglomerative clustering with cannot-link
constraints, the merge threshold set per clip without labels
(`cluster.auto_threshold`). Embedders compared: ours (trained from scratch,
self-supervised, on other clips; triplet and ArcFace-on-track-ids variants)
and OpenCV zoo's SFace.

Two protocols:
  tracks      the tracker's tracks as they are;
  tracklets   every track cut into 2-second pieces. The head-pose and
              walking clips have one unbroken track per person, which makes
              clustering trivial; cutting them up asks the embedding to
              link the same person across poses and distances, as a film's
              shots would.

Ground truth per track: the majority identity of the labelled boxes it covers
(IoU >= 0.5); tracks with no labelled frame or labelled "ignore" are
clustered but not scored. Purity and pairwise F1 are over tracks (each
counts once); `purity_time` weights tracks by their length.

-> p3_tracking/results/cluster.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cinescope.detect.boxes import iou  # noqa: E402
from cinescope.faces.clip import TrackCrops  # noqa: E402
from cinescope.faces.cluster import (  # noqa: E402
    agglomerative,
    auto_threshold,
    cooccurring_pairs,
    cosine_distances,
    inverse_purity,
    pairwise_f1,
    purity,
)
from cinescope.track.tracker import Track  # noqa: E402


def gt_identity(track: Track, labels: dict) -> str | None:
    votes = Counter()
    for f, b in zip(track.frames, track.boxes, strict=True):
        fr = labels.get(f)
        if not fr:
            continue
        gb = np.asarray([x["bbox"] for x in fr]).reshape(-1, 4)
        ov = iou(np.asarray(b)[None], gb)[0]
        j = int(np.argmax(ov))
        if ov[j] >= 0.5:
            votes[fr[j]["id"]] += 1
    if not votes:
        return None
    ident = votes.most_common(1)[0][0]
    return None if ident == "ignore" else ident


def split_tracklets(crops: list[TrackCrops], seconds: float, fps: float) -> list[TrackCrops]:
    n = max(1, int(round(seconds * fps)))
    out = []
    for c in crops:
        t = c.track
        fr = np.asarray(t.frames)
        rf = np.asarray(c.rgb_frames)
        for a in range(0, len(t), n):
            sl = slice(a, a + n)
            keep = (rf >= fr[sl][0]) & (rf <= fr[sl][-1])
            if keep.sum() == 0:
                continue
            sub = Track(t.id * 1000 + a // n, fr[sl].tolist(), t.boxes[sl], t.scores[sl], t.det_index[sl], t.shot)
            out.append(TrackCrops(sub, c.gray[sl], c.asd[sl], c.rgb[keep], rf[keep].tolist()))
    return out


def track_embeddings(crops: list[TrackCrops], embedder) -> np.ndarray:
    allc = np.concatenate([c.rgb for c in crops])
    e = embedder(allc)
    out, i = [], 0
    for c in crops:
        m = e[i:i + len(c.rgb)].mean(0)
        out.append(m / max(np.linalg.norm(m), 1e-12))
        i += len(c.rgb)
    return np.asarray(out)


def self_pairs(crops: list[TrackCrops], embedder) -> np.ndarray:
    """Distances between the two halves of each track (known same person)."""
    d = []
    for c in crops:
        if len(c.rgb) < 4:
            continue
        h = len(c.rgb) // 2
        a, b = embedder(c.rgb[:h]).mean(0), embedder(c.rgb[h:]).mean(0)
        a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
        d.append(1 - float(a @ b))
    return np.asarray(d)


def cluster_tracks(crops: list[TrackCrops], embedder, threshold: float | None = None,
                   same_from: list[TrackCrops] | None = None):
    """same_from: the tracks whose halves give the known-same pairs (default:
    `crops` itself; for tracklets, the whole tracks they were cut from)."""
    E = track_embeddings(crops, embedder)
    D = cosine_distances(E)
    cl = cooccurring_pairs([c.track.frames for c in crops])
    if threshold is None:
        same = self_pairs(same_from or crops, embedder)
        diff = np.asarray([D[a, b] for a, b in cl])
        threshold = auto_threshold(same, diff)
        if not np.isfinite(threshold):        # no co-occurring tracks: fall back to a fixed value
            threshold = 0.5
    labels = agglomerative(D, threshold, cannot_link=cl)
    return labels, float(threshold), D


def score(labels: np.ndarray, truth: list, lengths: np.ndarray) -> dict:
    m = np.asarray([t is not None for t in truth])
    p, tr, w = labels[m], [t for t in truth if t is not None], lengths[m]
    return {"purity": purity(p, tr), "purity_time": purity(p, tr, w), "inverse_purity": inverse_purity(p, tr),
            **{f"pair_{k}": v for k, v in pairwise_f1(p, tr).items()},
            "clusters": int(len(set(p.tolist()))), "identities": int(len(set(tr))),
            "tracks_scored": int(m.sum()), "tracks_total": int(len(truth))}


def embedders(ckpt_dir: Path) -> dict:
    from cinescope.faces.embed import SFaceEmbedder, embed, load

    out = {}
    for loss in ("triplet", "arcface"):
        p = ckpt_dir / f"embed_{loss}.pt"
        if p.exists():
            m = load(str(p))
            out[f"ours ({loss})"] = (lambda x, m=m: embed(m, x))
    out["SFace (opencv_zoo)"] = SFaceEmbedder()
    return out


def main(argv=None):
    from p3_tracking.train_embed import clip_crops

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--tracklet-seconds", type=float, default=2.0)
    ap.add_argument("--per-track", type=int, default=120,
                    help="aligned colour crops kept per track (enough for every 2 s tracklet to get several)")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--ckpt-dir", default=str(ROOT / "p3_tracking" / "checkpoints"))
    ap.add_argument("--save", default=str(ROOT / "p3_tracking" / "results" / "cluster.json"))
    args = ap.parse_args(argv)
    embs = embedders(Path(args.ckpt_dir))
    res = {}
    for v in args.videos:
        stem = Path(v).stem
        lab = json.loads((ROOT / "p3_tracking" / "labels" / f"{stem}.ids.json").read_text())
        labels = {f["index"]: f["boxes"] for f in lab["frames"]}
        crops = clip_crops(v, Path(args.out), args.per_track)
        res[stem] = {}
        for proto, cs in (("tracks", crops), ("tracklets", split_tracklets(crops, args.tracklet_seconds,
                                                                             lab["fps"]))):
            truth = [gt_identity(c.track, labels) for c in cs]
            lengths = np.asarray([len(c.track) for c in cs], float)
            res[stem][proto] = {}
            for name, emb in embs.items():
                lab_pred, thr, D = cluster_tracks(cs, emb, same_from=crops)
                r = {"threshold": round(thr, 4), **score(lab_pred, truth, lengths)}
                # Diagnostic only: the same embeddings cut at the true number of identities.
                k = len({t for t in truth if t is not None})
                if k:
                    orc = agglomerative(D, 0, cannot_link=cooccurring_pairs([c.track.frames for c in cs]),
                                        n_clusters=k)
                    r["oracle_k"] = {kk: score(orc, truth, lengths)[kk] for kk in ("purity", "pair_f1")}
                res[stem][proto][name] = r
                print(f"{stem:42s} {proto:9s} {name:20s} thr {thr:.3f} purity {r['purity']:.3f} "
                      f"inv {r['inverse_purity']:.3f} pairF1 {r['pair_f1']:.3f} "
                      f"clusters {r['clusters']}/{r['identities']} ids ({r['tracks_scored']} scored)")
    save = Path(args.save)
    save.parent.mkdir(parents=True, exist_ok=True)
    save.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
