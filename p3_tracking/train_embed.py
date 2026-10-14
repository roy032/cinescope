"""Train the face-embedding network from scratch on any clips, no labels (Week 10).

    python p3_tracking/train_embed.py trailers/Megamind.avi trailers/classroom.mp4 data/clips/*.mp4
    python p3_tracking/train_embed.py ... --loss arcface          # the naive baseline

Each clip is detected, cut into shots and tracked (Week 9); every track
contributes up to --per-track aligned face crops. Positives are crops of one
track; known negatives are tracks that share a frame, or come from different
clips. See cinescope/faces/embed.py.

The test clips of Weeks 9-11 (goldeneye, head-pose-face-detection-female-and-
male, face-demographics-walking) must not be passed here.

-> p3_tracking/checkpoints/embed_<loss>.pt, p3_tracking/results/embed_<loss>_history.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cinescope.faces.clip import extract_crops, load_crops, save_crops, track_clip  # noqa: E402
from cinescope.faces.cluster import cooccurring_pairs  # noqa: E402
from cinescope.faces.embed import ArcFaceLoss, EmbedNet, augment, batch_hard_triplet, to_input  # noqa: E402

TEST_CLIPS = {"goldeneye", "head-pose-face-detection-female-and-male", "face-demographics-walking"}


def clip_crops(video: str, out_root: Path, max_rgb: int, log=print):
    from p3_tracking.track_clip import load_or_cut, load_or_detect

    stem = Path(video).stem
    out = out_root / stem
    out.mkdir(parents=True, exist_ok=True)
    cache = out / f"crops_{max_rgb}.npz"
    if cache.exists():
        return load_crops(cache)
    dets = load_or_detect(video, out)
    cuts = load_or_cut(video, out)
    tracks = track_clip(dets, cuts)
    crops = extract_crops(video, dets, tracks, max_rgb=max_rgb)
    save_crops(cache, crops)
    log(f"  {stem}: {len(tracks)} tracks")
    return crops


def build_dataset(videos: list[str], out_root: Path, per_track: int, min_len: int, min_px: float):
    X, tid = [], []
    clip_of, frames_of = [], []
    for ci, v in enumerate(videos):
        if Path(v).stem in TEST_CLIPS:
            raise SystemExit(f"{v} is a test clip; refusing to train on it")
        for c in clip_crops(v, out_root, per_track):
            h = np.median([b[3] - b[1] for b in c.track.boxes])
            if len(c.track) < min_len or h < min_px or len(c.rgb) < 2:
                continue
            X.append(c.rgb)
            tid.append(np.full(len(c.rgb), len(clip_of)))
            clip_of.append(ci)
            frames_of.append(c.track.frames)
    clip_of = np.asarray(clip_of)
    n = len(clip_of)
    neg = clip_of[:, None] != clip_of[None, :]                 # different clips
    for ci in set(clip_of.tolist()):
        idx = np.flatnonzero(clip_of == ci)
        for a, b in cooccurring_pairs([frames_of[i] for i in idx]):
            neg[idx[a], idx[b]] = neg[idx[b], idx[a]] = True    # same frame, two people
    return np.concatenate(X), np.concatenate(tid), neg, clip_of, n


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--loss", default="triplet", choices=["triplet", "arcface"])
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--tracks-per-batch", type=int, default=32)
    ap.add_argument("--per-batch-track", type=int, default=4, help="crops per track in a batch")
    ap.add_argument("--per-track", type=int, default=64, help="crops kept per track")
    ap.add_argument("--min-len", type=int, default=8, help="frames")
    ap.add_argument("--min-px", type=float, default=32, help="median face height")
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--margin", type=float, default=0.3)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--ckpt", default=None)
    args = ap.parse_args(argv)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    g = torch.Generator().manual_seed(args.seed)

    X, tid, neg, clip_of, n_tracks = build_dataset(args.videos, Path(args.out), args.per_track, args.min_len,
                                                   args.min_px)
    by_track = [np.flatnonzero(tid == t) for t in range(n_tracks)]
    partners = [np.flatnonzero(neg[t] & (clip_of == clip_of[t])) for t in range(n_tracks)]
    print(f"{len(X)} crops, {n_tracks} tracks from {len(set(clip_of.tolist()))} clips; "
          f"{sum(len(p) > 0 for p in partners)} tracks have a co-occurring partner")

    model = EmbedNet()
    head = ArcFaceLoss(128, n_tracks) if args.loss == "arcface" else None
    params = list(model.parameters()) + (list(head.parameters()) if head else [])
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.1)
    ckpt = Path(args.ckpt or ROOT / "p3_tracking" / "checkpoints" / f"embed_{args.loss}.pt")
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    hist_path = ROOT / "p3_tracking" / "results" / f"embed_{args.loss}_history.csv"
    hist_path.parent.mkdir(parents=True, exist_ok=True)
    hist = []
    t0 = time.perf_counter()
    P, K = min(args.tracks_per_batch, n_tracks), args.per_batch_track
    for step in range(1, args.steps + 1):
        # Half the batch: random tracks plus one co-occurring partner each, so
        # the hard, same-scene negatives are present; the rest random.
        picked: list[int] = []
        for t in rng.permutation(n_tracks):
            if len(picked) >= P:
                break
            if t in picked:
                continue
            picked.append(int(t))
            if len(picked) < P // 2 * 2 and len(partners[t]):
                p = int(rng.choice(partners[t]))
                if p not in picked:
                    picked.append(p)
        idx = np.concatenate([rng.choice(by_track[t], K, replace=len(by_track[t]) < K) for t in picked])
        x = augment(to_input(X[idx], model.size), g)
        y = torch.as_tensor(tid[idx])
        model.train()
        emb = model(x)
        if head is None:
            kn = torch.as_tensor(neg[np.ix_(tid[idx], tid[idx])])
            loss, st = batch_hard_triplet(emb, y, kn, args.margin)
        else:
            loss, st = head(emb, y), {}
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % 50 == 0 or step == 1:
            row = {"step": step, "loss": round(loss.item(), 4), **{k: round(v, 4) for k, v in st.items()},
                   "lr": round(sched.get_last_lr()[0], 6), "sec": round(time.perf_counter() - t0, 1)}
            hist.append(row)
            print(row, flush=True)
    torch.save({"model": model.state_dict(), "config": {}, "loss": args.loss, "videos": args.videos,
                "steps": args.steps, "n_tracks": n_tracks, "n_crops": len(X)}, ckpt)
    with hist_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in hist for k in r)))
        w.writeheader()
        w.writerows(hist)
    print(f"saved {ckpt} ({time.perf_counter() - t0:.0f} s)")


if __name__ == "__main__":
    main()
