"""Sample frames from clips, embed them with open_clip, and caption them automatically.

    python p8_search/caption_frames.py data/train_clips/*.mp4 --out outputs/p8_train_frames.npz
    python p8_search/caption_frames.py trailers/*.mp4 trailers/*.avi --out outputs/p8_eval_frames.npz

Each sampled frame is cut into the same square tiles the search index uses
(encoders.square_tiles). Every tile gets
  * its open_clip ViT-B/32 embedding — the "real CLIP embeddings" of the ANN
    benchmark (bench_ann.py), and
  * a caption from open_clip zero-shot classification over a vocabulary fixed in
    advance: the 80 COCO and 1000 ImageNet class names (not words chosen after
    seeing the search queries). Caption = "a photo of a <top1>, a <top2> and a <top3>".
  * a small copy (64 x 64) for training the from-scratch dual encoder.

That is distillation, and the README says so: the from-scratch encoder learns
open_clip's opinion of these frames in COCO/ImageNet words, not human captions.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cinescope.io.video import VideoReader  # noqa: E402
from cinescope.search.encoders import get_encoder, square_tiles  # noqa: E402

COCO = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
        "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
        "scissors", "teddy bear", "hair drier", "toothbrush"]


def concept_vocab() -> list[str]:
    from open_clip.zero_shot_metadata import IMAGENET_CLASSNAMES

    seen, out = set(), []
    for w in COCO + list(IMAGENET_CLASSNAMES):
        if w.lower() not in seen:
            seen.add(w.lower())
            out.append(w)
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--max-per-clip", type=int, default=240, help="evenly thin long clips to this many frames")
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--encoder", default="openclip")
    args = ap.parse_args(argv)

    enc = get_encoder(args.encoder, tiles=False)
    vocab = concept_vocab()
    vt = enc.encode_texts([f"a photo of a {v}." for v in vocab])
    rows: dict[str, list] = {k: [] for k in ("images", "emb", "captions", "clips", "times", "tile")}
    t0 = time.perf_counter()
    for v in args.videos:
        frames = []
        with VideoReader(v, width=448) as vr:
            for f in vr.frames(sample_fps=args.fps):
                frames.append((f.t, f.rgb))
        if len(frames) > args.max_per_clip:
            keep = np.linspace(0, len(frames) - 1, args.max_per_clip).round().astype(int)
            frames = [frames[i] for i in keep]
        tiles, meta = [], []
        for t, rgb in frames:
            for j, tile in enumerate(square_tiles(rgb)):
                tiles.append(tile)
                meta.append((t, j))
        emb = enc.encode_images(tiles)
        logits = 100.0 * emb @ vt.T
        top = np.argsort(-logits, 1)[:, :3]
        for (t, j), tile, e, tp in zip(meta, tiles, emb, top, strict=True):
            rows["images"].append(np.asarray(Image.fromarray(tile).resize((args.size, args.size), Image.BILINEAR)))
            rows["emb"].append(e)
            rows["captions"].append(f"a photo of a {vocab[tp[0]]}, a {vocab[tp[1]]} and a {vocab[tp[2]]}")
            rows["clips"].append(Path(v).stem)
            rows["times"].append(t)
            rows["tile"].append(j)
        print(f"{Path(v).name}: {len(frames)} frames, {len(tiles)} tiles  e.g. '{rows['captions'][-1]}'  "
              f"({time.perf_counter() - t0:.0f} s)", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, images=np.stack(rows["images"]), emb=np.stack(rows["emb"]).astype(np.float32),
                        captions=np.asarray(rows["captions"]), clips=np.asarray(rows["clips"]),
                        times=np.asarray(rows["times"]), tile=np.asarray(rows["tile"]), encoder=np.asarray(enc.name))
    print(f"-> {args.out}: {len(rows['captions'])} tiles, {len(set(rows['captions']))} distinct captions")


if __name__ == "__main__":
    main()
