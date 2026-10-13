"""Train the from-scratch CLIP-style dual encoder (cinescope/search/dual_encoder.py) on a CPU.

    python p8_search/train_dual_encoder.py shapes                 # synthetic shapes & colours captions
    python p8_search/train_dual_encoder.py frames --frames outputs/p8_train_frames.npz
                                                                 # video frames + automatic captions

shapes  Images of 1-2 coloured shapes with captions such as "a red circle left of a
        blue square". Three (colour, shape) combinations never appear in training;
        the test measures retrieval on seen and on those unseen combinations, and
        whether the model can tell "A left of B" from "B left of A".
frames  Frames from public clips that are NOT in the search evaluation, each captioned
        automatically by open_clip zero-shot tags over the COCO + ImageNet class names
        (built by p8_search/caption_frames.py). The resulting encoder plugs into
        search.py/eval_search.py as `--encoder scratch:<ckpt>`.

Writes the checkpoint and a JSON of results to p8_search/results/.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cinescope.search.dual_encoder import (  # noqa: E402
    COLOURS,
    SHAPES,
    DualEncoder,
    DualEncoderConfig,
    WordTokenizer,
    shapes_sample,
    shapes_vocab,
    swap_relation,
    to_tensor,
    train_contrastive,
)

HELD_OUT = frozenset({("purple", "triangle"), ("yellow", "circle"), ("green", "square")})


@torch.no_grad()
def embed_images(model: DualEncoder, images: list[np.ndarray], bs: int = 256) -> np.ndarray:
    return np.concatenate([model.encode_image(to_tensor(images[i:i + bs])).numpy() for i in range(0, len(images), bs)])


@torch.no_grad()
def embed_texts(model: DualEncoder, texts: list[str], bs: int = 512) -> np.ndarray:
    return np.concatenate([model.encode_text(model.tokenizer(texts[i:i + bs])).numpy()
                           for i in range(0, len(texts), bs)])


def text_to_image_recall(model, images, caps, ks=(1, 5)) -> dict:
    """Each distinct caption queries the image set; a hit is any image with that caption."""
    uniq = sorted(set(caps))
    u, v = embed_images(model, images), embed_texts(model, uniq)
    order = np.argsort(-(v @ u.T), axis=1)
    caps_arr = np.asarray(caps)
    out = {}
    for k in ks:
        out[f"R@{k}"] = round(float(np.mean([(caps_arr[order[i, :k]] == c).any() for i, c in enumerate(uniq)])), 4)
    return out


def eval_shapes(model: DualEncoder, size: int, n: int = 1000, seed: int = 123) -> dict:
    rng = np.random.default_rng(seed)
    seen = [shapes_sample(rng, size, HELD_OUT) for _ in range(n)]
    unseen = [shapes_sample(rng, size, HELD_OUT, only_held_out=True) for _ in range(n)]
    res = {"seen_compositions": text_to_image_recall(model, [s[0] for s in seen], [s[1] for s in seen]),
           "held_out_compositions": text_to_image_recall(model, [s[0] for s in unseen], [s[1] for s in unseen])}
    # word order: true caption vs the same words with the two objects swapped
    for name, data in (("seen", seen), ("held_out", unseen)):
        pairs = [(im, c, swap_relation(c)) for im, c in data if swap_relation(c)]
        u = embed_images(model, [p[0] for p in pairs])
        a, b = embed_texts(model, [p[1] for p in pairs]), embed_texts(model, [p[2] for p in pairs])
        res[f"relation_order_acc_{name}"] = round(float(np.mean((u * a).sum(1) > (u * b).sum(1))), 4)
    # zero-shot classification of single-shape images over all 36 "a {size} {colour} {shape}" prompts
    singles = [(im, c) for im, c in seen + unseen if " of " not in c and "above" not in c]
    labels = [f"a {sz} {c} {s}" for sz in ("big", "small") for c in COLOURS for s in SHAPES]
    u, t = embed_images(model, [s[0] for s in singles]), embed_texts(model, labels)
    pred = np.asarray(labels)[(u @ t.T).argmax(1)]
    truth = np.asarray([s[1] for s in singles])
    res["zero_shot_36way_acc"] = round(float(np.mean(pred == truth)), 4)
    res["zero_shot_colour_acc"] = round(float(np.mean([p.split()[2] == q.split()[2] for p, q in zip(pred, truth, strict=True)])), 4)
    res["zero_shot_shape_acc"] = round(float(np.mean([p.split()[3] == q.split()[3] for p, q in zip(pred, truth, strict=True)])), 4)
    return res


def run_shapes(args) -> dict:
    cfg = DualEncoderConfig(image_size=args.size, embed_dim=128, vocab=shapes_vocab(), max_len=12,
                            widths=(32, 64, 128, 128))
    model = DualEncoder(cfg)
    rng = np.random.default_rng(args.seed)

    def batch(_step):
        s = [shapes_sample(rng, args.size, HELD_OUT) for _ in range(args.bs)]
        return [x[0] for x in s], [x[1] for x in s]

    before = eval_shapes(model, args.size, n=500)
    t = time.perf_counter()
    losses = train_contrastive(model, batch, args.steps, lr=args.lr, multi_positive=not args.plain_infonce)
    train_s = time.perf_counter() - t
    after = eval_shapes(model, args.size)
    return {"model": model, "untrained": before, "trained": after, "train_s": round(train_s, 1),
            "final_loss": round(float(np.mean(losses[-50:])), 4)}


def augment_frames(x: torch.Tensor) -> torch.Tensor:
    """Random resized crop (scale 0.6-1), horizontal flip, brightness/contrast jitter."""
    b, _, s, _ = x.shape
    out = torch.empty_like(x)
    for i in range(b):
        sc = float(np.random.uniform(0.6, 1.0))
        cs = max(8, int(s * sc))
        y0, x0 = np.random.randint(0, s - cs + 1, 2)
        crop = x[i:i + 1, :, y0:y0 + cs, x0:x0 + cs]
        crop = torch.nn.functional.interpolate(crop, size=(s, s), mode="bilinear", align_corners=False)
        if np.random.rand() < 0.5:
            crop = crop.flip(3)
        crop = (crop - 0.5) * float(np.random.uniform(0.8, 1.2)) + 0.5 + float(np.random.uniform(-0.1, 0.1))
        out[i] = crop[0].clamp(0, 1)
    return out


def run_frames(args) -> dict:
    z = np.load(args.frames, allow_pickle=False)
    images, caps = z["images"], [str(c) for c in z["captions"]]
    clips = [str(c) for c in z["clips"]]
    held = np.asarray([c == args.val_clip for c in clips])
    tok = WordTokenizer.build([c for c, h in zip(caps, held, strict=True) if not h], max_len=24)
    cfg = DualEncoderConfig(image_size=images.shape[1], embed_dim=256, vocab=tok.vocab, max_len=24,
                            widths=(32, 64, 128, 256))
    model = DualEncoder(cfg)
    tr = np.flatnonzero(~held)
    rng = np.random.default_rng(args.seed)

    def batch(_step):
        idx = rng.choice(tr, args.bs, replace=False)
        return list(images[idx]), [caps[i] for i in idx]

    t = time.perf_counter()
    losses = train_contrastive(model, batch, args.steps, lr=args.lr, augment=augment_frames,
                               multi_positive=not args.plain_infonce)
    res = {"model": model, "train_s": round(time.perf_counter() - t, 1),
           "final_loss": round(float(np.mean(losses[-50:])), 4), "n_train_frames": int(len(tr)),
           "vocab_size": len(tok.vocab), "val_clip": args.val_clip}
    if held.any():
        vi = np.flatnonzero(held)
        res["val_text_to_image"] = text_to_image_recall(model, list(images[vi]), [caps[i] for i in vi])
    return res


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("task", choices=["shapes", "frames"])
    ap.add_argument("--frames", help="npz from caption_frames.py (frames task)")
    ap.add_argument("--val-clip", default="", help="frames task: clip held out for a validation score")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--size", type=int, default=32, help="shapes task image size")
    ap.add_argument("--plain-infonce", action="store_true", help="diagonal-only positives (ablation)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", default=str(ROOT / "p8_search" / "checkpoints"))
    ap.add_argument("--tag", default="")
    args = ap.parse_args(argv)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(args.threads)

    res = run_shapes(args) if args.task == "shapes" else run_frames(args)
    model = res.pop("model")
    name = f"dual_{args.task}{args.tag}"
    Path(args.out).mkdir(parents=True, exist_ok=True)
    ckpt = Path(args.out).resolve() / f"{name}.pt"
    model.save(ckpt)
    res.update({"steps": args.steps, "batch": args.bs, "lr": args.lr,
                "multi_positive": not args.plain_infonce, "checkpoint": ckpt.relative_to(ROOT).as_posix()
                if ckpt.resolve().is_relative_to(ROOT) else str(ckpt),
                "params": sum(p.numel() for p in model.parameters())})
    out = ROOT / "p8_search" / "results" / f"{name}.json"
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))
    print(f"-> {ckpt}, {out}")


if __name__ == "__main__":
    main()
