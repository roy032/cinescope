"""Scene types from scratch: VideoMAE pre-training, then fine-tuning / probing (CPU).

    python p6_pose_video/train_scenes.py pretrain --minutes 30                  # VideoMAE on unlabelled windows
    python p6_pose_video/train_scenes.py finetune --init outputs/p6/videomae.pt --minutes 10
    python p6_pose_video/train_scenes.py finetune --minutes 10                  # same video ViT, random init
    python p6_pose_video/train_scenes.py vit --minutes 10                       # image ViT ([CLS]) on middle frames
    python p6_pose_video/train_scenes.py probe --init outputs/p6/videomae.pt    # frozen encoder + logistic regression
    python p6_pose_video/train_scenes.py evaluate                               # all of the above + CLIP on the test shots

Data: outputs/p6/scenes_train.npz / scenes_test.npz (build_scene_data.py).
Inputs are 4 frames of 64 x 112 px (tubelets 2 x 8 x 8 -> 2 x 8 x 14 = 224
tokens) for the video models and the middle frame for the image ViT. The
encoder is ViT-Tiny-like but half as deep: width 192, 6 blocks, 3 heads.

Training labels are the CLIP teacher's class probabilities on the training
clips (soft cross-entropy, i.e. distillation); the test clips' labels are by
eye and only used by `evaluate`. Augmentation: random 4 of the 8 stored frames
(consecutive), horizontal flip, random resized crop (scale 0.6-1), brightness
and contrast jitter.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cinescope.video.scenes import CLASSES, accuracy, softmax  # noqa: E402
from cinescope.video.videomae import VideoMAE  # noqa: E402
from cinescope.video.vit import ViT  # noqa: E402

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1)
HW = (64, 112)
ARCH = {"img_hw": HW, "patch": 8, "frames": 4, "tubelet": 2, "dim": 192, "depth": 6, "heads": 3}
LABELS = Path(__file__).resolve().parent / "labels"


def to_clip(frames: np.ndarray) -> torch.Tensor:
    """(B, T, H, W, 3) uint8 -> normalised (B, 3, T, H, W)."""
    x = torch.from_numpy(frames).float().div(255).permute(0, 4, 1, 2, 3)
    return (x - MEAN) / STD


def augment(frames: np.ndarray, rng: np.random.Generator, train: bool) -> np.ndarray:
    """(8, H, W, 3) -> (4, H, W, 3)."""
    import cv2

    s = int(rng.integers(0, 5)) if train else 2
    f = frames[s:s + 4]
    if not train:
        return f
    if rng.random() < 0.5:
        f = f[:, :, ::-1]
    sc = rng.uniform(0.6, 1.0)
    h, w = HW
    ch, cw = int(h * sc), int(w * sc)
    y, x = int(rng.integers(0, h - ch + 1)), int(rng.integers(0, w - cw + 1))
    f = np.stack([cv2.resize(np.ascontiguousarray(im[y:y + ch, x:x + cw]), (w, h)) for im in f])
    f = np.clip(f.astype(np.float32) * rng.uniform(0.7, 1.3) + rng.uniform(-20, 20), 0, 255).astype(np.uint8)
    return f


def load(path: str):
    d = np.load(path)
    return {k: d[k] for k in d.files}


def cosine_lr(base: float, frac: float, warm: float = 0.05) -> float:
    if frac < warm:
        return base * frac / warm
    return base * 0.5 * (1 + np.cos(np.pi * min(1.0, (frac - warm) / (1 - warm))))


def pretrain(args):
    d = load(args.train)
    frames = d["frames"]
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    mae = VideoMAE(**ARCH, mask_ratio=args.mask)
    opt = torch.optim.AdamW(mae.parameters(), args.lr, betas=(0.9, 0.95), weight_decay=0.05)
    t0, step, log = time.time(), 0, []
    budget = args.minutes * 60
    print(f"{len(frames)} windows; {sum(p.numel() for p in mae.parameters()) / 1e6:.2f} M parameters")
    while time.time() - t0 < budget:
        lr = cosine_lr(args.lr, (time.time() - t0) / budget)
        for g in opt.param_groups:
            g["lr"] = lr
        idx = rng.choice(len(frames), args.bs)
        x = to_clip(np.stack([augment(frames[i], rng, True) for i in idx]))
        loss, _, _ = mae(x)
        opt.zero_grad()
        loss.backward()
        opt.step()
        step += 1
        if step % 50 == 0:
            log.append({"step": step, "seconds": round(time.time() - t0), "loss": round(loss.item(), 4)})
            print(f"step {step} {time.time() - t0:.0f}s lr {lr:.2e} loss {loss.item():.4f}", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": mae.state_dict(), "arch": ARCH, "mask": args.mask, "steps": step, "log": log}, args.out)
    print(f"saved {args.out} after {step} steps")


def soft_targets(logits: np.ndarray) -> np.ndarray:
    return softmax(logits, 1).astype(np.float32)


def make_model(kind: str, init: str | None) -> ViT:
    if kind == "vit":
        return ViT(HW, 8, 1, 1, 192, 6, 3, num_classes=len(CLASSES), pool="cls", dropout=0.1)
    m = ViT(**ARCH, num_classes=len(CLASSES), pool="mean", dropout=0.1)
    if init:
        mae_state = torch.load(init, map_location="cpu", weights_only=False)["model"]
        enc = {k[len("encoder."):]: v for k, v in mae_state.items() if k.startswith("encoder.")}
        missing, unexpected = m.load_state_dict(enc, strict=False)
        assert not unexpected and set(missing) <= {"head.weight", "head.bias"}, (missing, unexpected)
    return m


def model_input(kind: str, frames4: np.ndarray) -> torch.Tensor:
    x = to_clip(frames4)
    return x[:, :, 1] if kind == "vit" else x      # image ViT: one frame


def finetune(args, kind: str):
    d = load(args.train)
    frames, y = d["frames"], soft_targets(d["logits"])
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    model = make_model(kind, args.init)
    opt = torch.optim.AdamW(model.parameters(), args.lr, weight_decay=0.05)
    t0, step, log = time.time(), 0, []
    budget = args.minutes * 60
    model.train()
    while time.time() - t0 < budget:
        lr = cosine_lr(args.lr, (time.time() - t0) / budget, warm=0.1)
        for g in opt.param_groups:
            g["lr"] = lr
        idx = rng.choice(len(frames), args.bs)
        x = model_input(kind, np.stack([augment(frames[i], rng, True) for i in idx]))
        loss = -(torch.from_numpy(y[idx]) * F.log_softmax(model(x), 1)).sum(1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        step += 1
        if step % 25 == 0:
            log.append({"step": step, "seconds": round(time.time() - t0), "loss": round(loss.item(), 4)})
            print(f"step {step} {time.time() - t0:.0f}s lr {lr:.2e} loss {loss.item():.4f}", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "kind": kind, "init": args.init, "steps": step, "log": log}, args.out)
    print(f"saved {args.out} after {step} steps")


@torch.no_grad()
def predict_windows(model: ViT, kind: str, frames: np.ndarray, bs: int = 32) -> np.ndarray:
    model.eval()
    out = []
    rng = np.random.default_rng(0)
    for i in range(0, len(frames), bs):
        x = model_input(kind, np.stack([augment(f, rng, False) for f in frames[i:i + bs]]))
        out.append(F.softmax(model(x), 1).numpy())
    return np.concatenate(out)


@torch.no_grad()
def encode(model: ViT, frames: np.ndarray, bs: int = 32) -> np.ndarray:
    model.eval()
    rng = np.random.default_rng(0)
    return np.concatenate([model.features(to_clip(np.stack([augment(f, rng, False) for f in frames[i:i + bs]])))
                           .numpy() for i in range(0, len(frames), bs)])


def probe(args):
    """Linear probe: frozen (pre-trained or random) encoder, logistic regression on
    the mean-pooled features, fitted to the teacher's hard labels."""
    from sklearn.linear_model import LogisticRegression

    tr = load(args.train)
    model = make_model("video", args.init)
    f_tr = encode(model, tr["frames"])
    y = tr["logits"].argmax(1)
    clf = LogisticRegression(max_iter=2000, C=args.C).fit(f_tr, y)
    torch.save({"model": model.state_dict(), "kind": "probe", "init": args.init,
                "coef": clf.coef_, "intercept": clf.intercept_, "classes": clf.classes_}, args.out)
    print(f"probe: train acc vs teacher {clf.score(f_tr, y):.3f} -> {args.out}")


def shot_predictions(test: dict, probs: np.ndarray) -> dict[str, dict[str, str]]:
    """Mean window probabilities per shot -> {clip: {shot: class}}."""
    out: dict[str, dict[str, str]] = {}
    for clip in np.unique(test["clip"]):
        m = test["clip"] == clip
        for s in np.unique(test["shot"][m]):
            w = m & (test["shot"] == s)
            out.setdefault(str(clip), {})[str(int(s))] = CLASSES[int(probs[w].mean(0).argmax())]
    return out


def evaluate(args):
    test = load(args.test)
    truth = {}
    for clip in np.unique(test["clip"]):
        p = LABELS / f"{clip}_scenes.json"
        truth[str(clip)] = json.loads(p.read_text(encoding="utf-8"))["shots"]
    systems: dict[str, np.ndarray] = {"open_clip zero-shot (teacher, library)": softmax(test["logits"], 1)}
    ck_dir = Path(args.ckpt_dir)
    for name, fn in [("VideoMAE pre-trained + fine-tuned", "scene_mae_ft.pt"),
                     ("video ViT, random init, fine-tuned", "scene_scratch_ft.pt"),
                     ("image ViT ([CLS]) from scratch", "scene_vit.pt"),
                     ("VideoMAE pre-trained + linear probe", "scene_mae_probe.pt"),
                     ("random-init encoder + linear probe", "scene_rand_probe.pt")]:
        p = ck_dir / fn
        if not p.exists():
            continue
        ck = torch.load(p, map_location="cpu", weights_only=False)
        if ck["kind"] == "probe":
            model = make_model("video", None)
            model.load_state_dict(ck["model"])
            f = encode(model, test["frames"])
            z = f @ ck["coef"].T + ck["intercept"]
            probs = np.zeros((len(f), len(CLASSES)))
            probs[:, ck["classes"]] = softmax(z, 1)
        else:
            model = make_model(ck["kind"], None)
            model.load_state_dict(ck["model"])
            probs = predict_windows(model, ck["kind"], test["frames"])
        systems[name] = probs
    teacher_shots = shot_predictions(test, systems["open_clip zero-shot (teacher, library)"])
    res = {}
    for name, probs in systems.items():
        shots_pred = shot_predictions(test, probs)
        merged_pred = {f"{c}/{s}": v for c, d in shots_pred.items() for s, v in d.items()}
        merged_truth = {f"{c}/{s}": v for c, d in truth.items() for s, v in d.items()}
        merged_teacher = {f"{c}/{s}": v for c, d in teacher_shots.items() for s, v in d.items()}
        r = accuracy(merged_pred, merged_truth)
        r["agreement_with_teacher"] = float(np.mean([merged_pred[k] == merged_teacher[k] for k in merged_pred]))
        r["predictions"] = merged_pred
        res[name] = r
        print(f"{name:45s} accuracy {r['accuracy']:.3f} (n={r['n']})  agreement with teacher "
              f"{r['agreement_with_teacher']:.3f}")
    tr = load(args.train)
    res["_teacher_label_distribution_train"] = {c: int((tr["logits"].argmax(1) == i).sum())
                                                for i, c in enumerate(CLASSES)}
    Path(args.results).parent.mkdir(parents=True, exist_ok=True)
    Path(args.results).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"-> {args.results}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["pretrain", "finetune", "vit", "probe", "evaluate"])
    ap.add_argument("--train", default="outputs/p6/scenes_train.npz")
    ap.add_argument("--test", default="outputs/p6/scenes_test.npz")
    ap.add_argument("--init", default=None, help="VideoMAE checkpoint for finetune / probe")
    ap.add_argument("--minutes", type=float, default=10)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--mask", type=float, default=0.9)
    ap.add_argument("--C", type=float, default=0.5)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", default=None)
    ap.add_argument("--ckpt-dir", default="outputs/p6")
    ap.add_argument("--results", default="p6_pose_video/results/scenes.json")
    args = ap.parse_args(argv)
    torch.set_num_threads(args.threads)
    if args.cmd == "pretrain":
        args.lr = args.lr or 1e-3
        args.out = args.out or "outputs/p6/videomae.pt"
        pretrain(args)
    elif args.cmd in ("finetune", "vit"):
        args.lr = args.lr or 5e-4
        default = "scene_vit.pt" if args.cmd == "vit" else ("scene_mae_ft.pt" if args.init else "scene_scratch_ft.pt")
        args.out = args.out or f"outputs/p6/{default}"
        finetune(args, "vit" if args.cmd == "vit" else "video")
    elif args.cmd == "probe":
        args.out = args.out or ("outputs/p6/scene_mae_probe.pt" if args.init else "outputs/p6/scene_rand_probe.pt")
        probe(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
