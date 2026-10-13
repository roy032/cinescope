"""Label every shot of data/own_shots with a shot scale, automatically (GPU).

    python p1_cnn/autolabel.py --root /tmp/own --face-ckpt centernet_r18.pt --out p1_cnn/results/own

Cues (see cinescope/framing/teacher.py): zero-shot SigLIP over class prompts,
and the largest-face-height ratio from the Phase 2 face detector, fused per
shot over its three key frames. Writes

  teacher.csv   every shot: split, per-cue class probabilities, face ratio, fused label + confidence
  index.csv     train/val shots with the fused label, in the format train_framing.py reads
                (test shots are left out: they are scored against hand labels only)
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cinescope.framing import SCALES  # noqa: E402
from cinescope.framing.teacher import (  # noqa: E402
    PROMPTS,
    class_scores,
    face_prior,
    face_ratio,
    fuse,
    softmax,
)


def load_shots(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["frames"] = r["frames"].split(";")
    return rows


def text_probs(paths: list[Path], model_name: str, bs: int = 64) -> dict[str, np.ndarray]:
    import torch
    from PIL import Image
    from transformers import AutoModel, AutoProcessor

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if dev == "cuda" else torch.float32
    model = AutoModel.from_pretrained(model_name, torch_dtype=dtype).to(dev).eval()
    proc = AutoProcessor.from_pretrained(model_name)

    def feats(out):
        out = out if torch.is_tensor(out) else out.pooler_output
        return torch.nn.functional.normalize(out.float(), dim=-1)

    prompts, prompt_class = [], []
    for c in SCALES:
        prompts += PROMPTS[c]
        prompt_class += [c] * len(PROMPTS[c])
    with torch.no_grad():
        t = proc(text=prompts, padding="max_length", return_tensors="pt").to(dev)
        tf = feats(model.get_text_features(**t))
        scale = model.logit_scale.exp().float()
        bias = model.logit_bias.float() if hasattr(model, "logit_bias") else 0.0
        out = {}
        for b in range(0, len(paths), bs):
            chunk = paths[b:b + bs]
            ims = [Image.open(p).convert("RGB") for p in chunk]
            px = proc(images=ims, return_tensors="pt")["pixel_values"].to(dev, dtype)
            logits = (feats(model.get_image_features(pixel_values=px)) @ tf.T * scale + bias).cpu().numpy()
            probs = softmax(class_scores(logits, prompt_class))
            for p, pr in zip(chunk, probs, strict=True):
                out[str(p)] = pr
            if b // bs % 50 == 0:
                print(f"  text cue {b + len(chunk)}/{len(paths)}", flush=True)
    del model
    return out


def face_ratios(paths: list[Path], ckpt: str) -> dict[str, float | None]:
    from PIL import Image

    from cinescope.detect.faces import FaceDetector

    det = FaceDetector(ckpt, threshold=0.3)
    out = {}
    for n, p in enumerate(paths):
        rgb = np.asarray(Image.open(p).convert("RGB"))
        d = det.detect([rgb])[0]
        out[str(p)] = face_ratio(d["boxes"], d["scores"], rgb.shape[0])
        if n % 2000 == 0:
            print(f"  face cue {n}/{len(paths)}", flush=True)
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="folder with shots.csv and frames/")
    ap.add_argument("--face-ckpt", required=True)
    ap.add_argument("--model", default="google/siglip-so400m-patch14-384")
    ap.add_argument("--w-face", type=float, default=1.0)
    ap.add_argument("--min-conf", type=float, default=0.0, help="drop train shots below this fused confidence")
    ap.add_argument("--out", default="p1_cnn/results/own")
    args = ap.parse_args(argv)

    root, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    shots = load_shots(root / "shots.csv")
    paths = sorted({root / f for s in shots for f in s["frames"]})
    print(f"{len(shots)} shots, {len(paths)} key frames", flush=True)
    t0 = time.time()
    tp = text_probs(paths, args.model)
    fr = face_ratios(paths, args.face_ckpt)
    print(f"cues done in {(time.time() - t0) / 60:.1f} min", flush=True)

    rows, index = [], []
    for s in shots:
        keys = [str(root / f) for f in s["frames"]]
        p_text = np.mean([tp[k] for k in keys], axis=0)
        ratios = [fr[k] for k in keys if fr[k] is not None]
        ratio = float(np.median(ratios)) if ratios else None
        p_face = face_prior(ratio)
        p = fuse(p_text, p_face, args.w_face)
        label, conf = SCALES[int(p.argmax())], float(p.max())
        row = {"split": s["split"], "episode": s["episode"], "shot": s["shot"], "frames": ";".join(s["frames"]),
               "face_ratio": "" if ratio is None else round(ratio, 4), "label": label, "conf": round(conf, 4)}
        for name, dist in (("text", p_text), ("face", p_face), ("fused", p)):
            for c, v in zip(SCALES, dist if dist is not None else [None] * len(SCALES), strict=True):
                row[f"{name}_{c}"] = "" if v is None else round(float(v), 4)
        rows.append(row)
        if s["split"] in ("train", "val") and (s["split"] == "val" or conf >= args.min_conf):
            index.append([s["split"], s["episode"], s["shot"], label, "", ";".join(s["frames"])])

    with (out / "teacher.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with (out / "index.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["split", "trailer", "shot", "scale", "movement", "frames"])
        w.writerows(index)
    dist = {c: sum(r["label"] == c for r in rows) for c in SCALES}
    faces = sum(r["face_ratio"] != "" for r in rows)
    print(f"teacher labels {dist}; a face on {faces}/{len(rows)} shots; index: {len(index)} train/val shots")


if __name__ == "__main__":
    main()
