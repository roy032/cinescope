"""Train the CRNN on synthetic subtitle lines, on CPU.

    python p7_ocr/train_crnn.py --bg-clips trailers/vtest.avi trailers/classroom.mp4 \
        trailers/face-demographics-walking.mp4 trailers/head-pose-face-detection-female-and-male.mp4 \
        trailers/people-detection.mp4 trailers/one-by-one-person-detection.mp4 --threads 1 --steps 3000
    # stops after --max-minutes (38); continue with --resume p7_ocr/checkpoints/crnn.pt
    # (copy it first: the run writes to the same --out path)

Text: *Pride and Prejudice* (public domain), downloaded once into data/text/;
the first 90% of the book gives training lines, the last 10% the validation
lines. The subtitle test set uses a different book (Sherlock Holmes), so no
test sentence was ever seen. Fonts: every font on the machine except the
held-out ones (default: DejaVu, which the yellow test style uses).
Backgrounds: frames of the --bg-clips only; keep the test clips out of it.

The loss is our own CTC (`cinescope/ocr/ctc.py`); --torch-ctc switches to
PyTorch's for a speed comparison.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from cinescope.ocr import charset  # noqa: E402
from cinescope.ocr.crnn import CRNN, Recognizer, SyntheticLines, batch_loss  # noqa: E402
from cinescope.ocr.metrics import corpus_cer, corpus_wer  # noqa: E402
from cinescope.ocr.synth import (  # noqa: E402
    corpus_lines,
    find_fonts,
    frames_from_clips,
    random_strings,
    render_line,
)

TRAIN_BOOK = ("https://raw.githubusercontent.com/GITenberg/Pride-and-Prejudice_1342/master/1342.txt",
              "pride_and_prejudice.txt")


def gutenberg_body(text: str) -> str:
    """Strip the Project Gutenberg licence header and footer."""
    lo = text.find("*** START")
    hi = text.find("*** END")
    if lo >= 0:
        text = text[text.find("\n", lo) + 1:hi if hi > lo else None]
    return text


def fetch_text(url: str, name: str) -> str:
    path = ROOT / "data" / "text" / name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {url}")
        urllib.request.urlretrieve(url, path)
    return gutenberg_body(path.read_text(encoding="utf-8", errors="ignore"))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bg-clips", nargs="+", required=True, help="clips whose frames are backgrounds")
    ap.add_argument("--frames-per-clip", type=int, default=60)
    ap.add_argument("--holdout-fonts", nargs="*", default=["DejaVu"])
    ap.add_argument("--steps", type=int, default=2400)
    ap.add_argument("--max-minutes", type=float, default=38.0)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--width", type=int, default=16, help="channels of the first conv")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--n-val", type=int, default=500)
    ap.add_argument("--torch-ctc", action="store_true")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "p7_ocr" / "checkpoints" / "crnn.pt"))
    args = ap.parse_args(argv)

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    book = fetch_text(*TRAIN_BOOK)
    cut = int(len(book) * 0.9)
    train_text, val_text = book[:cut], book[cut:]
    vocab = sorted(set(charset.normalize(train_text).split()))
    fonts = find_fonts(exclude=args.holdout_fonts)
    held = [f for f in find_fonts() if f not in fonts]
    print(f"{len(fonts)} training fonts, {len(held)} held out; {len(vocab)} words")
    frames = frames_from_clips(args.bg_clips, per_clip=args.frames_per_clip)
    print(f"{len(frames)} background frames from {len(args.bg_clips)} clips")

    def texts_fn(rng, n):
        k = int(n * 0.8)
        return corpus_lines(train_text, rng, k) + random_strings(rng, n - k, vocab)

    # validation: unseen text, held-out fonts when there are any
    vrng = random.Random(12345)
    val_texts = corpus_lines(val_text, vrng, args.n_val)
    val_fonts = held or fonts
    val_imgs = [np.asarray(render_line(t, vrng, val_fonts, frames)) for t in val_texts]

    model = CRNN(width=args.width)
    step0 = 0
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        step0 = ck["step"]
    warm = 200

    def lr_at(s):
        if s < warm:
            return (s + 1) / warm
        return 0.5 * (1 + math.cos(math.pi * min(1.0, (s - warm) / max(1, args.steps - warm)))) * 0.98 + 0.02

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at, last_epoch=step0 - 1 if step0 else -1)
    print(f"model: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M parameters")

    ds = SyntheticLines(texts_fn, fonts, frames, batch_size=args.bs, seed=args.seed + step0)
    dl = torch.utils.data.DataLoader(ds, batch_size=None, num_workers=args.workers,
                                     persistent_workers=args.workers > 0)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    log = []
    t0 = time.time()
    run = 0.0
    step = step0

    def evaluate():
        rec = Recognizer(model)
        hyps = [h for h, _ in rec.read(val_imgs)]
        model.train()
        return corpus_cer(val_texts, hyps), corpus_wer(val_texts, hyps), hyps

    def save(extra=None):
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step,
                    "config": {"width": args.width}, "charset": charset.CHARS,
                    "args": vars(args), **(extra or {})}, out)

    model.train()
    for x, widths, tg, tl in dl:
        loss = batch_loss(model, x, widths, tg, tl, use_torch_ctc=args.torch_ctc)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        sched.step()
        step += 1
        run = 0.98 * run + 0.02 * loss.item() if step > step0 + 1 else loss.item()
        elapsed = (time.time() - t0) / 60
        if step % 100 == 0:
            print(f"step {step} loss {run:.3f} lr {sched.get_last_lr()[0]:.2e} {elapsed:.1f} min", flush=True)
        if step % 1000 == 0 or step >= args.steps or elapsed > args.max_minutes:
            c, w, hyps = evaluate()
            log.append({"step": step, "loss": run, "val_cer": c, "val_wer": w, "minutes": elapsed})
            print(f"  val CER {c:.4f} WER {w:.4f}   e.g. {val_texts[0]!r} -> {hyps[0]!r}", flush=True)
            save({"log": log})
        if step >= args.steps or elapsed > args.max_minutes:
            break
    log_path = ROOT / "p7_ocr" / "results" / "crnn_train_log.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(log_path.read_text()) if args.resume and log_path.exists() else []
    log_path.write_text(json.dumps(prev + log, indent=1))
    print(f"saved {out}")
    os._exit(0)          # do not wait for the endless data worker


if __name__ == "__main__":
    main()
