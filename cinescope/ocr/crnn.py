"""CRNN line recogniser (Shi, Bai & Yao, 2015): conv backbone + BiLSTM + CTC.

    grey line, 32 px high, any width W
      -> conv/pool stack that halves height 5 times but width only twice
      -> feature map 1 × W/4: one 256-d column feature every 4 pixels
      -> 2-layer bidirectional LSTM over the columns (context: is this
         stroke the right half of an "m" or an "n" followed by a "l"?)
      -> linear layer to the alphabet + blank at every column
      -> CTC loss in training; greedy or prefix-beam decoding at test time.

Why the width stride is 4 and not 32: CTC needs at least one output step per
character, plus a blank between doubled letters ("ll", "ee", "oo"). A
32-px-high line has characters 6-20 px wide, so one step per 4 px gives ~2-5
steps per character. The pools after the second one are (2, 1): they reduce
height only.
"""
from __future__ import annotations

import math
import random
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from torch import nn

from . import charset
from .ctc import ctc_loss, greedy_decode, prefix_beam_search

HEIGHT = 32


def _block(cin, cout, pool):
    layers = [nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True)]
    if pool:
        layers.append(nn.MaxPool2d(pool, pool))
    return layers


class CRNN(nn.Module):
    def __init__(self, n_classes: int = charset.N_CLASSES, width: int = 16, hidden: int = 128):
        super().__init__()
        c = width
        self.cnn = nn.Sequential(
            *_block(1, c, (2, 2)),                 # 16 × W/2
            *_block(c, 2 * c, (2, 2)),             # 8 × W/4
            *_block(2 * c, 4 * c, None),
            *_block(4 * c, 4 * c, (2, 1)),         # 4 × W/4
            *_block(4 * c, 8 * c, (2, 1)),         # 2 × W/4
            *_block(8 * c, 8 * c, None),
            nn.Conv2d(8 * c, 8 * c, (2, 1), bias=False), nn.BatchNorm2d(8 * c), nn.ReLU(inplace=True),
        )                                          # 1 × W/4
        self.rnn = nn.LSTM(8 * c, hidden, num_layers=2, bidirectional=True, dropout=0.1)
        self.fc = nn.Linear(2 * hidden, n_classes)

    @staticmethod
    def out_len(widths: torch.Tensor) -> torch.Tensor:
        return widths // 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 1, 32, W) in [-1, 1] -> (T = W/4, B, classes) log-probabilities."""
        f = self.cnn(x)                            # (B, C, 1, T)
        f = f.squeeze(2).permute(2, 0, 1)          # (T, B, C)
        f, _ = self.rnn(f)
        return self.fc(f).log_softmax(-1)


# ---------------------------------------------------------------- batching
def to_height(img: np.ndarray, height: int = HEIGHT, max_width: int = 1200) -> np.ndarray:
    """Grey (h, w) uint8 crop -> (height, w') keeping the aspect ratio."""
    from PIL import Image

    im = Image.fromarray(np.asarray(img, dtype=np.uint8))
    if im.mode != "L":
        im = im.convert("L")
    w = max(height // 2, min(max_width, round(im.width * height / max(1, im.height))))
    return np.asarray(im.resize((w, height), Image.BILINEAR))


def collate(images: Sequence[np.ndarray], texts: Sequence[str] | None = None):
    """Pad a list of 32-high grey images to the widest (rounded up to 4) with
    the mean-free value 0 after normalisation. Returns (x, widths[, targets,
    target_lengths])."""
    widths = [im.shape[1] for im in images]
    wmax = int(math.ceil(max(widths) / 4) * 4)
    x = torch.zeros(len(images), 1, HEIGHT, wmax)
    for i, im in enumerate(images):
        x[i, 0, :, :im.shape[1]] = torch.from_numpy(im.astype(np.float32) / 127.5 - 1.0)
    widths_t = torch.tensor(widths)
    if texts is None:
        return x, widths_t
    enc = [charset.encode(t) for t in texts]
    lens = torch.tensor([len(e) for e in enc])
    tg = torch.zeros(len(enc), max(1, int(lens.max())), dtype=torch.long)
    for i, e in enumerate(enc):
        tg[i, :len(e)] = torch.tensor(e, dtype=torch.long)
    return x, widths_t, tg, lens


def batch_loss(model: CRNN, x, widths, targets, target_lengths, use_torch_ctc: bool = False):
    lp = model(x)
    il = CRNN.out_len(widths).clamp(max=lp.shape[0])
    if use_torch_ctc:
        return nn.functional.ctc_loss(lp, targets, il, target_lengths, zero_infinity=True)
    return ctc_loss(lp, targets, il, target_lengths, zero_infinity=True)


class SyntheticLines(torch.utils.data.IterableDataset):
    """Endless stream of width-bucketed batches of synthetic lines.

    Each worker renders `pool` lines, sorts them by width and cuts them into
    batches, so a batch's lines need little padding (padding is wasted
    compute and the LSTM would read it as content)."""

    def __init__(self, texts_fn, fonts, frames, batch_size: int = 32, pool: int = 8,
                 seed: int = 0, invert_p: float = 0.1):
        self.texts_fn, self.fonts, self.frames = texts_fn, fonts, frames
        self.batch_size, self.pool, self.seed, self.invert_p = batch_size, pool, seed, invert_p

    def __iter__(self):
        from .synth import render_line

        info = torch.utils.data.get_worker_info()
        rng = random.Random(self.seed * 1000 + (info.id if info else 0))
        while True:
            texts = self.texts_fn(rng, self.batch_size * self.pool)
            items = []
            for t in texts:
                im = np.asarray(render_line(t, rng, self.fonts, self.frames))
                if rng.random() < self.invert_p:
                    im = 255 - im
                items.append((im, t))
            items.sort(key=lambda it: it[0].shape[1])
            batches = [items[i:i + self.batch_size] for i in range(0, len(items), self.batch_size)]
            rng.shuffle(batches)
            for b in batches:
                yield collate([im for im, _ in b], [t for _, t in b])


# ---------------------------------------------------------------- inference
class Recognizer:
    """Read a list of crops with a trained CRNN.

        rec = Recognizer.load("p7_ocr/checkpoints/crnn.pt")
        rec.read([crop1, crop2])        # -> [("Hello there", 0.93), ...]
    """

    # confidence thresholds chosen on the tuning clips (p7_ocr/README.md)
    min_conf = 0.88
    credits_min_conf = 0.88

    def __init__(self, model: CRNN, decoder: str = "greedy", beam: int = 8):
        self.model = model.eval()
        self.decoder, self.beam = decoder, beam

    @classmethod
    def load(cls, path: str | Path, **kw) -> Recognizer:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        model = CRNN(**ck.get("config", {}))
        model.load_state_dict(ck["model"])
        return cls(model, **kw)

    @torch.no_grad()
    def read(self, crops: Sequence[np.ndarray], batch_size: int = 32) -> list[tuple[str, float]]:
        """crops: grey or RGB uint8 arrays of single text lines. Returns
        (text, confidence) per crop; confidence = geometric mean over the
        output steps of the winning class's probability."""
        ims = [to_height(np.asarray(c)) for c in crops]           # RGB -> L with the same weights as training
        order = np.argsort([im.shape[1] for im in ims])
        out: list[tuple[str, float]] = [("", 0.0)] * len(ims)
        for s in range(0, len(ims), batch_size):
            idx = order[s:s + batch_size]
            x, widths = collate([ims[i] for i in idx])
            lp = self.model(x)
            for j, i in enumerate(idx):
                t = int(CRNN.out_len(widths[j]))
                seq = lp[:t, j].numpy()
                if self.decoder == "beam":
                    labels = prefix_beam_search(seq, beam=self.beam)[0][0]
                else:
                    labels = greedy_decode(seq)
                conf = float(np.exp(seq.max(1).mean())) if t else 0.0
                out[i] = (charset.decode(labels), conf)
        return out
