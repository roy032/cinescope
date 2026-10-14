"""A small CLIP-style dual encoder, trained from scratch with InfoNCE.

CLIP (Radford et al., 2021) learns two encoders — image -> vector, text -> vector —
so that a matching (image, caption) pair has a large inner product and every
non-matching pair in the batch a small one. For a batch of B pairs with
L2-normalised embeddings u_i (images) and v_j (texts):

    logits[i, j] = exp(t) * <u_i, v_j>                 t: learned log-temperature
    loss = ( CE(logits, diag) + CE(logits.T, diag) ) / 2

Each row is a B-way classification "which caption is mine?" (and each column
"which image is mine?"): the InfoNCE bound of van den Oord et al. (2018). The
batch is the set of negatives, which is why CLIP used batches of 32k.

Two details that matter at this tiny scale:

* **Temperature.** Cosines live in [-1, 1]; a softmax over them is almost
  uniform. The learned scale (initialised to 1/0.07, clamped at 100 as in CLIP)
  sharpens it; fixing it at 1 makes training crawl.
* **Duplicate captions.** With a small caption vocabulary, two images in a batch
  often share a caption ("a red circle"). `multi_positive=True` treats every
  identical caption as a positive (the target spread uniformly over them, as in
  supervised contrastive learning, Khosla et al., 2020). A measured surprise: for
  *exact* duplicates this changes nothing. The text tower maps identical strings
  to identical vectors, so their logits tie, and summed over the group the plain
  and the multi-positive losses are the same number (ablation: 0.5782 vs 0.5780
  after 600 steps). What duplicates do cause is a loss floor — a caption shared by
  g images cannot pick one, so each such row costs at least log g — which is why
  the loss plateaus near 0.57 instead of going to 0. The false negatives that do
  hurt are *paraphrases* ("a car and a road" vs "a road and a car"), which no
  string-equality grouping catches.

The image tower is a plain 4-stage CNN, the text tower a 2-layer transformer
over word tokens with mean pooling: big enough to learn colours, shapes and
word order, small enough to train on a CPU in minutes. It cannot know about
anything its captions never mention — the reason a 400M-pair model exists.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

PAD, UNK = 0, 1
_WORD = re.compile(r"[a-z0-9]+")


class WordTokenizer:
    """Lower-case word tokens; ids 0 and 1 are <pad> and <unk>."""

    def __init__(self, vocab: list[str], max_len: int = 16) -> None:
        self.vocab = list(vocab)
        self.index = {w: i for i, w in enumerate(self.vocab)}
        self.max_len = max_len

    @classmethod
    def build(cls, texts: list[str], max_len: int = 16, min_count: int = 1) -> WordTokenizer:
        counts = Counter(w for t in texts for w in _WORD.findall(t.lower()))
        words = sorted(w for w, c in counts.items() if c >= min_count)
        return cls(["<pad>", "<unk>"] + words, max_len)

    def words(self, text: str) -> list[str]:
        return _WORD.findall(text.lower())

    def __call__(self, texts: list[str]) -> torch.Tensor:
        out = torch.full((len(texts), self.max_len), PAD, dtype=torch.long)
        for i, t in enumerate(texts):
            ids = [self.index.get(w, UNK) for w in self.words(t)][: self.max_len]
            if ids:
                out[i, : len(ids)] = torch.tensor(ids)
        return out

    def coverage(self, text: str) -> float:
        """Fraction of a text's words the model has ever seen."""
        w = self.words(text)
        return sum(x in self.index for x in w) / max(1, len(w))


@dataclass
class DualEncoderConfig:
    image_size: int = 64
    embed_dim: int = 128
    widths: tuple[int, ...] = (32, 64, 128, 256)
    convs: int = 1                     # conv-bn-relu layers per stage
    text_width: int = 128
    text_layers: int = 2
    text_heads: int = 4
    max_len: int = 16
    vocab: list[str] = field(default_factory=list)


class ImageTower(nn.Module):
    def __init__(self, widths: tuple[int, ...], out_dim: int, convs: int = 1) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        cin = 3
        for w in widths:            # conv-bn-relu x convs, then halve the resolution
            for _ in range(convs):
                layers += [nn.Conv2d(cin, w, 3, padding=1, bias=False), nn.BatchNorm2d(w), nn.ReLU(inplace=True)]
                cin = w
            layers.append(nn.MaxPool2d(2))
        self.body = nn.Sequential(*layers)
        self.proj = nn.Linear(cin, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.body(x).mean((2, 3)))


class TextTower(nn.Module):
    def __init__(self, vocab_size: int, width: int, layers: int, heads: int, max_len: int, out_dim: int) -> None:
        super().__init__()
        self.tok = nn.Embedding(vocab_size, width, padding_idx=PAD)
        self.pos = nn.Parameter(torch.randn(max_len, width) * 0.02)
        layer = nn.TransformerEncoderLayer(width, heads, 2 * width, dropout=0.0, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)
        self.proj = nn.Linear(width, out_dim)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        mask = ids == PAD
        h = self.enc(self.tok(ids) + self.pos[: ids.shape[1]], src_key_padding_mask=mask)
        keep = (~mask).unsqueeze(-1).float()
        h = (self.norm(h) * keep).sum(1) / keep.sum(1).clamp(min=1)      # mean over real tokens
        return self.proj(h)


class DualEncoder(nn.Module):
    MEAN = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
    STD = torch.tensor([0.25, 0.25, 0.25]).view(1, 3, 1, 1)

    def __init__(self, cfg: DualEncoderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tokenizer = WordTokenizer(cfg.vocab, cfg.max_len)
        self.image = ImageTower(tuple(cfg.widths), cfg.embed_dim, cfg.convs)
        self.text = TextTower(len(cfg.vocab), cfg.text_width, cfg.text_layers, cfg.text_heads, cfg.max_len,
                              cfg.embed_dim)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

    def encode_image(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, S, S) floats in [0, 1]."""
        return F.normalize(self.image((x - self.MEAN) / self.STD), dim=-1)

    def encode_text(self, ids: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.text(ids), dim=-1)

    def forward(self, images: torch.Tensor, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encode_image(images), self.encode_text(ids)

    # -- persistence
    def save(self, path: str | Path, extra: dict | None = None) -> None:
        torch.save({"config": asdict(self.cfg), "state": self.state_dict(), **(extra or {})}, path)

    @classmethod
    def load(cls, path: str | Path, map_location: str = "cpu") -> DualEncoder:
        ck = torch.load(path, map_location=map_location, weights_only=False)
        cfg = DualEncoderConfig(**{**ck["config"], "widths": tuple(ck["config"]["widths"])})
        m = cls(cfg)
        m.load_state_dict(ck["state"])
        return m.eval()


def info_nce(u: torch.Tensor, v: torch.Tensor, logit_scale: torch.Tensor,
             groups: torch.Tensor | None = None) -> torch.Tensor:
    """Symmetric InfoNCE. groups[i] == groups[j] marks pairs that are both positives
    (identical captions); None = only the diagonal is positive."""
    scale = logit_scale.clamp(max=math.log(100)).exp()
    logits = scale * u @ v.T
    if groups is None:
        target = torch.arange(len(u), device=u.device)
        return (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target)) / 2
    pos = (groups[:, None] == groups[None, :]).float()
    target = pos / pos.sum(1, keepdim=True)
    li = -(target * F.log_softmax(logits, 1)).sum(1).mean()
    lt = -(target * F.log_softmax(logits.T, 1)).sum(1).mean()
    return (li + lt) / 2


def caption_groups(captions: list[str]) -> torch.Tensor:
    ids: dict[str, int] = {}
    return torch.tensor([ids.setdefault(c, len(ids)) for c in captions])


# --------------------------------------------------------------------------- synthetic shapes task
COLOURS = {"red": (220, 40, 40), "green": (40, 190, 60), "blue": (50, 80, 230), "yellow": (240, 220, 40),
           "white": (240, 240, 240), "purple": (150, 60, 200)}
SHAPES = ("circle", "square", "triangle")


def _draw(d, shape: str, colour: tuple, cx: float, cy: float, r: float) -> None:
    if shape == "circle":
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=colour)
    elif shape == "square":
        d.rectangle([cx - r, cy - r, cx + r, cy + r], fill=colour)
    else:
        d.polygon([(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)], fill=colour)


def shapes_sample(rng: np.random.Generator, size: int = 64, held_out: frozenset = frozenset(),
                  only_held_out: bool = False) -> tuple[np.ndarray, str]:
    """One synthetic image and its caption: "a big red circle" or
    "a red circle left of a blue square" / "... above ...".

    held_out: (colour, shape) combinations never drawn in training (`only_held_out=False`),
    or always included at least once (`only_held_out=True`) — a test of compositional
    generalisation: the model has seen "purple" and "triangle", never "purple triangle".
    """
    from PIL import Image, ImageDraw

    def pick(held: bool) -> tuple[str, str]:
        while True:
            c, s = str(rng.choice(list(COLOURS))), str(rng.choice(SHAPES))
            if ((c, s) in held_out) == held:
                return c, s

    bg = tuple(int(v) for v in rng.integers(0, 70, 3))
    img = Image.new("RGB", (size, size), bg)
    d = ImageDraw.Draw(img)
    if rng.random() < 0.4:
        (c, s), big = pick(only_held_out), bool(rng.random() < 0.5)
        r = size * (0.32 if big else 0.16)
        cx, cy = rng.uniform(r, size - r, 2)
        _draw(d, s, COLOURS[c], cx, cy, r)
        cap = f"a {'big' if big else 'small'} {c} {s}"
    else:
        c1, s1 = pick(only_held_out)
        while True:
            c2, s2 = pick(False)
            if (c2, s2) != (c1, s1):
                break
        r = size * 0.15
        rel = str(rng.choice(["left of", "above"]))
        a, b = rng.uniform(r, size / 2 - r), rng.uniform(size / 2 + r, size - r)
        o1, o2 = rng.uniform(r, size - r, 2)
        p1, p2 = ((a, o1), (b, o2)) if rel == "left of" else ((o1, a), (o2, b))
        _draw(d, s1, COLOURS[c1], *p1, r)
        _draw(d, s2, COLOURS[c2], *p2, r)
        cap = f"a {c1} {s1} {rel} a {c2} {s2}"
    return np.asarray(img), cap


def shapes_vocab() -> list[str]:
    words = ["a", "big", "small", "left", "of", "above"] + list(COLOURS) + list(SHAPES)
    return ["<pad>", "<unk>"] + sorted(set(words))


def swap_relation(caption: str) -> str | None:
    """'a red circle left of a blue square' -> 'a blue square left of a red circle'
    (same words, different meaning: a bag-of-words model cannot tell them apart)."""
    for rel in (" left of ", " above "):
        if rel in caption:
            x, y = caption.split(rel)
            return f"{y}{rel}{x}"
    return None


def to_tensor(images: np.ndarray | list[np.ndarray]) -> torch.Tensor:
    return torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).float() / 255.0


def train_contrastive(model: DualEncoder, next_batch, steps: int, lr: float = 2e-3, wd: float = 0.05,
                      multi_positive: bool = True, augment=None, log_every: int = 100, log=print) -> list[float]:
    """AdamW + cosine schedule with warm-up. next_batch(step) -> (uint8 images list, captions)."""
    params_decay = [p for n, p in model.named_parameters() if p.ndim >= 2]
    params_other = [p for n, p in model.named_parameters() if p.ndim < 2]
    opt = torch.optim.AdamW([{"params": params_decay, "weight_decay": wd},
                             {"params": params_other, "weight_decay": 0.0}], lr=lr)
    warm = max(1, steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))
    model.train()
    losses = []
    for step in range(steps):
        images, caps = next_batch(step)
        x = to_tensor(images)
        if augment is not None:
            x = augment(x)
        u, v = model(x, model.tokenizer(caps))
        loss = info_nce(u, v, model.logit_scale, caption_groups(caps) if multi_positive else None)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        losses.append(float(loss.detach()))
        if log and (step % log_every == 0 or step == steps - 1):
            log(f"step {step:5d}  loss {np.mean(losses[-log_every:]):.3f}  "
                f"T {1 / model.logit_scale.exp().item():.3f}")
    model.eval()
    return losses
