"""MovieShots annotations -> a flat index, and the PyTorch datasets built on it.

MovieShots (Rao et al., ECCV 2020) ships ~46k shots from ~7k trailers, each
with three key frames and a scale and movement label. Its annotation JSON is
nested (split -> trailer -> shot -> labels); `read_movieshots` walks it without
assuming exact key names beyond "scale", finds each shot's frames on disk, and
`write_index` flattens everything to one CSV. Training only ever reads that
CSV, so a different release layout means changing this file only.

Index columns: split, trailer, shot, scale, movement, frames
(frames = up to three image paths relative to the dataset root, ';'-separated).
"""
from __future__ import annotations

import csv
import json
import random
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import SCALES

# label spellings seen in the wild -> canonical
_ALIASES = {
    "ecs": "ECS", "extreme close-up": "ECS", "extreme closeup": "ECS", "extreme_close_up": "ECS",
    "cs": "CS", "close-up": "CS", "closeup": "CS", "close_up": "CS",
    "ms": "MS", "medium": "MS", "medium shot": "MS", "medium_shot": "MS",
    "fs": "FS", "full": "FS", "full shot": "FS", "full_shot": "FS",
    "ls": "LS", "long": "LS", "long shot": "LS", "long_shot": "LS", "wide": "LS",
}
IMAGE_EXT = (".jpg", ".jpeg", ".png", ".webp")


def canonical_scale(label: str) -> str:
    key = str(label).strip().lower()
    if key not in _ALIASES:
        raise ValueError(f"unknown shot-scale label {label!r}; add it to _ALIASES")
    return _ALIASES[key]


def _label(entry: dict, key: str) -> str | None:
    v = entry.get(key)
    if isinstance(v, dict):
        v = v.get("label", v.get("name"))
    return None if v is None else str(v)


@dataclass
class ShotRecord:
    split: str
    trailer: str
    shot: str
    scale: str
    movement: str
    frames: list[str]


def _iter_entries(node: dict, path: tuple[str, ...] = ()) -> Iterator[tuple[tuple[str, ...], dict]]:
    """Yield (key path, entry) for every dict that carries a 'scale' label."""
    if "scale" in node and isinstance(node, dict):
        yield path, node
        return
    for k, v in node.items():
        if isinstance(v, dict):
            yield from _iter_entries(v, (*path, str(k)))


def find_frames(root: Path, trailer: str, shot: str) -> list[str]:
    """Key frames of one shot, relative to root. Looks for the MovieShots layout
    trailer/<id>/shot_<n>_img_<k>.jpg first, then any image of that shot."""
    for base in (root / "trailer" / trailer, root / trailer, root / "frames" / trailer):
        if not base.is_dir():
            continue
        hits = sorted(base.glob(f"shot_{shot}_img_*"))
        if not hits:
            hits = sorted(p for p in base.glob(f"*{shot}*") if p.suffix.lower() in IMAGE_EXT)
        if hits:
            return [p.relative_to(root).as_posix() for p in hits[:3]]
    return []


def read_movieshots(root: str | Path, annotations: str | Path | None = None) -> tuple[list[ShotRecord], Counter]:
    root = Path(root)
    if annotations is None:
        cands = sorted(root.glob("*split*.json")) or sorted(root.glob("*.json"))
        if not cands:
            raise FileNotFoundError(f"no annotation JSON in {root}; pass --annotations")
        annotations = cands[0]
    data = json.loads(Path(annotations).read_text(encoding="utf-8"))
    records, problems = [], Counter()
    for path, entry in _iter_entries(data):
        if len(path) < 2:
            problems["unrecognised nesting"] += 1
            continue
        split = path[0].lower() if len(path) >= 3 else "all"
        trailer, shot = path[-2], path[-1]
        split = {"validation": "val", "valid": "val"}.get(split, split)
        try:
            scale = canonical_scale(_label(entry, "scale"))
        except ValueError:
            problems["unknown scale label"] += 1
            continue
        frames = find_frames(root, trailer, shot)
        if not frames:
            problems["frames not found"] += 1
            continue
        records.append(ShotRecord(split, trailer, shot, scale, _label(entry, "movement") or "", frames))
    return records, problems


def assign_splits(records: list[ShotRecord], val: float = 0.1, test: float = 0.2, seed: int = 0) -> None:
    """For annotations without splits: split by *trailer*, never by shot —
    shots of one trailer share actors, sets and grading, and splitting them
    across train and test would leak."""
    trailers = sorted({r.trailer for r in records})
    random.Random(seed).shuffle(trailers)
    n = len(trailers)
    test_set = set(trailers[: int(n * test)])
    val_set = set(trailers[int(n * test): int(n * (test + val))])
    for r in records:
        r.split = "test" if r.trailer in test_set else "val" if r.trailer in val_set else "train"


def write_index(records: list[ShotRecord], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["split", "trailer", "shot", "scale", "movement", "frames"])
        for r in records:
            w.writerow([r.split, r.trailer, r.shot, r.scale, r.movement, ";".join(r.frames)])


def read_index(path: str | Path, split: str | None = None) -> list[ShotRecord]:
    with Path(path).open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [ShotRecord(r["split"], r["trailer"], r["shot"], r["scale"], r["movement"], r["frames"].split(";"))
            for r in rows if split is None or r["split"] == split]


# --- PyTorch side (imported lazily so the index tools work without torch) -------------

IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def build_transforms(train: bool, size: tuple[int, int] = (224, 384)):
    """Frames are resized to a 16:9-ish 224x384, not cropped square. Training
    crops keep >= 80% of the frame: the default RandomResizedCrop (down to 8%)
    would turn a long shot into a medium one and teach the model wrong labels."""
    from torchvision import transforms as T

    norm = [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    if not train:
        return T.Compose([T.Resize(size), *norm])
    ratio = size[1] / size[0]
    return T.Compose([
        T.RandomResizedCrop(size, scale=(0.8, 1.0), ratio=(ratio * 0.9, ratio * 1.1)),
        T.RandomHorizontalFlip(),                 # framing is mirror-invariant; vertical flips are not
        T.ColorJitter(0.3, 0.3, 0.2, 0.02),
        *norm,
    ])


class ShotDataset:
    """Train: one random key frame per shot (3x the views for free).
    Eval: all key frames, stacked, so predictions can be averaged per shot."""

    def __init__(self, records: list[ShotRecord], root: str | Path, train: bool,
                 size: tuple[int, int] = (224, 384), classes: tuple[str, ...] = SCALES):
        self.records, self.root, self.train = records, Path(root), train
        self.tf = build_transforms(train, size)
        self.cls = {c: i for i, c in enumerate(classes)}

    def __len__(self) -> int:
        return len(self.records)

    def _load(self, rel: str):
        from PIL import Image

        with Image.open(self.root / rel) as im:
            return self.tf(im.convert("RGB"))

    def __getitem__(self, i: int):
        import torch

        r = self.records[i]
        y = self.cls[r.scale]
        if self.train:
            return self._load(random.choice(r.frames)), y
        views = [self._load(f) for f in r.frames]
        views += [views[-1]] * (3 - len(views))          # fixed 3 views so batches stack
        return torch.stack(views), y

    def labels(self) -> np.ndarray:
        return np.array([self.cls[r.scale] for r in self.records])


class SyntheticShots:
    """Stand-in for MovieShots with the same interface: a lit disk (the subject)
    whose size sets the class. Used for smoke tests of the training script on a
    CPU (`--synthetic`), never for reported numbers."""

    RADII = ((0.9, 1.2), (0.55, 0.75), (0.32, 0.42), (0.18, 0.24), (0.06, 0.11))   # x frame height

    def __init__(self, n: int, train: bool, size: tuple[int, int] = (64, 112), seed: int = 0):
        self.n, self.train, self.size, self.seed = n, train, size, seed

    def __len__(self) -> int:
        return self.n

    def _img(self, rng: np.random.Generator, c: int):
        import torch

        h, w = self.size
        r = rng.uniform(*self.RADII[c]) * h
        cy, cx = rng.uniform(0.35 * h, 0.65 * h), rng.uniform(0.3 * w, 0.7 * w)
        yy, xx = np.mgrid[:h, :w] + 0.5
        disk = ((yy - cy) ** 2 + (xx - cx) ** 2) < r * r
        bg, fg = rng.uniform(0, 0.4, 3), rng.uniform(0.6, 1, 3)
        img = np.where(disk[None], fg[:, None, None], bg[:, None, None]) + rng.normal(0, 0.05, (3, h, w))
        return torch.tensor((img - 0.45) / 0.25, dtype=torch.float32)

    def __getitem__(self, i: int):
        import torch

        rng = np.random.default_rng((self.seed, i))
        c = int(rng.integers(0, len(SCALES)))
        if self.train:
            return self._img(rng, c), c
        return torch.stack([self._img(rng, c) for _ in range(3)]), c

    def labels(self) -> np.ndarray:
        return np.array([int(np.random.default_rng((self.seed, i)).integers(0, len(SCALES)))
                         for i in range(self.n)])
