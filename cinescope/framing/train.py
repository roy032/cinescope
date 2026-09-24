"""Training and evaluation loops for the shot-scale classifier.

Kept separate from the command-line script so the pieces (schedule, metrics,
one epoch) can be unit-tested and reused from a notebook on Kaggle/Colab.

What each training choice is for:
  * SGD + momentum, weight decay 5e-4 on conv/linear weights only (not BN or
    biases — decaying BN's scale fights the normalisation it exists for).
  * Linear warm-up, then cosine decay to zero: large early steps on randomly
    initialised BN statistics can blow up; cosine avoids hand-picked step drops.
  * Mixed precision (CUDA only): matmuls in float16, weights and the loss scale
    in float32. GradScaler multiplies the loss so small float16 gradients do not
    underflow to zero, and skips steps whose gradients overflowed.
  * Label smoothing 0.1: scale boundaries are fuzzy (medium vs. full shot is a
    judgement call), so the targets should not demand certainty.
"""
from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .metrics import confusion_matrix, metrics, plot_confusion  # noqa: F401  (re-exported)


def warmup_cosine(total_steps: int, warmup_steps: int) -> Callable[[int], float]:
    """Multiplier on the base learning rate at optimiser step `s` (LambdaLR)."""
    def f(s: int) -> float:
        if s < warmup_steps:
            return (s + 1) / warmup_steps
        p = (s - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, p)))
    return f


def param_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim <= 1 or name.endswith(".bias") else decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


@dataclass
class EpochStats:
    loss: float
    acc: float
    seconds: float
    lr: float


def train_one_epoch(model: nn.Module, loader: Iterable, opt: torch.optim.Optimizer,
                    sched: torch.optim.lr_scheduler.LRScheduler, device: torch.device,
                    scaler: torch.amp.GradScaler | None = None, label_smoothing: float = 0.1,
                    max_steps: int | None = None) -> EpochStats:
    model.train()
    crit = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    use_amp = scaler is not None and scaler.is_enabled()
    t0, tot_loss, correct, seen = time.perf_counter(), 0.0, 0, 0
    for step, (x, y) in enumerate(loader):
        if max_steps is not None and step >= max_steps:
            break
        x = x.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
        y = y.to(device, non_blocking=True)
        with torch.autocast(device.type, enabled=use_amp):          # float16 on CUDA
            logits = model(x)
            loss = crit(logits, y)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"loss became {loss.item()} at step {step}; lower the learning rate")
        opt.zero_grad(set_to_none=True)
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            opt.step()
        sched.step()
        tot_loss += loss.item() * len(y)
        correct += (logits.argmax(1) == y).sum().item()
        seen += len(y)
    return EpochStats(tot_loss / max(seen, 1), correct / max(seen, 1), time.perf_counter() - t0,
                      opt.param_groups[0]["lr"])


@torch.no_grad()
def predict(model: nn.Module, loader: Iterable, device: torch.device,
            max_steps: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Per-shot class probabilities (averaged over the shot's key frames) and labels.
    Batches are (B, V, C, H, W): V views of each shot."""
    model.eval()
    probs, labels = [], []
    for step, (x, y) in enumerate(loader):
        if max_steps is not None and step >= max_steps:
            break
        b, v = x.shape[:2]
        x = x.flatten(0, 1).to(device).contiguous(memory_format=torch.channels_last)
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            p = model(x).float().softmax(1)
        probs.append(p.view(b, v, -1).mean(1).cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(probs), np.concatenate(labels)
