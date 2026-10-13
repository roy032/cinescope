"""Connectionist Temporal Classification (Graves et al., 2006), from scratch.

A CRNN reads a text line as a sequence of T column features and outputs, at
every column, a distribution over the alphabet plus a *blank*. The label
("Hello", 5 characters) is shorter than T and nobody says which column holds
which letter. CTC sums the probability of every alignment that collapses to
the label — collapse = merge repeated symbols, then delete blanks:

    "HH-e-ll-llo--"  ->  "Hello"      ("ll-ll": the blank keeps the double l)

The sum is computed by dynamic programming over the *extended* label
l' = (-, H, -, e, -, l, -, l, -, o, -) of length 2S+1:

    alpha_t(s) = [alpha_{t-1}(s) + alpha_{t-1}(s-1) + alpha_{t-1}(s-2)] * y_t(l'_s)

where the s-2 skip is allowed only from one character to a *different*
character (from "l" to "l" a blank must sit in between). p(label | x) is
alpha_{T-1} at the last character plus at the final blank. Products of T
probabilities underflow float32 after a few dozen steps, so everything runs in
log space: products become sums, sums become logsumexp.

`ctc_loss` is the batched torch version (gradients by autograd) used for
training; `ctc_numpy` also runs the backward recursion beta and returns the
closed-form gradient with respect to the logits, y_t(k) - (1/p) * sum over
positions s with l'_s = k of alpha_t(s) beta_t(s) / y_t(k): the softmax
output minus how much posterior "occupancy" the alignments put on k at t.
Both are checked against torch.nn.functional.ctc_loss in tests/test_week21.py.
"""
from __future__ import annotations

import math

import numpy as np
import torch

NEG = -1e30          # log(0) stand-in: finite, so logsumexp of an all-"zero" set has no NaN gradient


def _extended(targets: torch.Tensor, target_lengths: torch.Tensor, blank: int):
    """(B, S) padded labels -> (B, 2S+1) blank-interleaved labels and the
    mask of positions where the s-2 skip is allowed."""
    b, s = targets.shape
    ext = torch.full((b, 2 * s + 1), blank, dtype=torch.long, device=targets.device)
    ext[:, 1::2] = targets
    skip = torch.zeros_like(ext, dtype=torch.bool)
    if s > 1:
        skip[:, 3::2] = targets[:, 1:] != targets[:, :-1]
    # positions beyond a sample's own 2*S_b+1 never receive mass (their alpha
    # stays NEG because nothing feeds them before the end positions are read)
    return ext, skip


def ctc_loss(log_probs: torch.Tensor, targets, input_lengths, target_lengths,
             blank: int = 0, reduction: str = "mean", zero_infinity: bool = False) -> torch.Tensor:
    """Same contract as torch.nn.functional.ctc_loss.

    log_probs: (T, B, C) log-softmax outputs. targets: (B, S) padded or the
    concatenated 1-D form. reduction "mean" divides each sample's loss by its
    label length, then averages over the batch (PyTorch's convention).
    """
    t_max, b, _ = log_probs.shape
    device = log_probs.device
    input_lengths = torch.as_tensor(input_lengths, dtype=torch.long, device=device)
    target_lengths = torch.as_tensor(target_lengths, dtype=torch.long, device=device)
    targets = torch.as_tensor(targets, dtype=torch.long, device=device)
    if targets.dim() == 1:                               # concatenated form -> padded
        s_max = int(target_lengths.max()) if b else 0
        padded = torch.full((b, max(1, s_max)), blank, dtype=torch.long, device=device)
        offsets = torch.cumsum(target_lengths, 0) - target_lengths
        for i in range(b):
            n = int(target_lengths[i])
            padded[i, :n] = targets[offsets[i]:offsets[i] + n]
        targets = padded
    ext, skip = _extended(targets, target_lengths, blank)
    n_ext = ext.shape[1]
    # emissions[t, b, s] = log y_t(l'_s)
    emit = log_probs.gather(2, ext.unsqueeze(0).expand(t_max, b, n_ext))

    neg = log_probs.new_full((b, n_ext), NEG)
    alpha = neg.clone()
    alpha[:, 0] = emit[0, :, 0]
    alpha[:, 1] = torch.where(target_lengths > 0, emit[0, :, 1], neg[:, 1]) if n_ext > 1 else alpha[:, 0]
    pad1 = log_probs.new_full((b, 1), NEG)
    pad2 = log_probs.new_full((b, 2), NEG)
    for t in range(1, t_max):
        stay = alpha
        step = torch.cat([pad1, alpha[:, :-1]], 1)
        jump = torch.where(skip, torch.cat([pad2, alpha[:, :-2]], 1), neg)
        new = torch.logaddexp(torch.logaddexp(stay, step), jump) + emit[t]
        # samples whose sequence already ended keep their final alpha
        alpha = torch.where((t < input_lengths).unsqueeze(1), new, alpha)

    last = 2 * target_lengths                            # final blank position
    a_blank = alpha.gather(1, last.unsqueeze(1)).squeeze(1)
    a_char = alpha.gather(1, (last - 1).clamp(min=0).unsqueeze(1)).squeeze(1)
    a_char = torch.where(target_lengths > 0, a_char, torch.full_like(a_char, NEG))
    loss = -torch.logaddexp(a_blank, a_char)
    # infeasible alignments (label needs more steps than T) give ~1e30, PyTorch says inf
    infeasible = loss > 1e29
    loss = torch.where(infeasible, torch.full_like(loss, math.inf), loss)
    if zero_infinity:
        loss = torch.where(infeasible, torch.zeros_like(loss), loss)
    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    return (loss / target_lengths.clamp(min=1).to(loss.dtype)).mean()


# --------------------------------------------------------------- NumPy, with beta
def _lse(*xs):
    m = np.maximum.reduce(xs)
    m_safe = np.where(np.isfinite(m), m, 0.0)
    with np.errstate(divide="ignore"):                  # log(0) = -inf is intended
        return m_safe + np.log(sum(np.exp(x - m_safe) for x in xs))


def ctc_numpy(logits: np.ndarray, label: list[int], blank: int = 0) -> tuple[float, np.ndarray]:
    """One sequence: logits (T, C) before softmax -> (loss = -log p, d loss / d logits).

    Forward-backward in log space. alpha_t(s) covers paths over steps 0..t
    ending at s, beta_t(s) paths over t..T-1 starting at s; both include the
    emission at t, so alpha+beta-emission is the log of the total probability
    of the paths that sit on position s at step t.
    """
    logits = np.asarray(logits, dtype=np.float64)
    m = logits.max(1, keepdims=True)
    lp = logits - (m + np.log(np.exp(logits - m).sum(1, keepdims=True)))     # log-softmax
    t_max = lp.shape[0]
    ext = [blank]
    for c in label:
        ext += [c, blank]
    n = len(ext)
    can_skip = np.array([s >= 2 and ext[s] != blank and ext[s] != ext[s - 2] for s in range(n)])
    e = lp[:, ext]                                       # (T, n) emissions
    alpha = np.full((t_max, n), -np.inf)
    alpha[0, 0] = e[0, 0]
    if n > 1:
        alpha[0, 1] = e[0, 1]
    for t in range(1, t_max):
        a = alpha[t - 1]
        prev1 = np.concatenate([[-np.inf], a[:-1]])
        prev2 = np.where(can_skip, np.concatenate([[-np.inf, -np.inf], a[:-2]]), -np.inf)
        alpha[t] = _lse(a, prev1, prev2) + e[t]
    beta = np.full((t_max, n), -np.inf)
    beta[-1, -1] = e[-1, -1]
    if n > 1:
        beta[-1, -2] = e[-1, -2]
    skip_from = np.concatenate([can_skip[2:], [False, False]])   # s -> s+2 allowed?
    for t in range(t_max - 2, -1, -1):
        b = beta[t + 1]
        nxt1 = np.concatenate([b[1:], [-np.inf]])
        nxt2 = np.where(skip_from, np.concatenate([b[2:], [-np.inf, -np.inf]]), -np.inf)
        beta[t] = _lse(b, nxt1, nxt2) + e[t]
    log_p = _lse(alpha[-1, -1], alpha[-1, -2]) if n > 1 else alpha[-1, -1]
    occupancy = alpha + beta - e - log_p                 # log posterior of (t, s)
    grad = np.exp(lp)                                    # d(-log p)/d logits = y - occupancy
    for s, k in enumerate(ext):
        grad[:, k] -= np.exp(occupancy[:, s])
    return float(-log_p), grad


# --------------------------------------------------------------- decoding
def greedy_decode(log_probs: np.ndarray, blank: int = 0) -> list[int]:
    """Best path: argmax per step, merge repeats, drop blanks. (T, C) -> labels."""
    best = np.asarray(log_probs).argmax(-1)
    out, prev = [], blank
    for k in best:
        if k != prev and k != blank:
            out.append(int(k))
        prev = k
    return out


def prefix_beam_search(log_probs: np.ndarray, beam: int = 10, blank: int = 0,
                       prune: float = -12.0) -> list[tuple[list[int], float]]:
    """CTC prefix beam search (Hannun et al., 2014), no language model.

    Greedy decoding picks the single most likely *path*; the most likely
    *label* can differ because many paths collapse to the same label. Each
    beam entry is a label prefix with two scores: log P(prefix, path ends in
    blank) and log P(prefix, path ends in its last character). The split is
    needed because extending "l" with "l" makes "ll" only after a blank;
    otherwise it is the same "l". Returns [(labels, log P)], best first.
    Symbols with log y < prune at a step are skipped (speed only).
    """
    lp = np.asarray(log_probs, dtype=np.float64)
    beams: dict[tuple[int, ...], list[float]] = {(): [0.0, -np.inf]}    # prefix -> [p_b, p_nb]
    for t in range(lp.shape[0]):
        row = lp[t]
        cand = [k for k in np.flatnonzero(row > prune) if k != blank]
        nxt: dict[tuple[int, ...], list[float]] = {}

        def add(prefix, i, val, store=nxt):
            cur = store.setdefault(prefix, [-np.inf, -np.inf])
            cur[i] = np.logaddexp(cur[i], val)

        for prefix, (p_b, p_nb) in beams.items():
            total = np.logaddexp(p_b, p_nb)
            add(prefix, 0, total + row[blank])               # emit blank: prefix unchanged
            last = prefix[-1] if prefix else None
            for k in cand:
                pk = row[k]
                if k == last:
                    add(prefix, 1, p_nb + pk)                # repeat merges into the same char
                    add(prefix + (int(k),), 1, p_b + pk)     # after a blank: a new char
                else:
                    add(prefix + (int(k),), 1, total + pk)
        scored = sorted(nxt.items(), key=lambda kv: -np.logaddexp(*kv[1]))[:beam]
        beams = dict(scored)
    return [(list(p), float(np.logaddexp(*v))) for p, v in
            sorted(beams.items(), key=lambda kv: -np.logaddexp(*kv[1]))]
