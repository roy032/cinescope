"""Edit distance, character and word error rates.

CER = (substitutions + deletions + insertions) / characters in the reference,
from the Levenshtein alignment. It can exceed 1 when the prediction is much
longer than the reference. Corpus CER sums edits and reference lengths over
all lines (long lines weigh more), which is how OCR papers report it; the mean
of per-line CERs would let a wrong one-word line count as much as a paragraph.
"""
from __future__ import annotations

from collections.abc import Sequence


def levenshtein(a: Sequence, b: Sequence) -> int:
    """Minimum number of single-element insertions, deletions and substitutions
    turning `a` into `b`. Works on strings (characters) and lists (words).
    Dynamic programming over one row: O(len(a)·len(b)) time, O(len(b)) memory."""
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1,              # delete x
                           cur[j - 1] + 1,           # insert y
                           prev[j - 1] + (x != y)))  # substitute (free if equal)
        prev = cur
    return prev[-1]


def cer(ref: str, hyp: str) -> float:
    """Character error rate of one line. An empty reference scores 0 if the
    hypothesis is empty too, else 1 per inserted character."""
    return levenshtein(ref, hyp) / max(1, len(ref)) if ref else float(len(hyp))


def wer(ref: str, hyp: str) -> float:
    """Word error rate: Levenshtein over whitespace-separated words."""
    r, h = ref.split(), hyp.split()
    return levenshtein(r, h) / max(1, len(r)) if r else float(len(h))


def corpus_cer(refs: Sequence[str], hyps: Sequence[str]) -> float:
    """Total character edits / total reference characters."""
    edits = sum(levenshtein(r, h) for r, h in zip(refs, hyps, strict=True))
    return edits / max(1, sum(len(r) for r in refs))


def corpus_wer(refs: Sequence[str], hyps: Sequence[str]) -> float:
    edits = sum(levenshtein(r.split(), h.split()) for r, h in zip(refs, hyps, strict=True))
    return edits / max(1, sum(len(r.split()) for r in refs))
