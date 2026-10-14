"""The alphabet the recogniser can output, and text normalisation.

Index 0 is the CTC blank; characters are 1..N. Subtitles in English use a
small alphabet: letters, digits, space and a dozen punctuation marks. Anything
else in source text (curly quotes, em dashes, accented letters) is mapped to
its nearest ASCII form or dropped, so training labels and test labels live in
the same alphabet and the CER measures reading, not typography.
"""
from __future__ import annotations

import string
import unicodedata

CHARS = string.ascii_letters + string.digits + " .,'\"!?-:;()&"
BLANK = 0
_INDEX = {c: i + 1 for i, c in enumerate(CHARS)}
N_CLASSES = len(CHARS) + 1          # + blank

_REPLACE = {
    "‘": "'", "’": "'", "“": '"', "”": '"', "—": "-", "–": "-",
    "…": "...", "_": "", "`": "'", "\t": " ", "\n": " ", "[": "(", "]": ")", "/": "-",
}


def normalize(text: str) -> str:
    """Map text into CHARS: ASCII-fold, replace typographic marks, collapse spaces."""
    out = []
    for ch in text:
        ch = _REPLACE.get(ch, ch)
        if len(ch) == 1 and ch not in _INDEX:
            ch = unicodedata.normalize("NFKD", ch).encode("ascii", "ignore").decode()
        out.append("".join(c for c in ch if c in _INDEX))
    return " ".join("".join(out).split())


def encode(text: str) -> list[int]:
    """Characters -> label indices (1..N). Unknown characters are skipped."""
    return [_INDEX[c] for c in text if c in _INDEX]


def decode(labels) -> str:
    """Label indices -> characters (blank and out-of-range indices dropped)."""
    return "".join(CHARS[i - 1] for i in labels if 0 < i <= len(CHARS))


def fold(text: str) -> str:
    """Lower-case letters and digits only — the comparison used for recognisers
    whose alphabet has no case, punctuation or spaces (the OpenCV zoo CRNN)."""
    return "".join(c for c in text.lower() if c.isalnum())
