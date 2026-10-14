"""Append-only arrays on disk, for per-frame data a film would not fit in memory.

A 2-hour film has ~170k frames. Phase 0 kept a 96 px thumbnail of every frame
in RAM (~2 GB for a film); the face crops of Phase 3 are ten times that. A
DiskArray appends fixed-shape items to a raw file and reads them back through
np.memmap, so memory stays flat however long the video is.

Pickling a DiskArray (the pipeline's decode checkpoints) stores its path and
length only; unpickling truncates the file to that length, so items written
after the checkpoint (before an interruption) are dropped and written again.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np


class DiskArray:
    def __init__(self, path: str | Path, item_shape: tuple[int, ...] | None = None, dtype=np.uint8):
        self.path = str(path)
        self.item_shape = tuple(item_shape) if item_shape is not None else None
        self.dtype = np.dtype(dtype)
        self.n = 0
        self._f = None
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        open(self.path, "wb").close()

    @property
    def item_bytes(self) -> int:
        return int(np.prod(self.item_shape)) * self.dtype.itemsize if self.item_shape else 0

    def append(self, x: np.ndarray) -> int:
        x = np.ascontiguousarray(x, dtype=self.dtype)
        if self.item_shape is None:
            self.item_shape = x.shape
        elif x.shape != self.item_shape:
            raise ValueError(f"{self.path}: item shape {x.shape} != {self.item_shape}")
        if self._f is None:
            self._f = open(self.path, "ab")      # noqa: SIM115 - kept open across appends
        self._f.write(x.tobytes())
        self.n += 1
        return self.n - 1

    def flush(self) -> None:
        if self._f is not None:
            self._f.flush()

    def close(self) -> None:
        if self._f is not None:
            self._f.close()
            self._f = None

    def array(self) -> np.ndarray:
        """(n, *item_shape) read-only memmap (an empty array when n == 0)."""
        self.flush()
        if self.n == 0 or self.item_shape is None:
            return np.zeros((0, *(self.item_shape or ())), self.dtype)
        return np.memmap(self.path, self.dtype, "r", shape=(self.n, *self.item_shape))

    def __len__(self) -> int:
        return self.n

    def __getstate__(self) -> dict:
        self.flush()
        return {"path": self.path, "item_shape": self.item_shape, "dtype": self.dtype.str, "n": self.n}

    def __setstate__(self, st: dict) -> None:
        self.path, self.item_shape, self.n = st["path"], st["item_shape"], st["n"]
        self.dtype = np.dtype(st["dtype"])
        self._f = None
        size = self.n * self.item_bytes
        if os.path.exists(self.path):
            with open(self.path, "r+b") as f:
                f.truncate(size)
        else:
            open(self.path, "wb").close()
