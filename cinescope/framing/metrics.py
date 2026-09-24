"""Classification metrics and the confusion-matrix plot (NumPy only)."""
from __future__ import annotations

import numpy as np


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n: int) -> np.ndarray:
    cm = np.zeros((n, n), dtype=int)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def metrics(probs: np.ndarray, labels: np.ndarray, classes: tuple[str, ...]) -> dict:
    pred = probs.argmax(1)
    cm = confusion_matrix(labels, pred, len(classes))
    per_class = {c: float(cm[i, i] / cm[i].sum()) if cm[i].sum() else float("nan")
                 for i, c in enumerate(classes)}
    off = [(int(cm[i, j]), classes[i], classes[j]) for i in range(len(classes))
           for j in range(len(classes)) if i != j and cm[i, j]]
    return {
        "accuracy": float((pred == labels).mean()),
        "mean_class_accuracy": float(np.nanmean(list(per_class.values()))),
        "per_class": per_class,
        "confusion": cm.tolist(),
        "top_confusions": [{"true": t, "pred": p, "count": n} for n, t, p in sorted(off, reverse=True)[:5]],
        "n": int(len(labels)),
    }


def plot_confusion(cm: np.ndarray, classes: tuple[str, ...], path: str, title: str = "") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = np.asarray(cm)
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(4.8, 4.2), dpi=150)
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)), classes)
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{norm[i, j]:.2f}\n({cm[i, j]})", ha="center", va="center", fontsize=7,
                    color="white" if norm[i, j] > 0.55 else "black")
    if title:
        ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
