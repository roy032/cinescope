"""Fill the README results tables from the files the training/evaluation runs wrote.

    python tools/update_results.py

Reads p1_cnn/results/own/hand_eval.json (or, for MovieShots,
p1_cnn/results/<tag>/metrics.json) + history.csv and
p2_detection/results/<tag>/wider_val.json + diagnosis.md; rewrites the blocks
between <!-- auto:NAME --> markers and the Phase 1/2 rows of the root README.
Missing results leave the existing text alone.
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_block(path: Path, name: str, body: str) -> bool:
    text = path.read_text(encoding="utf-8")
    pat = re.compile(rf"(<!-- auto:{name} -->\n).*?(<!-- /auto:{name} -->)", re.S)
    if not pat.search(text):
        return False
    path.write_text(pat.sub(lambda m: m.group(1) + body.rstrip() + "\n" + m.group(2), text), encoding="utf-8")
    return True


def hours(history: Path) -> str:
    if not history.exists():
        return "—"
    with history.open(encoding="utf-8") as f:
        secs = sum(float(r.get("seconds") or 0) for r in csv.DictReader(f))
    return f"{secs / 3600:.1f} h"


def phase1() -> dict:
    """Hand-labelled evaluation on our own episodes when it exists, else MovieShots test metrics."""
    res = ROOT / "p1_cnn" / "results"
    own = res / "own" / "hand_eval.json"
    if own.exists():
        d = json.loads(own.read_text(encoding="utf-8"))
        rows = ["| System | Trained on | Accuracy [95% CI] | Mean class acc. | Train time |", "|---|---|---|---|---|"]
        out = {}
        for key, m in d["systems"].items():
            lo, hi = m["accuracy_ci95"]
            trained = ("teacher labels, 30 epochs" if key == "pretrained" else
                       "teacher labels, 60 epochs" if key == "scratch" else "nothing (zero-shot / rule)")
            t = hours(res / key / "history.csv") + " (T4)" if key in ("pretrained", "scratch") else "—"
            rows.append(f"| {m['name']} | {trained} | {m['accuracy']:.3f} [{lo:.2f}–{hi:.2f}] | "
                        f"{m['mean_class_accuracy']:.3f} | {t} |")
            if key in ("pretrained", "scratch", "fused"):
                out[key] = m["accuracy"]
        dist = d["distribution"]
        rows += ["", f"{d['n']} hand-labelled shots from the held-out episodes; majority-class accuracy "
                     f"{max(dist.values()) / d['n']:.3f}. Face size only is scored on the shots where a face was found."]
        replace_block(ROOT / "p1_cnn" / "README.md", "p1-results", "\n".join(rows))
        return out
    rows = {}
    for tag, init in (("pretrained", "ImageNet"), ("scratch", "random")):
        m = res / tag / "metrics.json"
        if m.exists():
            d = json.loads(m.read_text(encoding="utf-8"))
            if d.get("synthetic") or "test" not in d:
                continue
            rows[tag] = (init, d["epochs"], d["test"]["accuracy"], d["test"]["mean_class_accuracy"],
                         hours(res / tag / "history.csv"))
    if not rows:
        return {}
    lines = ["| Model | Init | Epochs | Accuracy | Mean class acc. | Train time |", "|---|---|---|---|---|---|"]
    for tag in ("pretrained", "scratch"):
        if tag in rows:
            init, ep, acc, mca, t = rows[tag]
            lines.append(f"| ResNet-18 (ours) | {init} | {ep} | {acc:.3f} | {mca:.3f} | {t} (T4) |")
    replace_block(ROOT / "p1_cnn" / "README.md", "p1-results", "\n".join(lines))
    return {t: r[2] for t, r in rows.items()}


def phase2() -> dict:
    res, found = ROOT / "p2_detection" / "results", {}
    lines = ["| Model | Easy | Medium | Hard | ms/image |", "|---|---|---|---|---|"]
    diag = []
    for d in sorted(p for p in res.iterdir() if p.is_dir()) if res.exists() else []:
        w = d / "wider_val.json"
        if not w.exists():
            continue
        r = json.loads(w.read_text(encoding="utf-8"))
        if "easy" not in r:
            continue
        size = "log size" if d.name.endswith("_log") else "linear size"
        lines.append(f"| CenterNet-R18-FPN ({d.name}), {size}, 512 px training, ≤{r.get('max_side', 1024)} px test, "
                     f"top-{r.get('topk', 100)} peaks | {r['easy']:.3f} | {r['medium']:.3f} | {r['hard']:.3f} | "
                     f"{r.get('ms_per_image', '—')} (T4) |")
        found[d.name] = r
        md = d / "diagnosis.md"
        if md.exists():
            diag.append(f"**Where `{d.name}` loses AP** (per face size; from `diagnose_wider.py`):\n\n"
                        + md.read_text(encoding="utf-8").strip())
    if found:
        replace_block(ROOT / "p2_detection" / "README.md", "p2-wider", "\n".join(lines))
    if diag:
        replace_block(ROOT / "p2_detection" / "README.md", "p2-diagnosis", "\n\n".join(diag))
    return found


def root_table(p1: dict, p2: dict) -> None:
    path = ROOT / "README.md"
    text = path.read_text(encoding="utf-8")
    if p1:
        names = {"fused": "teacher", "pretrained": "ImageNet init", "scratch": "random init"}
        cell = " / ".join(f"{names.get(k, k)} {v:.3f}" for k, v in p1.items())
        text = re.sub(r"^\| 1 \| Shot-framing accuracy[^|]*\|.*$",
                      f"| 1 | Shot-framing accuracy (hand-labelled test episodes) | {cell} | — |", text, flags=re.M)
    if p2:
        best = max(p2.values(), key=lambda r: r["easy"])
        text = re.sub(r"^\| 2 \| Face AP \(WIDER FACE Easy[^)]*\) \|.*$",
                      f"| 2 | Face AP (WIDER FACE Easy / Medium / Hard) | {best['easy']:.3f} / {best['medium']:.3f} / "
                      f"{best['hard']:.3f} | — |", text, flags=re.M)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    p1, p2 = phase1(), phase2()
    root_table(p1, p2)
    print(f"phase 1 rows: {list(p1)}; phase 2 rows: {list(p2)}")


if __name__ == "__main__":
    main()
