"""Fill the README results tables from the files the training/evaluation runs wrote.

    python tools/update_results.py

Reads p1_cnn/results/own/hand_eval.json (or, for MovieShots,
p1_cnn/results/<tag>/metrics.json) + history.csv and
p2_detection/results/<tag>/wider_val.json + diagnosis.md and
p2_detection/results/compare_{faces,props}.json; rewrites the blocks
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
    return f"{secs / 3600:.1f} h" if secs >= 360 else f"{secs / 60:.0f} min"


SETS = {"own": "your own videos (held-out episodes)",
        "public": "public clips, 2-fold by time: each shot predicted by the network that did not train on its half"}


def _train_info(run: Path) -> tuple[str, str]:
    """('teacher labels, N epochs[, H x W px]', 'time (device)') from a training run's folder."""
    m = run / "metrics.json"
    d = json.loads(m.read_text(encoding="utf-8")) if m.exists() else {}
    trained = f"teacher labels, {d.get('epochs', '?')} epochs"
    if d.get("size") and list(d["size"]) != [224, 384]:
        trained += f", {d['size'][0]}×{d['size'][1]} px"
    if d.get("weights"):
        trained += f", init {Path(d['weights']).name}"
    return trained, f"{hours(run / 'history.csv')} ({d.get('device', 'T4')})"


def _hand_table(name: str, d: dict, res: Path) -> tuple[list[str], dict]:
    rows = ["| System | Trained on | Accuracy [95% CI] | Mean class acc. | Train time |", "|---|---|---|---|---|"]
    out = {}
    for key, m in d["systems"].items():
        lo, hi = m["accuracy_ci95"]
        trained, t = "nothing (zero-shot / rule)", "—"
        if key in ("pretrained", "scratch"):
            trained, t = _train_info(res / (key if name == "own" else f"{name}_{key}"))
        rows.append(f"| {m['name']} | {trained} | {m['accuracy']:.3f} [{lo:.2f}–{hi:.2f}] | "
                    f"{m['mean_class_accuracy']:.3f} | {t} |")
        if key in ("pretrained", "scratch", "fused"):
            out[key] = m["accuracy"]
    dist = d["distribution"]
    rows += ["", f"{d['n']} hand-labelled shots ({SETS.get(name, name)}; labels "
                 f"`{d.get('labels', 'p1_cnn/labels/')}`); majority-class accuracy "
                 f"{max(dist.values()) / d['n']:.3f}. Face size only is scored on the shots where a face was found."]
    if name == "public":
        rows[-1] += " Train time is the sum of the two folds."
    return rows, out


def phase1() -> dict:
    """Hand-labelled evaluations (your own videos, the public clips) when they exist, else MovieShots."""
    res = ROOT / "p1_cnn" / "results"
    found = [(s, res / s / "hand_eval.json") for s in SETS if (res / s / "hand_eval.json").exists()]
    if found:
        blocks, out = [], {}
        for name, path in found:
            d = json.loads(path.read_text(encoding="utf-8"))
            rows, acc = _hand_table(name, d, res)
            if len(found) > 1:
                rows.insert(0, f"**{SETS[name][0].upper() + SETS[name][1:]}**\n")
            blocks.append("\n".join(rows))
            if not out:                           # the root table shows the first set (own, then public)
                out = {**acc, "_set": name, "_n": d["n"]}
        replace_block(ROOT / "p1_cnn" / "README.md", "p1-results", "\n\n".join(blocks))
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


MODEL_NAMES = {"yolov8n": "YOLOv8n", "yolo11n": "YOLO11n", "rtdetr-l": "RT-DETR-L"}


def _tag(clip: str) -> str:
    """'goldeneye_dark' -> 'dark': the condition is the label file's tag, after the last underscore."""
    return clip.rsplit("_", 1)[-1]


def week8() -> bool:
    """The week-8 tables (compare_faces.json, compare_props.json) -> p2_detection/README.md."""
    res, done = ROOT / "p2_detection" / "results", False
    conds = ("dark", "profile", "crowd")
    faces = res / "compare_faces.json"
    if faces.exists():
        rep = json.loads(faces.read_text(encoding="utf-8"))
        lines = ["| Faces | dark AP50 | profile AP50 | crowd AP50 | all AP50 | all AP | ms/frame |", "|---|---|---|---|---|---|---|"]
        if not any(n.startswith("CenterNet-R18 (ours") for n in rep):
            lines.append("| CenterNet-R18 (ours) | — | — | — | — | — | — |")
        for name, r in rep.items():
            by = {_tag(c): v for c, v in r.items() if isinstance(v, dict) and c != "all"}
            lines.append(f"| {name} | " + " | ".join(f"{by[c]['AP50']:.3f}" if c in by else "—" for c in conds)
                         + f" | {r['all']['AP50']:.3f} | {r['all']['AP']:.3f} | {r['ms_per_frame']} |")
        done |= replace_block(ROOT / "p2_detection" / "README.md", "p2-faces", "\n".join(lines))
    props = res / "compare_props.json"
    if props.exists():
        rep = json.loads(props.read_text(encoding="utf-8"))
        lines = ["| Props (COCO classes) | dark AP50 | profile AP50 | crowd AP50 | dark AP | profile AP | crowd AP | ms/frame |",
                 "|---|---|---|---|---|---|---|---|"]
        for name, r in rep.items():
            by = {_tag(c): v for c, v in r["clips"].items()}
            cells = [f"{by[c]['AP50']:.3f}" if c in by else "—" for c in conds]
            cells += [f"{by[c]['AP']:.3f}" if c in by else "—" for c in conds]
            label = MODEL_NAMES.get(Path(name).stem, name)
            lines.append(f"| {label} | " + " | ".join(cells) + f" | {r['ms_per_frame']} |")
        done |= replace_block(ROOT / "p2_detection" / "README.md", "p2-props", "\n".join(lines))
    return done


def clip_ap50() -> dict:
    """{detector: (dark, profile, crowd) AP50} from compare_faces.json, for the root table."""
    faces = ROOT / "p2_detection" / "results" / "compare_faces.json"
    if not faces.exists():
        return {}
    out = {}
    for name, r in json.loads(faces.read_text(encoding="utf-8")).items():
        by = {_tag(c): v for c, v in r.items() if isinstance(v, dict) and c != "all"}
        out[name] = tuple(by[c]["AP50"] if c in by else None for c in ("dark", "profile", "crowd"))
    return out


def root_table(p1: dict, p2: dict, clips: dict | None = None) -> None:
    path = ROOT / "README.md"
    text = path.read_text(encoding="utf-8")
    if p1:
        what = {"own": "held-out episodes", "public": "public-clip shots"}.get(p1.get("_set"), "test shots")
        n = f", n={p1['_n']}" if "_n" in p1 else ""
        ours = " · ".join(f"{p1[k]:.3f} ({lab})" for k, lab in
                          (("pretrained", "ResNet-18, ImageNet init"), ("scratch", "random init")) if k in p1)
        base = f"{p1['fused']:.3f} (teacher: open_clip + YuNet)" if "fused" in p1 else "—"
        text = re.sub(r"^\| 1 \| Shot-framing accuracy[^|]*\|.*$",
                      f"| 1 | Shot-framing accuracy (hand-labelled {what}{n}) | {ours or '—'} | {base} |",
                      text, flags=re.M)
    if clips:
        def ap3(v: tuple) -> str:
            return " / ".join("—" if x is None else f"{x:.3f}" for x in v)

        ours = [f"{ap3(v)} ({n.removeprefix('CenterNet-R18 ').strip('()')})" for n, v in clips.items()
                if n.startswith("CenterNet-R18 (ours")]
        base = [f"{ap3(v)} (YuNet)" for n, v in clips.items() if "YuNet" in n]
        text = re.sub(r"^\| 2 \| Face AP50 on hand-labelled clip frames[^|]*\|.*$",
                      "| 2 | Face AP50 on hand-labelled clip frames (dark / profile / crowd) | "
                      f"{' · '.join(ours) or '—'} | {' · '.join(base) or '—'} |", text, flags=re.M)
    if p2:
        best = max(p2.values(), key=lambda r: r["easy"])
        text = re.sub(r"^\| 2 \| Face AP \(WIDER FACE Easy[^)]*\) \|.*$",
                      f"| 2 | Face AP (WIDER FACE Easy / Medium / Hard) | {best['easy']:.3f} / {best['medium']:.3f} / "
                      f"{best['hard']:.3f} | — |", text, flags=re.M)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    p1, p2 = phase1(), phase2()
    week8()
    root_table(p1, p2, clip_ap50())
    print(f"phase 1 rows: {[k for k in p1 if not k.startswith('_')]}; phase 2 rows: {list(p2)}")


if __name__ == "__main__":
    main()
