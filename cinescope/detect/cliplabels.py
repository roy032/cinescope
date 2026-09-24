"""Hand-drawn box labels (from p2_detection/label_boxes.py) as COCO ground truth."""
from __future__ import annotations

import json
from pathlib import Path


def load_labels(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def to_coco(labels: dict, classes: list[str], image_id_offset: int = 0) -> dict:
    """COCO ground truth for the given class names. Boxes marked ignore become
    crowd regions (iscrowd=1): matching them is neither rewarded nor punished."""
    cat = {c: i + 1 for i, c in enumerate(classes)}
    images, anns = [], []
    for k, f in enumerate(labels["frames"]):
        iid = image_id_offset + k + 1
        images.append({"id": iid, "file_name": f["file"], "width": f["width"], "height": f["height"]})
        for b in f["boxes"]:
            if b["label"] not in cat:
                continue
            x, y, w, h = b["bbox"]
            anns.append({"id": len(anns) + 1, "image_id": iid,
                         "category_id": cat[b["label"]], "bbox": [x, y, w, h], "area": w * h,
                         "iscrowd": int(b.get("ignore", 0))})
    return {"images": images, "annotations": anns,
            "categories": [{"id": i, "name": c} for c, i in cat.items()]}


def frames_dir(labels_path: str | Path, labels: dict, outputs: str | Path = "outputs") -> Path:
    for cand in (Path(outputs) / f"boxes_{labels['name']}" / "frames", Path(labels_path).parent / "frames"):
        if cand.is_dir():
            return cand
    raise FileNotFoundError(f"frames for {labels['name']} not found; re-run label_boxes.py with the same "
                            "arguments to regenerate them (the labels stay valid)")
