"""Score the from-scratch pose network against the teacher and against by-eye labels.

    python p6_pose_video/build_pose_data.py --clips goldeneye people-detection --fps 1 --keep-frames \
        --out outputs/p6/pose_heldout.npz
    python p6_pose_video/eval_pose.py --ckpt outputs/p6/pose_sb.pt

Two questions, kept apart because they measure different things:

1. **Does the student copy the teacher?** On held-out clips the network never
   trained on, YOLOv8n-pose's own output is the ground truth (keypoints with
   confidence >= 0.5 count as labelled; area = 0.53 x box area, the usual
   box-to-segment approximation). The student runs top-down on the teacher's
   boxes, so this isolates keypoint localisation. A perfect score means "as
   good as the teacher", not "correct".
2. **Is anyone right?** On goldeneye frames, keypoints labelled by eye
   (p6_pose_video/labels/goldeneye_pose.json) score both the student (on the
   YOLOv8n detector's boxes, end to end) and YOLOv8n-pose. People the labeller
   skipped are added as ignore regions so detecting them is not a false positive.
   It is a small set (13 people, 62 keypoints, mostly faces and shoulders: the
   clip is close-ups), so the numbers are indicative only.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cinescope.pose.oks import KeypointEvaluator, oks_matrix  # noqa: E402
from cinescope.pose.simplebaseline import load_checkpoint, predict  # noqa: E402
from cinescope.video.clips import resolve  # noqa: E402
from cinescope.video.frames import frames_at  # noqa: E402

AREA_FRAC = 0.53


def _ann(aid, iid, box, kp, ignore=False):
    x0, y0, x1, y1 = box
    k = np.asarray(kp, np.float64).reshape(17, 3)
    return {"id": aid, "image_id": iid, "category_id": 1, "bbox": [x0, y0, x1 - x0, y1 - y0],
            "area": AREA_FRAC * (x1 - x0) * (y1 - y0), "keypoints": k.ravel().tolist(),
            "num_keypoints": 0 if ignore else int((k[:, 2] > 0).sum()), "iscrowd": 0}


def _det(iid, kp, score):
    return {"image_id": iid, "category_id": 1, "keypoints": np.asarray(kp, np.float64).ravel().tolist(),
            "score": float(score)}


def mean_best_oks(gt: dict, dets: list[dict]) -> float:
    """Mean over labelled people of the best OKS any detection in the image reaches."""
    vals = []
    for a in gt["annotations"]:
        if a["num_keypoints"] == 0:
            continue
        ds = [d for d in dets if d["image_id"] == a["image_id"]]
        vals.append(float(oks_matrix(ds, [a]).max()) if ds else 0.0)
    return float(np.mean(vals)) if vals else float("nan")


def summary(gt, dets) -> dict:
    s = KeypointEvaluator(gt).evaluate(dets)
    return {"AP": round(s["AP"], 3), "AP50": round(s["AP50"], 3), "AP75": round(s["AP75"], 3),
            "AR": round(s["AR20"], 3), "mean_best_OKS": round(mean_best_oks(gt, dets), 3),
            "people": sum(a["num_keypoints"] > 0 for a in gt["annotations"])}


def teacher_agreement(net, input_hw, data_path: str) -> dict:
    d = np.load(data_path, allow_pickle=True)
    frames, people = d["frames"], d["people"]
    out = {}
    for clip in sorted({f["clip"] for f in frames}):
        images, anns, dets, n_ms = [], [], [], 0
        t0 = time.time()
        for iid, (f, p) in enumerate(zip(frames, people, strict=True)):
            if f["clip"] != clip:
                continue
            images.append({"id": iid, "width": f["rgb"].shape[1], "height": f["rgb"].shape[0]})
            boxes, scores, kpts = p["boxes"], p["scores"], p["kpts"]
            for b, k in zip(boxes, kpts, strict=True):
                kk = k.copy()
                kk[:, 2] = np.where(k[:, 2] >= 0.5, 2, 0)
                anns.append(_ann(len(anns) + 1, iid, b, kk))
            if len(boxes):
                pred = predict(net, f["rgb"], boxes, input_hw)
                n_ms += 1
                for s, k in zip(scores, pred, strict=True):
                    dets.append(_det(iid, k, s * k[:, 2].mean()))
        gt = {"images": images, "annotations": anns, "categories": [{"id": 1, "name": "person"}]}
        out[clip] = {**summary(gt, dets), "frames": len(images),
                     "ms_per_frame_student": round(1000 * (time.time() - t0) / max(1, n_ms), 1)}
    return out


def by_eye(net, input_hw, labels_path: str, index_path: str) -> dict:
    from cinescope.pose.yolo import YoloPose

    lab = json.loads(Path(labels_path).read_text(encoding="utf-8"))
    index = json.loads(Path(index_path).read_text(encoding="utf-8"))
    labelled = {p["id"]: p for p in lab["people"]}
    frame_keys = sorted({(e["clip"], e["t"]) for e in index if e["id"] in labelled})
    clip = frame_keys[0][0]
    frames = frames_at(resolve(clip), [t for _, t in frame_keys])
    images, anns = [], []
    for iid, ((_, t), fr) in enumerate(zip(frame_keys, frames, strict=True)):
        images.append({"id": iid, "width": fr.shape[1], "height": fr.shape[0]})
        for e in index:
            if (e["clip"], e["t"]) != (clip, t):
                continue
            if e["id"] in labelled:
                p = labelled[e["id"]]
                x, y, w, h = p["bbox"]
                anns.append(_ann(len(anns) + 1, iid, [x, y, x + w, y + h], p["keypoints"]))
            else:            # a person the labeller skipped: ignore region
                anns.append(_ann(len(anns) + 1, iid, e["box"], np.zeros(51), ignore=True))
    gt = {"images": images, "annotations": anns, "categories": [{"id": 1, "name": "person"}]}
    det = YoloPose(pose=False, conf=0.25)
    yolo = YoloPose(pose=True, conf=0.25)
    res = {}
    t0 = time.time()
    lib = yolo(frames)
    t_lib = (time.time() - t0) / len(frames)
    dets_lib = [_det(i, k, s) for i, p in enumerate(lib) for k, s in zip(p.kpts, p.scores, strict=True)]
    res["YOLOv8n-pose (library)"] = {**summary(gt, dets_lib), "ms_per_frame": round(1000 * t_lib, 1)}
    t0 = time.time()
    boxes = det(frames)
    dets_ours = []
    for i, (fr, p) in enumerate(zip(frames, boxes, strict=True)):
        if len(p):
            for k, s in zip(predict(net, fr, p.boxes, input_hw), p.scores, strict=True):
                dets_ours.append(_det(i, k, s * k[:, 2].mean()))
    t_ours = (time.time() - t0) / len(frames)
    res["SimpleBaseline (ours) on YOLOv8n boxes"] = {**summary(gt, dets_ours), "ms_per_frame": round(1000 * t_ours, 1)}
    res["_n"] = {"people": len(labelled), "keypoints": int(sum(p["num_keypoints"] for p in labelled.values())),
                 "frames": len(images)}
    return res


def example_sheet(net, input_hw, data_path: str, out: str, n: int = 6) -> None:
    """Teacher (left) vs student (right) on held-out frames, for the README."""
    from PIL import Image, ImageDraw

    from p6_pose_video.breakdown import draw_people

    d = np.load(data_path, allow_pickle=True)
    frames, people = d["frames"], d["people"]
    pick = []
    for clip in sorted({f["clip"] for f in frames}):
        idx = [i for i, (f, p) in enumerate(zip(frames, people, strict=True)) if f["clip"] == clip and len(p["boxes"])]
        pick += [idx[int(j)] for j in np.linspace(0, len(idx) - 1, n // 2)]
    w = 300
    tiles = []
    for i in pick:
        f, p = frames[i], people[i]
        sc = w / f["rgb"].shape[1]
        pair = []
        for k in (p["kpts"], predict(net, f["rgb"], p["boxes"], input_hw)):
            im = Image.fromarray(f["rgb"]).resize((w, round(f["rgb"].shape[0] * sc)))
            draw_people(im, p["boxes"], k, None, sc, min_conf=0.5 if k is p["kpts"] else 0.3)
            pair.append(im)
        tiles.append(pair)
    h = max(t[0].height for t in tiles)
    sheet = Image.new("RGB", (2 * w + 12, len(tiles) * (h + 4) + 22), (18, 18, 18))
    dr = ImageDraw.Draw(sheet)
    dr.text((4, 4), "YOLOv8n-pose (teacher)", fill=(255, 230, 120))
    dr.text((w + 12, 4), "SimpleBaseline (ours, same boxes)", fill=(255, 230, 120))
    for r, (a, b) in enumerate(tiles):
        sheet.paste(a, (4, 22 + r * (h + 4)))
        sheet.paste(b, (w + 8, 22 + r * (h + 4)))
    sheet.save(out, quality=80)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="outputs/p6/pose_sb.pt")
    ap.add_argument("--heldout", default="outputs/p6/pose_heldout.npz")
    ap.add_argument("--labels", default="p6_pose_video/labels/goldeneye_pose.json")
    ap.add_argument("--index", default="p6_pose_video/labels/goldeneye_pose_index.json")
    ap.add_argument("--out", default="p6_pose_video/results/pose.json")
    ap.add_argument("--sheet", default="p6_pose_video/results/pose_examples.jpg")
    ap.add_argument("--sheet-only", action="store_true")
    args = ap.parse_args(argv)
    import torch

    torch.set_num_threads(2)
    net, ck = load_checkpoint(args.ckpt)
    input_hw = tuple(ck.get("input_hw", (128, 96)))
    example_sheet(net, input_hw, args.heldout, args.sheet)
    if args.sheet_only:
        return
    res = {"checkpoint": {"steps": ck.get("steps"), "imagenet_init": ck.get("imagenet"),
                          "n_train_crops": ck.get("n_train")}}
    res["teacher_agreement"] = teacher_agreement(net, input_hw, args.heldout)
    print(json.dumps(res["teacher_agreement"], indent=1))
    res["by_eye_goldeneye"] = by_eye(net, input_hw, args.labels, args.index)
    print(json.dumps(res["by_eye_goldeneye"], indent=1))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
