# Phase 2 — Detection (Weeks 6–8)

**Feature:** CineScope finds the faces in a clip with a detector built and
evaluated here, and compares it (and production prop detectors) on movie frames.

```powershell
python p2_detection/detect_clip.py trailers/Ted_Lasso_120s.mkv --ckpt p2_detection/checkpoints/centernet_r18.pt
# -> outputs/Ted_Lasso_120s/faces.json, faces.jpg
```

## Week 6 — the metric code (NumPy)

| Built | Module | Checked against |
|---|---|---|
| IoU, crowd IoU, GIoU, box formats | `cinescope/detect/boxes.py` | hand-worked cases; `torchvision.ops.box_iou` |
| Greedy NMS, per-class NMS, Soft-NMS | `cinescope/detect/nms.py` | `torchvision.ops.nms` (identical indices on random boxes) |
| COCO evaluation: AP@[.5:.95], AP50/75, S/M/L, AR@1/10/100 | `cinescope/detect/coco_eval.py` | `pycocotools` on random ground truth + noisy detections (to 1e-10); hand-worked cases |
| WIDER FACE Easy/Medium/Hard protocol | `cinescope/detect/wider_eval.py` | the official toolkit's algorithm (score normalisation, ignore lists, 1000 thresholds) |

`pip install pycocotools` enables the comparison test (it may not have a
wheel for the newest Python; the test skips if it is missing).

**How mAP@0.5:0.95 is calculated.** For each category and each IoU threshold
t ∈ {0.50, 0.55, …, 0.95}: sort all detections by score; walk down the list,
matching each to the best still-unmatched ground truth with IoU ≥ t (TP) or
none (FP); compute precision and recall after every detection; replace each
precision by the best precision at any higher recall (the monotone envelope);
read it at 101 recall points and average → AP(category, t). mAP is the mean
over categories and thresholds. Ground truth marked crowd, or outside the
area range, is ignored: matching it counts neither way. AP@0.5 rewards
finding the object; the average up to 0.95 rewards tight boxes.

## Week 7 — CenterNet face detector (PyTorch)

| Built | Module |
|---|---|
| Heatmap/size/offset targets, exact Gaussian radius, NumPy reference decoder | `cinescope/detect/centernet_targets.py` |
| ResNet-18 + FPN (P2, stride 4) + three heads; penalty-reduced focal loss; L1 size/offset; max-pool peak decoding | `cinescope/detect/centernet.py` |
| WIDER FACE parsing, RetinaFace-style square-crop augmentation | `cinescope/detect/wider.py` |
| Training / evaluation | `p2_detection/train_faces.py`, `p2_detection/eval_wider.py` |

Worth knowing:

- **The Gaussian radius.** CenterNet's code (inherited from CornerNet) takes
  the wrong root of its quadratic and returns radii several times larger than
  the geometry implies. `gaussian_radius(exact=True)` solves the equation for
  "shift the centre diagonally by r, keep IoU ≥ 0.7" (the test checks IoU is
  exactly 0.7 at r); `exact=False` reproduces the original.
- **Heads start calibrated.** The heatmap bias starts at −2.19, so every cell
  begins at p = 0.1 instead of 0.5 — otherwise the ~16k negative cells per
  image swamp the first iterations' loss.
- **No NMS needed in principle**: a 3×3 max-pool keeps only local maxima. A
  light NMS still runs at inference for large faces with two adjacent peaks.

### Data and training (GPU: Kaggle / Colab)

WIDER FACE (Yang et al., 2016): 32k images, 393k faces, from the official site
— `WIDER_train.zip`, `WIDER_val.zip`, `wider_face_split.zip`, and the
evaluation toolkit (for the `.mat` ground truth of Easy/Medium/Hard).

```bash
python p2_detection/train_faces.py --wider /kaggle/input/wider-face --epochs 70 --bs 32
python p2_detection/eval_wider.py --wider /kaggle/input/wider-face --ckpt p2_detection/checkpoints/centernet_r18.pt \
    --eval-tools /kaggle/input/wider-face/eval_tools/ground_truth
```

CPU smoke test first: `python p2_detection/train_faces.py --synthetic --epochs 2`.

### Results — WIDER FACE val

| Model | Easy | Medium | Hard | ms/image |
|---|---|---|---|---|
| CenterNet-R18-FPN (ours), 512 px training, ≤1024 px test | — | — | — | — |

## Week 8 — against production detectors, on movie frames

Three clips that stress detectors differently — **dark**, **profile**,
**crowd** — about 30 frames each, labelled by hand:

```powershell
python p2_detection/label_boxes.py trailers/Lanterns_S01E01_900s.mkv --start 300 --seconds 60 --tag dark
# draw boxes in outputs/boxes_.../index.html, "Download labels", move the JSON to p2_detection/labels/
python p2_detection/compare_faces.py p2_detection/labels/*.boxes.json --ckpt p2_detection/checkpoints/centernet_r18.pt --yunet models/face_detection_yunet_2023mar.onnx
pip install ultralytics
python p2_detection/compare_props.py p2_detection/labels/*.boxes.json --models yolo11n.pt rtdetr-l.pt
```

| Faces | dark AP50 | profile AP50 | crowd AP50 | all AP | ms/frame |
|---|---|---|---|---|---|
| CenterNet-R18 (ours) | — | — | — | — | — |
| OpenCV YuNet | — | — | — | — | — |

| Props (COCO classes) | dark AP50 | profile AP50 | crowd AP50 | ms/frame |
|---|---|---|---|---|
| YOLO11n | — | — | — | — |
| RT-DETR-L | — | — | — | — |

COCO has no "gun" class — the roadmap's example prop. Recognising one needs a
fine-tuned model, which is exactly the argument for owning a training pipeline.

**DETR vs. the rest.** DETR (Carion et al., 2020) predicts a fixed set of N
boxes with a transformer decoder and trains with a one-to-one Hungarian
matching between predictions and ground truth. Because each object is matched
to exactly one prediction, duplicates are penalised during training and no
NMS is needed at inference. CenterNet removes anchors but keeps a dense
per-cell output; FCOS (Tian et al., 2019) is anchor-free and dense too, and
still uses NMS. RT-DETR makes the DETR recipe real-time.

## Interview questions

**Anchor-based vs. anchor-free — trade-offs?** Anchors give the regressor a
good starting box and handle several objects per location (different anchor
shapes), but they add hyper-parameters (sizes, ratios, IoU thresholds for
positive/negative assignment) and most anchors are easy negatives. Anchor-free
detectors (CenterNet, FCOS) predict per location directly: simpler, fewer
knobs, often faster; the cost is ambiguity when two objects share a location,
handled by FPN levels (FCOS) or accepted (CenterNet at stride 4).

**Why does NMS fail in crowds, and what replaces it?** NMS assumes two boxes
that overlap a lot are the same object. In a crowd two real faces can overlap
more than the threshold, and the lower-scoring one is deleted — raising the
threshold instead lets duplicates through. Soft-NMS decays overlapping scores
rather than deleting them (implemented and tested here); learned alternatives
are set prediction without NMS (DETR's one-to-one matching) and peak
extraction on a centre heatmap (CenterNet).
