# Phase 2 — Detection (Weeks 6–8)

**Feature:** CineScope finds the faces in a clip with a detector built and
evaluated here, and compares it (and production prop detectors) on movie frames.

```powershell
python p2_detection/detect_clip.py trailers/goldeneye.mp4 --ckpt p2_detection/checkpoints/centernet_r18.pt
# -> outputs/goldeneye/faces.json, faces.jpg
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

<!-- auto:p2-wider -->
| Model | Easy | Medium | Hard | ms/image |
|---|---|---|---|---|
| CenterNet-R18-FPN (centernet_r18), linear size, 512 px training, ≤1024 px test, top-1000 peaks | 0.575 | 0.713 | 0.784 | 59.2 (T4) |
| CenterNet-R18-FPN (centernet_r18_log), log size, 512 px training, ≤1024 px test, top-1000 peaks | 0.593 | 0.726 | 0.792 | 58.9 (T4) |
<!-- /auto:p2-wider -->

<!-- auto:p2-diagnosis -->
**Where `centernet_r18` loses AP** (per face size; from `diagnose_wider.py`):

### all faces

| size | faces | found_any_score | found_score>=0.5 | mislocalised | missed | median_iou_found | median_score_found | duplicates_per_face |
|---|---|---|---|---|---|---|---|---|
| 0-16 px | 18507 | 0.659 | 0.487 | 0.201 | 0.14 | 0.674 | 0.593 | 0.0 |
| 16-32 px | 10268 | 0.896 | 0.712 | 0.074 | 0.03 | 0.766 | 0.598 | 0.001 |
| 32-64 px | 6650 | 0.948 | 0.76 | 0.04 | 0.012 | 0.828 | 0.589 | 0.003 |
| 64-128 px | 2784 | 0.974 | 0.69 | 0.02 | 0.005 | 0.843 | 0.554 | 0.021 |
| 128-256 px | 1062 | 0.98 | 0.593 | 0.015 | 0.005 | 0.865 | 0.546 | 0.051 |
| 256-∞ px | 437 | 0.991 | 0.389 | 0.007 | 0.002 | 0.866 | 0.44 | 0.316 |

### easy

| size | faces | found_any_score | found_score>=0.5 | mislocalised | missed | median_iou_found | median_score_found | duplicates_per_face |
|---|---|---|---|---|---|---|---|---|
| 16-32 px | 15 | 0.8 | 0.0 | 0.2 | 0.0 | 0.622 | 0.225 | 0.0 |
| 32-64 px | 2915 | 0.952 | 0.736 | 0.036 | 0.012 | 0.836 | 0.581 | 0.004 |
| 64-128 px | 2782 | 0.974 | 0.69 | 0.02 | 0.005 | 0.843 | 0.554 | 0.021 |
| 128-256 px | 1062 | 0.98 | 0.593 | 0.015 | 0.005 | 0.865 | 0.546 | 0.051 |
| 256-∞ px | 437 | 0.991 | 0.389 | 0.007 | 0.002 | 0.866 | 0.44 | 0.316 |

### medium

| size | faces | found_any_score | found_score>=0.5 | mislocalised | missed | median_iou_found | median_score_found | duplicates_per_face |
|---|---|---|---|---|---|---|---|---|
| 0-16 px | 3 | 0.0 | 0.0 | 0.667 | 0.333 | — | — | 0.0 |
| 16-32 px | 2402 | 0.902 | 0.695 | 0.072 | 0.026 | 0.785 | 0.583 | 0.002 |
| 32-64 px | 6631 | 0.949 | 0.761 | 0.039 | 0.012 | 0.828 | 0.59 | 0.003 |
| 64-128 px | 2784 | 0.974 | 0.69 | 0.02 | 0.005 | 0.843 | 0.554 | 0.021 |
| 128-256 px | 1062 | 0.98 | 0.593 | 0.015 | 0.005 | 0.865 | 0.546 | 0.051 |
| 256-∞ px | 437 | 0.991 | 0.389 | 0.007 | 0.002 | 0.866 | 0.44 | 0.316 |

### hard

| size | faces | found_any_score | found_score>=0.5 | mislocalised | missed | median_iou_found | median_score_found | duplicates_per_face |
|---|---|---|---|---|---|---|---|---|
| 0-16 px | 10757 | 0.784 | 0.613 | 0.148 | 0.068 | 0.694 | 0.608 | 0.0 |
| 16-32 px | 10268 | 0.896 | 0.712 | 0.074 | 0.03 | 0.766 | 0.598 | 0.001 |
| 32-64 px | 6650 | 0.948 | 0.76 | 0.04 | 0.012 | 0.828 | 0.589 | 0.003 |
| 64-128 px | 2784 | 0.974 | 0.69 | 0.02 | 0.005 | 0.843 | 0.554 | 0.021 |
| 128-256 px | 1062 | 0.98 | 0.593 | 0.015 | 0.005 | 0.865 | 0.546 | 0.051 |
| 256-∞ px | 437 | 0.991 | 0.389 | 0.007 | 0.002 | 0.866 | 0.44 | 0.316 |

**Where `centernet_r18_log` loses AP** (per face size; from `diagnose_wider.py`):

### all faces

| size | faces | found_any_score | found_score>=0.5 | mislocalised | missed | median_iou_found | median_score_found | duplicates_per_face |
|---|---|---|---|---|---|---|---|---|
| 0-16 px | 18507 | 0.667 | 0.472 | 0.195 | 0.138 | 0.678 | 0.578 | 0.0 |
| 16-32 px | 10268 | 0.897 | 0.701 | 0.076 | 0.027 | 0.768 | 0.586 | 0.001 |
| 32-64 px | 6650 | 0.949 | 0.74 | 0.039 | 0.012 | 0.826 | 0.579 | 0.002 |
| 64-128 px | 2784 | 0.974 | 0.655 | 0.02 | 0.005 | 0.841 | 0.545 | 0.025 |
| 128-256 px | 1062 | 0.984 | 0.611 | 0.011 | 0.005 | 0.858 | 0.543 | 0.05 |
| 256-∞ px | 437 | 0.986 | 0.405 | 0.014 | 0.0 | 0.865 | 0.451 | 0.286 |

### easy

| size | faces | found_any_score | found_score>=0.5 | mislocalised | missed | median_iou_found | median_score_found | duplicates_per_face |
|---|---|---|---|---|---|---|---|---|
| 16-32 px | 15 | 0.733 | 0.0 | 0.267 | 0.0 | 0.625 | 0.177 | 0.067 |
| 32-64 px | 2915 | 0.953 | 0.708 | 0.037 | 0.011 | 0.837 | 0.571 | 0.003 |
| 64-128 px | 2782 | 0.974 | 0.655 | 0.02 | 0.005 | 0.841 | 0.545 | 0.025 |
| 128-256 px | 1062 | 0.984 | 0.611 | 0.011 | 0.005 | 0.858 | 0.543 | 0.05 |
| 256-∞ px | 437 | 0.986 | 0.405 | 0.014 | 0.0 | 0.865 | 0.451 | 0.286 |

### medium

| size | faces | found_any_score | found_score>=0.5 | mislocalised | missed | median_iou_found | median_score_found | duplicates_per_face |
|---|---|---|---|---|---|---|---|---|
| 0-16 px | 3 | 0.0 | 0.0 | 0.667 | 0.333 | — | — | 0.0 |
| 16-32 px | 2402 | 0.903 | 0.679 | 0.075 | 0.022 | 0.785 | 0.575 | 0.002 |
| 32-64 px | 6631 | 0.95 | 0.741 | 0.038 | 0.011 | 0.827 | 0.579 | 0.002 |
| 64-128 px | 2784 | 0.974 | 0.655 | 0.02 | 0.005 | 0.841 | 0.545 | 0.025 |
| 128-256 px | 1062 | 0.984 | 0.611 | 0.011 | 0.005 | 0.858 | 0.543 | 0.05 |
| 256-∞ px | 437 | 0.986 | 0.405 | 0.014 | 0.0 | 0.865 | 0.451 | 0.286 |

### hard

| size | faces | found_any_score | found_score>=0.5 | mislocalised | missed | median_iou_found | median_score_found | duplicates_per_face |
|---|---|---|---|---|---|---|---|---|
| 0-16 px | 10757 | 0.796 | 0.6 | 0.138 | 0.065 | 0.698 | 0.591 | 0.0 |
| 16-32 px | 10268 | 0.897 | 0.701 | 0.076 | 0.027 | 0.768 | 0.586 | 0.001 |
| 32-64 px | 6650 | 0.949 | 0.74 | 0.039 | 0.012 | 0.826 | 0.579 | 0.002 |
| 64-128 px | 2784 | 0.974 | 0.655 | 0.02 | 0.005 | 0.841 | 0.545 | 0.025 |
| 128-256 px | 1062 | 0.984 | 0.611 | 0.011 | 0.005 | 0.858 | 0.543 | 0.05 |
| 256-∞ px | 437 | 0.986 | 0.405 | 0.014 | 0.0 | 0.865 | 0.451 | 0.286 |
<!-- /auto:p2-diagnosis -->

Training: 60 epochs × 402 steps, batch 32, AdamW 5e-4 with cosine decay, mixed
precision, about 3.5 h on one Kaggle T4; loss 2.59 → 1.10.

**Open question — Easy is the lowest of the three.** For almost every published
detector Easy > Medium > Hard, because Easy contains only the large, clear
faces. An inversion means the model loses points on large faces, and the
aggregate AP cannot say how: missed, mislocalised (box too loose or tight for
IoU 0.5), found with low confidence, or found twice. `diagnose_wider.py`
answers that per face size; the second Kaggle run evaluates with it (and with
top-1000 peaks, since the first evaluation kept only 100 per image, which caps
recall on crowd images). The likely suspects and the fix for each:

- sizes regressed linearly in cells, where a 5% error on a 100-cell face is a
  5-cell loss — the loss is dominated by large faces yet still loose for them
  in relative terms → `--size-mode log` (scale-invariant, as CenterFace does);
- training crops that cut large faces, whose clipped boxes teach the model
  undersized boxes;
- duplicate peaks on large faces.

The log-size run moved Easy only from 0.575 to 0.593, and post-processing moved it by
0.001 (below), so the third suspect is next. **v3** (`kaggle/run_all.py --only p2_train_v3`,
about 4 h on a T4) trains with log sizes plus two options of `train_faces.py`:
`--keep-large 64` rejects a random crop that would cut through a face 64 px or taller
(a large face is either whole in the crop or not in it at all), and `--low-light 0.3`
turns 30% of the samples into night frames (gamma 1.8–3.2, a cool cast, sensor noise)
for the dark clip below, where ours misses 7 of 30 faces. Its WIDER rows and diagnosis
fill in the tables above by themselves; `python tools/run_everything.py p2` then scores
`checkpoints/centernet_r18_v3.pt` on the movie frames.

## Week 8 — against production detectors, on movie frames

Three clips that stress detectors differently — **dark**, **profile**,
**crowd** — about 30 frames each, labelled by hand (faces, and the COCO
classes visible in them: person everywhere, tv in the dark clip):

```powershell
python p2_detection/label_boxes.py trailers/your_clip.mp4 --start 300 --seconds 60 --tag dark
# draw boxes in outputs/boxes_.../index.html, "Download labels", move the JSON to p2_detection/labels/
python p2_detection/compare_faces.py p2_detection/labels/*.boxes.json --yunet data/weights/yunet.onnx \
    --ckpt p2_detection/checkpoints/centernet_r18.pt --ckpt p2_detection/checkpoints/centernet_r18_log.pt
python p2_detection/compare_props.py p2_detection/labels/*.boxes.json   # yolov8n, yolo11n, rtdetr-l from data/weights
python tools/update_results.py
```

The three labelled clips are public: GoldenEye's trailer (43–82 s: low-key
interiors, a snowy night, a face lit by a match), Intel's head-pose sample
video (tilts, three-quarter views, full profiles) and Intel's classroom video
(four people at desks; a small crowd, since no public clip at hand has a big
one with faces larger than a few pixels). The boxes were drawn **by eye**,
reading corners off a pixel grid on each frame, before any detector ran on
these frames, and were not changed afterwards: `labels/README.md` has the
details and the commands that regenerate the frames.

<!-- auto:p2-faces -->
| Faces | dark AP50 | profile AP50 | crowd AP50 | all AP50 | all AP | ms/frame |
|---|---|---|---|---|---|---|
| CenterNet-R18 (ours) | 0.915 | 0.956 | 0.544 | 0.658 | 0.163 | 528.7 |
| CenterNet-R18 (ours, log) | 0.912 | 0.936 | 0.524 | 0.635 | 0.165 | 573.9 |
| OpenCV YuNet | 0.989 | 0.980 | 0.605 | 0.748 | 0.216 | 37.1 |
<!-- /auto:p2-faces -->

<!-- auto:p2-props -->
| Props (COCO classes) | dark AP50 | profile AP50 | crowd AP50 | dark AP | profile AP | crowd AP | ms/frame |
|---|---|---|---|---|---|---|---|
| YOLOv8n | 0.933 | 1.000 | 1.000 | 0.662 | 0.990 | 0.850 | 1234.3 |
| YOLO11n | 0.978 | 1.000 | 0.990 | 0.737 | 1.000 | 0.842 | 1823.9 |
| RT-DETR-L | 0.977 | 1.000 | 1.000 | 0.780 | 0.995 | 0.866 | 5967.1 |
<!-- /auto:p2-props -->

AP50 / AP are COCO-style (the week-6 evaluator), per clip; "all" pools the
three clips. ms/frame is the median over the frames (1280 px wide, warm-up
excluded) with 2 CPU threads. The face rows were measured together on a
laptop CPU (Intel Core 5 120U); the prop rows on a 4-core cloud CPU **shared
with other jobs** (load average about 13), so compare face rows with face
rows and prop rows with prop rows, not across the two tables or with
published speeds.

- **Ours vs. YuNet: YuNet wins every clip, at 1/14 of the time.** Both
  CenterNet checkpoints score 0.64–0.66 AP50 over the three clips against
  YuNet's 0.75, and the `log` model, better on WIDER FACE val (0.593 vs.
  0.575 Easy), is slightly *worse* here: half a point of WIDER AP does not
  transfer to 91 movie frames. ms/frame is ~535 for ours (ResNet-18-FPN at
  1280 px, PyTorch) against 37 for YuNet (a 75k-parameter network built for
  this). Where ours loses, counting boxes with score ≥ 0.3:
  - **dark**: it misses 7 of the 30 faces outright (recall 0.77 at IoU 0.3;
    YuNet 1.00), the faces lit only by a match or seen against snow at night.
    WIDER FACE has few faces this dark; this is the gap a fine-tune on
    darker frames, or brightness augmentation, would target.
  - **crowd**: it finds 123 of 124 faces at IoU 0.3, as YuNet does, and
    loses at IoU 0.5 for the same reason (median IoU 0.55 vs. 0.56): its top
    edge sits a median 15 px below the hand boxes' hairline, exactly
    where YuNet's does. So the earlier guess that a WIDER-trained detector
    "draws the label convention" is wrong; at 50–90 px both detectors box
    from the eyebrows, and the crowd AP is a convention gap shared by both.
  - **profile**: close to YuNet (0.956 vs. 0.980 AP50), no confident false
    positives.
- **YuNet finds every crowd face, but AP50 says 0.6.** With score ≥ 0.5,
  every one of the 124 classroom faces has a YuNet box at IoU ≥ 0.3, yet only
  75% reach 0.5 (median IoU 0.56). The hand boxes run from the hairline to
  the chin, as WIDER FACE's do; YuNet's top edge sits a median 15 px lower,
  around the eyebrows. On faces 50–90 px tall that alone costs the IoU.
  The low AP@[.5:.95] in every clip is the same mismatch: box conventions,
  not detections.
- **Post-processing does not fix the large-face duplicates.** `tune_postproc.py`
  replays the saved WIDER val predictions through stricter suppression
  (NMS at 0.4/0.3, dropping boxes mostly inside a higher-scoring one, NMS
  only between large boxes) and re-scores them: every rule moves Easy by
  0.001 or less, on both checkpoints (`results/centernet_r18*/postproc.md`).
  The duplicates are too few to matter; the Easy deficit is in localisation
  and confidence, which needs training changes, not a filter.
- **Props.** At IoU 0.5 all three models nearly solve all three clips
  (AP50 ≥ 0.93); they differ at tight IoU and in the dark clip. There
  RT-DETR-L has the best tv AP (computer monitors, one filling the frame):
  0.82 against 0.65 (YOLO11n) and 0.48 (YOLOv8n), but the lowest person AP
  (0.74 against 0.82 and 0.84). It costs about 5× YOLOv8n's time per frame.
  YOLO11n beats YOLOv8n on the dark clip at about 1.5× the time.

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
