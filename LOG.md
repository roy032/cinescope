# Learning log

One entry per week: what I built, what broke, what I would explain in an interview.

## Week 1 — images as arrays

- Wrote correlation/convolution with a sliding-window sum (k² whole-image
  multiply-adds instead of H·W dot products) and checked it against
  `scipy.ndimage` and `cv2.filter2D` to 1e-9.
- Convolution flips the kernel, correlation does not. It only matters for
  asymmetric kernels — which is exactly Sobel. `cv2.filter2D` is correlation.
- Separable Gaussian: two 1-D passes cost 2k per pixel instead of k².
- sRGB must be linearised before XYZ/Lab. OpenCV's float Lab differs from the
  exact formula by up to ~0.3 ΔE because it approximates the gamma curve.

## Week 2 — shots and colour

- First cut detector used an absolute χ² floor of 0.25 and missed every cut in
  a shot/reverse-shot dialogue (same set, same colours). Switched to a
  threshold relative to the neighbourhood's 90th percentile.
- A flash comes back; a cut does not. The flash guard checks both the onset
  and the end of a short flash.
- k-means++ initialisation, empty-cluster re-seeding, Lab palettes, black
  letterbox bars excluded from palettes.

## Week 3 — motion

- Harris's M is the same matrix Lucas-Kanade inverts: good corners are where
  flow is well-conditioned.
- Pyramidal LK recovers an 11 px shift that single-level LK cannot.
- Hartley normalisation before DLT; RANSAC iteration count adapts to the best
  inlier ratio so far.
- Dark scenes: one practical lamp dominated the Harris response and a 1%
  quality cut-off left 4 corners. Contrast-normalising frames and a 0.1%
  cut-off fixed it.

## Week 4 — backprop by hand

- Scalar autograd engine: gradients accumulate (`+=`) because a value used
  twice gets a contribution from each use; the backward order has to be a
  topological sort. Made the sort iterative after a 5,000-step chain hit
  Python's recursion limit.
- NumPy Conv2d via im2col; its backward pass is col2im — scatter-add each
  receptive field's gradient back onto the pixels it came from.
- Every layer passes a central-difference gradient check (< 1e-6) and matches
  `torch.nn` forward and backward to 1e-10.
- A conv bias before BatchNorm has a gradient of exactly zero.
- BatchNorm running statistics lag when the learning rate is high: train-mode
  accuracy 0.74, eval-mode 0.55 on the same weights.

## Week 5 — ResNet-18

- ResNet-18 from scratch with torchvision's parameter names, so ImageNet
  weights load straight in and the two networks can be compared output for
  output. 11,689,512 parameters, as in torchvision.
- Frames kept at 16:9 (224×384) instead of cropped square; crops in
  augmentation keep ≥ 80% of the frame, or the label stops being true.
- MovieShots splits by trailer to avoid leakage between shots of one trailer.
- Training runs (ImageNet init vs. random init) go on Kaggle; results pending.

## Week 6 — measuring detectors

- COCO evaluation re-implemented step by step: greedy matching per IoU
  threshold with crowd/ignored ground truth, monotone precision envelope,
  101 recall points. Precision is tp / (tp + fp + eps), so a "perfect" AP is
  0.9999999999999999, in pycocotools too.
- NMS deletes one of two real faces that overlap by IoU 0.6 at threshold 0.5;
  Soft-NMS keeps it with a decayed score.
- WIDER FACE's official protocol differs from COCO's: global score
  normalisation, per-setting ignore lists, 1,000 thresholds, "+1" pixel widths.

## Week 7 — CenterNet

- The Gaussian radius in the CenterNet/CornerNet code takes the other root of
  its quadratic: for a 20×20 box it returns 5.5 cells where the geometry gives
  1.9. Implemented the exact radius and kept the original as an option.
- Heatmap bias initialised to −2.19 (p = 0.1).
- One random generator shared by DataLoader workers gives every worker the
  same crops — each sample gets its own generator.

## Week 7 — first WIDER FACE run

- 60 epochs on a Kaggle T4 (~3.5 h): WIDER val Easy 0.633 / Medium 0.740 /
  Hard 0.687. Easy below Medium is the wrong way round: large faces are being
  lost. Wrote a per-size diagnosis (found / mislocalised / missed / low score /
  duplicates) before changing anything, and a log-size option to test the
  most likely cause.
- The first evaluation kept only the top 100 peaks per image; crowd images in
  the Hard set have several hundred faces. Evaluation now keeps 1,000.
- A test asserted the *mean* initial heatmap probability was 0.1; random head
  weights plus a convex sigmoid make it ~0.15. The prior is in the bias —
  the test now checks sigmoid(bias) = 0.1.

## Week 5, revisited — any videos instead of MovieShots

- MovieShots' public folder holds only the annotation JSON; the frames are
  released on request. The first Kaggle run found 0 of 33,653 shots' frames.
- Switched to videos we can get: Phase 0 cuts them into shots, three key
  frames per shot, split by video.
- No labels exist, so a teacher labels the shots: SigLIP zero-shot prompts
  fused with the face-height ratio from the Phase 2 detector. The ResNet-18s
  learn from those labels; a hand-labelled sample of the test split (never
  shown any prediction while labelling) scores every system.


## Weeks 14–17 — calibration, SfM, depth, stereo, splatting

- Zhang calibration from scratch matches `cv2.calibrateCamera` exactly on the
  OpenCV chessboards (RMS 0.418 px); vanishing-point self-calibration gets the
  focal length within 7%, the image-size prior only within 43%.
- From-scratch SfM with a Schur-complement bundle adjustment: 0.19–0.24 px and
  ≤ 0.4% trajectory error on synthetic shots. On trailers it is conservative:
  it reconstructs 3 of 12 shots (COLMAP 10), with lower error on those (0.27
  vs 0.68 px). Pans and low-parallax shots get a rotation-only camera path.
- Plane-sweep stereo beats MiDaS small where it is confident (synthetic si-log
  0.087 vs 0.162); MiDaS covers every pixel.
- Filling DIBR holes from the background side is worth +3.5 dB inside the holes.
- CPU Gaussian splatting: 24.8 dB on held-out frames vs 17.4 dB for a point
  render; actors walking through the shot break the static-scene assumption.

## Weeks 23–24 — scene search and chat with the movie

- Exact, IVF and IVF-PQ search with asymmetric distance, in NumPy: recall
  matches FAISS, speed is 4–6x lower (Python loops over the lists).
- The benchmark caught two bugs: lists probed by L2 instead of inner product,
  and too few k-means samples per centroid.
- open_clip finds the right shot first for 85% of hand-written queries within
  a clip, and in the top 5 for 78% with nine clips pooled; misses are fine
  details about people (clothes, how many, head pose).
- The from-scratch CLIP-style encoder combines unseen colour-shape pairs but
  ignores word order, and is at chance on real queries: vocabulary and data,
  not the method.
- Chat with the movie: retrieval plus timestamped answers, an LLM optional;
  the "not found" threshold is the weak part.

## Weeks 9–11 — character timeline and who's talking

- ByteTrack-style face tracker with resets at shot cuts: MOTA 0.978 / IDF1
  0.989 on goldeneye, against 0.892 / 0.943 for ultralytics' ByteTrack with
  default settings. Hungarian and MOT metrics match scipy and motmetrics.
- A self-supervised face embedding trained on only 23 tracks splits people
  (purity 0.93, pair F1 0.32); SFace clusters goldeneye perfectly. Too little
  data, not the loss: the triplet loss hit zero early.
- Hand-built speaking score (voice activity + mouth motion + synchrony): AP
  0.39 against Light-ASD's 0.92. The audio cue does most of the work;
  synchrony adds nothing measurable.
- Decoding long clips grew the heap by about one frame per frame (glibc
  keeps freed arenas); mmap for large buffers fixed the out-of-memory kills.

## Weeks 21–22 — reading subtitles

- CTC as a log-space alpha recursion matches PyTorch's loss and gradient to
  1e-10, about 2.5x slower (a Python loop over time steps).
- The CRNN read synthetic validation lines at 3% CER but real subtitle crops
  at 68%: training crops were cut at the font's line height, the detector
  cuts around the ink. Cutting synthetic crops around the ink fixed it.
- On a busy CPU one PyTorch thread trained 3x faster than two.
- A subtitle stays still while the picture moves: per-line tracks plus a
  temporal median give one clean crop per line. Watermarks and screens in the
  same band are the main remaining error.

## Weeks 5 and 8, finished on public clips

- The teacher's cues are pluggable: open_clip ViT-B-32 when SigLIP is not
  available, YuNet when there is no CenterNet checkpoint.
- 66 public-clip shots labelled by eye before any prediction existed.
  ResNet-18s trained on a CPU in two folds by time: teacher 0.431, students
  0.176 / 0.137, below the 0.275 majority baseline. With 23–33 training shots
  per fold the students collapse onto two classes; the result needs the
  episode-scale shot set, not a different model.
- Week 8: dark / profile / crowd frames boxed by eye. YuNet face AP50 0.99 /
  0.98 / 0.61; the crowd loss is box convention (eyebrows vs hairline), not
  missed faces. YOLOv8n / YOLO11n / RT-DETR-L reach AP50 ≥ 0.93 on people;
  RT-DETR-L is best at tight IoU in the dark clip and about 5x slower.
