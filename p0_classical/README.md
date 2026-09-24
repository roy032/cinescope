# Phase 0 — Classical CV (Weeks 1–3)

**Feature:** CineScope splits a clip into shots, labels each shot's camera move,
and shows the colour palette of every shot plus a "movie barcode" of the whole
clip. No deep learning: NumPy for every algorithm, OpenCV only as the yardstick.

```powershell
python p0_classical/analyse.py trailers/Ted_Lasso_120s.mkv
# -> outputs/Ted_Lasso_120s/{shots.json, shots.jpg, barcode.png}
```

## What is implemented (and where)

| Week | Built from scratch | Module | Checked against |
|---|---|---|---|
| 1 | Video decoding + frame sampling (PyAV, OpenCV fallback) | `cinescope/io/video.py` | frame index/time round-trip test |
| 1 | 2-D convolution / correlation, separable Gaussian blur, Sobel | `cinescope/imgproc/filters.py` | `scipy.ndimage.convolve`, `cv2.filter2D`, `cv2.GaussianBlur`, `cv2.Sobel` (exact to 1e-6) |
| 1 | RGB↔HSV, RGB↔CIE Lab (with sRGB linearisation), ΔE*76 | `cinescope/imgproc/color.py` | published Lab values; `cv2.cvtColor` (≤0.5 ΔE — OpenCV approximates the gamma curve) |
| 2 | HSV joint histograms, χ² distance | `cinescope/shots/features.py` | |
| 2 | Hard cuts (relative spike threshold + flash guard), fades, dissolves (twin comparison) | `cinescope/shots/cuts.py` | PySceneDetect, FFmpeg scene filter (`eval_cuts.py`) |
| 2 | Boundary scoring (TRECVID-style one-to-one matching with tolerance) | `cinescope/shots/evaluate.py` | hand-worked unit tests |
| 2 | k-means with k-means++ init, Lab palettes, movie barcode | `cinescope/palette/kmeans.py` | scikit-learn inertia (within 5%) |
| 3 | Harris / Shi-Tomasi corners with non-max suppression | `cinescope/motion/harris.py` | checkerboard; `cv2.goodFeaturesToTrack` |
| 3 | Pyramidal Lucas-Kanade optical flow | `cinescope/motion/lk.py` | `cv2.calcOpticalFlowPyrLK` (median diff < 0.1 px) |
| 3 | Normalised DLT + adaptive RANSAC homography | `cinescope/motion/homography.py` | `cv2.findHomography`; 40%-outlier test |
| 3 | Camera-move classifier: static / pan / tilt / zoom / handheld | `cinescope/motion/camera.py` | synthetic shots with known motion |

`tests/` holds 37 tests (`python -m pytest`); every library comparison is
skipped cleanly if that library is not installed.

## Data and the test set

| Clip | Role |
|---|---|
| `Ted_Lasso_120s.mkv`, `Lanterns_S01E01_120s.mkv` | **development** — detector parameters were chosen by looking at these |
| `Ted_Lasso_900s.mkv`, `Lanterns_S01E01_900s.mkv` | **test** — hand-labelled, never used for tuning |
| `*_1800s.mkv` | spare / qualitative |

The test labels are made by hand with the labelling page:

```powershell
python p0_classical/label_cuts.py trailers/Ted_Lasso_900s.mkv
# open outputs/label_Ted_Lasso_900s/index.html, click the first frame of every new shot,
# "Download labels", move the file into p0_classical/labels/
```

The page shows every frame (12 per row) and deliberately shows **no detector
suggestions**, so the labels cannot inherit the detector's blind spots.

## Results

Run `python p0_classical/eval_cuts.py --all` once the labels exist.
Tolerance ±2 frames; gradual transitions match anywhere inside their span.

| Method | Precision | Recall | F1 | Time (5-min clip) |
|---|---|---|---|---|
| Ours (histogram spikes + flash guard + gradual) | — | — | — | — |
| PySceneDetect ContentDetector | — | — | — | — |
| PySceneDetect AdaptiveDetector | — | — | — | — |
| FFmpeg `scene > 0.3` | — | — | — | — |

*Target from the roadmap: F1 ≥ 0.85.*

Development-clip observations (no labels, inspected by eye):

- On `Ted_Lasso_120s`, every true cut scored at least ~12x its neighbourhood's
  90th-percentile distance, and every non-cut peak (people walking through
  frame, lamp flicker, a TV switching off) under ~2x. The first version used an
  absolute floor of 0.25 and missed 11 cuts in shot/reverse-shot dialogue
  (same room, same colours, χ² ≈ 0.15–0.24) — that is why the threshold is
  relative (`ratio=4`) with a low floor (`min_cut=0.08`).
- Timing on one CPU core, 640x480 source decoded at 160 px: ~50 s of
  decoding + features per 5-minute clip, <1 s for boundary detection, ~20 s for
  camera moves on 96 px thumbnails.

## What I learned / interview questions

**Why does Lab beat RGB for colour distance?** RGB is a device encoding: equal
steps are not equal perceived steps (a 40-level change in green is far more
visible than the same change in blue — `test_delta_e_is_perceptual_not_rgb`).
Lab was built so Euclidean distance tracks perceived difference, so k-means
clusters in Lab are colours a viewer would call different. The conversion must
linearise sRGB first; skipping that step gives plausible-looking, wrong values.

**What does RANSAC assume, and how many iterations do you need?** That the
model explains a large enough fraction *w* of the data, and that a minimal
sample (*s* = 4 correspondences for a homography) of inliers yields a good
model. For success probability *p*: N = log(1−p) / log(1−wˢ). At p = 0.995,
w = 0.5 needs 83 iterations; w = 0.2 needs over 3,300. The implementation
recomputes N every time a better inlier set appears.

**Why do dissolves break histogram-based cut detection?** During a dissolve
every frame is a blend of the two shots, so each adjacent-frame distance is
small — nothing spikes. Twin comparison instead watches for a run of
moderately raised distances and compares the frame *before* the run with the
frame *after* it; if those two are as different as a cut, it was a gradual
transition. The same logic has to reject camera moves, which also produce
runs — hence the "no single step explains the change" check.

**Why is a homography the right model for pan/tilt/zoom but not for a dolly?**
A rotating or zooming camera maps one image plane to another exactly by a
homography. A translating camera sees parallax — near objects move more than
far ones — which no single homography explains; it shows up as a low RANSAC
inlier ratio, which the classifier reports. Phase 5 (structure from motion)
handles it properly.

## Known limitations

- Histogram features are blind to cuts between visually identical shots
  (e.g. two angles of the same white wall). Phase 1's CNN features address this.
- Camera moves are measured on 96 px thumbnails; tracking noise sets a floor
  of ~3 % frame-width/s, so very slow moves read as static.
- There are no ground-truth camera-move labels yet; the classifier is
  validated on synthetic shots with known motion and by inspection.
