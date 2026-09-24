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
