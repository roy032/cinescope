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
