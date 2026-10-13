"""Multi-view geometry (Phase 5): calibration, two-view geometry, structure from
motion, bundle adjustment, plane-sweep stereo, depth-image-based rendering and
3D Gaussian splatting.

Conventions used everywhere in this package:

  * a camera is (K, R, t) and maps a world point X to x_cam = R X + t, then to
    pixels with K (x right, y down, z forward — OpenCV's convention);
  * rotations are passed around as 3x3 matrices, and as axis-angle vectors
    (Rodrigues) only inside optimisers;
  * image points are (N, 2) float arrays of (x, y) pixels.
"""
