"""Human pose (Phase 6, weeks 18-19): keypoint metrics, a top-down heatmap
network, pose tracks over time and action events.

Keypoints follow COCO's 17-point layout throughout, because that is what the
library model (YOLOv8-pose) and every public benchmark use.
"""
from __future__ import annotations

import numpy as np

KEYPOINTS: tuple[str, ...] = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle",
)
K = len(KEYPOINTS)
KP = {n: i for i, n in enumerate(KEYPOINTS)}

# Per-keypoint OKS falloff from the COCO keypoint task: the standard deviation
# of human annotators' clicks, relative to object scale. Eyes are easy to click
# consistently (0.025); hips are ambiguous under clothing (0.107).
COCO_SIGMAS = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62,
                        1.07, 1.07, .87, .87, .89, .89]) / 10.0

# Left/right pairs: swapped together when an image is mirrored.
FLIP_PAIRS = ((1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16))

SKELETON = ((15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), (5, 6),
            (5, 7), (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6))
