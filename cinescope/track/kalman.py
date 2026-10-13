"""A constant-velocity Kalman filter for bounding boxes, from scratch.

Detections jitter by a few pixels and drop out for a frame or two (a hand in
front of the face, motion blur, a profile turn). A Kalman filter keeps, for each
track, a Gaussian belief over the box *and its velocity*:

  predict:  x <- F x            P <- F P F^T + Q      (the box moves on; uncertainty grows)
  update:   K = P H^T (H P H^T + R)^-1
            x <- x + K (z - H x)  P <- (I - K H) P     (blend in the detection z)

The gain K weighs the prediction against the measurement by their
covariances: a track that has been predicted for several frames without a
detection is uncertain, so the next detection pulls it almost all the way.

State, as in DeepSORT/ByteTrack: (cx, cy, a, h, vcx, vcy, va, vh), with a =
width/height. Position and size noise scale with the box height, so a 300 px
face and a 30 px face get the same *relative* trust. (SORT uses area and
aspect instead; height-scaled noise behaves better when faces change size
quickly, as they do when a person walks toward the camera.)
"""
from __future__ import annotations

import numpy as np


def xyxy_to_xyah(b: np.ndarray) -> np.ndarray:
    b = np.asarray(b, dtype=np.float64)
    w, h = b[..., 2] - b[..., 0], b[..., 3] - b[..., 1]
    return np.stack([b[..., 0] + w / 2, b[..., 1] + h / 2, w / np.maximum(h, 1e-9), h], -1)


def xyah_to_xyxy(s: np.ndarray) -> np.ndarray:
    s = np.asarray(s, dtype=np.float64)
    h = s[..., 3]
    w = s[..., 2] * h
    return np.stack([s[..., 0] - w / 2, s[..., 1] - h / 2, s[..., 0] + w / 2, s[..., 1] + h / 2], -1)


class BoxKalman:
    """One track's filter. `std_pos`/`std_vel` are fractions of the box height
    (the ByteTrack defaults, 1/20 and 1/160)."""

    ndim = 4

    def __init__(self, box_xyxy: np.ndarray, std_pos: float = 1 / 20, std_vel: float = 1 / 160):
        self.std_pos, self.std_vel = std_pos, std_vel
        z = xyxy_to_xyah(box_xyxy)
        self.x = np.r_[z, np.zeros(4)]
        h = z[3]
        std = np.array([2 * std_pos * h, 2 * std_pos * h, 1e-2, 2 * std_pos * h,
                        10 * std_vel * h, 10 * std_vel * h, 1e-5, 10 * std_vel * h])
        self.P = np.diag(std ** 2)
        self.F = np.eye(8)
        self.F[:4, 4:] = np.eye(4)          # x_{t+1} = x_t + v_t  (dt = one frame)
        self.H = np.eye(4, 8)

    def predict(self) -> np.ndarray:
        h = self.x[3]
        q = np.array([self.std_pos * h, self.std_pos * h, 1e-2, self.std_pos * h,
                      self.std_vel * h, self.std_vel * h, 1e-5, self.std_vel * h])
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + np.diag(q ** 2)
        if self.x[3] <= 1:                  # a shrinking box must not invert
            self.x[3], self.x[7] = 1.0, 0.0
        return self.box()

    def project(self) -> tuple[np.ndarray, np.ndarray]:
        """Predicted measurement mean and covariance S = H P H^T + R."""
        h = self.x[3]
        r = np.array([self.std_pos * h, self.std_pos * h, 1e-1, self.std_pos * h]) ** 2
        return self.H @ self.x, self.H @ self.P @ self.H.T + np.diag(r)

    def update(self, box_xyxy: np.ndarray) -> np.ndarray:
        z = xyxy_to_xyah(box_xyxy)
        mean, S = self.project()
        # K = P H^T S^-1, solved rather than inverted (S is SPD).
        K = np.linalg.solve(S, self.H @ self.P).T
        self.x = self.x + K @ (z - mean)
        self.P = self.P - K @ S @ K.T
        return self.box()

    def mahalanobis(self, boxes_xyxy: np.ndarray) -> np.ndarray:
        """Squared Mahalanobis distance of measurements to the prediction
        (chi-square with 4 dof; 9.49 is the 95% gate DeepSORT uses)."""
        mean, S = self.project()
        d = xyxy_to_xyah(np.asarray(boxes_xyxy).reshape(-1, 4)) - mean
        return np.einsum("ni,ni->n", d, np.linalg.solve(S, d.T).T)

    def box(self) -> np.ndarray:
        return xyah_to_xyxy(self.x[:4])
