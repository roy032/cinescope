"""Library baselines for the SfM comparison.

* OpenCV: the same incremental pipeline (`sfm.run_sfm`) with its geometric
  solvers swapped for OpenCV's — findEssentialMat (five-point + RANSAC),
  recoverPose, triangulatePoints, solvePnPRansac — on the same tracks. OpenCV
  has no bundle adjuster in its main modules, so it runs without BA; ours is
  reported both without and with BA, which isolates what each part buys.
* COLMAP (pycolmap): the whole thing done the production way, from the frames
  themselves — SIFT features, sequential matching, geometric verification,
  incremental mapping with Ceres bundle adjustment. Its mean reprojection
  error is the number papers quote.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from .twoview import triangulate as dlt_triangulate


class OpenCVSolvers:
    """Drop-in replacements for ransac_essential / recover_pose / ransac_pnp / triangulate."""

    def __init__(self):
        import cv2

        self.cv2 = cv2

    def essential(self, xn1, xn2, focal, thresh_px, seed=0):
        cv2 = self.cv2
        e, mask = cv2.findEssentialMat(xn1, xn2, np.eye(3), method=cv2.RANSAC, prob=0.999,
                                       threshold=thresh_px / focal)
        if e is None or e.shape != (3, 3):
            return None, np.zeros(len(xn1), bool)
        return e, mask.ravel().astype(bool)

    def recover_pose(self, e, xn1, xn2):
        cv2 = self.cv2
        _, rot, t, mask, _ = cv2.recoverPose(e, xn1, xn2, np.eye(3), distanceThresh=100.0)
        return rot, t.ravel(), mask.ravel() > 0

    def pnp(self, x3, xn, focal, thresh_px, seed=0):
        cv2 = self.cv2
        ok, rvec, tvec, inl = cv2.solvePnPRansac(x3.astype(np.float64), xn.astype(np.float64), np.eye(3), None,
                                                 reprojectionError=thresh_px / focal, iterationsCount=1000,
                                                 confidence=0.999, flags=cv2.SOLVEPNP_EPNP)
        mask = np.zeros(len(x3), bool)
        if not ok or inl is None:
            return None, None, mask
        mask[inl.ravel()] = True
        rvec, tvec = cv2.solvePnPRefineLM(x3[mask], xn[mask], np.eye(3), None, rvec, tvec)
        return cv2.Rodrigues(rvec)[0], tvec.ravel(), mask

    def triangulate(self, pms, xs):
        if len(pms) == 2:
            xh = self.cv2.triangulatePoints(pms[0], pms[1], xs[0].T.astype(np.float64), xs[1].T.astype(np.float64))
            return (xh[:3] / xh[3]).T
        return dlt_triangulate(pms, xs)       # OpenCV's triangulatePoints is two-view only


def run_pycolmap(frames: np.ndarray, ids: list[int], work_dir: str | Path, focal: float | None = None,
                 overlap: int = 10) -> dict:
    """COLMAP incremental SfM on a shot's frames (single shared SIMPLE_PINHOLE camera)."""
    import pycolmap
    from PIL import Image

    work = Path(work_dir)
    shutil.rmtree(work, ignore_errors=True)
    (work / "images").mkdir(parents=True)
    for i, f in zip(ids, frames, strict=True):
        Image.fromarray(f).save(work / "images" / f"{i:06d}.png")
    db = work / "database.db"
    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = "SIMPLE_PINHOLE"
    h, w = frames.shape[1:3]
    if focal is not None:
        reader.camera_params = f"{focal},{w / 2},{h / 2}"
    pycolmap.extract_features(db, work / "images", camera_mode=pycolmap.CameraMode.SINGLE,
                              reader_options=reader, device=pycolmap.Device.cpu)
    pairing = pycolmap.SequentialPairingOptions()
    pairing.overlap = overlap
    pycolmap.match_sequential(db, pairing_options=pairing, device=pycolmap.Device.cpu)
    opts = pycolmap.IncrementalPipelineOptions()
    if focal is not None:
        opts.ba_refine_focal_length = False
    maps = pycolmap.incremental_mapping(db, work / "images", work / "sparse", options=opts)
    if not maps:
        return {"ok": False, "registered": 0}
    best = max(maps.values(), key=lambda r: r.num_reg_images())
    cam = next(iter(best.cameras.values()))
    poses = {}
    for img in best.images.values():
        if not img.has_pose:
            continue
        pose = img.cam_from_world()
        poses[int(Path(img.name).stem)] = (np.asarray(pose.rotation.matrix()), np.asarray(pose.translation))
    return {"ok": True, "registered": int(best.num_reg_images()), "images": len(ids),
            "points": int(best.num_points3D()),
            "mean_reproj_px": float(best.compute_mean_reprojection_error()),
            "mean_track_length": float(best.compute_mean_track_length()),
            "focal": float(cam.params[0]), "poses": poses}
