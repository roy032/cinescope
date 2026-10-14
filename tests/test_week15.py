"""Week 15: two-view geometry, PnP, bundle adjustment, incremental SfM, camera paths."""
import unittest

import numpy as np

from cinescope.geometry import ba, camera, pnp, twoview
from cinescope.geometry.campath import pan_tilt_roll, rotation_only_path
from cinescope.geometry.metrics import trajectory_errors, translation_direction_error_deg
from cinescope.geometry.sfm import SfMParams, run_sfm
from cinescope.geometry.tracks import Tracks

try:
    import cv2
except ImportError:
    cv2 = None

RNG = np.random.default_rng(15)
K = camera.intrinsics(400.0, 160, 120)


def scene(n=150, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform([-3, -2, 5], [3, 2, 11], (n, 3))


def two_views(x, eye=(1.0, 0.1, 0.2)):
    r2, t2 = camera.look_at(np.asarray(eye), [0, 0, 8.0])
    xs = []
    for rot, t in ((np.eye(3), np.zeros(3)), (r2, t2)):
        xc = x @ rot.T + t
        xs.append(xc[:, :2] / xc[:, 2:3])
    return xs, r2, t2


def synthetic_tracks(points, rots, ts, k, noise=0.2, width=320, height=240, seed=0):
    """Tracks as an LK tracker would deliver them: each point observed in every
    frame where it projects inside the image, plus pixel noise."""
    rng = np.random.default_rng(seed)
    tr = Tracks(list(range(len(rots))))
    for f, (rot, t) in enumerate(zip(rots, ts, strict=True)):
        uv = camera.project(points, k, rot, t) + rng.normal(0, noise, (len(points), 2))
        z = camera.depths(points, rot, t)
        inside = (z > 0) & (uv[:, 0] > 0) & (uv[:, 0] < width) & (uv[:, 1] > 0) & (uv[:, 1] < height)
        for i in np.flatnonzero(inside):
            tr.obs.setdefault(int(i), {})[f] = uv[i]
    return tr


class TestTwoView(unittest.TestCase):
    def test_eight_point_satisfies_epipolar_constraint(self):
        (x1, x2), _, _ = two_views(scene())
        f = twoview.eight_point(x1, x2)
        h1, h2 = np.c_[x1, np.ones(len(x1))], np.c_[x2, np.ones(len(x2))]
        self.assertLess(np.abs((h2 @ f * h1).sum(1)).max(), 1e-10)
        self.assertLess(twoview.sampson_distance(f, x1, x2).max(), 1e-10)

    def test_pose_and_cheirality(self):
        x = scene()
        (x1, x2), r2, t2 = two_views(x)
        e = twoview.eight_point(x1, x2, essential=True)
        rot, t, front = twoview.recover_pose(e, x1, x2)
        self.assertTrue(front.all())
        self.assertLess(camera.rotation_angle_deg(rot, r2), 1e-6)
        self.assertLess(translation_direction_error_deg(t, t2), 1e-6)
        # exactly one of the four decompositions puts the points in front of both cameras
        p1 = np.c_[np.eye(3), np.zeros(3)]
        n_front = []
        for rr, tt in twoview.decompose_essential(e):
            xx = twoview.triangulate([p1, np.c_[rr, tt]], [x1, x2])
            n_front.append(int(((xx[:, 2] > 0) & (xx @ rr[2] + tt[2] > 0)).sum()))
        self.assertEqual(sorted(n_front)[-1], len(x))
        self.assertEqual(sum(v == len(x) for v in n_front), 1)

    def test_triangulation_is_exact(self):
        x = scene()
        (x1, x2), r2, t2 = two_views(x)
        xx = twoview.triangulate([np.c_[np.eye(3), np.zeros(3)], np.c_[r2, t2]], [x1, x2])
        np.testing.assert_allclose(xx, x, atol=1e-8)

    def test_ransac_rejects_outliers(self):
        x = scene(200)
        (x1, x2), r2, t2 = two_views(x)
        x2 = x2 + RNG.normal(0, 0.3 / 400, x2.shape)
        bad = RNG.random(len(x)) < 0.3
        x2[bad] = RNG.uniform(-0.4, 0.4, (bad.sum(), 2))
        res = twoview.ransac_essential(x1, x2, 400.0, 1.0)
        self.assertGreater((res.inliers & ~bad).sum(), 0.8 * (~bad).sum())   # linear E: a few good ones lost
        self.assertLess((res.inliers & bad).sum(), 0.05 * bad.sum() + 2)
        rot, t, _ = twoview.recover_pose(res.e, x1[res.inliers], x2[res.inliers])
        rot, t = twoview.refine_relative_pose(rot, t, x1[res.inliers], x2[res.inliers])
        self.assertLess(camera.rotation_angle_deg(rot, r2), 0.5)
        self.assertLess(translation_direction_error_deg(t, t2), 5.0)

    @unittest.skipIf(cv2 is None, "OpenCV not installed")
    def test_agrees_with_opencv(self):
        from cinescope.geometry.baselines import OpenCVSolvers

        x = scene()
        (x1, x2), r2, t2 = two_views(x)
        cv = OpenCVSolvers()
        e, inl = cv.essential(x1, x2, 400.0, 1.0)
        rot, t, _ = cv.recover_pose(e, x1[inl], x2[inl])
        ours = twoview.recover_pose(twoview.eight_point(x1, x2, essential=True), x1, x2)
        self.assertLess(camera.rotation_angle_deg(rot, ours[0]), 1e-3)
        self.assertLess(translation_direction_error_deg(t, ours[1]), 1e-2)


class TestPnP(unittest.TestCase):
    def test_dlt_exact_and_ransac(self):
        x = scene(80)
        rot, t = camera.look_at(np.array([0.7, -0.3, 0.5]), [0, 0, 8.0])
        xc = x @ rot.T + t
        xn = xc[:, :2] / xc[:, 2:3]
        r_est, t_est = pnp.pnp_dlt(x, xn)
        np.testing.assert_allclose(r_est, rot, atol=1e-8)
        np.testing.assert_allclose(t_est, t, atol=1e-8)
        xn_noisy = xn + RNG.normal(0, 0.5 / 400, xn.shape)
        xn_noisy[:20] = RNG.uniform(-0.5, 0.5, (20, 2))
        res = pnp.ransac_pnp(x, xn_noisy, 400.0, 2.0)
        self.assertFalse(res.inliers[:20].any())
        self.assertGreater(res.inliers[20:].mean(), 0.9)
        self.assertLess(camera.rotation_angle_deg(res.rot, rot), 0.3)


class TestBundleAdjustment(unittest.TestCase):
    def setUp(self):
        self.x = scene(60)
        cams = [camera.look_at(np.array([0.3 * i - 0.6, 0.05 * i, 0]), [0, 0, 8.0]) for i in range(5)]
        self.rots = np.stack([c[0] for c in cams])
        self.ts = np.stack([c[1] for c in cams])
        n = len(self.x)
        self.ci, self.pi = np.repeat(np.arange(5), n), np.tile(np.arange(n), 5)
        self.obs = ba.reprojection(self.rots, self.ts, self.x, 400.0, np.array([160, 120.0]), self.ci, self.pi)

    def test_jacobian_matches_finite_differences(self):
        pb = ba.Problem(self.rots, self.ts, self.x, 410.0, (160, 120), self.ci, self.pi, self.obs, refine_focal=True)
        x0 = pb.x0 + RNG.normal(0, 0.01, pb.x0.shape)
        jac = pb.sparse_jacobian(x0).toarray()
        num = np.zeros_like(jac)
        for i in range(len(x0)):
            d = np.zeros_like(x0)
            d[i] = 1e-6
            num[:, i] = (pb.residuals(x0 + d).ravel() - pb.residuals(x0 - d).ravel()) / 2e-6
        np.testing.assert_allclose(jac, num, atol=1e-5)

    def test_converges_to_the_noise_floor(self):
        obs = self.obs + RNG.normal(0, 0.5, self.obs.shape)
        x_pert = self.x + RNG.normal(0, 0.05, self.x.shape)
        ts = self.ts + RNG.normal(0, 0.02, self.ts.shape)
        ts[0] = self.ts[0]
        res = ba.bundle_adjust(self.rots, ts, x_pert, 400.0, (160, 120), self.ci, self.pi, obs)
        ref = ba.bundle_adjust(self.rots, self.ts, self.x, 400.0, (160, 120), self.ci, self.pi, obs)
        self.assertGreater(res.initial_rmse, 3.0)
        self.assertAlmostEqual(res.final_rmse, ref.final_rmse, delta=0.01)


class TestSfM(unittest.TestCase):
    def _shot(self, translate=True, n_frames=16):
        x = np.random.default_rng(1).uniform([-4, -2, 4], [4, 2, 12], (400, 3))
        rots, ts = [], []
        for i in range(n_frames):
            a = i / (n_frames - 1)
            eye = np.array([-1.0 + 2.0 * a, -0.1, 0.0]) if translate else np.zeros(3)
            target = np.array([0.3 * a, 0.0, 8.0]) if translate else np.array([-1.5 + 3 * a, 0, 8.0])
            r, t = camera.look_at(eye, target)
            rots.append(r)
            ts.append(t)
        return x, np.stack(rots), np.stack(ts)

    def test_recovers_a_dolly(self):
        x, rots, ts = self._shot()
        tracks = synthetic_tracks(x, rots, ts, K)
        rec = run_sfm(tracks, K, 320, SfMParams())
        self.assertTrue(rec.ok, rec.log)
        self.assertLess(rec.mean_reproj_px, 0.5)
        fr = sorted(rec.poses)
        self.assertEqual(len(fr), len(rots))                         # every frame gets a pose
        gt_c = np.stack([camera.camera_centre(rots[f], ts[f]) for f in fr])
        err = trajectory_errors(rec.centres(fr), gt_c, np.stack([rec.poses[f][0] for f in fr]), rots[fr])
        self.assertLess(err["rot_err_deg_mean"], 0.2)
        self.assertLess(err["ate_rmse_rel_path"], 0.01)

    def test_refines_a_wrong_focal_length(self):
        x, rots, ts = self._shot()
        tracks = synthetic_tracks(x, rots, ts, K, noise=0.1)
        wrong = camera.intrinsics(480.0, 160, 120)
        rec = run_sfm(tracks, wrong, 320, SfMParams(refine_focal=True))
        self.assertTrue(rec.ok, rec.log)
        self.assertLess(abs(rec.k[0, 0] - 400) / 400, 0.05)

    def test_refuses_a_pure_pan(self):
        x, rots, ts = self._shot(translate=False)
        rec = run_sfm(synthetic_tracks(x, rots, ts, K), K, 320, SfMParams())
        self.assertFalse(rec.ok)

    def test_rotation_only_path_reads_the_pan(self):
        x, rots, ts = self._shot(translate=False)
        path = rotation_only_path(synthetic_tracks(x, rots, ts, K, noise=0.05), K)
        last = max(path)
        pan, tilt, _ = pan_tilt_roll(path[last]["R"])
        gt_pan, gt_tilt, _ = pan_tilt_roll(rots[last] @ rots[0].T)
        self.assertAlmostEqual(pan, gt_pan, delta=0.3)
        self.assertAlmostEqual(tilt, gt_tilt, delta=0.3)
        self.assertAlmostEqual(path[last]["zoom"], 1.0, delta=0.01)

    def test_alignment_is_similarity_invariant(self):
        _, rots, ts = self._shot()
        c = np.stack([camera.camera_centre(r, t) for r, t in zip(rots, ts, strict=True)])
        a = camera.rodrigues(np.array([0.3, -0.2, 0.5]))
        est_c = 2.5 * c @ a.T + np.array([1.0, 2.0, 3.0])
        est_r = np.stack([r @ a.T for r in rots])
        err = trajectory_errors(est_c, c, est_r, rots)
        self.assertLess(err["ate_rmse_rel_path"], 1e-9)
        self.assertLess(err["rot_err_deg_mean"], 1e-5)


if __name__ == "__main__":
    unittest.main()
