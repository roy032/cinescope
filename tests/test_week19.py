"""Week 19: pose tracks, motion features, action events on synthetic sequences."""
from __future__ import annotations

import numpy as np
import pytest

from cinescope.pose import KP
from cinescope.pose.events import EventParams, detect_events, features, match_events
from cinescope.pose.synthetic import Actor, Pose, random_scene, render_scene, skeleton
from cinescope.pose.track import track_people


def test_skeleton_geometry():
    k = skeleton(np.array([100.0, 200.0]), 300, 1, Pose())
    # standing: ankles below the hips by the leg length, head above the shoulders
    assert k[KP["left_ankle"], 1] - 200 == pytest.approx(0.49 * 300, rel=0.05)
    assert k[KP["nose"], 1] < k[KP["left_shoulder"], 1] < k[KP["left_hip"], 1]
    # lying: torso horizontal
    k2 = skeleton(np.array([100.0, 200.0]), 300, 1, Pose(torso=90))
    assert abs(k2[KP["left_shoulder"], 1] - 200) < 0.1 * 300


def test_tracker_keeps_identities_and_follows_runners():
    sc = render_scene([Actor(100, 200, 1, 500, [(0, 4, "run")]), Actor(900, 200, -1, 500, [(0, 4, "walk")])],
                      4.0, fps=8, rng=np.random.default_rng(0))
    tracks = track_people(sc.times, sc.boxes, sc.kpts)
    assert len(tracks) == 2
    assert min(len(t) for t in tracks) >= len(sc.times) - 2     # a runner may be lost for its first step
    # each track stays on one actor: x moves monotonically in its own direction
    for t in tracks:
        _, b, _ = t.arrays()
        cx = (b[:, 0] + b[:, 2]) / 2
        assert abs(np.sign(np.diff(cx)).sum()) >= len(cx) - 4


def test_limb_speeds_ignore_camera_pan():
    actors = lambda: [Actor(300, 250, 1, 600, [(0, 3, "walk")])]  # noqa: E731
    a = render_scene(actors(), 3.0, rng=np.random.default_rng(1))
    b = render_scene(actors(), 3.0, pan=400.0, rng=np.random.default_rng(1))
    p = EventParams()
    fa = features(track_people(a.times, a.boxes, a.kpts)[0], p)
    fb = features(track_people(b.times, b.boxes, b.kpts)[0], p)
    np.testing.assert_allclose(fa.limb_speed, fb.limb_speed, atol=1e-9, equal_nan=True)
    # ...while the body itself appears to move very differently
    assert abs(np.diff(fa.hip[:, 0]).mean() - np.diff(fb.hip[:, 0]).mean()) > 40


@pytest.mark.parametrize("kind,typ", [("fight", "strike"), ("fall", "fall"), ("chase", "run")])
def test_events_found_on_synthetic(kind, typ):
    found = total = 0
    for s in range(4):
        sc = random_scene(np.random.default_rng(100 + s), kind)
        pred = [e.to_dict() for e in detect_events(track_people(sc.times, sc.boxes, sc.kpts))]
        m = match_events(pred, sc.events).get(typ, {"tp": 0, "fn": 0})
        found += m["tp"]
        total += m["tp"] + m["fn"]
    assert found / total >= 0.6


@pytest.mark.parametrize("kind", ["talk", "walk", "idle"])
def test_no_events_in_calm_scenes(kind):
    n = 0
    for s in range(4):
        sc = random_scene(np.random.default_rng(200 + s), kind)
        n += len(detect_events(track_people(sc.times, sc.boxes, sc.kpts)))
    assert n <= 1


def test_match_events_counts():
    gt = [{"type": "strike", "t0": 1.0, "t1": 1.3}, {"type": "strike", "t0": 3.0, "t1": 3.3},
          {"type": "run", "t0": 5.0, "t1": 8.0}]
    pred = [{"type": "strike", "t": 1.2, "t0": 1.1, "t1": 1.2, "score": 2},
            {"type": "strike", "t": 1.25, "t0": 1.2, "t1": 1.25, "score": 1},    # duplicate -> FP
            {"type": "strike", "t": 6.0, "t0": 5.9, "t1": 6.0, "score": 1},      # nothing there -> FP
            {"type": "run", "t": 7.0, "t0": 6.5, "t1": 9.0, "score": 1}]
    r = match_events(pred, gt)
    assert (r["strike"]["tp"], r["strike"]["fp"], r["strike"]["fn"]) == (1, 2, 1)
    assert r["run"]["tp"] == 1 and r["run"]["f1"] == 1.0
