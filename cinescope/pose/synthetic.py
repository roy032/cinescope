"""Synthetic 2-D pose sequences with known action events.

A stick figure of height H (pixels) is driven by joint angles: torso lean,
and for each arm and leg the angle of the upper and the lower segment from
"straight down" (positive = towards the direction the person faces). Segment
lengths are fixed fractions of H, roughly anthropometric. Activities set the
angles over time:

  stand   small sway                      walk   legs +-25 deg at 0.9 Hz, 0.8 H/s
  run     legs +-45 deg at 1.4 Hz, knees and elbows bent, 2.5 H/s
  talk    hand gestures (0.6-1 H/s wrist speed) facing a partner: the hard negative for "strike"
  punch   one arm from guard to full extension in ~0.12 s and back
  kick    thigh and shin swing up to ~80 deg in ~0.2 s and back
  fall    torso tips from upright to lying over ~0.6 s, hips drop to the floor

Scenes put one or two people in a frame, add a camera pan (shifting every
point, which a camera-unaware "running" detector would mistake for motion),
keypoint noise and dropped keypoints, and return the ground-truth events with
their time spans. They are the tuning set for the event detector and a
check on it with exactly known answers.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import KP

SEG = {"torso": 0.30, "head": 0.13, "uarm": 0.17, "farm": 0.15, "thigh": 0.245, "shin": 0.245,
       "shoulder": 0.06, "hip": 0.045}


@dataclass
class Pose:
    """Joint angles in degrees (0 = pointing down; torso 0 = upright)."""
    torso: float = 0.0
    uarm: tuple[float, float] = (5.0, -5.0)      # left, right
    farm: tuple[float, float] = (5.0, -5.0)
    thigh: tuple[float, float] = (0.0, 0.0)
    shin: tuple[float, float] = (0.0, 0.0)


def skeleton(hip: np.ndarray, H: float, facing: int, p: Pose) -> np.ndarray:
    """(17, 2) COCO keypoints for a figure whose hip midpoint is at `hip`."""
    def vec(angle_deg, length):            # angle from straight down, forward = +facing x
        a = np.radians(angle_deg)
        return np.array([facing * np.sin(a), np.cos(a)]) * length * H

    k = np.zeros((17, 2))
    th = np.radians(p.torso)
    up = np.array([facing * np.sin(th), -np.cos(th)])
    side = np.array([-up[1], up[0]]) * facing                # perpendicular, 3/4 view offset
    sh_mid = hip + up * SEG["torso"] * H
    for s, sign in (("left", 1), ("right", -1)):
        i = 0 if s == "left" else 1
        sh = sh_mid + sign * side * SEG["shoulder"] * H
        hp = hip + sign * side * SEG["hip"] * H
        # limbs hang relative to the torso: rotate limb angles by the lean
        el = sh + vec(p.uarm[i] + p.torso, SEG["uarm"])
        wr = el + vec(p.farm[i] + p.torso, SEG["farm"])
        kn = hp + vec(p.thigh[i] + p.torso, SEG["thigh"])
        an = kn + vec(p.shin[i] + p.torso, SEG["shin"])
        k[KP[f"{s}_shoulder"]], k[KP[f"{s}_elbow"]], k[KP[f"{s}_wrist"]] = sh, el, wr
        k[KP[f"{s}_hip"]], k[KP[f"{s}_knee"]], k[KP[f"{s}_ankle"]] = hp, kn, an
    head = sh_mid + up * SEG["head"] * H
    fwd = np.array([facing * np.cos(th), np.sin(th) * facing * 0])
    k[KP["nose"]] = head + fwd * 0.03 * H
    k[KP["left_eye"]] = head + fwd * 0.02 * H + up * 0.015 * H + side * 0.015 * H
    k[KP["right_eye"]] = head + fwd * 0.02 * H + up * 0.015 * H - side * 0.015 * H
    k[KP["left_ear"]] = head - fwd * 0.02 * H + side * 0.03 * H
    k[KP["right_ear"]] = head - fwd * 0.02 * H - side * 0.03 * H
    return k


def leg_length() -> float:
    return SEG["thigh"] + SEG["shin"]


@dataclass
class Actor:
    x: float
    H: float
    facing: int
    ground: float                     # y of the floor (ankles when standing)
    script: list[tuple[float, float, str]] = field(default_factory=list)   # (t0, t1, activity)
    seed: int = 0

    def activity(self, t: float) -> tuple[str, float]:
        for t0, t1, a in self.script:
            if t0 <= t < t1:
                return a, t - t0
        return "stand", t


def _pose_at(actor: Actor, act: str, u: float, rng_phase: float, fallen: bool) -> tuple[Pose, float, float]:
    """Pose, forward speed (H/s) and hip height above the floor (fraction of H)."""
    stand_h = leg_length()
    if fallen:
        return Pose(torso=88, uarm=(-80, -70), farm=(-85, -75), thigh=(-88, -85), shin=(-90, -88)), 0.0, 0.05
    if act == "walk" or act == "run":
        A, f, knee, arm, v, lean, elb = ((25, 0.9, 30, 15, 0.8, 0, 10) if act == "walk"
                                         else (45, 1.4, 75, 40, 2.5, 10, 80))
        ph = 2 * np.pi * f * u + rng_phase
        s = np.sin(ph)
        thigh = (A * s, -A * s)
        shin = (A * s - knee * max(0.0, -np.cos(ph)), -A * s - knee * max(0.0, np.cos(ph)))
        uarm = (-arm * s, arm * s)
        farm = (uarm[0] + elb, uarm[1] + elb)
        hip_h = stand_h * (np.cos(np.radians(A)) if act == "run" else 0.98)
        return Pose(lean, uarm, farm, thigh, shin), v, hip_h
    if act == "talk":
        g = np.sin(2 * np.pi * 0.7 * u + rng_phase)
        g2 = np.sin(2 * np.pi * 0.45 * u + 2 * rng_phase)
        return Pose(0, (35 + 15 * g, 5 + 8 * g2), (110 + 30 * g, 15 + 15 * g2)), 0.0, stand_h
    if act in ("punch", "kick"):
        # guard -> extension in 0.12 s (kick 0.2 s), hold 0.05 s, back over 0.2 s
        ext_t = 0.12 if act == "punch" else 0.2
        if u < ext_t:
            e = u / ext_t
        elif u < ext_t + 0.05:
            e = 1.0
        else:
            e = max(0.0, 1 - (u - ext_t - 0.05) / 0.2)
        e = 0.5 - 0.5 * np.cos(np.pi * e)                # smooth start and stop
        guard = Pose(5, (20, 15), (165, 160))          # fists at the chin
        if act == "punch":
            return Pose(5 + 5 * e, (20 + 70 * e, 15), (165 - 75 * e, 160)), 0.0, stand_h
        return Pose(-10 * e, guard.uarm, guard.farm, (80 * e, 0), (80 * e - 30 * (1 - e) * e, 0)), 0.0, stand_h
    if act == "fall":
        e = min(1.0, u / 0.6)
        e = e * e                                         # accelerates, like gravity
        return (Pose(88 * e, (-60 * e, -50 * e), (-70 * e, -60 * e), (-80 * e, -70 * e), (-85 * e, -80 * e)),
                0.0, stand_h * (1 - e) + 0.05 * e)
    if act == "fight_stance":
        b = np.sin(2 * np.pi * 1.2 * u + rng_phase)
        return Pose(5, (20 + 5 * b, 15), (165, 160), (5 * b, -5 * b)), 0.0, stand_h * 0.97
    sway = np.sin(2 * np.pi * 0.3 * u + rng_phase)
    return Pose(2 * sway), 0.0, stand_h


@dataclass
class Scene:
    fps: float
    times: np.ndarray                 # (T,)
    kpts: list[np.ndarray]            # per frame (P, 17, 3)
    boxes: list[np.ndarray]           # per frame (P, 4)
    events: list[dict]                # {"type", "t0", "t1", "actor"}
    actors: int


def render_scene(actors: list[Actor], seconds: float, fps: float = 8.0, pan: float = 0.0,
                 noise: float = 0.01, drop: float = 0.05, rng: np.random.Generator | None = None) -> Scene:
    """Simulate actors; `pan` = camera pan in pixels/second; noise in units of H."""
    rng = rng or np.random.default_rng(0)
    times = np.arange(0, seconds, 1 / fps)
    kpts_all, boxes_all = [], []
    xs = [a.x for a in actors]
    fallen_at = [None] * len(actors)
    phases = [rng.uniform(0, 2 * np.pi) for _ in actors]
    for i, t in enumerate(times):
        dt = 0 if i == 0 else times[i] - times[i - 1]
        frame_k, frame_b = [], []
        for j, a in enumerate(actors):
            act, u = a.activity(t)
            if act == "fall" and fallen_at[j] is None:
                fallen_at[j] = t - u
            fallen = fallen_at[j] is not None and t - fallen_at[j] >= 0.6
            pose, v, hip_h = _pose_at(a, act, u, phases[j], fallen)
            xs[j] += a.facing * v * a.H * dt
            hip = np.array([xs[j] - pan * t, a.ground - hip_h * a.H])
            k = skeleton(hip, a.H, a.facing, pose)
            k = k + rng.normal(0, noise * a.H, k.shape)
            conf = np.where(rng.random(17) < drop, rng.uniform(0, 0.3, 17), rng.uniform(0.6, 1.0, 17))
            kk = np.concatenate([k, conf[:, None]], 1)
            frame_k.append(kk)
            frame_b.append(np.r_[k.min(0) - 0.03 * a.H, k.max(0) + 0.03 * a.H])
        kpts_all.append(np.stack(frame_k))
        boxes_all.append(np.stack(frame_b))
    events = []
    for j, a in enumerate(actors):
        for t0, t1, act in a.script:
            if act in ("punch", "kick"):
                events.append({"type": "strike", "t0": t0, "t1": t0 + 0.3, "actor": j})
            elif act == "fall":
                events.append({"type": "fall", "t0": t0, "t1": t0 + 0.8, "actor": j})
            elif act == "run":
                events.append({"type": "run", "t0": t0, "t1": t1, "actor": j})
    return Scene(fps, times, kpts_all, boxes_all, events, len(actors))


def random_scene(rng: np.random.Generator, kind: str | None = None, fps: float = 8.0) -> Scene:
    """One of: fight, talk, chase, fall, walk, idle — randomised sizes, places, timings, pans."""
    kind = kind or str(rng.choice(["fight", "talk", "chase", "fall", "walk", "idle"]))
    H = rng.uniform(150, 400)
    ground = rng.uniform(1.0, 1.3) * H + 50
    seconds = float(rng.uniform(4, 7))
    pan = float(rng.choice([0.0, 0.0, rng.uniform(-1.2, 1.2) * H]))
    x0 = rng.uniform(200, 600)
    gap = rng.uniform(0.35, 0.55) * H
    if kind == "fight":
        a = Actor(x0, H, 1, ground)
        b = Actor(x0 + gap, H * rng.uniform(0.9, 1.1), -1, ground)
        t = rng.uniform(0.5, 1.0)
        for actor in (a, b):
            actor.script.append((0, seconds, "fight_stance"))
        while t < seconds - 0.6:
            who = a if rng.random() < 0.5 else b
            who.script.insert(0, (t, t + 0.4, str(rng.choice(["punch", "punch", "kick"]))))
            t += rng.uniform(0.8, 1.6)
        if rng.random() < 0.4:
            loser = b if rng.random() < 0.5 else a
            tf = seconds - 1.5
            loser.script = [s for s in loser.script if s[0] < tf - 0.5] + [(tf, seconds, "fall")]
            loser.script.sort(key=lambda s: (s[2] != "fall", s[0]))
        actors = [a, b]
    elif kind == "talk":
        actors = [Actor(x0, H, 1, ground, [(0, seconds, "talk")]),
                  Actor(x0 + gap * 1.2, H, -1, ground, [(0, seconds, "talk")])]
    elif kind == "chase":
        t0 = rng.uniform(0.5, 1.5)
        actors = [Actor(x0, H, 1, ground, [(t0, seconds, "run")]),
                  Actor(x0 - 1.5 * H, H, 1, ground, [(t0 + 0.3, seconds, "run")])]
    elif kind == "fall":
        tf = rng.uniform(1.0, seconds - 1.5)
        actors = [Actor(x0, H, int(rng.choice([-1, 1])), ground, [(0, tf, "walk"), (tf, seconds, "fall")])]
    elif kind == "walk":
        actors = [Actor(x0, H, 1, ground, [(0, seconds, "walk")]),
                  Actor(x0 + 2 * H, H * 0.8, -1, ground, [(0, seconds, "walk")])]
    else:
        actors = [Actor(x0, H, 1, ground, [(0, seconds, "stand")])]
    return render_scene(actors, seconds, fps, pan=pan, rng=rng)
