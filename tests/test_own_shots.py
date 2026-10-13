"""Phase 1 on your own videos: video discovery and splits, the automatic
labeller's cue fusion, the labelling page and the hand evaluation."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "p1_cnn"))

import build_own_shots as bos  # noqa: E402

from cinescope.framing import SCALES  # noqa: E402
from cinescope.framing.teacher import (  # noqa: E402
    FACE_CENTRES,
    class_scores,
    face_prior,
    face_ratio,
    fuse,
)


class TestEpisodes:
    def test_show_and_id(self):
        p = Path("videos/Some.Show.S02E07.1080p.WEB.mkv")
        assert bos.show_of(p) == "some_show"
        assert bos.episode_id(p) == "some_show_s02e07"
        assert bos.episode_id(Path("videos/My Series 1x03.mp4")) == "my_series_s01e03"

    def test_any_other_video_is_its_own_group(self):
        assert bos.show_of(Path("videos/holiday.mp4")) is None
        assert bos.episode_id(Path("videos/Tears of Steel (2012).mkv")) == "tears_of_steel_2012"
        assert bos.show_of(Path("videos/1920x1080 test.mp4")) is None      # a resolution, not 19x10

    def test_finds_every_video(self, tmp_path):
        for name in ("a.mp4", "sub/b.MKV", "notes.txt", "sub/c.webm"):
            (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
            (tmp_path / name).write_bytes(b"x" * 10)
        assert [p.name for p in bos.find_videos(tmp_path)] == ["a.mp4", "b.MKV", "c.webm"]
        assert bos.find_videos(tmp_path, min_mb=1) == []

    def test_searches_inside_the_project_but_not_its_copies(self, tmp_path, monkeypatch):
        for name in ("trailers/clip.mp4", "data/episodes/copy.mp4"):
            (tmp_path / name).parent.mkdir(parents=True)
            (tmp_path / name).write_bytes(b"x")
        monkeypatch.setattr(bos, "ROOT", tmp_path)
        assert [p.name for p in bos.find_videos(tmp_path)] == ["clip.mp4"]

    @staticmethod
    def _ep(name, show, n=10):
        return {"episode": name, "show": show,
                "shots": [{"shot": f"{i:04d}", "t0": float(i), "t1": float(i) + 0.9} for i in range(n)]}

    def test_episode_level_split(self):
        eps = [self._ep(f"some_show_s01e0{i}", "some_show") for i in range(1, 5)]
        split = bos.assign_splits(eps)
        per_ep = {e["episode"]: {split[(e["episode"], s["shot"])] for s in e["shots"]} for e in eps}
        assert per_ep == {"some_show_s01e01": {"train"}, "some_show_s01e02": {"train"},
                          "some_show_s01e03": {"val"}, "some_show_s01e04": {"test"}}

    def test_time_split_for_a_single_episode(self):
        e = self._ep("tears_of_steel", "tears_of_steel", n=100)
        split = bos.assign_splits([e])
        seq = [split[(e["episode"], s["shot"])] for s in e["shots"]]
        assert seq[0] == "train" and seq[-1] == "test"
        # contiguous blocks, in order: no test shot sits between two training shots
        assert seq == sorted(seq, key=["train", "val", "test"].index)


class TestTeacher:
    def test_class_scores_average_prompts(self):
        prompt_class = ["ECS", "ECS", "CS", "MS", "FS", "LS"]
        logits = np.array([[1.0, 3.0, 0.0, 0.0, 0.0, 5.0]])
        s = class_scores(logits, prompt_class)
        assert s.shape == (1, 5)
        assert s[0, 0] == pytest.approx(2.0) and s[0, 4] == pytest.approx(5.0)

    def test_face_ratio_uses_largest_confident_face(self):
        boxes = np.array([[0, 0, 10, 20], [0, 0, 50, 90], [0, 0, 80, 150]])
        scores = np.array([0.9, 0.8, 0.2])            # the biggest box is not confident
        assert face_ratio(boxes, scores, 180) == pytest.approx(0.5)
        assert face_ratio(boxes, np.zeros(3), 180) is None
        assert face_ratio(np.zeros((0, 4)), np.zeros(0), 180) is None

    def test_face_prior_peaks_at_each_centre(self):
        for i, c in enumerate(SCALES):
            p = face_prior(FACE_CENTRES[c])
            assert p.argmax() == i and p.sum() == pytest.approx(1.0)
        assert face_prior(None) is None and face_prior(0.0) is None

    def test_face_prior_orders_with_size(self):
        """Bigger faces never move the prediction towards wider shots."""
        preds = [face_prior(r).argmax() for r in np.geomspace(1.5, 0.01, 40)]
        assert preds == sorted(preds)

    def test_fuse(self):
        p_text = np.array([0.1, 0.4, 0.4, 0.05, 0.05])
        assert np.allclose(fuse(p_text, None), p_text)
        fused = fuse(p_text, face_prior(FACE_CENTRES["CS"]))
        assert SCALES[fused.argmax()] == "CS" and fused.sum() == pytest.approx(1.0)


def _write_shots(root: Path, n: int = 40) -> None:
    (root / "data" / "own_shots").mkdir(parents=True)
    with (root / "data" / "own_shots" / "shots.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "episode", "shot", "start", "end", "frames"])
        for i in range(n):
            w.writerow(["test" if i % 2 else "train", "ep", f"{i:04d}", 0, 10,
                        f"frames/ep/{i:04d}_0.jpg;frames/ep/{i:04d}_1.jpg"])


def test_label_page_has_only_test_shots_and_no_predictions(tmp_path, monkeypatch):
    import label_scales

    _write_shots(tmp_path)
    monkeypatch.setattr(label_scales, "SHOTS", tmp_path / "data" / "own_shots" / "shots.csv")
    label_scales.main(["--n", "5", "--out", str(tmp_path / "page")])
    html = (tmp_path / "page" / "index.html").read_text(encoding="utf-8")
    items = json.loads(html.split("const ITEMS = ", 1)[1].split(";\n", 1)[0])
    assert len(items) == 5
    assert all(int(it["key"].split("/")[1]) % 2 == 1 for it in items)      # odd = test
    assert all(f.startswith("../../data/own_shots/frames/") for it in items for f in it["frames"])
    assert "teacher" not in html.lower() and "fused" not in html.lower()


def test_hand_eval(tmp_path, monkeypatch):
    import eval_own

    res = tmp_path / "res"
    res.mkdir()
    rng = np.random.default_rng(0)
    y = rng.integers(0, 5, 60)
    labels = {f"ep/{i:04d}": SCALES[c] for i, c in enumerate(y)}
    labels["ep/9999"] = "skip"
    lab = tmp_path / "own_test.labels.json"
    lab.write_text(json.dumps({"labels": labels}))
    fields = ["split", "episode", "shot", "frames", "face_ratio", "label", "conf"] + \
        [f"{p}_{c}" for p in ("text", "face", "fused") for c in SCALES]
    with (res / "teacher.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i, c in enumerate(y):
            row = {"split": "test", "episode": "ep", "shot": f"{i:04d}", "frames": "", "face_ratio": "",
                   "label": SCALES[c], "conf": 1}
            onehot = np.eye(5)[c]                         # fused = perfect; text = always MS; face = none
            for j, s in enumerate(SCALES):
                row[f"fused_{s}"], row[f"text_{s}"], row[f"face_{s}"] = onehot[j], float(s == "MS"), ""
            w.writerow(row)
    monkeypatch.setattr(eval_own, "RES", res)
    monkeypatch.setattr(eval_own, "LABELS", lab)
    eval_own.main()
    out = json.loads((res / "hand_eval.json").read_text())
    assert out["n"] == 60                                   # the skipped shot is not scored
    assert out["systems"]["fused"]["accuracy"] == 1.0
    assert out["systems"]["text"]["accuracy"] == pytest.approx(np.mean(y == 2))
    assert "face" not in out["systems"]                     # no shot had a face
    lo, hi = out["systems"]["text"]["accuracy_ci95"]
    assert lo <= out["systems"]["text"]["accuracy"] <= hi
