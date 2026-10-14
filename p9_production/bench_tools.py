"""The "before" measurement: every phase's own tool, run one after the other on a clip.

    python p9_production/bench_tools.py trailers/goldeneye.mp4

This is how a clip was processed before Phase 9: each phase decodes the video
itself (Phase 3 twice, plus a Phase 0 pass for its shot cuts) and runs its
models in PyTorch the way its own script does. Wall-clock time per tool ->
p9_production/results/tools_<clip>.json. bench_pipeline.py measures the same
features through cinescope.pipeline.

Phase 1 has no trained checkpoint in this repo (it is trained on the owner's
videos), so its timing uses a ResNet-18 with random weights: the cost of a
forward pass does not depend on the weights.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--skip", nargs="*", default=[], help="tools to skip (p0 p1 p3 p5 p6 p7 p8)")
    ap.add_argument("--asd", default="lightasd", choices=["ours", "lightasd"])
    ap.add_argument("--out", help="result file (default p9_production/results/tools_<clip>.json)")
    args = ap.parse_args(argv)

    import torch

    torch.set_num_threads(args.threads)
    from cinescope.io.video import VideoReader
    from cinescope.shots.pipeline import analyse

    with VideoReader(args.video) as vr:
        info = vr.info
    clip = Path(args.video).stem
    scratch = ROOT / "outputs" / "bench_tools" / clip
    shutil.rmtree(scratch, ignore_errors=True)
    scratch.mkdir(parents=True)
    times: dict[str, float] = {}
    state: dict = {}

    def run(name, fn):
        if name in args.skip:
            return
        t0 = time.perf_counter()
        fn()
        times[name] = round(time.perf_counter() - t0, 2)
        print(f"{name}: {times[name]} s", flush=True)

    def shots():
        if "a" not in state:
            state["a"] = analyse(args.video, with_palettes=False)
        return state["a"].shots

    def p0():
        state["a"] = analyse(args.video, with_camera=True)

    def p1():
        from cinescope.framing import SCALES
        from cinescope.framing.classify import FramingClassifier
        from cinescope.framing.resnet import resnet18

        ck = scratch / "random_framing.pt"
        torch.save({"classes": list(SCALES), "model": resnet18(len(SCALES)).state_dict()}, ck)
        FramingClassifier(ck).label_shots(args.video, shots())

    def p3():
        from cinescope.faces.characters import character_timeline
        from cinescope.speaker import lightasd

        # Phase 3 as it shipped: Light-ASD's reference recipe, one chunk at a time
        # (Phase 9 added the batched path, which is now the default).
        reference = lightasd.LightASD.score
        lightasd.LightASD.score = lambda self, *a, **k: reference(self, *a, **{**k, "batched": False})
        character_timeline(args.video, scratch / "p3", embedder="sface", asd=args.asd, log=lambda *a: None)

    def p5():
        from cinescope.geometry.campath import path_from_rotations, rotation_only_path
        from cinescope.geometry.shot import default_k, load_shot, reconstruct

        longest = max(shots(), key=lambda s: s.end - s.start)
        sf = load_shot(args.video, longest.index, width=480, stride=2, max_frames=120,
                       start=longest.start, end=longest.end)
        rec, tracks = reconstruct(sf)
        if not rec.ok:
            path_from_rotations(rotation_only_path(tracks, default_k(sf.frames.shape[2], sf.frames.shape[1])))

    def p7():
        from cinescope.ocr.credits import CreditsReader
        from cinescope.ocr.libs import make_reader
        from cinescope.ocr.subtitles import SubtitleReader

        ck = ROOT / "p7_ocr" / "checkpoints" / "crnn.pt"
        rec = make_reader("ours" if ck.exists() else "rapidocr", str(ck) if ck.exists() else None)

        def frames(fps):
            with VideoReader(args.video) as vr:
                for f in vr.frames(sample_fps=fps):
                    yield f.t, f.rgb

        SubtitleReader(rec.read).run(frames(8.0), 1 / 8.0)
        CreditsReader(rec.read).run(frames(4.0))

    def p8():
        from cinescope.search.encoders import get_encoder
        from cinescope.search.rag import TAG_VOCAB
        from cinescope.search.shotindex import build_clip_index

        build_clip_index(args.video, get_encoder("openclip"), scratch / "index", tag_vocab=TAG_VOCAB, log=None)

    def p6():
        from cinescope.video.clips import clip_name
        from p6_pose_video.breakdown import main as breakdown

        (Path("outputs") / "shots" / f"{clip_name(Path(args.video))}.json").unlink(missing_ok=True)  # its own Phase 0 pass

        breakdown([args.video, "--pose", "yolo", "--scene", "clip", "--out", str(scratch / "p6"),
                   "--threads", str(args.threads)])

    for name, fn in (("p0", p0), ("p1", p1), ("p3", p3), ("p5", p5), ("p6", p6), ("p7", p7), ("p8", p8)):
        run(name, fn)

    total = round(sum(times.values()), 2)
    out = {"video": Path(args.video).name, "duration_s": round(info.duration_s, 2), "fps": info.fps,
           "size": [info.width, info.height], "threads": args.threads, "asd": args.asd,
           "seconds": times, "total_s": total,
           "video_s_per_wall_s": round(info.duration_s / total, 4) if total else None}
    res = ROOT / "p9_production" / "results"
    res.mkdir(parents=True, exist_ok=True)
    out["loadavg_after"] = [round(x, 2) for x in os.getloadavg()]
    out["when"] = time.strftime("%Y-%m-%d %H:%M")
    Path(args.out or res / f"tools_{clip}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    main()
