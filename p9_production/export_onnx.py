"""Export the pipeline's PyTorch models to ONNX; check the numbers; time both engines.

    python p9_production/export_onnx.py                 # all models whose weights are present
    python p9_production/export_onnx.py --models resnet18 clip_visual --threads 2
    python p9_production/export_onnx.py --backend tensorrt --device cuda   # NVIDIA GPU: TensorRT vs. PyTorch on CUDA

For each model: export with dynamic batch (and sequence / width) axes, run the
same random input through PyTorch and ONNX Runtime -- with a batch size (and
length) *different* from the one used for tracing, so a dynamic axis that
was silently frozen would show up -- and report the largest absolute
difference and the lowest cosine similarity per output row. Then time both
engines (median of `--reps` runs after warm-up) at batch 1 and a typical batch.

-> p9_production/results/onnx.json (+ a markdown table on stdout)

Models (weights from data/weights or the phases' checkpoints):
  resnet18     Phase 1 framing classifier, 224x384 frames. No trained checkpoint ships with
               the repo, so the weights are timm's ImageNet ResNet-18 if present, else random:
               the speed and the numerical agreement do not depend on what the weights learnt.
  embednet     Phase 3 face embedder (p3_tracking/checkpoints/embed_triplet.pt), 64x64 faces
  clip_visual  Phase 8 open_clip ViT-B/32 image tower, 224x224 crops
  clip_text    Phase 8 text tower, 77 tokens
  lightasd     Phase 3 Light-ASD (who is talking), 1 s chunks of 25 crops + 100 MFCC steps
  crnn         Phase 7 CRNN line reader (p7_ocr/checkpoints/crnn.pt), 32 x 400 lines
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cinescope.pipeline.accel import OnnxModule, _asd_batched, _clip_parts, compare  # noqa: E402


def _median_ms(fn, reps: int, warmup: int = 2) -> float:
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts) * 1000)


def models(names: list[str]):
    """name -> (module, trace example, check input, input names, output names, dynamic axes, batch for timing)."""
    import torch

    from cinescope.faces.detectors import weight_path

    g = torch.Generator().manual_seed(0)
    out = {}
    if "resnet18" in names:
        from cinescope.framing.resnet import resnet18

        m = resnet18(5)
        timm = weight_path("resnet18_a1_0-d63eafa0.pth")
        src = "random"
        if timm.exists():
            st = torch.load(timm, map_location="cpu", weights_only=True)
            m.load_state_dict({k: v for k, v in st.items() if not k.startswith("fc.")}, strict=False)
            src = "timm ImageNet (fc random)"
        xr = lambda b: torch.randn(b, 3, 224, 384, generator=g)  # noqa: E731
        out["resnet18"] = (m.eval(), (xr(2),), lambda b: (xr(b),), ["frame"], ["logits"],
                           {"frame": {0: "batch"}, "logits": {0: "batch"}}, 16, src)
    if "embednet" in names:
        ck = ROOT / "p3_tracking" / "checkpoints" / "embed_triplet.pt"
        if ck.exists():
            from cinescope.faces.embed import load

            m = load(str(ck))
            xe = lambda b: torch.rand(b, 3, 64, 64, generator=g) * 4 - 2  # noqa: E731
            out["embednet"] = (m, (xe(2),), lambda b: (xe(b),), ["face"], ["embedding"],
                               {"face": {0: "batch"}, "embedding": {0: "batch"}}, 24, ck.name)
    if "clip_visual" in names or "clip_text" in names:
        from cinescope.search.encoders import get_encoder

        enc = get_encoder("openclip")
        vis, txt = _clip_parts(enc.model)
        xv = lambda b: torch.randn(b, 3, 224, 224, generator=g)  # noqa: E731
        if "clip_visual" in names:
            out["clip_visual"] = (vis.eval(), (xv(2),), lambda b: (xv(b),), ["image"], ["embedding"],
                                  {"image": {0: "batch"}, "embedding": {0: "batch"}}, 16, enc.name)
        words = ["a man walks away from an explosion", "two people talking in a car", "a plane in fog",
                 "a close-up of a face", "a crowd at night", "a red car", "someone typing", "a dog"]
        t = lambda b: enc.tokenizer([words[i % len(words)] for i in range(b)])  # noqa: E731
        if "clip_text" in names:
            out["clip_text"] = (txt.eval(), (t(2),), lambda b: (t(b),), ["tokens"], ["embedding"],
                                {"tokens": {0: "batch"}, "embedding": {0: "batch"}}, 64, enc.name)
    if "lightasd" in names and weight_path("light_asd_ava.model").exists():
        from cinescope.speaker.lightasd import LightASD

        m = _asd_batched(LightASD().net).eval()
        ex = (torch.randn(2, 100, 13, generator=g), torch.rand(2, 25, 112, 112, generator=g) * 255)

        def inp(b, n=50):          # a different length (2 s) from the traced one
            return (torch.randn(b, 4 * n, 13, generator=g), torch.rand(b, n, 112, 112, generator=g) * 255)

        out["lightasd"] = (m, ex, inp, ["mfcc", "crops"], ["logits"],
                           {"mfcc": {0: "batch", 1: "audio_steps"}, "crops": {0: "batch", 1: "frames"},
                            "logits": {0: "batch", 1: "frames"}}, 8, "light_asd_ava.model")
    if "crnn" in names:
        ck = ROOT / "p7_ocr" / "checkpoints" / "crnn.pt"
        if ck.exists():
            from cinescope.ocr.crnn import Recognizer

            m = Recognizer.load(ck).model
            ex = (torch.zeros(2, 1, 32, 160),)
            inp = lambda b: (torch.rand(b, 1, 32, 400, generator=g) * 2 - 1,)  # noqa: E731
            out["crnn"] = (m, ex, inp, ["line"], ["log_probs"],
                           {"line": {0: "batch", 3: "width"}, "log_probs": {0: "steps", 1: "batch"}}, 16, ck.name)
    return out


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+",
                    default=["resnet18", "embednet", "clip_visual", "clip_text", "lightasd", "crnn"])
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--out", default=str(ROOT / "p9_production" / "results" / "onnx.json"))
    ap.add_argument("--backend", default="onnx", choices=["onnx", "cuda", "tensorrt"],
                    help="ONNX Runtime provider (cuda/tensorrt need an NVIDIA GPU and onnxruntime-gpu)")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="where PyTorch runs for the comparison")
    args = ap.parse_args(argv)

    import onnxruntime
    import torch

    torch.set_num_threads(args.threads)
    rows = []
    for name, (m, ex, inp, ins, outs, dyn, batch, src) in models(args.models).items():
        t0 = time.perf_counter()
        om = OnnxModule(m, ex, ins, outs, dyn, tag=f"bench-{name}", backend=args.backend, threads=args.threads)
        t_export = time.perf_counter() - t0
        m = m.to(args.device)
        inp = _on(inp, args.device)
        # crnn's output is (T, B, C): compare per batch row
        check_in = inp(3)
        if name == "crnn":
            def tfn(*a, m=m):
                return m(*a).transpose(0, 1).cpu()

            def ofn(*a, om=om):
                return om(*a).transpose(0, 1)
        else:
            def tfn(*a, m=m):
                return m(*a).cpu()

            ofn = om
        num = compare(tfn, ofn, check_in)
        row = {"model": name, "weights": src, "onnx_mb": round(om.path.stat().st_size / 2**20, 1),
               "provider": om.sess.get_providers()[0],          # what actually ran (ORT falls back silently)
               "export_s": round(t_export, 1), **{k: float(f"{v:.3g}") for k, v in num.items()}}
        for b in (1, batch):
            x = inp(b)
            with torch.no_grad():
                pt = _median_ms(lambda x=x, m=m: _sync(m(*x), args.device), args.reps)
            ort = _median_ms(lambda x=x, om=om: om(*x), args.reps)
            row[f"torch_ms_b{b}" if b == 1 else "torch_ms_bN"] = round(pt, 2)
            row[f"onnx_ms_b{b}" if b == 1 else "onnx_ms_bN"] = round(ort, 2)
        row["batch_N"] = batch
        row["speedup_b1"] = round(row["torch_ms_b1"] / row["onnx_ms_b1"], 2)
        row["speedup_bN"] = round(row["torch_ms_bN"] / row["onnx_ms_bN"], 2)
        rows.append(row)
        print(json.dumps(row), flush=True)
    res = {"threads": args.threads, "torch": torch.__version__, "onnxruntime": onnxruntime.__version__,
           "cpu": _cpu_name(), "reps": args.reps, "backend": args.backend, "torch_device": args.device,
           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "rows": rows}
    Path(args.out).write_text(json.dumps(res, indent=1))
    print("\n| Model | max abs diff | min cosine | PyTorch b=1 (ms) | ONNX b=1 (ms) | speed-up | "
          "PyTorch b=N (ms) | ONNX b=N (ms) | speed-up |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['model']} | {r['max_abs']:.1e} | {r['min_cosine']:.6f} | {r['torch_ms_b1']} | {r['onnx_ms_b1']} | "
              f"{r['speedup_b1']}x | {r['torch_ms_bN']} (N={r['batch_N']}) | {r['onnx_ms_bN']} | {r['speedup_bN']}x |")
    return res


def _on(make_inputs, device: str):
    """Wrap an input factory so its tensors land on `device` (ONNX Runtime copies them back to numpy)."""
    if device == "cpu":
        return make_inputs
    return lambda b: tuple(t.to(device) if hasattr(t, "to") else t for t in make_inputs(b))


def _sync(out, device: str):
    if device == "cuda":
        import torch

        torch.cuda.synchronize()
    return out


def _cpu_name() -> str:
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    import platform

    return platform.processor()


if __name__ == "__main__":
    main()
