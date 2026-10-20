"""Model inference backends: PyTorch, ONNX Runtime, TensorRT (through ONNX Runtime).

    from cinescope.pipeline.accel import OnnxModule
    fast = OnnxModule(model, (example,), ["x"], ["y"], {"x": {0: "b"}, "y": {0: "b"}}, tag="resnet18")
    y = fast(x)                         # torch in, torch out: a drop-in for model(x)

Why ONNX at all on a CPU: PyTorch's eager mode runs one Python-dispatched
kernel per layer, with no fusion across layers. Exporting the traced graph
lets ONNX Runtime fold BatchNorm into the convolutions, fuse conv + ReLU and
the attention blocks, and pre-pack weights for its own GEMM/conv kernels.
The model is the same numbers in a different engine, so every export is
checked against PyTorch on inputs of a *different* shape from the one it was
traced with (dynamic axes really dynamic): see p9_production/export_onnx.py.

Exports are cached in outputs/onnx/ (or Options.onnx_dir / $CINESCOPE_ONNX_DIR)
under a name that includes a fingerprint of the weights, so a retrained
checkpoint is re-exported and a known one is not.

TensorRT. `backend="tensorrt"` asks ONNX Runtime for its TensorRT execution
provider (then CUDA, then CPU), with fp16 and an engine cache. It needs an
NVIDIA GPU and onnxruntime-gpu built with TensorRT; measured on a Kaggle T4
(p9_production/README.md). Models with a dynamic axis other than the batch run on
CUDA instead, since TensorRT builds a new engine for every new shape.

The factories at the bottom (clip_encoder, lightasd, embednet,
framing_classifier, ocr_reader) return the object each phase already uses,
with its network swapped for the ONNX one when the backend is not torch. They
are cached per process: the web app's search endpoint and the pipeline share
one loaded encoder.
"""
from __future__ import annotations

import contextlib
import hashlib
import os
import warnings
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
_CACHE: dict = {}


def onnx_dir(options=None) -> Path:
    d = (getattr(options, "onnx_dir", None) or os.environ.get("CINESCOPE_ONNX_DIR")
         or str(REPO / "outputs" / "onnx"))
    p = Path(d)
    p.mkdir(parents=True, exist_ok=True)
    return p


def providers(backend: str, cache_dir: Path | None = None) -> list:
    if backend == "onnx":
        return ["CPUExecutionProvider"]
    if backend == "tensorrt":                       # needs an NVIDIA GPU (timed on a Kaggle T4)
        trt = {"trt_fp16_enable": True, "trt_engine_cache_enable": True,
               "trt_engine_cache_path": str(cache_dir or onnx_dir() / "trt")}
        return [("TensorrtExecutionProvider", trt), "CUDAExecutionProvider", "CPUExecutionProvider"]
    if backend == "cuda":
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    raise ValueError(f"unknown ONNX backend {backend!r} (onnx | tensorrt | cuda)")


def fingerprint(module) -> str:
    """Cheap, stable weight fingerprint: every tensor's name, shape, sum and
    first values (hashing all 600 MB of CLIP would take longer than exporting)."""
    h = hashlib.sha1()
    for k, v in module.state_dict().items():
        a = v.detach().float().reshape(-1).cpu().numpy() if v.numel() else np.zeros(0)
        h.update(k.encode())
        h.update(str(tuple(v.shape)).encode())
        h.update(np.float64(a.sum() if a.size else 0).tobytes())
        h.update(a[:16].astype(np.float32).tobytes())
    return h.hexdigest()[:12]


def export(module, example: tuple, path: Path, input_names: list[str], output_names: list[str],
           dynamic_axes: dict, opset: int = 17) -> Path:
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.onnx")
    module.eval()
    # nn.MultiheadAttention's inference fast path is one fused op with no ONNX
    # equivalent; the exporter needs the plain matmul/softmax path (same maths).
    fast = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        with torch.no_grad():
            torch.onnx.export(module, example, str(tmp), input_names=input_names, output_names=output_names,
                              dynamic_axes=dynamic_axes, opset_version=opset, do_constant_folding=True,
                              dynamo=False)
    finally:
        torch.backends.mha.set_fastpath_enabled(fast)
    os.replace(tmp, path)
    return path


def session(path: str | Path, backend: str = "onnx", threads: int = 2):
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.log_severity_level = 3
    if backend != "onnx" and hasattr(ort, "preload_dlls"):
        # CUDA / cuDNN from the nvidia-* pip packages (PyTorch's); best effort, the check below reports
        with contextlib.suppress(Exception):
            ort.preload_dlls()
    sess = ort.InferenceSession(str(path), so, providers=providers(backend))
    want = providers(backend)[0]
    want = want[0] if isinstance(want, tuple) else want
    if sess.get_providers()[0] != want:        # ONNX Runtime falls back to CPU without a word
        warnings.warn(f"{Path(path).name}: asked for {want}, running on {sess.get_providers()[0]} "
                      "(GPU libraries not found or not matching this onnxruntime-gpu build)", stacklevel=2)
    return sess


class OnnxModule:
    """A torch module's forward, run by ONNX Runtime. Attribute access falls
    through to the original module (so code reading `model.size` still works)."""

    def __init__(self, module, example: tuple, input_names: list[str], output_names: list[str],
                 dynamic_axes: dict, tag: str, backend: str = "onnx", threads: int = 2, options=None):
        self.module = module
        if backend == "tensorrt" and any(n != "batch" for ax in dynamic_axes.values() for n in ax.values()):
            # TensorRT builds a new engine for every new sequence length or line width it meets;
            # on a T4 that turned Light-ASD's 54 s into 28 min. Those models run on CUDA instead.
            backend = "cuda"
        path = onnx_dir(options) / f"{tag}-{fingerprint(module)}.onnx"
        if not path.exists():
            export(module, example, path, input_names, output_names, dynamic_axes)
        self.path = path
        try:
            self.sess = session(path, backend, threads)
        except Exception:                       # a stale or broken cached export: export again
            export(module, example, path, input_names, output_names, dynamic_axes)
            self.sess = session(path, backend, threads)
        self.input_names, self.output_names = input_names, output_names

    def run_numpy(self, *arrays: np.ndarray) -> list[np.ndarray]:
        feed = {}
        for n, a in zip(self.input_names, arrays, strict=True):
            a = np.asarray(a)
            feed[n] = a.astype(np.float32) if a.dtype == np.float64 else a
        return self.sess.run(self.output_names, feed)

    def __call__(self, *tensors):
        import torch

        outs = self.run_numpy(*[t.detach().cpu().numpy() if hasattr(t, "detach") else t for t in tensors])
        res = [torch.from_numpy(o) for o in outs]
        return res[0] if len(res) == 1 else tuple(res)

    def eval(self):
        return self

    def __getattr__(self, name):
        if name == "module":
            raise AttributeError(name)
        return getattr(self.module, name)


def compare(torch_fn, onnx_fn, inputs: tuple) -> dict:
    """Max absolute / relative difference and min cosine similarity (per row) of two implementations."""
    import torch

    with torch.no_grad():
        a = torch_fn(*inputs)
    b = onnx_fn(*inputs)
    a = a.detach().float().cpu().numpy().reshape(len(a), -1)
    b = np.asarray(b.detach().cpu().numpy() if hasattr(b, "detach") else b, np.float32).reshape(len(a), -1)
    cos = (a * b).sum(1) / np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-12)
    return {"max_abs": float(np.abs(a - b).max()), "max_rel": float(np.abs(a - b).max() / max(np.abs(a).max(), 1e-12)),
            "min_cosine": float(cos.min())}


# --------------------------------------------------------------------------- wrappers per model
def _clip_parts(model):
    import torch

    class Visual(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            return self.m.encode_image(x)

    class Text(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, tokens):
            return self.m.encode_text(tokens)

    return Visual(model), Text(model)


def onnx_clip(enc, backend: str, threads: int, options=None):
    """Swap an OpenClipEncoder's image and text towers for ONNX sessions (same .name,
    so indexes built either way are interchangeable)."""
    import torch

    vis, txt = _clip_parts(enc.model)
    tag = enc.name.split(":")[1] if ":" in enc.name else "clip"
    ov = OnnxModule(vis, (torch.randn(2, 3, 224, 224),), ["image"], ["embedding"],
                    {"image": {0: "batch"}, "embedding": {0: "batch"}}, f"clip-visual-{tag}", backend, threads,
                    options)
    tokens = enc.tokenizer(["a photo", "a film still of a man walking"])
    ot = OnnxModule(txt, (tokens,), ["tokens"], ["embedding"], {"tokens": {0: "batch"}, "embedding": {0: "batch"}},
                    f"clip-text-{tag}", backend, threads, options)
    # instance attributes shadow the methods; OpenClipEncoder calls model.encode_image / encode_text
    object.__setattr__(enc.model, "encode_image", ov)
    object.__setattr__(enc.model, "encode_text", ot)
    enc.onnx = (ov, ot)
    return enc


def _asd_batched(net):
    """Light-ASD's whole forward as one graph. Its audio encoder applies MaxPool3d
    to 4-D tensors (PyTorch reads them as unbatched 3-D volumes); ONNX has no
    such reading, so on a copy those pools become the equivalent MaxPool2d
    (pool along the last axis only, stride 2)."""
    import copy

    import torch

    net = copy.deepcopy(net)
    enc = net.model.audioEncoder
    for k in ("pool1", "pool2"):
        setattr(enc, k, torch.nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)))

    class ASD(torch.nn.Module):
        """(B, 4n, 13) MFCCs + (B, n, 112, 112) crops -> (B, n) speaking logits."""

        def __init__(self, net):
            super().__init__()
            self.net = net

        def forward(self, a, v):
            m = self.net.model
            out = m.forward_audio_visual_backend(m.forward_audio_frontend(a), m.forward_visual_frontend(v))
            return self.net.lossAV.FC(out)[:, 1].reshape(a.shape[0], -1)

    return ASD(net)


def onnx_lightasd(m, backend: str, threads: int, options=None):
    import torch

    mod = OnnxModule(_asd_batched(m.net), (torch.randn(2, 100, 13), torch.rand(2, 25, 112, 112) * 255),
                     ["mfcc", "crops"], ["logits"],
                     {"mfcc": {0: "batch", 1: "audio_steps"}, "crops": {0: "batch", 1: "frames"},
                      "logits": {0: "batch", 1: "frames"}}, "lightasd", backend, threads, options)

    def runner(a: np.ndarray, v: np.ndarray) -> np.ndarray:
        batched = a.ndim == 3
        a = np.asarray(a, np.float32) if batched else np.asarray(a, np.float32)[None]
        v = np.asarray(v, np.float32) if batched else np.asarray(v, np.float32)[None]
        out = mod.run_numpy(a, v)[0]
        return out if batched else out[0]

    m.runner = runner
    m.onnx = mod
    return m


# --------------------------------------------------------------------------- factories used by the stages
def _threads(options) -> int:
    return int(getattr(options, "threads", 2) or 2)


def _backend(options) -> str:
    return getattr(options, "backend", "torch") or "torch"


def _set_torch_threads(options) -> None:
    import torch

    torch.set_num_threads(_threads(options))


def clip_encoder(spec: str = "openclip", options=None):
    key = ("clip", spec, _backend(options))
    if key not in _CACHE:
        from ..search.encoders import get_encoder

        _set_torch_threads(options)
        enc = get_encoder(spec)
        if _backend(options) != "torch" and hasattr(enc, "tokenizer"):
            enc = onnx_clip(enc, _backend(options), _threads(options), options)
        _CACHE[key] = enc
    return _CACHE[key]


def lightasd(options=None):
    key = ("lightasd", _backend(options))
    if key not in _CACHE:
        from ..speaker.lightasd import LightASD

        _set_torch_threads(options)
        m = LightASD()
        if _backend(options) != "torch":
            m = onnx_lightasd(m, _backend(options), _threads(options), options)
        _CACHE[key] = m
    return _CACHE[key]


def embednet(ckpt: str, options=None):
    """Phase 3's own face embedder as a crops -> unit vectors function."""
    import torch

    from ..faces.embed import embed, load

    _set_torch_threads(options)
    m = load(ckpt)
    if _backend(options) != "torch":
        m = OnnxModule(m, (torch.randn(2, 3, m.size, m.size),), ["face"], ["embedding"],
                       {"face": {0: "batch"}, "embedding": {0: "batch"}}, "embednet", _backend(options),
                       _threads(options), options)
    return lambda crops: embed(m, crops)


def framing_classifier(ckpt: str, options=None):
    import torch

    from ..framing.classify import FramingClassifier

    _set_torch_threads(options)
    clf = FramingClassifier(ckpt, device="cpu")
    if _backend(options) != "torch":
        clf.model = OnnxModule(clf.model, (torch.randn(2, 3, 224, 384),), ["frame"], ["logits"],
                               {"frame": {0: "batch", 2: "height", 3: "width"}, "logits": {0: "batch"}},
                               "framing-resnet18", _backend(options), _threads(options), options)
    return clf


def ocr_reader(name: str = "auto", ckpt: str | None = None, options=None):
    """(reader object with .read, its name). auto: our CRNN when its checkpoint
    exists (p7_ocr/checkpoints/crnn.pt), else RapidOCR's recogniser (already ONNX)."""
    import torch

    from ..ocr.libs import make_reader

    default_ckpt = REPO / "p7_ocr" / "checkpoints" / "crnn.pt"
    if name == "auto":
        name = "ours" if (ckpt and Path(ckpt).exists()) or default_ckpt.exists() else "rapidocr"
    _set_torch_threads(options)
    rec = make_reader(name, ckpt)
    if name == "ours" and _backend(options) != "torch":
        rec.model = OnnxModule(rec.model, (torch.zeros(2, 1, 32, 160),), ["line"], ["log_probs"],
                               {"line": {0: "batch", 3: "width"}, "log_probs": {0: "steps", 1: "batch"}},
                               "crnn", _backend(options), _threads(options), options)
    return rec, name
