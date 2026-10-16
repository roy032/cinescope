# Phase 9 on an NVIDIA GPU: the TensorRT and CUDA backends, checked against PyTorch and timed.
# The development laptop has no NVIDIA GPU, so this is where `--backend tensorrt` gets tested.
import json

find_code("cinescope/pipeline/accel.py")
OUT = "/kaggle/working/results/tensorrt"
os.makedirs(OUT, exist_ok=True)

# The CPU and GPU wheels of ONNX Runtime install the same module: keep only the GPU one.
sh("pip install -q -r requirements-full.txt")
sh("pip uninstall -y -q onnxruntime")
sh("pip install -q --force-reinstall --no-deps onnxruntime-gpu")
sh("pip install -q 'tensorrt-cu12>=10,<11'")
py("tools/fetch_assets.py")

# ONNX Runtime finds TensorRT, cuDNN and cuBLAS through the loader path: the pip packages' lib folders.
site = subprocess.run([sys.executable, "-c", "import site; print(site.getsitepackages()[0])"],
                      capture_output=True, text=True, check=True).stdout.strip()
libs = sorted({os.path.dirname(p) for p in glob.glob(f"{site}/tensorrt_libs/*.so*")
               + glob.glob(f"{site}/nvidia/*/lib/*.so*")})
os.environ["LD_LIBRARY_PATH"] = ":".join(libs + [os.environ.get("LD_LIBRARY_PATH", "")])
os.environ["CINESCOPE_ONNX_DIR"] = "/kaggle/working/onnx"
sh("nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv")

# 1. Which providers actually load (ONNX Runtime silently falls back to CPU when one fails).
probe = r"""
import json, onnxruntime as ort, torch
from cinescope.pipeline.accel import OnnxModule
m = torch.nn.Sequential(torch.nn.Conv2d(3, 8, 3), torch.nn.ReLU(), torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten())
res = {"onnxruntime": ort.__version__, "available": ort.get_available_providers()}
for b in ("cuda", "tensorrt"):
    om = OnnxModule(m, (torch.randn(2, 3, 32, 32),), ["x"], ["y"], {"x": {0: "b"}, "y": {0: "b"}}, tag=f"probe-{b}", backend=b)
    om(torch.randn(4, 3, 32, 32))
    res[b] = om.sess.get_providers()
print(json.dumps(res))
open("/kaggle/working/results/tensorrt/providers.json", "w").write(json.dumps(res, indent=1))
"""
py("-c", probe)

# 2. Every pipeline model: export, agreement with PyTorch on the GPU, timing (TensorRT, then CUDA).
for backend in ("tensorrt", "cuda"):
    py("p9_production/export_onnx.py", "--backend", backend, "--device", "cuda", "--reps", "20",
       "--out", f"{OUT}/onnx_{backend}.json")

# 3. The whole pipeline on a public clip with each backend (CPU ONNX as the reference).
for backend in ("onnx", "tensorrt"):
    py("p9_production/bench_pipeline.py", "trailers/goldeneye.mp4", "--backend", backend,
       "--workers", "2", "--tag", f"kaggle-{backend}")
for f in glob.glob("p9_production/results/pipeline_goldeneye_kaggle-*.json"):
    shutil.copy(f, OUT)
summary = {b: json.load(open(f"{OUT}/onnx_{b}.json")) for b in ("tensorrt", "cuda")}
print(json.dumps({b: [(r["model"], r["provider"], r["onnx_ms_b1"], r["torch_ms_b1"]) for r in s["rows"]]
                  for b, s in summary.items()}, indent=1))
