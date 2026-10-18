# Phase 9 on an NVIDIA GPU: the TensorRT and CUDA backends, checked against PyTorch and timed.
# The development laptop has no NVIDIA GPU, so this is where `--backend tensorrt` gets tested.
import json

find_code("tools/fetch_assets.py")          # one folder deep: find_code copies the marker's grandparent
# Work in /tmp: clips, weights and engines would otherwise all land in the job's downloadable output.
shutil.copytree("/kaggle/working/cinescope", "/tmp/cinescope", dirs_exist_ok=True)
os.chdir("/tmp/cinescope")
shutil.rmtree("/kaggle/working/cinescope")
OUT = "/kaggle/working/results/tensorrt"
os.makedirs(OUT, exist_ok=True)
ERRORS = {}


def step(name, fn):
    """Run one measurement; a failure is recorded in errors.json and the later steps still run."""
    try:
        fn()
    except Exception as e:  # noqa: BLE001
        ERRORS[name] = repr(e)
        print(f"!!! {name} failed: {e!r}", flush=True)
    with open(f"{OUT}/errors.json", "w") as f:
        json.dump(ERRORS, f, indent=1)

# The CPU and GPU wheels of ONNX Runtime install the same module: keep only the GPU one.
sh("pip install -q -r requirements.txt -r requirements-full.txt")
sh("pip uninstall -y -q onnxruntime")
# A CUDA 12 build to match Kaggle's PyTorch (a newer default wheel loaded neither CUDA nor TensorRT on
# the first run: both fell back to CPU), and the TensorRT 10 it was built against.
sh("pip install -q --force-reinstall --no-deps onnxruntime-gpu==1.22.0 || pip install -q --force-reinstall --no-deps onnxruntime-gpu")
sh("pip install -q 'tensorrt-cu12==10.9.*' || pip install -q 'tensorrt-cu12>=10,<11'")
py("tools/fetch_assets.py")

# ONNX Runtime finds TensorRT, cuDNN and cuBLAS through the loader path: the pip packages' lib folders.
site = subprocess.run([sys.executable, "-c", "import site; print(site.getsitepackages()[0])"],
                      capture_output=True, text=True, check=True).stdout.strip()
libs = sorted({os.path.dirname(p) for p in glob.glob(f"{site}/tensorrt_libs/*.so*")
               + glob.glob(f"{site}/nvidia/*/lib/*.so*")})
os.environ["LD_LIBRARY_PATH"] = ":".join(libs + [os.environ.get("LD_LIBRARY_PATH", "")])
os.environ["CINESCOPE_ONNX_DIR"] = "/tmp/onnx"
sh("nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv")

# 1. Which providers actually load (ONNX Runtime silently falls back to CPU when one fails).
probe = r"""
import json, onnxruntime as ort, torch
from cinescope.pipeline.accel import OnnxModule
m = torch.nn.Sequential(torch.nn.Conv2d(3, 8, 3), torch.nn.ReLU(), torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten())
res = {"onnxruntime": ort.__version__, "available": ort.get_available_providers(),
       "torch": torch.__version__, "torch_cuda": torch.version.cuda}
ort.set_default_logger_severity(1)                # print why a provider fails to load
for b in ("cuda", "tensorrt"):
    om = OnnxModule(m, (torch.randn(2, 3, 32, 32),), ["x"], ["y"], {"x": {0: "b"}, "y": {0: "b"}}, tag=f"probe-{b}", backend=b)
    om(torch.randn(4, 3, 32, 32))
    res[b] = om.sess.get_providers()
print(json.dumps(res))
open("/kaggle/working/results/tensorrt/providers.json", "w").write(json.dumps(res, indent=1))
"""
step("providers", lambda: py("-c", probe))

# 2. Every pipeline model: export, agreement with PyTorch on the GPU, timing (TensorRT, then CUDA).
for backend in ("tensorrt", "cuda"):
    step(f"export_{backend}", lambda b=backend: py("p9_production/export_onnx.py", "--backend", b, "--device", "cuda",
                                                   "--reps", "20", "--out", f"{OUT}/onnx_{b}.json"))

# 3. The whole pipeline on a public clip with each backend (CPU ONNX as the reference).
for backend in ("onnx", "tensorrt"):
    step(f"pipeline_{backend}", lambda b=backend: py("p9_production/bench_pipeline.py", "trailers/goldeneye.mp4",
                                                     "--backend", b, "--workers", "2", "--tag", f"kaggle-{b}"))
for f in glob.glob("p9_production/results/pipeline_goldeneye_kaggle-*.json"):
    shutil.copy(f, OUT)
for b in ("tensorrt", "cuda"):
    if os.path.exists(f"{OUT}/onnx_{b}.json"):
        rows = json.load(open(f"{OUT}/onnx_{b}.json"))["rows"]
        print(b, json.dumps([(r["model"], r["provider"], r["onnx_ms_b1"], r["torch_ms_b1"]) for r in rows]))
print("errors:", ERRORS)
