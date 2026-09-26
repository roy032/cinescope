# CineScope Phase 2 — evaluate + diagnose an already-trained face detector (one Kaggle cell).
# Inputs to attach: the "cinescope-code" dataset (latest version) and the finished Phase 2
# notebook's output (Add Input -> Your Work / Notebook Output). Settings: GPU on, Internet on.
import glob
import os
import shutil
import subprocess
import sys

hits = glob.glob("/kaggle/input/**/p2_detection/diagnose_wider.py", recursive=True)
assert hits, "Attach the NEW version of the cinescope-code dataset (it must contain diagnose_wider.py)"
src = os.path.dirname(os.path.dirname(hits[0]))
shutil.copytree(src, "/kaggle/working/cinescope", dirs_exist_ok=True)
os.chdir("/kaggle/working/cinescope")

ckpts = glob.glob("/kaggle/input/**/centernet_r18.pt", recursive=True)
assert ckpts, "Attach the finished Phase 2 notebook's output (it contains checkpoints/centernet_r18.pt)"
CKPT = ckpts[0]

W = "/tmp/wider"                       # only the validation half is needed (~370 MB)
os.makedirs(W, exist_ok=True)
hf = "https://huggingface.co/datasets/CUHK-CSE/wider_face/resolve/main/data"
for z in ["WIDER_val.zip", "wider_face_split.zip"]:
    subprocess.run(["wget", "-q", "-L", "-O", f"{W}/{z}", f"{hf}/{z}"], check=True)
    subprocess.run(["unzip", "-q", "-o", f"{W}/{z}", "-d", W], check=True)
subprocess.run(["wget", "-q", "-O", "/tmp/eval_tools.zip",
                "http://shuoyang1213.me/WIDERFACE/support/eval_script/eval_tools.zip"], check=True)
subprocess.run(["unzip", "-q", "-o", "/tmp/eval_tools.zip", "-d", "/tmp"], check=True)
GT = os.path.dirname(glob.glob("/tmp/**/wider_easy_val.mat", recursive=True)[0])

OUT = "/kaggle/working/results"
def run(*args):
    print("$", " ".join(args), flush=True)
    subprocess.run([sys.executable, *args], check=True)

run("p2_detection/eval_wider.py", "--wider", W, "--ckpt", CKPT, "--eval-tools", GT,
    "--topk", "1000", "--out", OUT)
run("p2_detection/diagnose_wider.py", "--wider", W, "--eval-tools", GT,
    "--preds", f"{OUT}/centernet_r18/wider_val_preds.npz", "--out", f"{OUT}/centernet_r18/diagnosis.md")
subprocess.run("cd /kaggle/working && zip -qr phase2_diagnosis.zip results && ls -lh phase2_diagnosis.zip",
               shell=True, check=True)
