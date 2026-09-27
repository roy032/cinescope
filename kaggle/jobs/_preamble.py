# --- shared start of every CineScope Kaggle job (copied in by kaggle/run_all.py) ---
import glob
import os
import shutil
import subprocess
import sys
import zipfile


def find_code(marker: str) -> str:
    """Copy the cinescope code (dataset folder, or the zip inside it) to /kaggle/working/cinescope."""
    hits = glob.glob(f"/kaggle/input/**/{marker}", recursive=True)
    if not hits:
        for z in glob.glob("/kaggle/input/**/cinescope-code*.zip", recursive=True):
            zipfile.ZipFile(z).extractall("/kaggle/working/_code")
        hits = glob.glob(f"/kaggle/working/_code/**/{marker}", recursive=True)
    assert hits, "the cinescope-code dataset is not attached"
    src = os.path.dirname(os.path.dirname(hits[0]))
    shutil.copytree(src, "/kaggle/working/cinescope", dirs_exist_ok=True)
    os.chdir("/kaggle/working/cinescope")
    return src


def py(*args: str) -> None:
    print("$ python", " ".join(args), flush=True)
    subprocess.run([sys.executable, *args], check=True)


def sh(cmd: str) -> None:
    print("$", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)


def get_wider(train: bool = True) -> tuple[str, str]:
    """WIDER FACE (val, and train if asked) + the official eval ground truth, under /tmp."""
    w = "/tmp/wider"
    os.makedirs(w, exist_ok=True)
    hf = "https://huggingface.co/datasets/CUHK-CSE/wider_face/resolve/main/data"
    for z in (["WIDER_train.zip"] if train else []) + ["WIDER_val.zip", "wider_face_split.zip"]:
        sh(f"wget -q -L -O {w}/{z} {hf}/{z} && unzip -q -o {w}/{z} -d {w} && rm {w}/{z}")
    sh("wget -q -O /tmp/eval_tools.zip http://shuoyang1213.me/WIDERFACE/support/eval_script/eval_tools.zip"
       " && unzip -q -o /tmp/eval_tools.zip -d /tmp")
    gt = os.path.dirname(glob.glob("/tmp/**/wider_easy_val.mat", recursive=True)[0])
    return w, gt
# --- end of preamble ---
