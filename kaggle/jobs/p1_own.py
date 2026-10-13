# Phase 1 on your own videos: auto-label the shots, train ResNet-18 (ImageNet init 30 epochs,
# random init 60), and predict the held-out test split for the hand-labelled evaluation.
find_code("p1_cnn/train_framing.py")
zips = glob.glob("/kaggle/input/**/own_shots.zip", recursive=True)
csvs = glob.glob("/kaggle/input/**/own_shots/shots.csv", recursive=True) + \
    glob.glob("/kaggle/input/**/shots.csv", recursive=True)
if zips:
    ROOT = "/tmp/own"
    zipfile.ZipFile(zips[0]).extractall(ROOT)
else:
    assert csvs, "the cinescope-own-shots dataset is not attached"
    ROOT = os.path.dirname(csvs[0])
ckpt = glob.glob("/kaggle/input/**/centernet_r18.pt", recursive=True)
assert ckpt, "the cinescope-ckpt dataset (centernet_r18.pt) is not attached"
OUT, CK = "/kaggle/working/results", "/kaggle/working/checkpoints"
OWN = f"{OUT}/own"
py("p1_cnn/autolabel.py", "--root", ROOT, "--face-ckpt", ckpt[0], "--out", OWN)
common = ["--root", ROOT, "--index", f"{OWN}/index.csv", "--out", OUT, "--ckpt-dir", CK, "--workers", "4"]
py("p1_cnn/train_framing.py", *common, "--pretrained", "--epochs", "30", "--tag", "pretrained")
py("p1_cnn/train_framing.py", *common, "--epochs", "60", "--tag", "scratch")
for tag in ("pretrained", "scratch"):
    py("p1_cnn/predict_shots.py", "--ckpt", f"{CK}/{tag}.pt", "--root", ROOT, "--shots", f"{ROOT}/shots.csv",
       "--split", "test", "--out", f"{OWN}/student_{tag}.csv")
