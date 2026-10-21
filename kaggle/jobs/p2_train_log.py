# Retrain CenterNet with scale-invariant (log) size regression, then evaluate and diagnose it.
# A job that starts with "# extends: p2_train_log" sets TAG / SIZE_MODE / EPOCHS / EXTRA first.
TAG = globals().get("TAG", "centernet_r18_log")
SIZE_MODE = globals().get("SIZE_MODE", "log")
EPOCHS = globals().get("EPOCHS", 60)
EXTRA = globals().get("EXTRA", [])
find_code("p2_detection/train_faces.py")
W, GT = get_wider(train=True)
OUT, CK = "/kaggle/working/results", "/kaggle/working/checkpoints"
prev = glob.glob(f"/kaggle/input/**/{TAG}.pt", recursive=True)      # resume if a previous run is attached
os.makedirs(CK, exist_ok=True)
if prev:
    shutil.copy(prev[0], f"{CK}/{TAG}.pt")
py("p2_detection/train_faces.py", "--wider", W, "--epochs", str(EPOCHS), "--bs", "32", "--workers", "4",
   "--tag", TAG, "--size-mode", SIZE_MODE, "--ckpt-dir", CK, "--out", OUT, "--resume", *EXTRA)
py("p2_detection/eval_wider.py", "--wider", W, "--ckpt", f"{CK}/{TAG}.pt", "--eval-tools", GT,
   "--topk", "1000", "--out", OUT)
py("p2_detection/diagnose_wider.py", "--wider", W, "--eval-tools", GT,
   "--preds", f"{OUT}/{TAG}/wider_val_preds.npz", "--out", f"{OUT}/{TAG}/diagnosis.md")
