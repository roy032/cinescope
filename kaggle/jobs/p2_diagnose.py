# Evaluate the trained CenterNet on WIDER val (top-1000 peaks) and break results down by face size.
find_code("p2_detection/diagnose_wider.py")
ckpt = glob.glob("/kaggle/input/**/centernet_r18.pt", recursive=True)
assert ckpt, "the cinescope-ckpt dataset (centernet_r18.pt) is not attached"
W, GT = get_wider(train=False)
OUT = "/kaggle/working/results"
py("p2_detection/eval_wider.py", "--wider", W, "--ckpt", ckpt[0], "--eval-tools", GT, "--topk", "1000", "--out", OUT)
py("p2_detection/diagnose_wider.py", "--wider", W, "--eval-tools", GT,
   "--preds", f"{OUT}/centernet_r18/wider_val_preds.npz", "--out", f"{OUT}/centernet_r18/diagnosis.md")
