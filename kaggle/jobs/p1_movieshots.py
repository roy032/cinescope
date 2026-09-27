# Phase 1: MovieShots -> index -> ResNet-18 shot scale, ImageNet init (30 epochs) and from scratch (60).
find_code("p1_cnn/train_framing.py")
sh("pip -q install gdown")
MS = None
for j in glob.glob("/kaggle/input/**/*.json", recursive=True):       # an attached copy wins
    if "trailer" in os.path.basename(j).lower():
        MS = os.path.dirname(j)
        break
if MS is None:
    MS = "/tmp/movieshots"
    sh(f'gdown --folder "https://drive.google.com/drive/folders/1wAviJV0fuISKG6RpCAqiwG7lwtfxktkQ" -O {MS} --remaining-ok')
    for z in glob.glob(f"{MS}/**/*.zip", recursive=True) + glob.glob(f"{MS}/**/*.tar*", recursive=True):
        d = os.path.dirname(z)
        sh(f'unzip -q -o "{z}" -d "{d}"' if z.endswith(".zip") else f'tar -xf "{z}" -C "{d}"')
        os.remove(z)
for root, _dirs, files in os.walk(MS):                               # layout, for the log
    if root[len(MS):].count(os.sep) <= 2:
        print(root, len(files), files[:3])
jsons = glob.glob(f"{MS}/**/*split*.json", recursive=True) or glob.glob(f"{MS}/**/*.json", recursive=True)
ROOT = os.path.dirname(jsons[0])
OUT, CK = "/kaggle/working/results", "/kaggle/working/checkpoints"
py("p1_cnn/prepare_movieshots.py", "--root", ROOT, "--annotations", jsons[0], "--out", "/kaggle/working/index.csv")
common = ["--root", ROOT, "--index", "/kaggle/working/index.csv", "--out", OUT, "--ckpt-dir", CK, "--workers", "4"]
py("p1_cnn/train_framing.py", *common, "--pretrained", "--epochs", "30", "--tag", "pretrained")
py("p1_cnn/train_framing.py", *common, "--epochs", "60", "--tag", "scratch")
