# Phase 1 — CNN fundamentals (Weeks 4–5)

**Feature:** every shot gets a framing label (extreme close-up, close-up,
medium, full, long) from a ResNet-18 written from scratch.

```powershell
python p0_classical/analyse.py trailers/goldeneye.mp4 --framing p1_cnn/checkpoints/pretrained.pt
# shots.json gains  "framing": {"label": "CS", "name": "close-up", "confidence": 0.91, ...}
```

## Week 4 — backprop by hand (NumPy)

```powershell
python p1_cnn/week4_backprop.py        # ~10 s on a CPU; writes p1_cnn/results/week4.json
```

Measured (`results/week4.json`): every layer's gradient check within 3.5e-10;
the NumPy strided conv matches `torch.nn.Conv2d` to 2e-16 (forward), 4e-16
(dL/dx) and 7e-15 (dL/dW); the tiny ResNet reaches 100% on the synthetic test
set in 5 epochs.

| Built | Module | Checked against |
|---|---|---|
| Scalar reverse-mode autograd (micrograd-style), tiny MLP | `cinescope/nn/autograd.py` | finite differences; PyTorch autograd |
| Linear, Conv2d (im2col / col2im, stride, padding), ReLU, MaxPool2d, BatchNorm2d (train/eval), GlobalAvgPool, Residual block, softmax cross-entropy, SGD + momentum | `cinescope/nn/layers.py` | central-difference gradient check (< 1e-6 relative error); `torch.nn` forward **and** backward to 1e-10 |
| Gradient checker | `cinescope/nn/gradcheck.py` | |
| Tiny NumPy ResNet on a synthetic shot-scale task | `cinescope/nn/toy.py` | trains to ~100% in 5 epochs on a CPU |

Things the gradient check taught me:

- A conv bias in front of BatchNorm gets a gradient of **exactly zero** — BN
  subtracts the batch mean and the bias with it. That's why ResNet convs have
  `bias=False`.
- BatchNorm's backward pass has three terms, not one: the mean and the
  variance are themselves functions of every input in the batch.
- The first NumPy CNN scored 0.74 on the test set in train mode but 0.55 in
  `eval()` mode: BatchNorm's running statistics lagged behind weights that were
  still moving fast (lr 0.2, momentum 0.1 on the averages). Decaying the
  learning rate to zero (cosine) lets the averages catch up.
- The first version of the toy task (random subject and set colours) was too
  hard for an 8-channel network; making the subject lit and the set darker, as
  in most shots, made it learnable in seconds. It is a test of the machinery,
  not a benchmark.
- The gradient checker must not use the same random seed for the input and
  for the projection of the output: with `sum(f(x) * x)` a correct BatchNorm
  showed a 5e-6 error, because the loss nearly cancelled and the finite
  differences lost precision.

## Week 5 — ResNet-18 shot-scale classifier (PyTorch)

| Built | Module |
|---|---|
| ResNet-18 from scratch (BasicBlock, projection shortcuts, Kaiming init, optional zero-init residual); parameter names match torchvision so ImageNet weights load directly | `cinescope/framing/resnet.py` |
| MovieShots annotations → flat CSV index; trailer-level splits; datasets (3 key frames per shot) | `cinescope/framing/data.py` |
| Training loop: SGD + Nesterov, warm-up + cosine LR, mixed precision (GradScaler), label smoothing, no weight decay on BN/bias | `cinescope/framing/train.py` |
| Metrics, confusion matrix | `cinescope/framing/metrics.py` |
| Shot labelling in the Phase 0 pipeline | `cinescope/framing/classify.py` |

Design choices worth defending:

- **Frames are not cropped square.** Frames are resized to 224×384 (16:9) and
  global average pooling makes the network size-agnostic. A square centre crop
  cuts off the sides of a wide frame — exactly the evidence for "long shot".
- **Augmentation keeps ≥ 80% of the frame.** torchvision's default
  `RandomResizedCrop` samples crops down to 8% of the image, which turns a long
  shot into a medium one while keeping the "long shot" label. Horizontal flips
  and colour jitter are safe; vertical flips are not (no film is upside down).
- **Splits are by trailer, not by shot.** Shots of one trailer share actors,
  sets and colour grading; mixing them across train and test inflates accuracy.
- **Three key frames per shot**: one random frame per step while training,
  probabilities averaged over all three at test time.

### Data: any videos, labelled automatically

MovieShots (Rao et al., ECCV 2020), the usual benchmark, is released by its
authors on request; its public folder holds only the annotation files. The
classifier is therefore trained on whatever videos you point it at (TV
episodes, films, clips), with labels made automatically and a test set labelled
by hand. The footage and key frames are never committed.

1. **Shots.** `build_own_shots.py` finds every video under a folder, copies
   them into `data/episodes/`, cuts them with the Phase 0 shot detector and
   keeps three key frames per shot (25/50/75% of its length; shots under half a
   second and black frames are dropped). Splits are by video: files named like
   episodes of one show (`Show.S01E02`, `Show 1x02`) are split by episode, the
   last one test and the one before it validation; any other video is split by
   time (70/10/20%), so no scene is on both sides.
2. **Automatic labels** (`autolabel.py`, `cinescope/framing/teacher.py`). Two
   independent cues, fused per shot:
   - zero-shot image-text matching, three prompts per class: SigLIP (so400m)
     by default; open_clip ViT-B-32-quickgelu from a local LAION-400M file
     (`--text openclip`, also the automatic fallback when SigLIP cannot be
     loaded, e.g. without HuggingFace access);
   - the height of the largest face relative to the frame height, turned into
     a soft class prior (a Gaussian in log space around each class's typical
     ratio). Scale is defined by how much of a person the frame shows, so this
     cue is strong when a face is visible and silent when none is. The face
     detector is our Phase 2 CenterNet (`--face-ckpt`) or, without a
     checkpoint, OpenCV's YuNet (`cinescope/detect/yunet.py`).
3. **Hand-labelled test set.** `label_scales.py` builds a keyboard page with
   300 random shots from the test split; it never shows a prediction, so
   the labels do not lean towards any system. `eval_own.py` scores every cue,
   the fused teacher and both networks against it.

The networks learn from noisy labels. Whether a student ends up better or worse
than its teacher on the hand labels is the interesting number: better means
the CNN averaged away some of the teacher's errors; worse means it mostly
copied them.

```powershell
python p1_cnn/build_own_shots.py --search path/to/videos   # find + copy + cut, zip for Kaggle
RUN_OWN_SHOTS.bat path\to\videos                          # Windows: that, the labelling page and the Kaggle run
python p1_cnn/eval_own.py  # after labelling: every system against the hand labels
```

MovieShots still works unchanged (`prepare_movieshots.py`, then
`kaggle/run_all.py --only p1_movieshots` with its frames attached).

### Training (Kaggle GPU)

`kaggle/jobs/p1_own.py`: auto-labelling (~15 min on a T4), then the two runs
the roadmap asks for, ResNet-18 from ImageNet weights (30 epochs) and from
random init (60), selected on the validation split (whose labels are the
teacher's), then predictions for the test split. Before spending GPU hours,
check the loop on a CPU:

```powershell
python p1_cnn/train_framing.py --synthetic --epochs 2 --tag smoke
```

### The public-clip run (CPU only)

The whole experiment also runs on the public sample clips, on a CPU, with the
offline back-ends of both cues (open_clip + YuNet):

```powershell
python p1_cnn/build_own_shots.py --videos data/public_clips --out data/public_shots --no-zip
python p1_cnn/label_scales.py --shots data/public_shots/shots.csv --set public --split all  # labelling page
python p1_cnn/run_public.py      # teacher, 2 folds x 2 inits, predictions, eval_own.py --set public; ~57 min
python tools/update_results.py
```

- **Clips**: 19 public videos (the trailers in `trailers/`, Big Buck Bunny,
  and more of Intel's sample videos for detection demos), cut into 66
  shots. Most of the sample videos are a single static shot.
- **Hand labels** (`labels/public_test.labels.json`): all 66 shots, labelled
  by eye from contact sheets of their three key frames **before the teacher
  or any network had run** (51 usable; 15 marked *skip*: black frames,
  kinetic-typography titles). How each borderline case was decided is in the
  file's note.
- **2-fold cross-validation by time.** 66 shots are too few for a held-out
  20%, so each clip is split in two halves by time; fold 0 trains on the
  second halves and predicts the first, fold 1 the reverse (validation is
  the 10% of a clip just before its test half, wrapping round to the clip's
  last 10% for fold 0). Every shot is predicted once, by
  a network that never saw it.
- **Hyper-parameters fixed before the run**, not tuned: batch 8, lr 0.01
  (ImageNet init, 30 epochs) / 0.05 (random init, 60 epochs), 2 warm-up
  epochs, 224×384, 2 CPU threads.
- **ImageNet weights**: download.pytorch.org is unreachable from the build
  machine, so the ImageNet init is timm's ResNet-18 `resnet18_a1_0-d63eafa0.pth`
  (*ResNet strikes back*, A1 recipe), from the official pytorch-image-models
  GitHub release `v0.1-rsb-weights`
  (`github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet18_a1_0-d63eafa0.pth`,
  the URL timm itself uses for `resnet18.a1_in1k`). Its SHA-256 begins with
  `d63eafa0`, matching the hash in the file name. It uses torchvision's
  parameter names, so `resnet18(pretrained=True, weights=path)` loads it
  directly (`train_framing.py --weights`).

### Results

Shot-level accuracy on the hand-labelled test shots (3 key frames averaged;
95% CI by bootstrap over shots). Filled in by `tools/update_results.py`.

<!-- auto:p1-results -->
| System | Trained on | Accuracy [95% CI] | Mean class acc. | Train time |
|---|---|---|---|---|
| open_clip ViT-B/32 zero-shot (prompts only) | nothing (zero-shot / rule) | 0.412 [0.27–0.55] | 0.379 | — |
| Face size only (OpenCV YuNet), shots with a face | nothing (zero-shot / rule) | 0.158 [0.00–0.32] | 0.155 | — |
| Teacher: prompts + face size | nothing (zero-shot / rule) | 0.431 [0.29–0.57] | 0.396 | — |
| ResNet-18, ImageNet init | teacher labels, 30 epochs, init resnet18_a1_0-d63eafa0.pth | 0.176 [0.08–0.29] | 0.131 | 0.3 h (CPU) |
| ResNet-18, random init | teacher labels, 60 epochs | 0.137 [0.06–0.24] | 0.100 | 0.5 h (CPU) |

51 hand-labelled shots (public clips, 2-fold by time: each shot predicted by the network that did not train on its half; labels `p1_cnn/labels/public_test.labels.json`); majority-class accuracy 0.275. Face size only is scored on the shots where a face was found. Train time is the sum of the two folds.
<!-- /auto:p1-results -->

**What the public-clip run shows.** The teacher (0.43) beats the majority
class (0.27) and its own text cue alone (0.41), but its confidence interval is
wide: 51 shots. Both students do **worse than the majority class** — they
collapsed onto the teacher's most frequent labels (the ImageNet-init student
predicts MS or CS for 65 of 66 shots, the random-init one MS for 49). This is
a data failure, not a model one: each fold trains on 23–33 shots, and because
most sample videos are a single shot, a whole clip lands in one half — fold 0
saw no FS or LS shot at all in training. The validation halves hold 5 shots,
so "best epoch" is noise (every run peaks at 4/5). The face cue alone is
weak (0.16 on the 19 shots with a face): it reads 16 of them one class too
close (CS as ECS, MS as CS). The typical face ratio per class was set by hand,
not fitted, and is evidently a class off for YuNet's boxes; fitting it on
training shots (never the test ones) is the obvious next step. The experiment is the pipeline working end to end on a CPU;
the result that counts needs the hundreds of shots of the episode set.

Confusion matrices: `p1_cnn/results/own/confusion_<system>.png` (public run:
`p1_cnn/results/public/`). Expected
confusions are between neighbours on the scale (ECS↔CS, MS↔FS): the classes
are points on a continuum, and annotators disagree at the boundaries too.

## Interview questions

**Why do residual connections make deep networks trainable?** A block outputs
`x + F(x)`, so its Jacobian is `I + ∂F/∂x`. The identity term carries the
gradient from the loss to early layers unchanged, instead of multiplying it
through dozens of weight matrices where it vanishes or explodes. It also makes
"do nothing" easy to learn (F = 0): adding blocks can't make the network worse
than a shallower one, which is the degradation problem He et al. set out to fix.
`zero_init_residual` starts every block at exactly that point.

**What does BatchNorm do at train vs. inference time?** Training: normalise
each channel with the current batch's mean and variance, then scale and shift
(γ, β), and update running averages of mean and variance. Inference: use the
running averages, so the output for one image doesn't depend on the others in
its batch. Forgetting `model.eval()` means predictions change with batch
composition; tiny batches while training make the statistics noisy.

**Your loss is flat at epoch 1 — what do you check first?** (1) Overfit one
batch: if the loss can't reach ~0 on 32 examples, the bug is in the model or
loss, not the data. (2) The initial loss: with 5 classes it should be about
ln 5 ≈ 1.61; far off means bad init or wrong labels. (3) The learning rate
(too low → flat; too high → flat after a spike or NaN) and whether the
optimiser actually has the parameters. (4) Data: labels shuffled
independently of images, normalisation wrong, every image identical after a
broken transform. (5) `zero_grad` in the loop, and the model in train mode.
