# Phase 1 — CNN fundamentals (Weeks 4–5)

**Feature:** every shot gets a framing label (extreme close-up, close-up,
medium, full, long) from a ResNet-18 written from scratch.

```powershell
python p0_classical/analyse.py trailers/Ted_Lasso_120s.mkv --framing p1_cnn/checkpoints/pretrained.pt
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

### Data: our own episodes, labelled automatically

MovieShots (Rao et al., ECCV 2020), the usual benchmark, is released by its
authors on request; its public folder holds only the annotation files. The
classifier is therefore trained on footage we have: full episodes of *Ted
Lasso* and *Lanterns*, with labels made automatically and a test set labelled
by hand. The footage and key frames are never committed.

1. **Shots.** `build_own_shots.py` finds the episodes, copies them into
   `data/episodes/`, cuts them with the Phase 0 shot detector and keeps three
   key frames per shot (25/50/75% of its length; shots under half a second and
   black frames are dropped). Splits are by episode: per show, the last episode
   is test and the one before it validation, so no scene is on both sides.
2. **Automatic labels** (`autolabel.py`, `cinescope/framing/teacher.py`). Two
   independent cues, fused per shot:
   - zero-shot image-text matching with SigLIP (so400m), three prompts per class;
   - the height of the largest face from the Phase 2 detector relative to the
     frame height, turned into a soft class prior (a Gaussian in log space
     around each class's typical ratio). Scale is defined by how much of a
     person the frame shows, so this cue is strong when a face is visible and
     silent when none is.
3. **Hand-labelled test set.** `label_scales.py` builds a keyboard page with
   300 random shots from the test episodes; it never shows a prediction, so
   the labels do not lean towards any system. `eval_own.py` scores every cue,
   the fused teacher and both networks against it.

The networks learn from noisy labels. Whether a student ends up better or worse
than its teacher on the hand labels is the interesting number: better means
the CNN averaged away some of the teacher's errors; worse means it mostly
copied them.

```powershell
RUN_OWN_SHOTS.bat          # find + copy + cut the episodes, build the labelling page, train on Kaggle
python p1_cnn/eval_own.py  # after labelling: every system against the hand labels
```

MovieShots still works unchanged (`prepare_movieshots.py`, then
`kaggle/run_all.py --only p1_movieshots` with its frames attached).

### Training (Kaggle GPU)

`kaggle/jobs/p1_own.py`: auto-labelling (~15 min on a T4), then the two runs
the roadmap asks for, ResNet-18 from ImageNet weights (30 epochs) and from
random init (60), selected on the validation episodes (whose labels are the
teacher's), then predictions for the test episodes. Before spending GPU hours,
check the loop on a CPU:

```powershell
python p1_cnn/train_framing.py --synthetic --epochs 2 --tag smoke
```

### Results

Shot-level accuracy on the hand-labelled test shots (3 key frames averaged).
Filled in by `tools/update_results.py`.

<!-- auto:p1-results -->
| System | Trained on | Accuracy [95% CI] | Mean class acc. | Train time |
|---|---|---|---|---|
| SigLIP zero-shot (prompts only) | nothing | — | — | — |
| Teacher: prompts + face size | nothing | — | — | — |
| ResNet-18, ImageNet init | teacher labels, 30 epochs | — | — | — |
| ResNet-18, random init | teacher labels, 60 epochs | — | — | — |
<!-- /auto:p1-results -->

Confusion matrices: `p1_cnn/results/own/confusion_<system>.png`. Expected
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
