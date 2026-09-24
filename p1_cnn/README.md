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

## Week 5 — ResNet-18 on MovieShots (PyTorch)

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

### Data

MovieShots (Rao et al., *A Unified Framework for Shot Type Classification
Based on Subject Centric Lens*, ECCV 2020): about 46k shots from trailers, with
scale and movement labels. It is released by the authors on request through
the project page; it is never committed to this repo.

```powershell
python p1_cnn/prepare_movieshots.py --root D:\datasets\MovieShots     # -> data/movieshots/index.csv
```

The script prints how many shots it used per split and class, and how many it
skipped and why.

### Training (Kaggle / Colab GPU)

```bash
# the two runs the roadmap asks for
python p1_cnn/train_framing.py --root /kaggle/input/movieshots --index data/movieshots/index.csv --pretrained --tag pretrained --epochs 30
python p1_cnn/train_framing.py --root /kaggle/input/movieshots --index data/movieshots/index.csv --tag scratch --epochs 60
```

On Kaggle: upload MovieShots and a zip of this repo as two private datasets,
turn on the GPU, `pip install -e .`, run `prepare_movieshots.py` then the two
commands above, and download `p1_cnn/results/` and `p1_cnn/checkpoints/`.
Before spending GPU hours, check the loop on a CPU:

```powershell
python p1_cnn/train_framing.py --synthetic --epochs 2 --tag smoke
```

### Results

Test split, shot-level accuracy (3 key frames averaged). Filled in after the runs.

| Model | Init | Epochs | Accuracy | Mean class acc. | Train time |
|---|---|---|---|---|---|
| ResNet-18 (ours) | ImageNet | 30 | — | — | — |
| ResNet-18 (ours) | random | 60 | — | — | — |
| MovieShots paper (scale) | — | — | *copy from the paper's results table* | | |

Confusion matrix: `p1_cnn/results/pretrained/confusion.png`. Expected
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
