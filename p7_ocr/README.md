# Phase 7 — Reading text: subtitles and credits (Weeks 21–22)

**Feature:** CineScope reads the burned-in subtitles of any clip into a timed
`.srt`, and lists the other text that appears on screen (title cards, rolling
credits, signs and screens) in `credits.txt`. The recogniser (a CRNN with a
CTC loss written here) and the subtitle detector/tracker are built from
scratch, then compared with PaddleOCR's PP-OCRv3 and the OpenCV zoo CRNN.

```powershell
python p7_ocr/read_subtitles.py trailers/goldeneye.mp4
# -> outputs/goldeneye/subtitles.srt, credits.txt, credits.json
python p7_ocr/read_subtitles.py trailers/goldeneye.mp4 --reader rapidocr      # library recogniser
```

The CRNN checkpoint (`p7_ocr/checkpoints/crnn.pt`, 12 MB with optimiser state, not committed) comes from
`train_crnn.py` below; about 95 minutes on a shared laptop-class CPU.

## Week 21 — recognition from scratch

| Built | Module | Checked against |
|---|---|---|
| CTC loss: batched alpha recursion in log space (autograd backward) | `cinescope/ocr/ctc.py` `ctc_loss` | `torch.nn.functional.ctc_loss`: value and gradient equal to 1e-10 (float64), all three reductions, padded and concatenated targets, repeated labels, `inf` for infeasible alignments |
| CTC forward-backward in NumPy with the closed-form gradient y − occupancy/p | `ctc.py` `ctc_numpy` | brute-force sum over every path (T=5); finite differences; PyTorch's gradient to 1e-9 |
| Greedy (best-path) and prefix beam-search decoding | `ctc.py` | a case where the best label is not the best path (p 0.64 vs 0.36); beam probabilities equal brute-force sums |
| Levenshtein distance, CER, WER (line and corpus level) | `cinescope/ocr/metrics.py` | hand-worked cases |
| Alphabet (75 symbols + blank), text normalisation | `cinescope/ocr/charset.py` | |
| Synthetic subtitle lines: system fonts minus held-out ones, white/yellow/grey text, outline, shadow, translucent box, backgrounds cropped from video frames, blur, down-sampling, JPEG, noise, crops around the ink with random margins | `cinescope/ocr/synth.py` | by eye (contact sheets) |
| CRNN: 7 conv layers (32 px high → 1, width stride 4) + 2-layer BiLSTM + linear, 0.99 M parameters | `cinescope/ocr/crnn.py` | a tiny model overfits two images with our CTC (test) |
| Training on CPU | `p7_ocr/train_crnn.py` | validation CER on unseen text in held-out fonts |

**How CTC works.** The network emits a distribution over the alphabet plus a
*blank* at each of T columns. A label is the *collapse* of a path (merge
repeats, then drop blanks: `HH-e-ll-llo` → `Hello`), and the loss is −log of
the summed probability of every path that collapses to it. Interleaving blanks
gives an extended label of length 2S+1, and the sum is a dynamic programme:
alpha_t(s) = (alpha_{t−1}(s) + alpha_{t−1}(s−1) + alpha_{t−1}(s−2)) · y_t(l'_s),
where the skip from s−2 is allowed only between *different* characters —
"ll" needs a blank in between, which is also why a line needs at least one
output step per character plus one per doubled letter. Everything is in log
space (logaddexp), or the product of a few dozen probabilities underflows.

### Data and training

- **Text:** *Pride and Prejudice* (public domain), downloaded once from a
  GitHub mirror of Project Gutenberg into `data/text/`: random word spans from
  the first 90% (80% of lines), plus shuffled words, numbers and random
  characters (20%) so the network cannot lean on English. Validation uses the
  last 10% of the book. **The subtitle test text is a different book**
  (Sherlock Holmes).
- **Fonts:** the 74 font files found under `/usr/share/fonts` (Liberation,
  FreeFont, Carlito/Caladea, Inter, …) minus **DejaVu, held out**: validation
  and one test style use only DejaVu.
- **Backgrounds:** 240 frames from six clips — vtest, classroom,
  face-demographics-walking, head-pose-face-detection, people-detection,
  one-by-one-person-detection. **goldeneye, testvideo and Megamind are never
  used** — they are the test clips.
- **Schedule:** AdamW 1e-3, 200-step warm-up, cosine to 3,000 steps of 32
  lines (96k rendered lines), gradient clipping 5. CPU, three sessions of
  ≤ 38 minutes each (`--resume`). For the last 900 steps the crops were
  changed to be cut around the ink rather than the font's line height, after
  the first checkpoint read clean real crops badly (see *What broke*).

<!-- results: p7_ocr/results/crnn_train_log.json -->
| Step | Train loss | Val CER | Val WER |
|---|---|---|---|
| 442 | 2.98 | 0.772 | 0.979 |
| 1000 | 0.68 | 0.080 | 0.262 |
| 2115 | 0.25 | 0.030 | 0.120 |
| 3000 (ink crops) | 0.30 | 0.035 | 0.130 |

Our CTC costs 39 ms per forward+backward at T=75, B=32 on one core, against
15 ms for PyTorch's fused kernel: a Python loop over T against one C++ kernel.
It was used for all training anyway.

### Recognition alone: line CER on rendered test crops

`eval_lines.py`: 400 Sherlock Holmes lines rendered in the held-out DejaVu
fonts over frames of the three test clips — no detection or tracking, so this
isolates the recogniser.

<!-- results: p7_ocr/results/lines_test.json -->
| Recogniser | CER | CER folded¹ | WER | lines exact | ms/line |
|---|---|---|---|---|---|
| **Ours**, CRNN 1 M params, greedy | 0.027 | 0.016 | 0.118 | 0.532 | 17 |
| Ours, prefix beam search (8) | 0.027 | 0.016 | 0.117 | 0.540 | 80 |
| PP-OCRv3 recogniser (rapidocr-onnxruntime) | **0.019** | **0.007** | **0.098** | **0.695** | 337 |
| OpenCV zoo CRNN (`crnn_en.onnx`), words cut by column gaps | 0.644 | 0.591 | 0.921 | 0.000 | 261 |

¹ lower-case letters and digits only. Timings are on a heavily shared CPU.
Beam search buys almost nothing without a language model: the greedy path
already carries most of the probability mass on clean crops. The OpenCV CRNN
reads single words, lower-case only, and our word splitter (column gaps)
is the weak link — it gets the word count right on 43% of tuning lines.

## Week 22 — subtitles in video

| Built | Module |
|---|---|
| Text map: morphological top-hat (bright and dark), "near the local peak" test, stroke density and on/off transitions per row window, row/column projections, two-line splitting at the gap between line cores | `cinescope/ocr/textmap.py` |
| Line tracking over time (box IoU + stroke IoU), temporal median of each track's crops, per-line reading, grouping lines into cues, de-duplication, timings | `cinescope/ocr/subtitles.py` |
| .srt read/write, reading-speed schedule, cue matching and scoring (CER with misses and false cues, timing errors) | `cinescope/ocr/srt.py` |
| Credits / on-screen text: whole-frame lines, vertical scroll from row-profile cross-correlation, tracks in page coordinates, best reading per track | `cinescope/ocr/credits.py` |
| Library readers behind one interface: PP-OCRv3 recogniser, PP-OCRv3 detector+recogniser, OpenCV zoo CRNN, TrOCR (optional) | `cinescope/ocr/libs.py` |
| Test-set builder (burns known subtitles into any clip), credits clip builder, evaluators | `p7_ocr/make_testset.py`, `make_credits_clip.py`, `eval_subtitles.py`, `eval_credits.py`, `eval_lines.py` |

**Why a subtitle is easy to find.** It is thin bright strokes (top-hat) that
alternate with background many times along a row (a comb, not an edge), laid
out horizontally, near the bottom centre — and it **stays still while the
picture moves**. Each line becomes a *track*: the next sample continues it if
its box overlaps and its stroke pixels agree (IoU). When the subtitle changes,
the strokes change even if the box does not. Tracking lines rather than whole
frames matters on static shots, where texture mistaken for text is as still
as a subtitle. The **temporal median** of a track's crops keeps the text
(identical in every frame) and flattens the moving background, so the
recogniser reads one clean image per line. Lines that come and go together
form a cue; a cue first seen at sample t_a and last at t_b gets
[t_a − P/2, t_b + P/2] with P = 1/8 s.

### The test set any clip can produce

`make_testset.py` burns known subtitles into a clip with FFmpeg/libass, so
any video becomes a subtitle test with exact text and timings:

- **Text:** 400 dialogue lines from *The Adventures of Sherlock Holmes*
  (`labels/sherlock_lines.txt`), timed at reading speed (17 chars/s,
  1.2–3.2 s, gaps 0.25–1.2 s), long ones wrapped to two lines.
- **Styles:** `white` (Arial → Liberation Sans, thin black outline), `yellow`
  (**DejaVu Sans Bold — a font never seen in training**, drop shadow),
  `boxed` (Liberation Serif in a translucent box).
- **Test clips:** goldeneye (82 s), testvideo (24 s), Megamind (11 s) — 122
  cues, 4,526 characters. No frame of these clips was a training background.
- **Tuning clips:** the same on people-detection, classroom,
  face-demographics-walking and head-pose-face-detection (training-background
  clips) with held-out *Pride and Prejudice* lines. Every threshold
  (text map, tracking, confidence) was chosen there.

### Results — burned-in subtitles (test clips)

`python p7_ocr/eval_subtitles.py --readers ours rapidocr opencv rapidocr-det`.
**CER** is over the whole clip: a missed cue counts all its characters as
deleted, a false cue (text that is not a subtitle) all its characters as
inserted — so it can exceed 1. *CER on found cues* isolates reading.

<!-- results: p7_ocr/results/subtitles_test.json -->
| Reader (same detector/tracker for the first three) | CER | CER on found cues | cue recall | cue precision | start err (s) | end err (s) | CER goldeneye | CER testvideo | CER Megamind |
|---|---|---|---|---|---|---|---|---|---|
| **Ours** (CRNN, greedy) | **0.460** | 0.235 | 0.902 | **0.827** | 0.252 | 0.150 | 0.315 | **0.961** | 0.405 |
| PP-OCRv3 recogniser on our lines | 0.478 | 0.224 | 0.967 | 0.781 | 0.153 | 0.182 | 0.321 | 1.162 | 0.117 |
| OpenCV zoo CRNN on our lines (folded²) | 1.026 | 0.633 | 0.992 | 0.419 | 0.185 | 0.230 | 0.984 | 1.132 | 1.290 |
| PP-OCRv3 detector + recogniser, every frame (our grouping) | 0.494 | **0.201** | **1.000** | 0.767 | **0.064** | **0.079** | **0.249** | 1.528 | **0.007** |

² The OpenCV CRNN has no capitals, punctuation or spaces, so its row is
scored on lower-case letters and digits. By style (ours / PP-OCRv3 rec):
white 0.451 / 0.387, yellow (unseen font) 0.369 / 0.407, boxed 0.554 / 0.631.

Per-clip, the story is clearer than the pooled number:

- **goldeneye** (dark interiors, computer screens): ours 0.315 vs the
  library recogniser 0.321 on the same lines. Most of the residual is not
  reading: the clip's own on-screen text (`PREFS`, `FBI HEADQUARTERS`, …) sits
  in the subtitle band and is read as extra cues, and tracks that break in
  the middle of a cue leave duplicate fragments, which count as false cues.
- **testvideo** carries a burned-in watermark (`Copyright © 2013 Brandon
  Castellano  www.bcastell.com`) exactly where subtitles go; the two lines
  touch, and every reader returns watermark + subtitle mixtures. CER > 1 for
  every reader; it is a fair failure, not a scoring bug.
- **Megamind** (3 cues per style): the library detector is perfect; ours
  splits two of nine cues into fragments with different readings.

Our recogniser is within 1–2 points of PP-OCRv3 on reading (0.235 vs 0.224 on
found cues; 0.027 vs 0.019 on rendered lines) with 1 M parameters trained for
95 CPU-minutes on synthetic data, against a model trained on millions of real
lines. PP-OCRv3's characteristic error is dropped spaces (`Thenext room`); ours
are i/l/I confusions and letters lost at crop edges. The library's own detector
gives the best timing (it never fragments a track because it reads every
frame) at 16× our pipeline's run time.

**Tuning-set numbers** (`results/subtitles_tune.json`, 6 videos, 133 cues):
ours 0.073 CER, cue precision 0.99, start/end error 0.046/0.041 s — the
confidence threshold 0.88 was chosen here (0.35 → 0.194, 0.55 → 0.100,
0.8 → 0.075, 0.88 → 0.073; beam search 0.076). PP-OCRv3's recogniser on the
same lines: 0.033. The gap between tuning and test (0.07 → 0.46) is the
honest headline of this phase: the tuning clips are surveillance-style
footage with no text of their own; film footage has screens, signs and
watermarks in the subtitle band, and darker, busier backgrounds.

**A first test run** (`results/subtitles_test_v1.json`, PP-OCRv3 recogniser
only) exposed two text-map failures — white subtitles over a light wall
became solid blobs, and texture touching a line made its band too tall and it
was dropped. Both were reproduced on new *tuning* clips
(face-demographics-walking, head-pose-face-detection), fixed there (the
"near the local peak" stroke test and trimming over-tall bands to the line
core), and the test rerun once. That run's numbers were CER 0.420, cue recall
0.77, start error 0.33 s; after the fix, recall 0.97 and start error 0.15 s,
but CER 0.478, because the better detector also finds more of the clips'
non-subtitle text. Both runs are reported because the test set influenced
which failures got fixed.

### Credits and on-screen text

`make_credits_clip.py` builds a 52 s clip with two static title cards over
Megamind frames and rolling credits (37 lines, 60 px/s); the tuning version
uses people-detection frames and other names. `credits.txt` is scored line by
line: a true line is found when an output line reads it with CER ≤ 0.2.

<!-- results: p7_ocr/results/credits_credits_test.json, credits_goldeneye_text.json -->
| Reader | Credits found (of 37) | precision | mean CER | GoldenEye real text found (of 14, folded) | precision | mean CER |
|---|---|---|---|---|---|---|
| **Ours** | 34 | 0.971 | 0.098 | 4 | 0.167 | 0.561 |
| PP-OCRv3 recogniser | **35** | **1.000** | **0.063** | **7** | **0.500** | **0.466** |
| OpenCV zoo CRNN (folded) | 6 | 0.150 | 0.602 | 3 | 0.014 | 0.532 |

All three use our text map and tracks. The GoldenEye labels
(`labels/goldeneye_text.json`) were made by eye from full-resolution frames
before any reader ran on the clip: the computer screen (`Send Spike`,
`Initiate Search Program`, `Enter Password`, the menu bar), the FBI screen
and `Better Luck Next Time / Slug-heads!`.

The rolling part of the synthetic credits is solved by both CTC readers; the
misses are on the two title cards over film frames — the large serif title
`THE SPECKLED BAND` (missed by both) and, for ours, the director card
(`prected by ee e`). Real on-screen text is not solved:
our CRNN reads `lnitiate Seurch Proqrum`, `FEI HEADOUARTERS`,
`coMFUTER FRAUD DVSON` — close, but over the 0.2 CER bar — because the
computer screen uses a squared display font, low-resolution and slightly
blurred, unlike any training font; and 20 of its 24 output lines are texture
or fragments. PP-OCRv3, trained on real photos, reads twice as many
(`FBI HEADQUARTERS`, `Slug-heads!`), but also merges the menu buttons into
`BBURNUUTLINEUFTLINEMESSHGE`. Ours ran 5× faster (120 s vs 649 s).

### TrOCR

TrOCR (Li et al., 2021) is a ViT encoder with an autoregressive text decoder,
trained on printed and handwritten lines; it needs no CTC because the decoder
emits characters one at a time and stops itself. `--reader trocr` loads it
through `transformers` from a local folder (`TROCR_MODEL`, or
`data/weights/trocr-base-printed`), but the HuggingFace hub is blocked on the
machine these results were made on, so the path is written and untested and
there is no TrOCR row. Expect it to read better than both CTC models on
clean crops and to be one to two orders of magnitude slower on CPU (334 M
parameters for the base model, one decoder pass per character).

## What broke, and what it taught

- **CPU contention decides speed, not FLOPs.** On a machine with load 15 on
  4 cores, two PyTorch threads ran a training step 3× *slower* than one
  thread. All later runs used `--threads 1`.
- **A recogniser can ace its validation set and fail on real crops.** After
  2,100 steps the CRNN read synthetic validation lines at 3% CER and real
  burned-in crops at 68% (three tuning videos). The crops looked clean; the difference was
  geometry: training crops were cut from the font's line height (always some
  space above the capitals), the detector cuts around the ink. Re-cutting the
  synthetic crops around the ink with random margins and fine-tuning for 900
  steps brought the tuning-clip CER to 0.07. Dumping the actual crops the
  pipeline sends to the recogniser found it in minutes.
- **The same grey conversion.** Training used PIL's luma (0.299 R + 0.587 G
  + 0.114 B); inference first averaged the channels, turning yellow text
  (luma 219) into grey 177.
- **CTC's blank plateau.** A freshly initialised CRNN first learns to output
  only blanks (loss drops 15×, CER stays 100%), and only then starts placing
  letters; the tiny overfit test shows it.
- **Confidence is reader-specific.** Our score (mean per-step max
  probability) and PP-OCR's are on different scales; texture read as
  `-mmmm` or `tm` by our CRNN mostly scores below 0.8, real lines above 0.9. Each
  reader carries its own threshold, chosen on the tuning clips.

## Known limitations

- Subtitles are looked for in the bottom 38% of the frame, centred; top-of-
  frame subtitles (used when the bottom has text) are missed.
- Any other text in that band — a watermark, a screen, a sign — is read as a
  subtitle cue. A style model (colour, outline, font size consistent across
  cues) would separate them.
- Track fragmentation: when a line's detection drops out for more than one
  sample, the cue is split and the pieces may be read differently; the
  de-duplication only merges near-identical readings.
- English alphabet only; italics are read but not marked as `<i>`.

## Interview questions

**Why CTC rather than per-character classification?** Per-character training
needs each character's position, which nobody labels. CTC needs only the
string: it marginalises over every alignment of the string to the T outputs,
so the network learns where characters are as a by-product. The cost is the
conditional-independence assumption (each step's output is independent given
the image), which is why CTC models gain from an external language model and
why attention decoders (TrOCR) can do better on ambiguous text.

**Why does CTC need the blank?** For two reasons: to emit nothing between
characters (most columns are between letters), and to separate genuinely
repeated characters — without a blank, `ll` would collapse to `l`.

**Greedy vs beam search?** Greedy picks the best path; the best *label* is the
sum over many paths. Prefix beam search keeps, for each candidate prefix, the
probability of paths ending in blank and in its last character separately (so
`l`+`l` is only a new letter after a blank) and merges paths that collapse to
the same prefix. It matters most with a language model in the beam; without
one, on clean crops, it changed our CER by 0.0006.

**How would you detect subtitles without a learned detector?** They are
high-contrast thin strokes (top-hat), dense alternations along rows (a comb,
not an edge), horizontally laid out, near the bottom centre, and static over
a second or more while the image changes. Temporal stability also gives a
cleaner image to read: the median over the frames of one subtitle.
