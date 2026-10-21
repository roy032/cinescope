# CineScope

**An X-Ray engine for Hollywood films, built from pixels up.**

Give CineScope a movie or TV episode and it watches the whole thing, then returns an interactive breakdown:

- **Who's on screen when:** a timeline for every character, with screen time and who's talking
- **How it was shot:** every cut, every shot's framing (close-up, medium, wide) and camera moves
- **The film's color palette:** a "movie barcode" of its signature colors
- **Scene search in plain English:** *"the scene where he walks away from the explosion"*
- **VFX tools:** cut actors out of shots, remove objects, convert 2D to stereo 3D
- **3D fly-through:** turn a long single take into a set you can explore from any angle


## Why this project

I'm building CineScope to learn computer vision end to end, from convolution in NumPy to 3D Gaussian splatting and multimodal retrieval. Each phase follows one rule:

> **Implement the core idea from scratch first, then switch to the production library and compare.**

My own implementation is how I learn the idea. The comparison with the library gives real benchmarks, which each phase's README reports.

---

## Roadmap

| Phase | Weeks | Feature | CV area | Status |
|---|---|---|---|---|
| [0](p0_classical/) | 1–3 | Shot cuts, camera moves, color palettes | Classical CV | Built; scored on public clips |
| [1](p1_cnn/) | 4–5 | Shot-framing labels | CNNs | Built; scored on public clips and on 289 hand-labelled shots of held-out episodes (students match their teacher, 0.80) |
| [2](p2_detection/) | 6–8 | Face and prop detection | Object detection | Built; trained on WIDER FACE; ours and production detectors scored on hand-labelled dark/profile/crowd frames (YuNet ahead) |
| [3](p3_tracking/) | 9–11 | Character timeline, who's talking | Tracking, metric learning, audio-visual | Built; scored on hand-labelled public clips |
| [4](p4_segmentation/) | 12–13 | Actor cutouts, object removal | Segmentation, matting, inpainting | Built; compared with SAM 2, MobileSAM, YOLOv8n-seg and OpenCV on public clips |
| [5](p5_3d/) | 14–17 | Camera path, 2D→3D, set fly-through | Calibration, SfM, depth, Gaussian splatting | Built; scored on synthetic ground truth and public clips |
| [6](p6_pose_video/) | 18–20 | Fight breakdowns, scene types | Pose, ViT, VideoMAE | Built; scored on public clips with by-eye labels; from-scratch scene models trail CLIP |
| [7](p7_ocr/) | 21–22 | Subtitles and credits read | OCR (CRNN + CTC, TrOCR) | Built; scored on subtitles burned into public clips |
| [8](p8_search/) | 23–24 | Scene search, chat with the movie | CLIP/SigLIP, multimodal RAG | Built; scored on hand-written queries over public clips (LLM answers optional) |
| [9](p9_production/) | 25–26 | Full-film pipeline, web app | ONNX/TensorRT, FastAPI, React | Built; one-pass pipeline with cache and resume, ONNX Runtime, FastAPI + React app |

## Results

Filled in as each phase ships.

| Phase | Metric | From scratch | Library baseline |
|---|---|---|---|
| 0 | Shot-cut F1 (public clips) | 0.945 | 0.960 (PySceneDetect) |
| 1 | Shot-framing accuracy (hand-labelled held-out episodes, n=289) | 0.796 (ResNet-18, ImageNet init) · 0.775 (random init) | 0.806 (teacher: zero-shot prompts + face size) |
| 2 | Face AP (WIDER FACE Easy / Medium / Hard) | 0.593 / 0.726 / 0.792 | — |
| 2 | Face AP50 on hand-labelled clip frames (dark / profile / crowd) | 0.915 / 0.956 / 0.544 (ours) · 0.912 / 0.936 / 0.524 (ours, log) | 0.989 / 0.980 / 0.605 (YuNet) |
| 3 | Character clustering purity / pair F1 (goldeneye, 8 people) | 0.93 / 0.32 (self-supervised triplet; 15 clusters) | 1.00 / 1.00 (SFace; 8 clusters) |
| 4 | Mask IoU vs. SAM 2 on public clips / IoU on exact-mask composites / CPU time | 0.73 / 0.91 / 0.6 s (GrabCut) · 0.71 / 0.76 / 0.09 s (U-Net) | 1.00 (reference) / 0.97 / 6.0 s (SAM 2 tiny) · 0.92 / 0.96 / 3.8 s (MobileSAM) |
| 5 | SfM reprojection error (px), on the public-clip shots both reconstruct | 0.27 (3 of 12 shots) | 0.68 (COLMAP; 10 of 12 shots) |
| 6 | Scene-type accuracy (27 hand-labelled public-clip shots) | 0.370 (VideoMAE, fine-tuned) | 0.852 (OpenCLIP zero-shot) |
| 7 | Subtitle character error rate (burned-in subtitles, 3 public clips) | 0.460 | 0.478 (PP-OCRv3 rec) / 0.494 (PP-OCRv3 det+rec) |
| 8 | Scene search Recall@5 (46 test queries, 9 public clips pooled) | 0.783 (own exact index) · 0.457 (own IVF-PQ) · 0.000 (own dual encoder) | 0.783 (open_clip + FAISS Flat) · 0.435 (FAISS IVF-PQ) |
| 9 | Time to process a 2-hour film (4-core CPU, extrapolated from goldeneye) | 15.0 h (phase tools one after another, PyTorch) | 5.0 h (one-pass pipeline, ONNX Runtime) |

### Beyond the roadmap: hand the cut to an editor

A breakdown is most useful inside the tools editors already use. `cinescope.editorial`
turns any report into:

- **EDL** (CMX 3600), **FCPXML 1.10** and **OpenTimelineIO**: the original file cut at every
  shot CineScope found, ready to relink in Premiere, DaVinci Resolve, Avid or Final Cut.
  Framing, camera move, scene type and the characters on screen travel along as comments,
  keywords and metadata; scene-type changes become chapter markers. Drop-frame timecode for
  29.97 and 59.94 fps.
- **Shot list CSV** and a **printable storyboard PDF**: every shot's frame, timecode, length
  and labels, with shot numbers that match the cut files.
- **WebVTT subtitles** from the text read off the picture (the web app's player shows them).
- **Pace**: average and median shot length and a cuts-per-minute curve, in the report and the app.

The web app has an Export panel with all of these. The EDL and OTIO files are checked with
OpenTimelineIO's own readers.

---

## Film language

Beyond the roadmap, the pipeline describes a film the way its crew would. These stages
need no models and run in seconds on a CPU (`cinescope/cinema/`):

| Stage | What it reports |
|---|---|
| `look` | Per shot and for the whole clip: lighting key (low-key / high-key), contrast, tonal range, saturation, colour temperature, teal-and-orange grade, symmetry, crushed blacks and blown highlights, and the aspect ratio inside any letterbox bars (2.39:1 scope, 1.85:1 flat, 16:9 ...) |
| `editing` | Recurring camera set-ups; shot/reverse-shot exchanges, montages and long takes |
| `sound` | Loudness in broadcast units (EBU R 128 style integrated LUFS, loudness range, per shot), loud peaks and silences, the share that is speech, and how often the picture cuts on a sound onset compared with chance |
| `story` | Scenes (shots in one place and time: no shared camera set-up and no shared colours across the cut, or a silence), an intensity curve from cutting rate, loudness and motion, the three acts, plot points 1 and 2, the midpoint and the climax |
| `composition` | Per shot, from a saliency map (spectral residual + colour contrast) of its key frames: the centre of interest, rule-of-thirds placement (centre, thirds, power point, edge), left/right balance, headroom and shallow or deep focus; a heat map of where the eye goes |
| `rhythm` | The soundtrack's beat: tempo (BPM) and pulse clarity from the onset envelope's autocorrelation, dynamic-programming beat tracking, the sections that are music with a beat, and how often the picture cuts on the beat compared with chance |

`python -m cinescope.editorial outputs/<clip>/report.json -f html` (or **Export → Breakdown** in the app)
writes a shareable one-file HTML breakdown: the facts, barcode, pace, look, loudness, beat and story structure over time,
the characters and every shot with its frame and timecode. It works offline and can be mailed as is.

---

## Repo structure

```
cinescope/
├── cinescope/            # Shared package used by the phase tools, the pipeline and the app
│   ├── io/               # Video decoding, frame sampling, previews
│   ├── imgproc/ palette/ motion/ shots/   # Phase 0: filters, colour, flow, shot cuts
│   ├── nn/ framing/      # Phase 1: autograd, CNNs, shot-framing classifier
│   ├── detect/ faces/    # Phase 2: CenterNet face detector, face crops and embeddings
│   ├── track/ speaker/   # Phase 3: tracking, clustering, who's talking
│   ├── segment/          # Phase 4: GrabCut, U-Net, matting, inpainting
│   ├── geometry/         # Phase 5: calibration, SfM, depth, stereo, splats
│   ├── pose/ video/      # Phase 6: pose, action events, ViT/VideoMAE scene types
│   ├── ocr/              # Phase 7: CRNN + CTC
│   ├── search/           # Phase 8: CLIP index, scene search, chat
│   ├── pipeline/         # Phase 9: one-pass pipeline (python -m cinescope.pipeline)
│   └── editorial/        # Hand-off to editors: EDL, FCPXML, OTIO, shot list, storyboard, pace
├── p0_classical/ … p9_production/   # One folder per phase: scripts, labels, results and README
├── app/                  # FastAPI back end + React front end (Phase 9)
├── kaggle/               # GPU training jobs that run on Kaggle
├── tests/                # pytest suite, mostly one file per roadmap week
├── tools/                # run_everything.py, fetch_assets.py, update_results.py
├── Dockerfile            # CPU image of the web app and pipeline
├── data/ trailers/ outputs/   # Local only, git-ignored: weights, clips, results
├── LOG.md                # Weekly learning log
├── requirements.txt      # Core + dev dependencies
└── requirements-full.txt # Everything the full pipeline and the web app need
```

Every phase folder has its own README with example output, its metrics and what I learned.

## Getting started

One command sets everything up and runs it: creates `.venv`, installs the dependencies,
downloads the public test clips and library weights, lints, runs the tests, builds the web
app, scores Phases 1–2, runs the whole pipeline on a clip, checks the ONNX export (and
TensorRT, on an NVIDIA GPU) and builds the Docker image when a Docker daemon is present.
Steps that cannot run on the machine are skipped with the reason.

```bash
git clone https://github.com/roy032/cinescope.git
cd cinescope
python tools/run_everything.py              # Windows: double-click RUN_EVERYTHING.bat; Unix: make all
python tools/run_everything.py test pipeline --clip trailers/any_clip.mp4   # just some steps
python tools/run_everything.py kaggle       # the GPU training jobs on Kaggle (hours; needs a Kaggle API token)
```

By hand:

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt -r requirements-full.txt
python tools/fetch_assets.py                        # public clips -> trailers/, weights -> data/weights/
```

Put a video clip in `trailers/` (or `data/clips/`) and run:

```bash
python -m cinescope.io.preview trailers/your_clip.mkv      # thumbnail grid -> outputs/
python p0_classical/analyse.py trailers/your_clip.mkv      # shots, camera moves, palettes
python -m pytest                                           # the test suite
python -m cinescope.pipeline trailers/your_clip.mkv       # every phase in one pass -> outputs/<clip>/report.json
python -m cinescope.editorial outputs/your_clip            # EDL, FCPXML, OTIO, CSV, subtitles, PDF storyboard
```

The web app (FastAPI + React) and the Docker image are described in [p9_production/](p9_production/).
The GPU runs that need your Kaggle account (TensorRT timing, detector retraining) are one command:
see [kaggle/README.md](kaggle/README.md).

**Main tools:** Python · NumPy · PyTorch · OpenCV · PyAV · FAISS/Qdrant · FastAPI · React

---

## Development

```bash
pip install -e ".[video,cv,dl,app,dev]"     # or the requirements files above
ruff check .                                 # lint (CI pins ruff 0.16.8)
python -m pytest -q                          # tests; ones needing an optional library skip
cd app/frontend && npm ci && npm test && npm run build
cinescope trailers/your_clip.mp4             # same as python -m cinescope.pipeline
```

CI runs lint, the Python tests, the front end's tests and build, and a Docker smoke test
on every pull request.

---

## Footage and copyright

- No film footage is committed to this repo. `data/`, `trailers/` and `outputs/` are git-ignored.
- Every tool takes any video as an argument. Scores are measured on public clips, such as
  PySceneDetect's test videos (`goldeneye.mp4`, `testvideo.mp4`) and Intel's sample videos.
- Public demos use open films such as Blender's [*Tears of Steel*](https://mango.blender.org/) and [*Sintel*](https://durian.blender.org/) (CC BY), plus short excerpts only.
- Film titles are the property of their respective owners.

---

## Learning resources

- [Stanford CS231n](https://cs231n.github.io/): deep learning for computer vision
- Richard Szeliski, [*Computer Vision: Algorithms and Applications*](https://szeliski.org/Book/), 2nd ed.
- Hartley & Zisserman, *Multiple View Geometry in Computer Vision*
- Papers for each phase are listed in that phase's README.

---

## Author

**Jayanta Roy** · CS, BRAC University
[GitHub](https://github.com/roy032) · [LinkedIn](https://www.linkedin.com/in/jayanta-roy-b80357407)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup, tests and the house rules.

## License

Code released under the [MIT License](LICENSE).
