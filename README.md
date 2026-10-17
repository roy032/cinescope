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

---

## Repo structure

```
cinescope/
├── cinescope/            # Shared pipeline package used by the app
│   ├── io/               # Video decoding, frame sampling
│   ├── shots/            # Shot detection and classification
│   ├── faces/            # Detection, tracking, embeddings
│   └── ...
├── p0_classical/         # Phase notebooks, experiments and README
├── p1_cnn/
├── ...
├── p9_production/
├── app/                  # FastAPI back end + React front end (Phase 9)
├── data/                 # Local only — git-ignored
├── assets/               # GIFs and images for READMEs
├── LOG.md                # Weekly learning log
└── requirements.txt
```

Every phase folder has its own README with a GIF, its metrics and what I learned.

---

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
```

The web app (FastAPI + React) and the Docker image are described in [p9_production/](p9_production/).

**Main tools:** Python · NumPy · PyTorch · OpenCV · PyAV · FAISS/Qdrant · FastAPI · React

---

## Footage and copyright

- No film footage is committed to this repo. `data/` is git-ignored.
- Development uses trailers and short clips.
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

## License

Code released under the [MIT License](LICENSE).
