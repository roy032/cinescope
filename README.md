# 🎬 CineScope

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
| [0](p0_classical/) | 1–3 | Shot cuts, camera moves, color palettes | Classical CV | 🚧 Built, awaiting test labels |
| [1](p1_cnn/) | 4–5 | Shot-framing labels | CNNs | 🚧 Built, awaiting MovieShots training runs |
| [2](p2_detection/) | 6–8 | Face and prop detection | Object detection | 🚧 Built, awaiting WIDER FACE training |
| [3](p3_tracking/) | 9–11 | Character timeline, who's talking | Tracking, metric learning, audio-visual | ⏳ Planned |
| [4](p4_segmentation/) | 12–13 | Actor cutouts, object removal | Segmentation, matting, inpainting | ⏳ Planned |
| [5](p5_3d/) | 14–17 | Camera path, 2D→3D, set fly-through | Calibration, SfM, depth, Gaussian splatting | ⏳ Planned |
| [6](p6_pose_video/) | 18–20 | Fight breakdowns, scene types | Pose, ViT, VideoMAE | ⏳ Planned |
| [7](p7_ocr/) | 21–22 | Subtitles and credits read | OCR (CRNN + CTC, TrOCR) | ⏳ Planned |
| [8](p8_search/) | 23–24 | Scene search, chat with the movie | CLIP/SigLIP, multimodal RAG | ⏳ Planned |
| [9](p9_production/) | 25–26 | Full-film pipeline, web app | ONNX/TensorRT, FastAPI, React | ⏳ Planned |

## Results

Filled in as each phase ships.

| Phase | Metric | From scratch | Library baseline |
|---|---|---|---|
| 0 | Shot-cut F1 | — | — |
| 1 | Shot-framing accuracy (MovieShots) | — | — |
| 2 | Face AP (WIDER FACE Easy) | — | — |
| 3 | Character clustering purity | — | — |
| 4 | Mask IoU / speed vs. SAM 2 | — | — |
| 5 | SfM reprojection error (px) | — | — |
| 6 | Scene-type accuracy | — | — |
| 7 | Subtitle character error rate | — | — |
| 8 | Scene search Recall@5 | — | — |
| 9 | Time to process a 2-hour film | — | — |

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

```bash
git clone https://github.com/roy032/cinescope.git
cd cinescope
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Put a video clip in `trailers/` (or `data/clips/`) and run:

```bash
python -m cinescope.io.preview trailers/your_clip.mkv      # thumbnail grid -> outputs/
python p0_classical/analyse.py trailers/your_clip.mkv      # shots, camera moves, palettes
python -m pytest                                           # the test suite
```

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
