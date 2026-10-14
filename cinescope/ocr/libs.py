"""Production OCR models behind the same `read(crops) -> [(text, conf)]`
interface as our CRNN, so the subtitle pipeline can swap recognisers.

  rapidocr   PP-OCRv3 (Baidu PaddleOCR) recogniser, ONNX, bundled in the
             rapidocr-onnxruntime wheel: an SVTR-LCNet recogniser trained on
             millions of real and synthetic Chinese + English lines.
             `RapidOCRFull` also uses its DB text *detector*, for an
             end-to-end baseline that does not depend on our text map.
  opencv     The OpenCV zoo CRNN (VGG + BiLSTM + CTC, `crnn_en.onnx`) through
             cv2.dnn.TextRecognitionModel. It reads single *words*, 100×32,
             lower-case letters and digits only — so lines are cut into words
             at wide column gaps first, and it is scored case- and
             punctuation-blind (charset.fold).
  trocr      TrOCR (Li et al., 2021; ViT encoder + text transformer decoder)
             through HuggingFace transformers, if installed and the weights
             are on disk (`microsoft/trocr-base-printed` saved locally). The
             HuggingFace hub is unreachable from the machine these results
             were made on, so this path is untested here.
"""
from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from .textmap import TextParams, luma, stroke_map


def weights_dir() -> Path:
    return Path(os.environ.get("CINESCOPE_WEIGHTS", Path(__file__).resolve().parents[2] / "data" / "weights"))


class RapidOCRRec:
    name = "rapidocr"
    min_conf = 0.35            # thresholds on its score, checked on the tuning clips
    credits_min_conf = 0.6

    def __init__(self):
        from rapidocr_onnxruntime import RapidOCR

        self.engine = RapidOCR()

    def read(self, crops: Sequence[np.ndarray]) -> list[tuple[str, float]]:
        if not crops:
            return []
        bgr = [np.ascontiguousarray(np.asarray(c)[:, :, ::-1]) if np.asarray(c).ndim == 3
               else np.repeat(np.asarray(c)[:, :, None], 3, 2) for c in crops]
        res, _ = self.engine.text_recognizer(bgr)
        return [(str(t), float(s)) for t, s in res]


class RapidOCRFull:
    """DB detector + recogniser on a whole image: [(box xyxy, text, score)]."""
    name = "rapidocr-det"

    def __init__(self):
        from rapidocr_onnxruntime import RapidOCR

        self.engine = RapidOCR()

    def detect_read(self, rgb: np.ndarray) -> list[tuple[tuple[int, int, int, int], str, float]]:
        res, _ = self.engine(np.ascontiguousarray(rgb[:, :, ::-1]))
        out = []
        for quad, text, score in res or []:
            q = np.asarray(quad)
            out.append(((int(q[:, 0].min()), int(q[:, 1].min()), int(q[:, 0].max()), int(q[:, 1].max())),
                        str(text), float(score)))
        return out


def split_words(crop: np.ndarray, gap: float = 0.2) -> list[np.ndarray]:
    """Cut a line crop into words at column gaps wider than `gap` × height.

    Strokes come from the top-hat of the polarity (bright or dark text) that
    yields the cleaner column profile — the one with more separate runs —
    measured on the middle rows only, so a neighbouring line's descenders do
    not bridge the gaps."""
    y = luma(crop)
    h = y.shape[0]
    mid = slice(int(0.2 * h), int(0.8 * h))
    best = None
    for pol in ("bright", "dark"):
        s = stroke_map(y, int(h / 0.06), TextParams(polarity=pol, tophat_thr=35))[mid]
        idx = np.flatnonzero(s.sum(0) > 0)
        if not idx.size:
            continue
        cuts = np.flatnonzero(np.diff(idx) > gap * h)
        spans = [(int(a), int(b)) for a, b in zip(np.r_[idx[0], idx[cuts + 1]], np.r_[idx[cuts] + 1, idx[-1] + 1],
                                                  strict=True)]
        if best is None or len(spans) > len(best):
            best = spans
    if not best:
        return [crop]
    pad = int(0.1 * h)
    return [crop[:, max(0, a - pad):b + pad] for a, b in best if b - a > 0.15 * h]


class OpenCVCRNN:
    name = "opencv-crnn"

    def __init__(self, path: str | None = None):
        import cv2

        path = path or str(weights_dir() / "crnn_en.onnx")
        self.model = cv2.dnn.TextRecognitionModel(path)
        self.model.setDecodeType("CTC-greedy")
        self.model.setVocabulary(list("0123456789abcdefghijklmnopqrstuvwxyz"))
        self.model.setInputParams(1.0 / 127.5, (100, 32), (127.5, 127.5, 127.5))

    def read(self, crops: Sequence[np.ndarray]) -> list[tuple[str, float]]:
        import cv2

        out = []
        for c in crops:
            words = []
            for w in split_words(np.asarray(c)):
                g = cv2.cvtColor(np.ascontiguousarray(w), cv2.COLOR_RGB2GRAY) if w.ndim == 3 else w
                words.append(self.model.recognize(g))
            out.append((" ".join(x for x in words if x), 1.0))
        return out


class TrOCR:
    name = "trocr"

    def __init__(self, path: str | None = None):
        try:
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        except ImportError as e:
            raise RuntimeError("TrOCR needs `pip install transformers`") from e
        path = path or os.environ.get("TROCR_MODEL", str(weights_dir() / "trocr-base-printed"))
        if not Path(path).exists():
            raise RuntimeError(f"TrOCR weights not found at {path}: save microsoft/trocr-base-printed "
                               "there (processor + model, save_pretrained) or set TROCR_MODEL")
        self.processor = TrOCRProcessor.from_pretrained(path)
        self.model = VisionEncoderDecoderModel.from_pretrained(path).eval()

    def read(self, crops: Sequence[np.ndarray]) -> list[tuple[str, float]]:
        import torch
        from PIL import Image

        if not crops:
            return []
        ims = [Image.fromarray(np.asarray(c)).convert("RGB") for c in crops]
        with torch.no_grad():
            px = self.processor(images=ims, return_tensors="pt").pixel_values
            out = self.model.generate(px, max_new_tokens=64, output_scores=True,
                                      return_dict_in_generate=True)
        texts = self.processor.batch_decode(out.sequences, skip_special_tokens=True)
        return [(t, 1.0) for t in texts]


def make_reader(name: str, ckpt: str | None = None, decoder: str = "greedy"):
    """Recogniser by name: ours | rapidocr | opencv | trocr."""
    if name == "ours":
        from .crnn import Recognizer

        if ckpt is None:
            ckpt = str(Path(__file__).resolve().parents[2] / "p7_ocr" / "checkpoints" / "crnn.pt")
        return Recognizer.load(ckpt, decoder=decoder)
    return {"rapidocr": RapidOCRRec, "opencv": OpenCVCRNN, "trocr": TrOCR}[name]()
