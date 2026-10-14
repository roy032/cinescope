"""Scene types and the zero-shot CLIP teacher.

Classes (one per shot; definitions also in p6_pose_video/README.md):

  dialogue      a character scene: one to three people are the subject, no fast
                action (conversation, reaction shots, people walking or working)
  action        fights, chases, explosions, vehicles at speed, fast physical motion
  establishing  a place is the subject: landscapes, exteriors, cityscapes, wide
                interiors where people are absent or tiny
  crowd         many people (about five or more) with no single subject
  text          the frame is there to be read: title and credit cards, logos,
                screens, documents, maps and other graphics filling the frame

The teacher is OpenCLIP ViT-B/32 (LAION-400M) used zero-shot: each class is
described by several prompts; a frame's class score is the mean over its
prompts of the image-text cosine similarity times CLIP's logit scale, and a
softmax over classes gives probabilities. A shot's distribution is the mean
over its sampled frames.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

CLASSES: tuple[str, ...] = ("dialogue", "action", "establishing", "crowd", "text")

PROMPTS: dict[str, list[str]] = {
    "dialogue": ["a film still of two people talking", "a close-up of a person's face in a movie",
                 "a medium shot of a person in a room", "a conversation scene from a film",
                 "a person walking in a hallway"],
    "action": ["an action scene with a fight", "a car chase in a movie", "an explosion in a film",
               "people running and fighting", "a fast action shot with motion blur"],
    "establishing": ["a wide establishing shot of a landscape", "an aerial view of a city",
                     "a landscape with no people", "nature scenery in a film", "the outside of a building"],
    "crowd": ["a crowd of people", "many people walking in a street", "a busy public place full of people",
              "a large group of people in a hall", "a surveillance camera view of many people"],
    "text": ["a title card with text", "white text on a black screen", "a computer screen showing text",
             "a logo on a plain background", "a map graphic on a screen"],
}


def weights_dir() -> Path:
    from ..pose.simplebaseline import weights_dir as wd

    return wd()


class ClipTeacher:
    """open_clip ViT-B-32-quickgelu zero-shot classifier over CLASSES."""

    def __init__(self, weights: str | Path | None = None, model: str = "ViT-B-32-quickgelu"):
        import open_clip
        import torch

        path = Path(weights) if weights else weights_dir() / "vit_b_32-quickgelu-laion400m_e32.pt"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model, pretrained=str(path))
        self.model.eval()
        tok = open_clip.get_tokenizer(model)
        self.prompt_class = [c for c in CLASSES for _ in PROMPTS[c]]
        texts = [p for c in CLASSES for p in PROMPTS[c]]
        with torch.no_grad():
            t = self.model.encode_text(tok(texts))
        self.text = t / t.norm(dim=-1, keepdim=True)
        self.scale = float(self.model.logit_scale.detach().exp())

    def frame_logits(self, frames: list[np.ndarray], bs: int = 32) -> np.ndarray:
        """RGB frames -> (N, n_classes) mean prompt logits per class."""
        import torch
        from PIL import Image

        out = []
        with torch.no_grad():
            for i in range(0, len(frames), bs):
                x = torch.stack([self.preprocess(Image.fromarray(f)) for f in frames[i:i + bs]])
                e = self.model.encode_image(x)
                e = e / e.norm(dim=-1, keepdim=True)
                out.append((self.scale * e @ self.text.T).numpy())
        logits = np.concatenate(out) if out else np.zeros((0, len(self.prompt_class)))
        return np.stack([logits[:, [i for i, pc in enumerate(self.prompt_class) if pc == c]].mean(1)
                         for c in CLASSES], 1)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = np.asarray(x, np.float64)
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def accuracy(pred: dict[str, str], truth: dict[str, str]) -> dict:
    """Shot-level accuracy over shots with a real class (label 'none' excluded),
    plus a per-class breakdown and confusion counts."""
    keys = [k for k, v in truth.items() if v in CLASSES and k in pred]
    correct = [pred[k] == truth[k] for k in keys]
    per = {}
    for c in CLASSES:
        ks = [k for k in keys if truth[k] == c]
        if ks:
            per[c] = {"n": len(ks), "correct": int(sum(pred[k] == c for k in ks))}
    conf = {}
    for k in keys:
        conf.setdefault(truth[k], {}).setdefault(pred[k], 0)
        conf[truth[k]][pred[k]] += 1
    return {"accuracy": float(np.mean(correct)) if keys else float("nan"), "n": len(keys),
            "per_class": per, "confusion": conf}
