"""Light-ASD (Liao et al., CVPR 2023) -- the library baseline for Week 11.

The network below is vendored, with only formatting changes, from
https://github.com/Junhua-Liao/Light-ASD (model/Model.py, model/Encoder.py,
model/Classifier.py, loss.py), released under the MIT License:

    MIT License
    Copyright (c) 2023 Liao Junhua
    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:
    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

What it is: a 1 M-parameter network. A visual encoder (3-D convolutions split
into 2-D spatial and 1-D temporal parts) reads 112x112 grayscale face crops at
25 fps; an audio encoder (2-D convolutions split the same way) reads 13 MFCCs
at 100 fps; the two 128-d per-frame embeddings are *added* and a
bidirectional GRU (two one-way GRUs) produces a per-frame speaking logit.
Trained on AVA-ActiveSpeaker (weights: `light_asd_ava.model`).

`LightASD.score(asd_crops, times, audio)` reproduces the inference recipe of
the authors' Columbia_test.py: crops resampled to 25 fps, MFCCs from
python_speech_features' recipe (`audio.mfcc`, verified identical), scores
averaged over chunkings of 1-6 s.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

from .audio import SR, mfcc


class _AudioBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.relu = nn.ReLU()
        self.m_3 = nn.Conv2d(cin, cout, kernel_size=(3, 1), padding=(1, 0), bias=False)
        self.bn_m_3 = nn.BatchNorm2d(cout, momentum=0.01, eps=0.001)
        self.t_3 = nn.Conv2d(cout, cout, kernel_size=(1, 3), padding=(0, 1), bias=False)
        self.bn_t_3 = nn.BatchNorm2d(cout, momentum=0.01, eps=0.001)
        self.m_5 = nn.Conv2d(cin, cout, kernel_size=(5, 1), padding=(2, 0), bias=False)
        self.bn_m_5 = nn.BatchNorm2d(cout, momentum=0.01, eps=0.001)
        self.t_5 = nn.Conv2d(cout, cout, kernel_size=(1, 5), padding=(0, 2), bias=False)
        self.bn_t_5 = nn.BatchNorm2d(cout, momentum=0.01, eps=0.001)
        self.last = nn.Conv2d(cout, cout, kernel_size=(1, 1), padding=(0, 0), bias=False)
        self.bn_last = nn.BatchNorm2d(cout, momentum=0.01, eps=0.001)

    def forward(self, x):
        x3 = self.relu(self.bn_m_3(self.m_3(x)))
        x3 = self.relu(self.bn_t_3(self.t_3(x3)))
        x5 = self.relu(self.bn_m_5(self.m_5(x)))
        x5 = self.relu(self.bn_t_5(self.t_5(x5)))
        return self.relu(self.bn_last(self.last(x3 + x5)))


class _VisualBlock(nn.Module):
    def __init__(self, cin, cout, is_down=False):
        super().__init__()
        self.relu = nn.ReLU()
        st = (1, 2, 2) if is_down else (1, 1, 1)
        self.s_3 = nn.Conv3d(cin, cout, kernel_size=(1, 3, 3), stride=st, padding=(0, 1, 1), bias=False)
        self.bn_s_3 = nn.BatchNorm3d(cout, momentum=0.01, eps=0.001)
        self.t_3 = nn.Conv3d(cout, cout, kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=False)
        self.bn_t_3 = nn.BatchNorm3d(cout, momentum=0.01, eps=0.001)
        self.s_5 = nn.Conv3d(cin, cout, kernel_size=(1, 5, 5), stride=st, padding=(0, 2, 2), bias=False)
        self.bn_s_5 = nn.BatchNorm3d(cout, momentum=0.01, eps=0.001)
        self.t_5 = nn.Conv3d(cout, cout, kernel_size=(5, 1, 1), padding=(2, 0, 0), bias=False)
        self.bn_t_5 = nn.BatchNorm3d(cout, momentum=0.01, eps=0.001)
        self.last = nn.Conv3d(cout, cout, kernel_size=(1, 1, 1), padding=(0, 0, 0), bias=False)
        self.bn_last = nn.BatchNorm3d(cout, momentum=0.01, eps=0.001)

    def forward(self, x):
        x3 = self.relu(self.bn_s_3(self.s_3(x)))
        x3 = self.relu(self.bn_t_3(self.t_3(x3)))
        x5 = self.relu(self.bn_s_5(self.s_5(x)))
        x5 = self.relu(self.bn_t_5(self.t_5(x5)))
        return self.relu(self.bn_last(self.last(x3 + x5)))


class _VisualEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = _VisualBlock(1, 32, is_down=True)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.block2 = _VisualBlock(32, 64)
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.block3 = _VisualBlock(64, 128)
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))

    def forward(self, x):
        x = self.pool1(self.block1(x))
        x = self.pool2(self.block2(x))
        x = self.block3(x).transpose(1, 2)
        B, T, C, W, H = x.shape
        x = self.maxpool(x.reshape(B * T, C, W, H))
        return x.view(B, T, C)


class _AudioEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = _AudioBlock(1, 32)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 1, 3), stride=(1, 1, 2), padding=(0, 0, 1))
        self.block2 = _AudioBlock(32, 64)
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 1, 3), stride=(1, 1, 2), padding=(0, 0, 1))
        self.block3 = _AudioBlock(64, 128)

    def forward(self, x):
        x = self.pool1(self.block1(x))
        x = self.pool2(self.block2(x))
        x = self.block3(x)
        return torch.mean(x, dim=2, keepdim=True).squeeze(2).transpose(1, 2)


class _BGRU(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.gru_forward = nn.GRU(input_size=c, hidden_size=c, num_layers=1, bidirectional=False,
                                  bias=True, batch_first=True)
        self.gru_backward = nn.GRU(input_size=c, hidden_size=c, num_layers=1, bidirectional=False,
                                   bias=True, batch_first=True)
        self.gelu = nn.GELU()

    def forward(self, x):
        x, _ = self.gru_forward(x)
        x = torch.flip(self.gelu(x), dims=[1])
        x, _ = self.gru_backward(x)
        return self.gelu(torch.flip(x, dims=[1]))


class _ASDModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.visualEncoder = _VisualEncoder()
        self.audioEncoder = _AudioEncoder()
        self.GRU = _BGRU(128)

    def forward_visual_frontend(self, x):
        B, T, W, H = x.shape
        x = x.view(B, 1, T, W, H)
        x = (x / 255 - 0.4161) / 0.1688
        return self.visualEncoder(x)

    def forward_audio_frontend(self, x):
        return self.audioEncoder(x.unsqueeze(1).transpose(2, 3))

    def forward_audio_visual_backend(self, x1, x2):
        return torch.reshape(self.GRU(x1 + x2), (-1, 128))


class _LossAV(nn.Module):
    def __init__(self):
        super().__init__()
        self.FC = nn.Linear(128, 2)


class _ASD(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _ASDModel()
        self.lossAV = _LossAV()
        self.lossV = _LossAV()


class LightASD:
    DURATIONS = (1, 1, 1, 2, 2, 2, 3, 3, 4, 5, 6)     # Columbia_test.py's "more reliable" set

    def __init__(self, weights: str = "light_asd_ava.model"):
        from ..faces.detectors import weight_path

        self.net = _ASD()
        state = torch.load(str(weight_path(weights)), map_location="cpu", weights_only=False)
        state = {k.replace("module.", ""): v for k, v in state.items()}
        missing, unexpected = self.net.load_state_dict(state, strict=False)
        if missing:
            raise RuntimeError(f"Light-ASD weights missing keys: {missing[:5]}")
        self.net.eval()

    @torch.no_grad()
    def _run(self, a: np.ndarray, v: np.ndarray) -> np.ndarray:
        m = self.net.model
        ea = m.forward_audio_frontend(torch.FloatTensor(a).unsqueeze(0))
        ev = m.forward_visual_frontend(torch.FloatTensor(v).unsqueeze(0))
        out = m.forward_audio_visual_backend(ea, ev)
        logits = self.net.lossAV.FC(out)
        return logits[:, 1].numpy()                    # what Columbia_test.py thresholds at 0

    def score(self, crops: np.ndarray, t: np.ndarray, audio: np.ndarray, sr: int = SR) -> np.ndarray:
        """crops (T, 112, 112) Light-ASD crops at track frame times t (seconds);
        audio: the clip's mono 16 kHz track. Returns a logit per track frame."""
        t = np.asarray(t, dtype=np.float64)
        t0, t1 = t[0], t[-1] + 1 / 25
        grid = np.arange(t0, t1, 1 / 25)                         # 25 fps
        vid = crops[np.clip(np.searchsorted(t, grid, side="right") - 1, 0, len(t) - 1)]
        seg = audio[int(t0 * sr):int(np.ceil(t1 * sr))] * 32768.0  # the model saw int16 wav samples
        if len(seg) < sr // 25:
            return np.zeros(len(t))
        a = mfcc(seg, sr)
        length = min((a.shape[0] - a.shape[0] % 4) / 100, vid.shape[0] / 25)
        a = a[:int(round(length * 100))]
        vid = vid[:int(round(length * 25))]
        if len(vid) == 0:
            return np.zeros(len(t))
        all_scores = []
        for dur in self.DURATIONS:
            sc = []
            for i in range(int(math.ceil(length / dur))):
                ai = a[i * dur * 100:(i + 1) * dur * 100]
                vi = vid[i * dur * 25:(i + 1) * dur * 25]
                if len(vi) == 0 or len(ai) < 4:
                    continue
                # The audio encoder downsamples by 4; keep the pair consistent.
                n = min(len(vi), len(ai) // 4)
                sc.extend(self._run(ai[:4 * n], vi[:n]))
            all_scores.append(np.asarray(sc[:len(vid)]))
        n = min(len(s) for s in all_scores)
        s25 = np.mean([s[:n] for s in all_scores], axis=0)
        k = np.clip(np.round((t - t0) * 25).astype(int), 0, max(n - 1, 0))
        return s25[k] if n else np.zeros(len(t))
