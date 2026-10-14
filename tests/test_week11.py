"""Week 11: MFCC, voice activity, mouth motion, active-speaker score, AP, Light-ASD shapes."""
import unittest

import numpy as np

from cinescope.speaker.active import MOUTH, UPPER, mouth_activity, score_track
from cinescope.speaker.audio import SR, at_times, mfcc, vad
from cinescope.speaker.metrics import average_precision, f1_at, roc_auc


def syllables(seconds=3.0, rate=4.0, f0=140.0, seed=0):
    """Speech-like: a harmonic voice switched on and off ~4 times a second."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR)) / SR
    f = f0 * (1 + 0.1 * np.sin(2 * np.pi * 0.7 * t))
    phase = 2 * np.pi * np.cumsum(f) / SR
    voice = sum(np.sin(k * phase) / k for k in range(1, 15))
    env = (np.sin(2 * np.pi * rate * t) > 0).astype(float)
    env = np.convolve(env, np.ones(160) / 160, mode="same")
    return (0.3 * voice * env + 0.003 * rng.standard_normal(len(t))).astype(np.float32)


class TestAudio(unittest.TestCase):
    def test_mfcc_matches_python_speech_features(self):
        psf = __import__("pytest").importorskip("python_speech_features")
        x = np.random.default_rng(0).standard_normal(SR) * 3000
        ref = psf.mfcc(x, SR, numcep=13, winlen=0.025, winstep=0.010)
        np.testing.assert_allclose(mfcc(x), ref, atol=1e-8)

    def test_mfcc_rate(self):
        self.assertEqual(mfcc(np.zeros(2 * SR) + 0.1).shape, (199, 13))   # 100 frames per second

    def test_vad_prefers_syllabic_voice(self):
        rng = np.random.default_rng(1)
        silence = 0.003 * rng.standard_normal(2 * SR)
        noise = 0.3 * rng.standard_normal(2 * SR)                          # broadband, steady
        t = np.arange(2 * SR) / SR
        hum = 0.3 * np.sin(2 * np.pi * 440 * t)                            # steady tone (music-like)
        x = np.concatenate([silence, syllables(2.0), silence, noise, silence, hum, silence]).astype(np.float32)
        v = vad(x)
        seg = [v["score"][i * 200 + 20:(i + 1) * 200 - 20].mean() for i in range(7)]
        mod = [v["modulation"][i * 200 + 50:(i + 1) * 200 - 50].mean() for i in range(7)]
        self.assertGreater(seg[1], 0.5)
        self.assertLess(seg[0], 0.1)
        self.assertGreater(seg[1], 2 * seg[3])                             # noise: flat spectrum
        self.assertGreater(mod[1], 2 * mod[5])                             # tone: no syllable rhythm...
        # ...but a loud steady tone is harmonic, so it still passes the VAD: a known
        # weakness (sustained music) recorded in the README, not hidden here.
        self.assertGreater(seg[5], 0.5)

    def test_at_times(self):
        v = np.arange(10.0)
        np.testing.assert_allclose(at_times(v, 10, np.array([0.0, 0.5])), [0, 5])
        np.testing.assert_allclose(at_times(v, 10, np.array([0.0]), span=0.3), [1.0])


def faces(n=48, mouth_moves=True, head_moves=False, seed=0):
    """Aligned 112x112 'faces': a textured background, a mouth that opens and closes."""
    rng = np.random.default_rng(seed)
    base = rng.integers(80, 160, (112, 112)).astype(np.uint8)
    out = []
    for k in range(n):
        f = base.copy()
        if mouth_moves:
            h = int(4 + 8 * (np.sin(2 * np.pi * k / 6) > 0))
            f[92 - h // 2:92 + h // 2, 42:70] = 20
        if head_moves:
            f[UPPER[0], UPPER[1]] = np.roll(f[UPPER[0], UPPER[1]], k % 3, axis=1)
        out.append(f)
    return np.asarray(out)


class TestActiveSpeaker(unittest.TestCase):
    def test_mouth_activity_is_relative(self):
        talking = mouth_activity(faces(mouth_moves=True)).mean()
        still = mouth_activity(faces(mouth_moves=False)).mean()
        nodding = mouth_activity(faces(mouth_moves=False, head_moves=True)).mean()
        self.assertGreater(talking, still + 1)
        self.assertLess(nodding, still)                # upper-face motion lowers the ratio
        self.assertEqual(MOUTH[0].start, 78)

    def test_score_needs_speech_and_mouth(self):
        fps, n = 12.0, 96
        times = np.arange(n) / fps
        speech = np.zeros(800)
        speech[200:600] = 1.0                                   # speech from 2 s to 6 s
        v = {"fps": 100.0, "score": speech, "energy_db": speech * 20 - 40}
        frames = list(range(n))
        # A face whose mouth moves only while speech is heard.
        g = faces(n, mouth_moves=False)
        g[24:72] = faces(48, mouth_moves=True)
        r = score_track(g, frames, times, v, fps)["score"]
        self.assertGreater(r[30:66].mean(), 0.3)
        self.assertLess(r[:20].mean(), 0.05)                     # mouth still, no speech
        # A listener: speech heard, mouth still.
        r2 = score_track(faces(n, mouth_moves=False), frames, times, v, fps)["score"]
        self.assertLess(r2[30:66].mean(), r[30:66].mean() / 2)


class TestMetrics(unittest.TestCase):
    def test_ap_against_sklearn(self):
        skm = __import__("pytest").importorskip("sklearn.metrics")
        rng = np.random.default_rng(0)
        for _ in range(20):
            y = rng.random(200) < 0.2
            s = y * rng.random(200) + rng.random(200)
            s[::7] = np.round(s[::7], 1)                         # ties
            self.assertAlmostEqual(average_precision(y, s), skm.average_precision_score(y, s), places=10)
            self.assertAlmostEqual(roc_auc(y, s), skm.roc_auc_score(y, s), places=10)

    def test_f1(self):
        r = f1_at(np.array([1, 1, 0, 0], bool), np.array([0.9, 0.2, 0.8, 0.1]), 0.5)
        self.assertEqual((r["tp"], r["fp"], r["fn"]), (1, 1, 1))
        self.assertAlmostEqual(r["f1"], 0.5)


class TestLightASDNetwork(unittest.TestCase):
    def test_shapes(self):
        torch = __import__("pytest").importorskip("torch")
        torch.set_num_threads(1)
        from cinescope.speaker.lightasd import _ASD

        net = _ASD().eval()
        T = 10                                                  # video frames (25 fps) -> 40 MFCC frames
        with torch.no_grad():
            a = net.model.forward_audio_frontend(torch.randn(1, 4 * T, 13))
            v = net.model.forward_visual_frontend(torch.rand(1, T, 112, 112) * 255)
            out = net.lossAV.FC(net.model.forward_audio_visual_backend(a, v))
        self.assertEqual(tuple(a.shape), (1, T, 128))
        self.assertEqual(tuple(v.shape), (1, T, 128))
        self.assertEqual(tuple(out.shape), (T, 2))
        # The vendored network has exactly the published weight names.
        keys = set(net.state_dict())
        self.assertIn("model.visualEncoder.block1.s_3.weight", keys)
        self.assertIn("model.GRU.gru_backward.bias_hh_l0", keys)
        self.assertIn("lossAV.FC.weight", keys)


if __name__ == "__main__":
    unittest.main()
