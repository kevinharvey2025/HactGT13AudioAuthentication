"""Shared synthetic-signal helpers for DSP tests (fixtures only; no detection claims)."""
import numpy as np

SR = 16000


def speechlike(dur=3.0, sr=SR, f0=140.0, seed=0, noise_db=-50.0):
    """Syllable-like voiced bursts (harmonic pulse train) over low-level noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(dur * sr)) / sr
    phase = 2 * np.pi * np.cumsum(f0 * (1 + 0.05 * np.sin(2 * np.pi * 0.7 * t))) / sr
    src = sum(np.sin(k * phase) / k for k in range(1, 30))
    env = (np.sin(2 * np.pi * 2.5 * t) > 0.2).astype(float)
    env = np.convolve(env, np.hanning(400) / np.hanning(400).sum(), mode="same")
    x = 0.3 * src * env / (np.max(np.abs(src)) + 1e-9)
    x += 10 ** (noise_db / 20) * rng.standard_normal(t.size)
    return x
