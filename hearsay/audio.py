"""Audio decoding and the canonical form every model sees.

The NSA test files are 16 kHz mono PCM16 written by ffmpeg (Lavf58.29.100), peak-normalized,
mostly 3-4 s long, and low-passed near 7.5 kHz. The 242 real LJSpeech clips are ffmpeg's default
resample of the LJSpeech originals (reproduced here to within 3 LSB), which keeps energy up to
~7.8 kHz; every DiffSSD fake is in its native format (22.05/24/44.1 kHz, some mp3), and some
end in exact digital zeros. To keep format, bandwidth, level, duration and digital silence
from becoming class shortcuts, *both* classes (and the test set) go through the same steps:
ffmpeg -> 16 kHz mono PCM16 -> trim edge silence -> [train: crop to a test-like duration]
-> 7 kHz low-pass -> peak-normalize -> 1-LSB dither.
"""
import subprocess
from functools import lru_cache
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve, firwin

from . import config

LOWPASS_HZ = 7000.0


def decode(path, sr=config.SR):
    """Any container/codec -> mono int16 at `sr`, resampled by ffmpeg like the test set."""
    path = str(path)
    if path.endswith(".wav"):
        info = sf.info(path)
        if info.samplerate == sr and info.channels == 1 and info.subtype == "PCM_16":
            return sf.read(path, dtype="int16")[0]
    out = subprocess.run(
        [config.ffmpeg_bin(), "-v", "error", "-nostdin", "-i", path, "-ac", "1", "-ar", str(sr),
         "-f", "s16le", "-acodec", "pcm_s16le", "-"],
        capture_output=True, check=True).stdout
    return np.frombuffer(out, np.int16).copy()


def to_float(x):
    return x.astype(np.float32) / 32768.0 if x.dtype == np.int16 else x.astype(np.float32)


def trim_silence(x, sr=config.SR, top_db=40.0, pad=0.05):
    """Drop leading/trailing frames more than `top_db` below the clip's loudest frame."""
    frame = int(0.02 * sr)
    n = len(x) // frame
    if n < 3:
        return x
    e = to_float(x[: n * frame]).reshape(n, frame)
    db = 10 * np.log10(np.mean(e ** 2, axis=1) + 1e-12)
    active = np.where(db > db.max() - top_db)[0]
    if len(active) == 0:
        return x
    p = int(pad * sr)
    return x[max(0, active[0] * frame - p): min(len(x), (active[-1] + 1) * frame + p)]


def crop(x, dur, rng, sr=config.SR):
    n = int(round(dur * sr))
    if len(x) <= n:
        return x
    start = int(rng.integers(0, len(x) - n + 1))
    return x[start: start + n]


def peak_normalize(x, peak=0.99):
    x = to_float(x)
    m = np.abs(x).max()
    return x * (peak / m) if m > 0 else x


@lru_cache(maxsize=4)
def _lowpass_taps(cutoff, sr):
    # 513-tap Kaiser FIR: flat to ~6.9 kHz, about -80 dB from ~7.1 kHz
    return firwin(513, cutoff, fs=sr, window=("kaiser", 8.0)).astype(np.float32)


def lowpass(x, cutoff=LOWPASS_HZ, sr=config.SR):
    return fftconvolve(to_float(x), _lowpass_taps(cutoff, sr), mode="same").astype(np.float32)


def dither(x, rng):
    """TPDF dither at 1 LSB of int16 so no clip contains exact digital zeros."""
    return x + ((rng.random(len(x)) - rng.random(len(x))) / 32768.0).astype(np.float32)


class TestLikeDurations:
    """Sample crop lengths from the empirical test-set duration distribution."""

    def __init__(self, durations):
        self.d = np.sort(np.asarray(durations, dtype=np.float64))

    @classmethod
    def from_cache(cls):
        return cls(np.load(config.CACHE / "test_durations.npy"))

    def sample(self, rng):
        return float(self.d[rng.integers(len(self.d))])


def canonical(x, rng, durations=None, trim=True, aug=None):
    """The view every model sees (float32). Pass `durations` for train/dev crops; None for test.

    `aug(x, rng) -> (x, params)` is applied after cropping and before the low-pass, so augmented
    views obey the same bandwidth/level contract as clean ones.
    """
    if trim:
        x = trim_silence(x)
    if durations is not None:
        x = crop(x, durations.sample(rng), rng)
    if aug is not None:
        x, _ = aug(peak_normalize(x, 0.9), rng)
    x = lowpass(x)
    x = peak_normalize(x - x.mean(), peak=float(rng.uniform(0.97, 1.0)))  # DC offsets are generator-specific
    return dither(x, rng)


def uid_rng(uid, view=0, seed=0):
    """Deterministic per-clip RNG so crops/augmentations are reproducible across stages."""
    import zlib
    return np.random.default_rng([seed, view, zlib.crc32(uid.encode())])


def cache_path(uid, root=None):
    return Path(root or config.CACHE / "wav16k") / f"{uid}.wav"


def load_cached(uid):
    return sf.read(cache_path(uid), dtype="int16")[0]
