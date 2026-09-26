"""Channel augmentations, applied to BOTH classes so "noisy/compressed = real" can't be learned.

Every call returns (audio, params). `params["channel"]` is the channel label used by the D5
channel prototypes; the other keys are logged so any view can be reproduced. Covers plan
section 6.1: codecs at several bitrates, telephony, band-limiting and resampling, noise (white,
pink, brown, babble, mains hum), room reverb, gain/clipping, plus spectral tilt (the NSA test
set is darker than every training source).
"""
import subprocess

import numpy as np
from scipy.signal import butter, fftconvolve, firwin, sosfilt

from . import config
from .audio import to_float

SR = config.SR

CODECS = {
    # name: (ffmpeg encode args, container, encode sample rate)
    "mp3": (lambda b: ["-c:a", "libmp3lame", "-b:a", f"{b}k"], "mp3", SR, [24, 32, 48, 64, 128]),
    "aac": (lambda b: ["-c:a", "aac", "-b:a", f"{b}k"], "adts", SR, [16, 24, 32, 64]),
    "opus": (lambda b: ["-c:a", "libopus", "-b:a", f"{b}k"], "ogg", SR, [8, 12, 16, 24, 32]),
    "g711": (lambda b: ["-c:a", "pcm_mulaw"], "wav", 8000, [64]),
    "g726": (lambda b: ["-c:a", "g726", "-b:a", f"{b}k"], "wav", 8000, [16, 24, 32]),
    "g722": (lambda b: ["-c:a", "g722"], "wav", SR, [64]),
}


def _ffmpeg_roundtrip(x, enc_args, fmt, enc_sr):
    raw = np.clip(x, -1, 1).astype(np.float32).tobytes()
    enc = subprocess.run([config.ffmpeg_bin(), "-v", "error", "-f", "f32le", "-ar", str(SR), "-ac", "1", "-i", "-",
                          "-ar", str(enc_sr), *enc_args, "-f", fmt, "-"], input=raw, capture_output=True, check=True).stdout
    dec = subprocess.run([config.ffmpeg_bin(), "-v", "error", "-i", "-", "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"],
                         input=enc, capture_output=True, check=True).stdout
    y = np.frombuffer(dec, np.float32).copy()
    return _align(y, len(x))


def _align(y, n):
    return y[:n] if len(y) >= n else np.pad(y, (0, n - len(y)))


def codec(x, rng, name=None, kbps=None):
    name = name or str(rng.choice(list(CODECS)))
    args, fmt, enc_sr, rates = CODECS[name]
    kbps = kbps or int(rng.choice(rates))
    return _ffmpeg_roundtrip(x, args(kbps), fmt, enc_sr), dict(channel=f"codec_{name}", codec=name, kbps=kbps)


def telephony(x, rng):
    sos = butter(6, [300, 3400], btype="bandpass", fs=SR, output="sos")
    return sosfilt(sos, x).astype(np.float32), dict(channel="telephony")


def bandlimit(x, rng):
    cut = float(rng.uniform(3500, 6500))
    y = fftconvolve(x, firwin(255, cut, fs=SR), mode="same").astype(np.float32)
    return y, dict(channel="bandlimit", cutoff_hz=round(cut))


def resample_chain(x, rng):
    import soxr
    mid = int(rng.choice([8000, 11025, 12000]))
    return _align(soxr.resample(soxr.resample(x, SR, mid), mid, SR).astype(np.float32), len(x)), dict(channel="resample", mid_sr=mid)


def tilt(x, rng):
    """Smooth spectral tilt of `db_per_oct` dB/octave above 500 Hz (negative = darker)."""
    db = float(rng.uniform(-6.0, 2.0))
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / SR)
    g = 10 ** (db * np.log2(np.maximum(f, 500) / 500) / 20)
    return np.fft.irfft(X * g, n=len(x)).astype(np.float32), dict(channel="tilt", db_per_oct=round(db, 2))


def _colored(n, rng, kind):
    w = rng.standard_normal(n)
    if kind == "white":
        return w
    W = np.fft.rfft(w)
    f = np.maximum(np.fft.rfftfreq(n, 1 / SR), 20.0)
    return np.fft.irfft(W / (np.sqrt(f) if kind == "pink" else f), n=n)


def _mix(x, n, snr_db):
    px, pn = np.mean(x ** 2) + 1e-12, np.mean(n ** 2) + 1e-12
    return (x + n * np.sqrt(px / (pn * 10 ** (snr_db / 10)))).astype(np.float32)


def noise(x, rng, babble_pool=None):
    kinds = ["white", "pink", "brown"] + (["babble"] if babble_pool else [])
    kind = str(rng.choice(kinds))
    snr = float(rng.uniform(5, 30))
    if kind == "babble":
        n = np.zeros(len(x))
        for _ in range(int(rng.integers(3, 7))):
            b = to_float(babble_pool[int(rng.integers(len(babble_pool)))])
            b = np.resize(b, len(x)) if len(b) < len(x) else b[int(rng.integers(0, len(b) - len(x) + 1)):][: len(x)]
            n += b / (np.std(b) + 1e-6)
        snr = float(rng.uniform(8, 25))
    else:
        n = _colored(len(x), rng, kind)
    return _mix(x, n, snr), dict(channel=f"noise_{kind}", snr_db=round(snr, 1))


def hum(x, rng):
    f0 = float(rng.choice([50.0, 60.0])) * float(rng.uniform(0.998, 1.002))
    t = np.arange(len(x)) / SR
    h = sum(rng.uniform(0.1, 1.0) / k * np.sin(2 * np.pi * k * f0 * t + rng.uniform(0, 2 * np.pi)) for k in range(1, 6))
    snr = float(rng.uniform(10, 35))
    return _mix(x, h, snr), dict(channel="hum", hum_hz=round(f0, 2), snr_db=round(snr, 1))


def reverb(x, rng):
    rt60 = float(rng.uniform(0.15, 0.9))
    n = int(rt60 * SR)
    t = np.arange(n) / SR
    rir = rng.standard_normal(n) * np.exp(-6.9 * t / rt60)
    rir[0] = float(rng.uniform(2, 8)) * np.abs(rir).max()                 # direct path
    rir = sosfilt(butter(2, float(rng.uniform(3000, 7000)), fs=SR, output="sos"), rir)
    y = fftconvolve(x, rir)[: len(x)]
    return (y / (np.abs(y).max() + 1e-9) * np.abs(x).max()).astype(np.float32), dict(channel="reverb", rt60=round(rt60, 2))


def clip(x, rng):
    g = 10 ** (float(rng.uniform(3, 12)) / 20)
    return np.clip(x * g, -1, 1).astype(np.float32), dict(channel="clipping", gain_db=round(float(20 * np.log10(g)), 1))


AUGS = {"codec": codec, "telephony": telephony, "bandlimit": bandlimit, "resample": resample_chain, "tilt": tilt,
        "noise": noise, "hum": hum, "reverb": reverb, "clip": clip}
WEIGHTS = {"codec": 3, "telephony": 1, "bandlimit": 1, "resample": 1, "tilt": 2, "noise": 2, "hum": 1, "reverb": 1.5, "clip": 0.5}


def random_chain(x, rng, n_ops=None, babble_pool=None):
    """1-2 random augmentations in sequence. Returns (audio, params) with a combined channel label."""
    n_ops = n_ops or int(rng.choice([1, 1, 2]))
    names = list(WEIGHTS)
    p = np.array([WEIGHTS[k] for k in names], float)
    ops = rng.choice(names, size=n_ops, replace=False, p=p / p.sum())
    params = []
    for op in ops:
        x, prm = noise(x, rng, babble_pool) if op == "noise" else AUGS[op](x, rng)
        params.append(prm)
    return x, dict(channel="+".join(p["channel"] for p in params), ops=params)
