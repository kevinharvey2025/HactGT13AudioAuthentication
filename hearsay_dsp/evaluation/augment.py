"""Label-preserving channel transforms applied identically to both classes.

Every transformed file is written as 16-bit PCM WAV at the analysis rate
(like the HEARSAY test files) with a JSON sidecar recording the source
content hash, condition, parameters and seed, then analyzed by the normal
pipeline. Derivatives inherit the source row's split and group_id.

Conditions are named parameter sets. Training-time augmentation and
robustness evaluation use different settings (see TRAIN_CONDITIONS vs
EVAL_CONDITIONS), so robustness is measured on unseen transform settings.
"""

from __future__ import annotations

import hashlib
import io
from pathlib import Path

import av
import numpy as np
import soundfile as sf
from scipy.signal import butter, fftconvolve, resample_poly, sosfiltfilt

from ..io.cache import atomic_write_json
from ..io.decode import decode_audio

AUG_VERSION = "1"

EVAL_CONDITIONS = {
    "clean": [],
    "crop_3to4s": [("crop", {"min_s": 3.0, "max_s": 4.0}), ("peaknorm", {"peak": 0.99})],
    "noise_white_10db": [("noise", {"color": "white", "snr_db": 10.0})],
    "noise_pink_5db": [("noise", {"color": "pink", "snr_db": 5.0})],
    "mp3_24k": [("codec", {"codec": "libmp3lame", "bitrate": 24000})],
    "opus_12k": [("codec", {"codec": "libopus", "bitrate": 12000})],
    "telephone_mulaw": [("telephone", {})],
    "reverb_rt60_0.6": [("reverb", {"rt60_s": 0.6})],
    "gain_-20db": [("gain", {"db": -20.0})],
    "clip_25pct": [("clip", {"level": 0.25})],
    "lowpass_7500": [("lowpass", {"cutoff_hz": 7500.0})],
}

TRAIN_CONDITIONS = {
    "t_crop": [("crop", {"min_s": 2.5, "max_s": 5.0}), ("peaknorm", {"peak": 0.99})],
    "t_noise": [("noise", {"color": "random", "snr_db": [12.0, 30.0]})],
    "t_mp3": [("codec", {"codec": "libmp3lame", "bitrate": 48000})],
    "t_aac": [("codec", {"codec": "aac", "bitrate": 32000})],
    "t_reverb": [("reverb", {"rt60_s": [0.2, 0.45]})],
    "t_band": [("bandpass", {"low_hz": 200.0, "high_hz": 3600.0})],
}


def _uniform(v, rng):
    return float(rng.uniform(v[0], v[1])) if isinstance(v, (list, tuple)) else v


def active_power(x: np.ndarray, sr: int) -> float:
    n = int(0.025 * sr)
    h = int(0.010 * sr)
    if x.size < n:
        return float(np.mean(x ** 2) + 1e-20)
    idx = np.arange(0, x.size - n + 1, h)[:, None] + np.arange(n)[None, :]
    e = np.mean(x[idx] ** 2, axis=1)
    return float(np.mean(e[e >= np.median(e)]) + 1e-20)


def colored_noise(n: int, color: str, rng) -> np.ndarray:
    white = rng.standard_normal(n)
    if color == "white":
        return white
    spec = np.fft.rfft(white)
    f = np.arange(spec.size, dtype=float)
    f[0] = 1.0
    spec = spec / (np.sqrt(f) if color == "pink" else f)
    y = np.fft.irfft(spec, n)
    return y / (np.std(y) + 1e-12)


def codec_roundtrip(x: np.ndarray, sr: int, codec: str, bitrate: int) -> np.ndarray:
    fmt = {"libmp3lame": "mp3", "aac": "adts", "libopus": "ogg"}[codec]
    buf = io.BytesIO()
    pcm = (np.clip(x, -1, 1) * 32767).astype(np.int16)[None, :]
    with av.open(buf, mode="w", format=fmt) as out:
        st = out.add_stream(codec, rate=sr)
        st.bit_rate = bitrate
        st.layout = "mono"
        frame = av.AudioFrame.from_ndarray(pcm, format="s16", layout="mono")
        frame.sample_rate = sr
        for pkt in st.encode(frame):
            out.mux(pkt)
        for pkt in st.encode(None):
            out.mux(pkt)
    buf.seek(0)
    chunks = []
    with av.open(buf, mode="r") as inp:
        s = inp.streams.audio[0]
        out_sr = s.codec_context.sample_rate
        for fr in inp.decode(s):
            arr = fr.to_ndarray()
            if not fr.format.is_planar:
                arr = arr.reshape(-1, len(fr.layout.channels)).T
            arr = arr.astype(np.float64)
            if fr.format.name.startswith("s16"):
                arr /= 32768.0
            chunks.append(arr.mean(axis=0))
    y = np.concatenate(chunks)
    if out_sr != sr:
        y = resample_poly(y, sr, out_sr)
    if codec == "aac":  # ADTS does not signal the encoder's 1024-sample priming delay
        y = y[1024:]
    # length-align with the source (codec padding at the end is removed)
    return y[: x.size] if y.size >= x.size else np.concatenate([y, np.zeros(x.size - y.size)])


def telephone(x: np.ndarray, sr: int) -> np.ndarray:
    sos = butter(4, [300, 3400], btype="bandpass", fs=sr, output="sos")
    y = resample_poly(sosfiltfilt(sos, x), 1, sr // 8000)
    mu = 255.0
    y = np.clip(y, -1, 1)
    q = np.round((np.sign(y) * np.log1p(mu * np.abs(y)) / np.log1p(mu)) * 127) / 127
    y = np.sign(q) * ((1 + mu) ** np.abs(q) - 1) / mu
    return resample_poly(y, sr // 8000, 1)


def synthetic_rir(sr: int, rt60: float, rng) -> np.ndarray:
    n = int(1.2 * rt60 * sr)
    t = np.arange(n) / sr
    h = rng.standard_normal(n) * np.exp(-6.908 * t / rt60)
    h[0] = 1.0 + abs(h[0])
    return h / np.sqrt(np.sum(h ** 2))


def apply_chain(x: np.ndarray, sr: int, chain: list, rng) -> tuple[np.ndarray, list]:
    applied = []
    for name, params in chain:
        p = dict(params)
        if name == "crop":
            dur = _uniform([p["min_s"], p["max_s"]], rng)
            n = int(dur * sr)
            if x.size > n:
                start = int(rng.integers(0, x.size - n + 1))
                x = x[start:start + n]
                p.update(start_s=start / sr, dur_s=dur)
        elif name == "peaknorm":
            x = x / (np.max(np.abs(x)) + 1e-12) * p["peak"]
        elif name == "noise":
            color = p["color"] if p["color"] != "random" else str(rng.choice(["white", "pink", "brown"]))
            snr = _uniform(p["snr_db"], rng)
            nz = colored_noise(x.size, color, rng)
            x = x + nz * np.sqrt(active_power(x, sr) / 10 ** (snr / 10.0))
            p.update(color=color, snr_db=snr)
        elif name == "codec":
            x = codec_roundtrip(x, sr, p["codec"], int(p["bitrate"]))
        elif name == "telephone":
            x = telephone(x, sr)
            p.update(band_hz=[300, 3400], rate=8000, companding="mu-law 8 bit")
        elif name == "reverb":
            rt60 = _uniform(p["rt60_s"], rng)
            rms = np.sqrt(np.mean(x ** 2))
            x = fftconvolve(x, synthetic_rir(sr, rt60, rng))[: x.size]
            x = x * rms / (np.sqrt(np.mean(x ** 2)) + 1e-12)
            p.update(rt60_s=rt60, rir="exponentially decaying Gaussian noise + direct path")
        elif name == "gain":
            x = x * 10 ** (p["db"] / 20.0)
        elif name == "clip":
            lim = p["level"] * np.max(np.abs(x))
            x = np.clip(x, -lim, lim)
        elif name == "lowpass":
            x = sosfiltfilt(butter(8, p["cutoff_hz"], fs=sr, output="sos"), x)
        elif name == "bandpass":
            x = sosfiltfilt(butter(6, [p["low_hz"], p["high_hz"]], btype="bandpass", fs=sr,
                                   output="sos"), x)
        else:
            raise ValueError(name)
        applied.append({"op": name, **p})
    peak = np.max(np.abs(x))
    if peak > 0.999:  # keep inside full scale without clipping
        x = x / peak * 0.999
        applied.append({"op": "rescale_to_full_scale", "factor": float(0.999 / peak)})
    return x, applied


def variant_seed(seed: int, src_sha: str, condition: str) -> int:
    return int(hashlib.sha256(f"{seed}:{src_sha}:{condition}".encode()).hexdigest()[:16], 16)


def make_variant(src_path: str, src_sha: str, condition: str, chain: list, cfg: dict,
                 out_root: str | Path, seed: int) -> str:
    """Create (or reuse) the transformed WAV for one source file; returns its path."""
    out_dir = Path(out_root) / f"{condition}.v{AUG_VERSION}"
    out = out_dir / f"{src_sha}.wav"
    side = out_dir / f"{src_sha}.json"
    if out.exists() and side.exists():
        return str(out)
    dec = decode_audio(src_path, cfg)
    rng = np.random.default_rng(variant_seed(seed, src_sha, condition))
    y, applied = apply_chain(dec.analysis.copy(), dec.analysis_sr, chain, rng)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / f".{src_sha}.tmp.wav"
    sf.write(tmp, y, dec.analysis_sr, subtype="PCM_16")
    tmp.replace(out)
    atomic_write_json(side, {"source_path": src_path, "source_sha256": src_sha, "condition": condition,
                             "chain": applied, "seed": seed, "aug_version": AUG_VERSION,
                             "sample_rate": dec.analysis_sr})
    return str(out)
