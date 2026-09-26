"""Decoding with provenance.

Amplitude convention: float64 samples in [-1, 1); integer PCM is divided by
2**(bits-1) (int16 -> /32768). Originals are only ever opened read-only.

Decoder policy: the container is sniffed from the file's magic bytes (not
its extension). WAV/FLAC/AIFF/OGG go to soundfile (libsndfile) first; MP3,
MP4/M4A, AAC and anything else go to PyAV (bundled FFmpeg) first. If the
primary decoder fails, the other one is tried and both attempts are logged.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from math import gcd

import av
import numpy as np
import scipy
import soundfile as sf
from scipy.signal import resample_poly

SOUNDFILE_FIRST = {"wav", "flac", "aiff", "ogg"}


class DecodeError(RuntimeError):
    pass


@dataclass
class DecodedAudio:
    native: np.ndarray            # mono, float64, native rate
    native_sr: int
    analysis: np.ndarray          # mono, float64, analysis rate (resampled, quantized)
    analysis_sr: int
    provenance: dict = field(default_factory=dict)
    quality: dict = field(default_factory=dict)


def file_sha256(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def sniff_format(path: str) -> str:
    with open(path, "rb") as fh:
        head = fh.read(64)
    if len(head) < 4:
        return "unknown"
    if head[:4] in (b"RIFF", b"RF64", b"RIFX") and head[8:12] == b"WAVE":
        return "wav"
    if head[:4] == b"fLaC":
        return "flac"
    if head[:4] == b"OggS":
        return "ogg"
    if head[:4] == b"FORM" and head[8:12] in (b"AIFF", b"AIFC"):
        return "aiff"
    if head[4:8] == b"ftyp":
        return "mp4"
    if head[:3] == b"ID3":
        return "mp3"
    if head[0] == 0xFF and (head[1] & 0xE0) == 0xE0:
        layer = (head[1] >> 1) & 0x3
        return "aac_adts" if layer == 0 else "mp3"
    if head[:4] == b"\x30\x26\xb2\x75":
        return "asf"
    if head[:4] == b"\x1a\x45\xdf\xa3":
        return "matroska"
    return "unknown"


def _pcm_scale(fmt_name: str) -> tuple[float, float]:
    """(offset, divisor) converting a PyAV sample format to [-1, 1)."""
    base = fmt_name.rstrip("p")
    if base == "u8":
        return 128.0, 128.0
    if base == "s16":
        return 0.0, 32768.0
    if base == "s32":
        return 0.0, 2147483648.0
    if base == "s64":
        return 0.0, 9223372036854775808.0
    if base in ("flt", "dbl"):
        return 0.0, 1.0
    raise DecodeError(f"unsupported sample format {fmt_name}")


def _decode_pyav(path: str, stream_policy: str):
    with av.open(path, mode="r") as container:
        streams = list(container.streams.audio)
        if not streams:
            raise DecodeError("no audio stream")
        if stream_policy == "most_channels":
            stream = sorted(streams, key=lambda s: (-(s.codec_context.channels or 0), s.index))[0]
        elif stream_policy == "first":
            stream = min(streams, key=lambda s: s.index)
        else:
            raise DecodeError(f"unknown stream policy {stream_policy}")
        cc = stream.codec_context
        chunks = []
        for frame in container.decode(stream):
            arr = frame.to_ndarray()
            nch = len(frame.layout.channels)
            if not frame.format.is_planar:
                arr = arr.reshape(-1, nch).T
            offset, div = _pcm_scale(frame.format.name)
            chunks.append((arr.astype(np.float64) - offset) / div)
        if not chunks:
            raise DecodeError("stream decoded to zero frames")
        x = np.concatenate(chunks, axis=1).T
        header_dur = None
        if stream.duration is not None and stream.time_base is not None:
            header_dur = float(stream.duration * stream.time_base)
        elif container.duration is not None:
            header_dur = container.duration / 1e6
        prov = {
            "decoder": "pyav",
            "decoder_version": (f"PyAV {av.__version__}; libavcodec "
                                + ".".join(map(str, av.library_versions["libavcodec"]))),
            "container": container.format.name,
            "codec": cc.name,
            "bit_rate": cc.bit_rate or container.bit_rate,
            "stream_index": stream.index,
            "n_audio_streams": len(streams),
            "native_sr": int(cc.sample_rate),
            "channels": int(x.shape[1]),
            "header_duration_s": header_dur,
            "tags": {**{str(k): str(v) for k, v in container.metadata.items()},
                     **{f"stream:{k}": str(v) for k, v in stream.metadata.items()}},
        }
        return x, int(cc.sample_rate), prov


def _decode_soundfile(path: str, stream_policy: str):
    info = sf.info(path)
    x, sr = sf.read(path, dtype="float64", always_2d=True)
    prov = {
        "decoder": "soundfile",
        "decoder_version": f"soundfile {sf.__version__}; {sf.__libsndfile_version__}",
        "container": info.format,
        "codec": info.subtype,
        "bit_rate": None,
        "stream_index": 0,
        "n_audio_streams": 1,
        "native_sr": int(sr),
        "channels": int(info.channels),
        "header_duration_s": float(info.frames) / sr if info.frames else None,
        "tags": {},
    }
    return x, int(sr), prov


_DECODERS = {"pyav": _decode_pyav, "soundfile": _decode_soundfile}


def resample(x: np.ndarray, sr_in: int, sr_out: int, window=("kaiser", 5.0)):
    """Anti-aliased polyphase resampling (scipy.signal.resample_poly)."""
    if sr_in == sr_out:
        return x.astype(np.float64, copy=True), {"resampled": False}
    g = gcd(int(sr_in), int(sr_out))
    up, down = int(sr_out) // g, int(sr_in) // g
    y = resample_poly(x, up, down, window=tuple(window))
    return y, {"resampled": True, "up": up, "down": down, "window": list(window),
               "impl": f"scipy {scipy.__version__} signal.resample_poly"}


def quantize(x: np.ndarray, bits: int | None) -> np.ndarray:
    if not bits:
        return x
    scale = float(2 ** (bits - 1))
    return np.clip(np.round(x * scale), -scale, scale - 1) / scale


def probe_file(path: str) -> dict:
    """Header-level technical profile without full decoding (validation/profiling)."""
    out = {"path": path, "sha256": file_sha256(path), "size_bytes": os.path.getsize(path),
           "sniffed_format": sniff_format(path), "extension": os.path.splitext(path)[1].lower()}
    try:
        if out["sniffed_format"] in SOUNDFILE_FIRST:
            i = sf.info(path)
            out.update(container=i.format, codec=i.subtype, sr=int(i.samplerate), channels=int(i.channels),
                       duration_s=float(i.duration), bit_rate=None)
        else:
            with av.open(path) as c:
                s = c.streams.audio[0]
                dur = (float(s.duration * s.time_base) if s.duration is not None
                       else (c.duration or 0) / 1e6)
                out.update(container=c.format.name, codec=s.codec_context.name,
                           sr=int(s.codec_context.sample_rate), channels=int(s.codec_context.channels),
                           duration_s=dur, bit_rate=s.codec_context.bit_rate or c.bit_rate)
        out["probe_status"] = "ok"
    except Exception as exc:  # noqa: BLE001 - reported per file
        out.update(probe_status="error", probe_error=f"{type(exc).__name__}: {exc}")
    return out


def decode_audio(path: str, cfg: dict) -> DecodedAudio:
    dcfg = cfg["decode"]
    fmt = sniff_format(path)
    order = ["soundfile", "pyav"] if fmt in SOUNDFILE_FIRST else ["pyav", "soundfile"]
    attempts = []
    for name in order:
        try:
            x, sr, prov = _DECODERS[name](path, dcfg["stream_policy"])
            break
        except Exception as exc:  # noqa: BLE001 - every failure is recorded
            attempts.append({"decoder": name, "error": f"{type(exc).__name__}: {exc}"})
    else:
        raise DecodeError(f"all decoders failed: {attempts}")
    prov["sniffed_format"] = fmt
    prov["extension"] = os.path.splitext(path)[1].lower()
    prov["failed_attempts"] = attempts
    if x.size == 0 or x.shape[0] == 0:
        raise DecodeError("decoded zero samples")

    quality: dict = {}
    finite = np.isfinite(x)
    n_bad = int((~finite).sum())
    quality["nonfinite_samples"] = n_bad
    if n_bad:
        if n_bad / x.size > dcfg["max_nonfinite_frac"]:
            raise DecodeError(f"{n_bad} nonfinite samples ({n_bad / x.size:.2%})")
        x = np.where(finite, x, 0.0)
    ch_rms = np.sqrt(np.mean(x ** 2, axis=0))
    quality["channel_rms_dbfs"] = [float(20 * np.log10(r + 1e-12)) for r in ch_rms]
    quality["clipped_frac"] = float(np.mean(np.abs(x) >= dcfg["clip_level"]))
    mono = x.mean(axis=1)
    if x.shape[1] > 1:
        ratio = float(np.sqrt(np.mean(mono ** 2)) / (np.mean(ch_rms) + 1e-12))
        quality["downmix_rms_ratio"] = ratio
        quality["downmix_cancellation_risk"] = bool(ratio < 0.5)
    quality["native_duration_s"] = mono.size / sr
    hdr = prov.get("header_duration_s")
    quality["header_duration_mismatch_s"] = (None if hdr is None
                                             else float(abs(hdr - mono.size / sr)))

    target = int(dcfg["analysis_sr"])
    analysis, rs = resample(mono, sr, target, dcfg["resampler"]["window"])
    analysis = quantize(analysis, dcfg.get("quantize_bits"))
    prov["resampling"] = rs
    prov["analysis_sr"] = target
    prov["quantize_bits"] = dcfg.get("quantize_bits")
    prov["downmix"] = "mean of channels" if x.shape[1] > 1 else "mono input"
    quality["analysis_duration_s"] = analysis.size / target
    quality["peak_dbfs"] = float(20 * np.log10(np.max(np.abs(analysis)) + 1e-12))
    quality["rms_dbfs"] = float(10 * np.log10(np.mean(analysis ** 2) + 1e-20))
    return DecodedAudio(native=mono, native_sr=sr, analysis=analysis, analysis_sr=target,
                        provenance=prov, quality=quality)
