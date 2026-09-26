"""T0 triage: container/codec facts, filesystem times and cheap signal statistics for every file.

These fields drive routing and the per-file trace. They are NOT classifier inputs: in DiffSSD the
format alone (sample rate, codec, encoder tag) separates real from fake perfectly, while every
test file shares one format, so a model that used them would learn nothing transferable.
"""
import json
import os
import subprocess

import numpy as np
from scipy.signal import welch

from .. import config


def probe(path):
    out = subprocess.run([config.ffmpeg_bin("ffprobe"), "-v", "error", "-show_format", "-show_streams",
                          "-of", "json", str(path)], capture_output=True, text=True)
    j = json.loads(out.stdout or "{}")
    fmt = j.get("format", {})
    st = next((s for s in j.get("streams", []) if s.get("codec_type") == "audio"), {})
    tags = {k.lower(): v for k, v in {**fmt.get("tags", {}), **st.get("tags", {})}.items()}
    s = os.stat(path)
    return dict(
        container=fmt.get("format_name"), codec=st.get("codec_name"), native_sr=int(st.get("sample_rate", 0) or 0),
        channels=int(st.get("channels", 0) or 0), sample_fmt=st.get("sample_fmt"),
        bits=int(st.get("bits_per_sample", 0) or st.get("bits_per_raw_sample", 0) or 0),
        bitrate=int(fmt.get("bit_rate", 0) or 0), header_duration=float(fmt.get("duration", 0) or 0),
        encoder=tags.get("encoder") or tags.get("software") or "", n_tags=len(tags),
        size_bytes=s.st_size, mtime=s.st_mtime, ctime=s.st_ctime, atime=s.st_atime,
        birthtime=getattr(s, "st_birthtime", np.nan), probe_ok=bool(st))


def signal_stats(x, sr=config.SR):
    """Cheap signal facts on the decoded 16 kHz int16 clip (before canonicalization)."""
    xf = x.astype(np.float64) / 32768.0
    n = len(xf)
    rms = np.sqrt(np.mean(xf ** 2)) + 1e-12
    f, p = welch(xf, sr, nperseg=1024)
    ref = np.median(10 * np.log10(p[(f > 300) & (f < 3000)] + 1e-20))
    above = np.where(10 * np.log10(p + 1e-20) > ref - 45)[0]
    band = lambda lo, hi: 10 * np.log10(p[(f >= lo) & (f < hi)].mean() + 1e-20)
    zeros = (x == 0)
    # longest run of exact digital zeros
    run = 0
    if zeros.any():
        edges = np.diff(np.concatenate([[0], zeros.astype(np.int8), [0]]))
        run = int((np.where(edges == -1)[0] - np.where(edges == 1)[0]).max())
    return dict(
        decoded_duration=n / sr, peak=float(np.abs(xf).max()), rms_dbfs=float(20 * np.log10(rms)),
        dc_offset=float(xf.mean() / rms), clip_frac=float(np.mean(np.abs(x.astype(np.int32)) >= 32767)),
        zero_frac=float(zeros.mean()), zero_run_s=run / sr, cutoff_hz=float(f[above.max()]) if len(above) else 0.0,
        hf_7k_drop_db=float(band(6500, 7000) - band(7500, 8000)), hf_ratio_4k=float(p[f > 4000].sum() / p.sum()))


def triage(path, x16=None):
    t = probe(path)
    if x16 is not None:
        t.update(signal_stats(x16))
        t["duration_mismatch_s"] = abs(t["header_duration"] - t["decoded_duration"]) if t["header_duration"] else np.nan
    return t
