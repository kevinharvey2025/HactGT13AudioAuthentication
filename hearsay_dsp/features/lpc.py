"""Linear-prediction residual measurements.

Autocorrelation-method LPC (Hamming-windowed frame, Levinson-Durbin) per
active frame; the residual is obtained by inverse filtering the unwindowed
frame with its preceding samples as filter history. Order defaults to
fs/1000 + 2 (18 at 16 kHz). Frames with a non-positive prediction error or a
reflection coefficient |k| >= 1 are counted as unstable and skipped.

These measure how predictable the waveform is and how impulsive/periodic the
excitation is. They also depend on phonetic content and recording noise; the
ablation decides whether they carry authenticity information.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import lfilter
from scipy.stats import kurtosis

from .common import INSUFFICIENT, OK, ModuleResult, robust_stats


def levinson_durbin(r: np.ndarray, order: int):
    """Returns (a, err, k) with a[0] = 1 and A(z) = sum a_i z^-i."""
    a = np.zeros(order + 1)
    a[0] = 1.0
    err = float(r[0])
    ks = np.zeros(order)
    for i in range(1, order + 1):
        if err <= 0:
            return a, err, ks
        acc = r[i] + np.dot(a[1:i], r[i - 1:0:-1])
        k = -acc / err
        a[1:i] = a[1:i] + k * a[i - 1:0:-1]
        a[i] = k
        ks[i - 1] = k
        err *= (1.0 - k * k)
    return a, err, ks


def autocorr(x: np.ndarray, max_lag: int) -> np.ndarray:
    n = x.size
    nfft = int(2 ** np.ceil(np.log2(2 * n)))
    spec = np.fft.rfft(x, nfft)
    return np.fft.irfft(spec * np.conj(spec), nfft)[: max_lag + 1]


def lpc_module(x: np.ndarray, stft, act, cfg: dict) -> ModuleResult:
    lcfg = cfg["features"]["lpc"]
    sr = stft.sr
    order = int(lcfg["order"])
    max_hz = float(cfg["features"]["max_hz"])
    params = {"order": order, "method": "autocorrelation + Levinson-Durbin",
              "frame_s": stft.win_len / sr, "hop_s": stft.hop / sr, "window": "hamming",
              "residual_acf_lag_ms": [2.5, 15.0], "flatness_band_hz": [50, max_hz]}
    if act is None or act.active.sum() < lcfg["min_active_frames"]:
        return ModuleResult(status=INSUFFICIENT,
                            reason=f"fewer than {lcfg['min_active_frames']} active frames",
                            params=params)
    win = np.hamming(stft.win_len)
    lag_lo, lag_hi = int(0.0025 * sr), int(0.015 * sr)
    freqs = np.fft.rfftfreq(stft.win_len, 1.0 / sr)
    band = (freqs >= 50) & (freqs <= max_hz)
    hann = np.hanning(stft.win_len)
    gains, kurts, peaks, flats = [], [], [], []
    unstable = 0
    for i in np.where(act.active)[0]:
        start = int(stft.starts[i])
        frame = x[start: start + stft.win_len]
        if frame.size < stft.win_len:
            continue
        r = autocorr(frame * win, order)
        r[0] *= 1.0 + 1e-9
        a, err, ks = levinson_durbin(r, order)
        if err <= 0 or np.any(np.abs(ks) >= 1.0):
            unstable += 1
            continue
        hist = x[max(0, start - order): start]
        hist = np.concatenate([np.zeros(order - hist.size), hist])
        e = lfilter(a, [1.0], np.concatenate([hist, frame]))[order:]
        ee = float(np.sum(e ** 2))
        if ee <= 0:
            unstable += 1
            continue
        gains.append(10 * np.log10(np.sum(frame ** 2) / ee))
        kurts.append(kurtosis(e, fisher=True, bias=True))
        ac = autocorr(e - e.mean(), lag_hi)
        peaks.append(float(np.max(ac[lag_lo:lag_hi + 1]) / (ac[0] + 1e-20)))
        pe = np.abs(np.fft.rfft(e * hann)) ** 2 + 1e-20
        pe = pe[band]
        flats.append(10 * np.log10(np.exp(np.mean(np.log(pe))) / np.mean(pe)))
    n = len(gains)
    quality = {"q.lpc.n_frames": n,
               "q.lpc.unstable_frac": float(unstable / max(unstable + n, 1))}
    if n < 20:
        return ModuleResult(status=INSUFFICIENT, reason=f"only {n} stable LPC frames",
                            quality=quality, params=params)
    feats = {}
    feats.update(robust_stats(gains, "lpc.pred_gain_db", which=("med", "iqr")))
    feats.update(robust_stats(kurts, "lpc.res_kurtosis", which=("med", "iqr")))
    feats.update(robust_stats(peaks, "lpc.res_acf_peak", which=("med", "iqr")))
    feats.update(robust_stats(flats, "lpc.res_flatness_db", which=("med", "iqr")))
    return ModuleResult(status=OK, features=feats, quality=quality, params=params)
