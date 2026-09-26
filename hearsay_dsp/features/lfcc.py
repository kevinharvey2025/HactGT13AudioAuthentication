"""LFCC front-end.

`lfcc_bp` reproduces the ASVspoof 2021 LA MATLAB baseline front-end
(Baseline-LFCC-GMM/matlab/LFCC/lfcc_bp.m, Sahidullah; pipeline by Todisco,
EURECOM, BSD-3-Clause), step by step:

  buffer(x, L, L/2, 'nodelay')    frames with 50% overlap, last one zero-padded
  hamming(L)                      symmetric Hamming window
  |fft(frame, NFFT)|^2            power spectrum, bins 0..NFFT/2
  bins closest to low/high Hz     inclusive band selection
  trimf on linspace(low, high, n_filter + 2) edges   triangular filterbank
  log10(energy + eps), dct        MATLAB dct = orthonormal DCT-II; first n_coeff
  Deltas(hlen=1)                  (x[t+1] - x[t-1]) / 2, edges replicated; twice

Official configuration: 30 ms window, NFFT 1024, 70 filters over 0-4 kHz,
19 coefficients including c0, plus deltas and delta-deltas (57 dims). The
building blocks are unit-tested against MATLAB semantics; numerical parity with
MATLAB itself was not tested (MATLAB/Octave unavailable).

The "custom" variant keeps the official framing but analyses 0-7 kHz and drops
c0 (a pure gain change only shifts c0 under an orthonormal DCT).
"""

from __future__ import annotations

import numpy as np
from scipy.fft import dct

from .common import DIGITAL_SILENCE_RMS, INSUFFICIENT, OK, ModuleResult

MATLAB_EPS = float(np.finfo(float).eps)


def matlab_buffer_nodelay(x: np.ndarray, n: int, p: int) -> np.ndarray:
    """MATLAB buffer(x, n, p, 'nodelay') returned as rows [frames, n]."""
    length, hop = x.size, n - p
    m = max(1, int(np.ceil((length - p) / hop)))
    total = (m - 1) * hop + n
    padded = np.zeros(max(total, length))
    padded[:length] = x
    idx = np.arange(m)[:, None] * hop + np.arange(n)[None, :]
    return padded[idx]


def trimf(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """MATLAB Fuzzy Logic Toolbox trimf(x, [a b c])."""
    y = np.zeros_like(x, dtype=float)
    if a != b:
        m = (a < x) & (x < b)
        y[m] = (x[m] - a) / (b - a)
    if b != c:
        m = (b < x) & (x < c)
        y[m] = (c - x[m]) / (c - b)
    y[x == b] = 1.0
    return y


def matlab_deltas(feat: np.ndarray, hlen: int = 1) -> np.ndarray:
    """Deltas() from lfcc_bp.m along time; feat is [frames, dims]."""
    win = np.arange(hlen, -hlen - 1, -1, dtype=float)
    xx = np.concatenate([np.repeat(feat[:1], hlen, axis=0), feat,
                         np.repeat(feat[-1:], hlen, axis=0)], axis=0)
    t = feat.shape[0]
    d = np.zeros_like(feat, dtype=float)
    for k, wk in enumerate(win):
        if wk != 0:
            d += wk * xx[2 * hlen - k: 2 * hlen - k + t]
    return d / (2.0 * np.sum(np.arange(1, hlen + 1) ** 2))


def linear_filterbank(fs: int, nfft: int, n_filter: int, low_hz: float, high_hz: float):
    f = (fs / 2.0) * np.linspace(0.0, 1.0, nfft // 2 + 1)
    lo = int(np.argmin(np.abs(f - low_hz)))
    hi = int(np.argmin(np.abs(f - high_hz)))
    fsub = f[lo:hi + 1]
    edges = np.linspace(low_hz, high_hz, n_filter + 2)
    fb = np.stack([trimf(fsub, edges[i], edges[i + 1], edges[i + 2]) for i in range(n_filter)],
                  axis=1)
    return lo, hi, fb


def lfcc_bp(x: np.ndarray, fs: int, window_ms: float = 30, nfft: int = 1024,
            n_filter: int = 70, n_coeff: int = 19, low_hz: float = 0.0,
            high_hz: float = 4000.0):
    """Returns (stat, delta, double_delta, frames) with features as [frames, n_coeff]."""
    frame_len = int(round(fs / 1000.0 * window_ms))
    frames = matlab_buffer_nodelay(x, frame_len, frame_len // 2)
    y = frames * np.hamming(frame_len)
    power = np.abs(np.fft.fft(y, n=nfft, axis=1)[:, : nfft // 2 + 1]) ** 2
    lo, hi, fb = linear_filterbank(fs, nfft, n_filter, low_hz, high_hz)
    energies = power[:, lo:hi + 1] @ fb
    stat = dct(np.log10(energies + MATLAB_EPS), type=2, norm="ortho", axis=1)[:, :n_coeff]
    delta = matlab_deltas(stat, 1)
    double_delta = matlab_deltas(delta, 1)
    return stat, delta, double_delta, frames


def lfcc_module(x: np.ndarray, sr: int, act, cfg: dict) -> ModuleResult:
    lcfg = cfg["features"]["lfcc"]
    off = lcfg["official"]
    s, d, dd, frames = lfcc_bp(x, sr, off["window_ms"], off["nfft"], off["n_filter"],
                               off["n_coeff"], off["low_hz"], off["high_hz"])
    official = np.concatenate([s, d, dd], axis=1).astype(np.float32)

    cus = lcfg["custom"]
    cs, cd, cdd, _ = lfcc_bp(x, sr, cus["window_ms"], cus["nfft"], cus["n_filter"],
                             cus["n_coeff"], cus["low_hz"], cus["high_hz"])
    if cus.get("drop_c0", True):
        cs, cd, cdd = cs[:, 1:], cd[:, 1:], cdd[:, 1:]
    custom = np.concatenate([cs, cd, cdd], axis=1).astype(np.float32)

    ms = np.mean(frames ** 2, axis=1)
    nonsilent = np.sqrt(ms) >= DIGITAL_SILENCE_RMS
    e_db = 10.0 * np.log10(ms + 1e-20)
    active = nonsilent & (e_db >= act.thr_active_db) if act is not None else nonsilent.copy()

    arrays = {"official": official, "custom": custom,
              "nonsilent": nonsilent, "active": active}
    quality = {"q.lfcc.n_frames": int(frames.shape[0]),
               "q.lfcc.n_nonsilent": int(nonsilent.sum()),
               "q.lfcc.n_active": int(active.sum())}
    params = {"official": off, "custom": cus, "hop_ms": off["window_ms"] / 2.0,
              "reference": "ASVspoof 2021 LA Baseline-LFCC-GMM matlab/LFCC/lfcc_bp.m"}
    use = active if active.sum() >= 20 else nonsilent
    if use.sum() < 20:
        return ModuleResult(status=INSUFFICIENT, reason="fewer than 20 non-silent LFCC frames",
                            quality=quality, params=params, arrays=arrays)
    feats = {}
    stat = cs[use]
    dl = cd[use]
    for i in range(stat.shape[1]):
        feats[f"cep.c{i + 1:02d}_mean"] = float(stat[:, i].mean())
        feats[f"cep.c{i + 1:02d}_std"] = float(stat[:, i].std())
        feats[f"cep.d{i + 1:02d}_std"] = float(dl[:, i].std())
    quality["q.lfcc.summary_frames"] = "active" if use is active else "nonsilent"
    return ModuleResult(status=OK, features=feats, quality=quality, params=params, arrays=arrays)
