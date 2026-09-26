"""Phase features: modified group delay cepstra and instantaneous-frequency continuity.

Modified group delay (Hegde & Murthy; used for synthetic speech detection by
Wu et al.) is computed without phase unwrapping:

    tau(k) = (X_R Y_R + X_I Y_I) / S(k)^(2 gamma),   Y = DFT(n x[n])
    tau_m(k) = sign(tau) |tau|^alpha,  MGDCC = DCT(tau_m)[1 : n_ceps + 1]

where S is the cepstrally smoothed magnitude (lifter length L). S^(2 gamma)
also keeps the ratio stable near spectral zeros. Frames are scaled to unit
RMS first, so the features do not depend on gain.

Instantaneous frequency uses the STFT phase advance between frames minus the
expected advance 2 pi k H / N, wrapped to [-pi, pi). A frame's IF jump is the
power-weighted mean |IF_t - IF_{t-1}| over strong bins. Spikes are reported as
possible phase discontinuities (diagnostic): natural onsets and plosives also
produce them.
"""

from __future__ import annotations

import numpy as np
from scipy.fft import dct

from .common import INSUFFICIENT, OK, ModuleResult, robust_stats, robust_z


def princarg(phase: np.ndarray) -> np.ndarray:
    return (phase + np.pi) % (2 * np.pi) - np.pi


def modified_group_delay(frames: np.ndarray, nfft: int, alpha: float, gamma: float,
                         lifter: int) -> np.ndarray:
    frame_len = frames.shape[1]
    rms = np.sqrt(np.mean(frames ** 2, axis=1, keepdims=True)) + 1e-12
    xw = frames / rms * np.hamming(frame_len)
    n = np.arange(frame_len)
    spec_x = np.fft.rfft(xw, nfft, axis=1)
    spec_y = np.fft.rfft(xw * n, nfft, axis=1)
    logmag = np.log(np.abs(spec_x) + 1e-10)
    ceps = np.fft.irfft(logmag, nfft, axis=1)
    lift = np.zeros(nfft)
    lift[:lifter] = 1.0
    lift[nfft - lifter + 1:] = 1.0
    smooth = np.exp(np.fft.rfft(ceps * lift, nfft, axis=1).real)
    tau = (spec_x.real * spec_y.real + spec_x.imag * spec_y.imag) / (smooth ** (2 * gamma))
    return np.sign(tau) * np.abs(tau) ** alpha


def instantaneous_frequency(spec: np.ndarray, hop: int, nfft: int, sr: int) -> np.ndarray:
    """IF in Hz for frames 1..T-1 (row t uses frames t-1 and t)."""
    k = np.arange(spec.shape[1])
    phi = np.angle(spec)
    expected = 2 * np.pi * k * hop / nfft
    dev = princarg(phi[1:] - phi[:-1] - expected[None, :])
    return (k[None, :] / nfft + dev / (2 * np.pi * hop)) * sr


def if_jump_series(spec: np.ndarray, power: np.ndarray, freqs: np.ndarray, hop: int, nfft: int,
                   sr: int, max_hz: float, rel_power_db: float) -> np.ndarray:
    """Per-frame weighted mean |delta IF| (Hz); entry t compares IF(t) with IF(t-1); NaN for t < 2."""
    inst = instantaneous_frequency(spec, hop, nfft, sr)
    band = (freqs >= 50) & (freqs <= max_hz)
    p = power * band[None, :]
    thr = p.max(axis=1, keepdims=True) * 10 ** (rel_power_db / 10.0)
    w = np.where(p >= thr, p, 0.0)
    out = np.full(spec.shape[0], np.nan)
    diff = np.abs(inst[1:] - inst[:-1])            # compares IF at t and t-1, t >= 2
    wt = w[2:]
    denom = wt.sum(axis=1)
    ok = denom > 0
    vals = np.full(diff.shape[0], np.nan)
    vals[ok] = (diff[ok] * wt[ok]).sum(axis=1) / denom[ok]
    out[2:] = vals
    return out


def phase_module(stft, act, cfg: dict) -> ModuleResult:
    pcfg = cfg["features"]["phase"]
    max_hz = float(cfg["features"]["max_hz"])
    params = {**pcfg, "frame_s": stft.win_len / stft.sr, "hop_s": stft.hop / stft.sr,
              "nfft": stft.nfft, "band_hz": [50, max_hz],
              "reference": "modified group delay (Hegde & Murthy 2007); IF by phase vocoder"}
    if act is None or act.active.sum() < pcfg["min_active_frames"]:
        return ModuleResult(status=INSUFFICIENT,
                            reason=f"fewer than {pcfg['min_active_frames']} active frames",
                            params=params)
    band = (stft.freqs >= 50) & (stft.freqs <= max_hz)
    mgd = modified_group_delay(stft.frames[act.active], stft.nfft, pcfg["alpha"], pcfg["gamma"],
                               int(pcfg["lifter"]))[:, band]
    ceps = dct(mgd, type=2, norm="ortho", axis=1)[:, 1:int(pcfg["n_ceps"]) + 1]
    feats = {}
    for i in range(ceps.shape[1]):
        feats[f"phase.mgdcc{i + 1:02d}_mean"] = float(ceps[:, i].mean())
        feats[f"phase.mgdcc{i + 1:02d}_std"] = float(ceps[:, i].std())

    jumps = if_jump_series(stft.spec, stft.power, stft.freqs, stft.hop, stft.nfft, stft.sr,
                           max_hz, pcfg["rel_power_db"])
    a = act.active
    usable = np.zeros_like(a)
    usable[2:] = a[2:] & a[1:-1] & a[:-2]
    usable &= np.isfinite(jumps)
    quality = {"q.phase.n_mgd_frames": int(a.sum()), "q.phase.n_if_frames": int(usable.sum())}
    candidates = []
    if usable.sum() >= 10:
        j = jumps[usable]
        feats.update(robust_stats(j, "phase.ifjump_hz", which=("med", "p90")))
        z = robust_z(j)
        feats["phase.ifjump_maxz"] = float(np.max(z))
        times = stft.times[usable]
        for t, zz, jj in zip(times, z, j):
            if zz >= pcfg["jump_z"]:
                candidates.append({"time_s": float(t), "if_jump_hz": float(jj), "robust_z": float(zz),
                                   "label": "possible phase discontinuity"})
    else:
        feats.update({"phase.ifjump_hz_med": np.nan, "phase.ifjump_hz_p90": np.nan,
                      "phase.ifjump_maxz": np.nan})
    return ModuleResult(status=OK, features=feats, quality=quality, params=params,
                        candidates=candidates)
