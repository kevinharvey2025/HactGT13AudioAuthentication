"""Electric network frequency (mains hum): presence check and local continuity.

Presence: narrowband peak power at 50 Hz and 60 Hz (both tested: location is
unknown) against the median power 2-6 Hz away, from a Welch spectrum of the
native-rate signal decimated to 1 kHz. Harmonics are reported as diagnostics
only, because male-voice harmonics can sit near 100-180 Hz.

Continuity tracking runs only when a component is present and the clip is at
least `min_duration_s` long; two-to-five-second clips cannot support
meaningful ENF analysis. Presence proves neither authenticity nor location,
and absence is neutral. This module is diagnostic only, never a classifier
input, and does no timestamp verification (that needs reference ENF data).
"""

from __future__ import annotations

from math import gcd

import numpy as np
from scipy.signal import resample_poly, welch

from .common import INSUFFICIENT, OK, ModuleResult

ENF_SR = 1000


def _peak_snr(f: np.ndarray, p: np.ndarray, fc: float) -> float:
    peak = p[(f >= fc - 0.6) & (f <= fc + 0.6)]
    side = p[((f >= fc - 6) & (f <= fc - 2)) | ((f >= fc + 2) & (f <= fc + 6))]
    if peak.size == 0 or side.size == 0:
        return np.nan
    return float(10 * np.log10(peak.max() / (np.median(side) + 1e-30) + 1e-30))


def track_frequency(y: np.ndarray, sr: int, fc: float, win_s: float, hop_s: float) -> np.ndarray:
    """Peak frequency near fc per frame, quadratic interpolation on a zero-padded FFT."""
    win = int(win_s * sr)
    hop = int(hop_s * sr)
    nfft = 1 << 15
    freqs = np.fft.rfftfreq(nfft, 1.0 / sr)
    band = np.where((freqs >= fc - 1.0) & (freqs <= fc + 1.0))[0]
    out = []
    for s in range(0, y.size - win + 1, hop):
        mag = np.abs(np.fft.rfft(y[s:s + win] * np.hanning(win), nfft))
        i = band[np.argmax(mag[band])]
        a, b, c = np.log(mag[i - 1: i + 2] + 1e-30)
        denom = a - 2 * b + c
        delta = 0.5 * (a - c) / denom if denom != 0 else 0.0
        out.append(freqs[i] + delta * (freqs[1] - freqs[0]))
    return np.array(out)


def enf_module(native: np.ndarray, native_sr: int, cfg: dict) -> ModuleResult:
    ec = cfg["features"]["enf"]
    params = {**ec, "analysis_sr": ENF_SR, "presence": "peak vs median at 2-6 Hz offset"}
    g = gcd(int(native_sr), ENF_SR)
    y = resample_poly(native, ENF_SR // g, int(native_sr) // g)
    dur = y.size / ENF_SR
    nper = min(y.size, 2 * ENF_SR)
    f, p = welch(y, fs=ENF_SR, nperseg=nper, noverlap=nper // 2)
    snrs = {}
    for nom in ec["nominal_hz"]:
        for h in range(1, int(ec["harmonics"]) + 1):
            snrs[f"{int(nom)}x{h}"] = _peak_snr(f, p, nom * h)
    fund = {nom: snrs[f"{int(nom)}x1"] for nom in ec["nominal_hz"]}
    best_nom = max(fund, key=lambda k: -np.inf if np.isnan(fund[k]) else fund[k])
    diagnostics = {"diag.enf.peak_snr_db": snrs, "diag.enf.best_nominal_hz": best_nom,
                   "diag.enf.best_snr_db": fund[best_nom], "diag.enf.duration_s": dur}
    quality = {"q.enf.present": bool(fund[best_nom] >= ec["presence_snr_db"])}
    if not quality["q.enf.present"]:
        return ModuleResult(status=INSUFFICIENT, reason="no reliable narrowband mains component",
                            quality=quality, diagnostics=diagnostics, params=params)
    if dur < ec["min_duration_s"]:
        return ModuleResult(
            status=INSUFFICIENT,
            reason=(f"mains-like component at {best_nom:.0f} Hz ({fund[best_nom]:.1f} dB) but clip is "
                    f"{dur:.1f} s (< {ec['min_duration_s']} s) for continuity analysis"),
            quality=quality, diagnostics=diagnostics, params=params)
    track = track_frequency(y, ENF_SR, best_nom, ec["track_win_s"], ec["track_hop_s"])
    steps = np.abs(np.diff(track))
    diagnostics.update({"diag.enf.track_std_hz": float(track.std()),
                        "diag.enf.max_step_hz": float(steps.max()) if steps.size else np.nan,
                        "diag.enf.n_track_frames": int(track.size)})
    return ModuleResult(status=OK, quality=quality, diagnostics=diagnostics, params=params)
