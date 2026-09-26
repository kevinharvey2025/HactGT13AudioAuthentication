"""Spectral structure: band energy, shape, flux and long-term fine structure.

Measurements are computed on active frames of the analysis-rate STFT, over an
eligible band [50 Hz, eligible_max]. eligible_max is the configured limit
(7 kHz) reduced to the estimated occupied bandwidth, so a band-limited clip
gets NaN (unavailable) for full-band features instead of zero-energy
evidence. Low-band (50 Hz - 4 kHz) versions stay available for telephone-band
audio. Bandwidth itself reflects channel and processing as well as synthesis
and is reported as a diagnostic, never as a classifier input.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d
from scipy.signal import welch

from .common import INSUFFICIENT, OK, ModuleResult, robust_stats

BAND_EDGES_HZ = [50, 250, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000]
TINY = 1e-20


def occupied_bandwidth(x: np.ndarray, sr: int, rel_db: float = 50.0, cliff_db: float = 25.0,
                       below_hz: float = 700.0, gap_hz: float = 300.0,
                       min_edge_hz: float = 2500.0) -> dict:
    """Estimated upper band edge of the content, in Hz.

    Two rules, the lower result wins:
      level: highest frequency whose smoothed PSD is within rel_db of the median
             0.3-3 kHz level (deep band limits over a very low floor);
      cliff: highest frequency f_e >= min_edge_hz such that everything from
             f_e + gap_hz up to Nyquist is at least cliff_db below the median level
             of [f_e - below_hz, f_e] (band limits over a moderate noise floor, e.g.
             telephone, codec or resampler low-pass). Natural spectral tilt and
             high-frequency fricative energy do not satisfy it; the minimum edge
             keeps narrowband signals (tones) from triggering it.
    """
    nper = int(2 ** np.ceil(np.log2(0.032 * sr)))
    if x.size < nper:
        nper = int(2 ** np.floor(np.log2(max(x.size, 16))))
    f, p = welch(x, fs=sr, nperseg=nper, noverlap=nper // 2, window="hann", detrend=False)
    pdb = 10 * np.log10(p + 1e-30)
    df = f[1] - f[0]
    pdb_s = uniform_filter1d(pdb, max(1, int(round(200.0 / df))))
    ref_sel = (f >= 300) & (f <= 3000)
    ref = float(np.median(pdb_s[ref_sel])) if ref_sel.any() else float(np.median(pdb_s))
    above = np.where(pdb_s >= ref - rel_db)[0]
    bw_level = float(f[above.max()]) if above.size else 0.0
    wb = max(2, int(round(below_hz / df)))
    gap = max(1, int(round(gap_hz / df)))
    lo_limit = max(wb, int(np.searchsorted(f, min_edge_hz)))
    bw_cliff = None
    for i in range(f.size - gap - 1, lo_limit - 1, -1):
        if np.median(pdb_s[i - wb:i]) - pdb_s[i + gap:].max() >= cliff_db:
            bw_cliff = float(f[i])
            break
    bw = min(bw_level, bw_cliff) if bw_cliff is not None else bw_level
    lo_i, hi_i = np.searchsorted(f, bw - 500), np.searchsorted(f, bw + 500)
    drop = float(pdb_s[max(lo_i, 0)] - pdb_s[min(hi_i, f.size - 1)]) if bw > 0 else np.nan
    return {"bw_hz": bw, "bw_level_rule_hz": bw_level, "bw_cliff_rule_hz": bw_cliff, "ref_db": ref,
            "edge_drop_db": drop, "nyquist_hz": sr / 2.0}


def _frame_measures(power: np.ndarray, freqs: np.ndarray, fmax: float) -> dict:
    sel = (freqs >= 50.0) & (freqs <= fmax)
    p = power[:, sel] + TINY
    f = freqs[sel]
    tot = p.sum(axis=1)
    centroid = (p * f).sum(axis=1) / tot
    spread = np.sqrt((p * (f[None, :] - centroid[:, None]) ** 2).sum(axis=1) / tot)
    cum = np.cumsum(p, axis=1) / tot[:, None]
    roll85 = f[np.argmax(cum >= 0.85, axis=1)]
    roll95 = f[np.argmax(cum >= 0.95, axis=1)]
    flat = 10 * np.log10(np.exp(np.mean(np.log(p), axis=1)) / np.mean(p, axis=1))
    mag = np.sqrt(p)
    mag = mag / mag.sum(axis=1, keepdims=True)
    return {"centroid_hz": centroid, "spread_hz": spread, "rolloff85_hz": roll85,
            "rolloff95_hz": roll95, "flatness_db": flat, "mag": mag, "tot": tot}


def spectral_module(stft, act, x: np.ndarray, native: np.ndarray, native_sr: int,
                    cfg: dict) -> ModuleResult:
    fcfg = cfg["features"]
    max_hz = float(fcfg["max_hz"])
    lb_max = float(fcfg["low_band_max_hz"])
    obw_native = occupied_bandwidth(native, native_sr)
    obw = occupied_bandwidth(x, stft.sr)
    eligible_max = float(min(max_hz, 0.95 * native_sr / 2.0, obw["bw_hz"]))
    diagnostics = {
        "diag.spec.native_occupied_bw_hz": obw_native["bw_hz"],
        "diag.spec.native_edge_drop_db": obw_native["edge_drop_db"],
        "diag.spec.native_nyquist_hz": obw_native["nyquist_hz"],
        "diag.spec.eligible_range_hz": [50.0, eligible_max],
    }
    quality = {"q.spec.occupied_bw_hz": obw["bw_hz"], "q.spec.eligible_max_hz": eligible_max}
    params = {"stft": fcfg["stft"], "band_edges_hz": BAND_EDGES_HZ, "max_hz": max_hz,
              "low_band_max_hz": lb_max, "summaries": "median/IQR/p10/p90 over active frames",
              **fcfg["spectral"]}
    if act is None or act.active.sum() < 20:
        return ModuleResult(status=INSUFFICIENT, reason="fewer than 20 active frames",
                            quality=quality, diagnostics=diagnostics, params=params)
    a = act.active
    power, freqs = stft.power[a], stft.freqs
    feats: dict = {}
    full_ok = eligible_max >= 0.98 * max_hz
    low_ok = eligible_max >= 0.98 * lb_max
    quality["q.spec.full_band_available"] = bool(full_ok)
    quality["q.spec.low_band_available"] = bool(low_ok)

    fb = _frame_measures(power, freqs, max_hz)
    names = ["centroid_hz", "spread_hz", "rolloff85_hz", "rolloff95_hz", "flatness_db"]
    for n in names:
        stats = robust_stats(fb[n], f"spec.fb_{n}")
        feats.update(stats if full_ok else {k: np.nan for k in stats})
    sel_hi = (freqs >= 4000.0) & (freqs <= max_hz)
    ph = power[:, sel_hi] + TINY
    flat_hi = 10 * np.log10(np.exp(np.mean(np.log(ph), axis=1)) / np.mean(ph, axis=1))
    stats = robust_stats(flat_hi, "spec.fb_hiflatness_db")
    feats.update(stats if full_ok else {k: np.nan for k in stats})
    # Flux between consecutive frames that are both active.
    idx = np.where(a)[0]
    pair = np.where(np.diff(idx) == 1)[0]
    if pair.size >= 5:
        flux = np.linalg.norm(fb["mag"][pair + 1] - fb["mag"][pair], axis=1)
        stats = robust_stats(flux, "spec.fb_flux")
    else:
        stats = {f"spec.fb_flux_{w}": np.nan for w in ("med", "iqr", "p10", "p90")}
    feats.update(stats if full_ok else {k: np.nan for k in stats})

    lb = _frame_measures(power, freqs, lb_max)
    for n in ("centroid_hz", "rolloff85_hz", "flatness_db"):
        stats = robust_stats(lb[n], f"spec.lb_{n}", which=("med", "iqr"))
        feats.update(stats if low_ok else {k: np.nan for k in stats})

    band_tot = power[:, (freqs >= 50) & (freqs <= max_hz)].sum(axis=1) + TINY
    for lo, hi_edge in zip(BAND_EDGES_HZ[:-1], BAND_EDGES_HZ[1:]):
        sel = (freqs >= lo) & (freqs < hi_edge)
        ratio = 10 * np.log10(power[:, sel].sum(axis=1) / band_tot + TINY)
        stats = robust_stats(ratio, f"spec.band_{lo}_{hi_edge}_db", which=("med", "iqr"))
        available = full_ok or hi_edge <= eligible_max
        feats.update(stats if available else {k: np.nan for k in stats})

    ns = ~act.digital_silence
    feats["spec.energy_std_db"] = float(np.std(act.e_db[ns])) if ns.sum() > 1 else np.nan

    # Exploratory: periodic ripple in the long-term spectrum after removing the envelope.
    sel = (freqs >= 50) & (freqs <= (max_hz if full_ok else eligible_max))
    ltas_db = 10 * np.log10(power[:, sel].mean(axis=0) + TINY)
    df = freqs[1] - freqs[0]
    k = int(round(fcfg["spectral"]["ltas_smooth_hz"] / df)) | 1
    resid = ltas_db - median_filter(ltas_db, size=k, mode="nearest")
    resid = resid - resid.mean()
    acf = np.correlate(resid, resid, mode="full")[resid.size - 1:]
    acf = acf / (acf[0] + TINY)
    lag_lo, lag_hi = (int(round(h / df)) for h in fcfg["spectral"]["ripple_lag_hz"])
    lag_hi = min(lag_hi, acf.size - 1)
    if lag_hi > lag_lo:
        j = lag_lo + int(np.argmax(acf[lag_lo:lag_hi + 1]))
        feats["spec.ltas_ripple_acf_max"] = float(acf[j])
        diagnostics["diag.spec.ltas_ripple_lag_hz"] = float(j * df)
    else:
        feats["spec.ltas_ripple_acf_max"] = np.nan
    feats["spec.ltas_ripple_std_db"] = float(resid.std())
    return ModuleResult(status=OK, features=feats, quality=quality, diagnostics=diagnostics,
                        params=params)
