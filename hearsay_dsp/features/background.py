"""Background (low-energy frame) statistics and consistency over time.

Low-energy frames come from the classical activity estimate; they are not
guaranteed non-speech. Clip-level measurements describe the background
spectrum relative to the active level (gain invariant). Temporal consistency
compares adjacent ~1 s windows for jumps in background level, background
spectral shape and DC offset, using a robust local-difference rule:

    candidate if jump >= max(absolute_min, median + k * 1.4826 * MAD)

Candidates within merge_s are merged. They are reported as "possible
discontinuity" with their supporting measurements; without boundary labels
there is no claim of localization accuracy, and a candidate is not a finding
of synthesis.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter

from .common import INSUFFICIENT, OK, ModuleResult

BG_BANDS_HZ = [50, 250, 500, 1000, 2000, 3000, 4000, 5500, 7000]
TINY = 1e-20


def _robust_threshold(values: np.ndarray, abs_min: float, k: float) -> float:
    med = float(np.median(values))
    mad = 1.4826 * float(np.median(np.abs(values - med)))
    return max(abs_min, med + k * mad)


def background_module(x: np.ndarray, stft, act, cfg: dict) -> ModuleResult:
    bcfg = cfg["features"]["background"]
    max_hz = float(cfg["features"]["max_hz"])
    params = {**bcfg, "bands_hz": BG_BANDS_HZ, "method": "robust local difference of adjacent windows"}
    if act is None:
        return ModuleResult(status=INSUFFICIENT, reason="no activity estimate", params=params)
    bg = act.background
    n_bg = int(bg.sum())
    quality = {"q.bg.n_frames": n_bg}
    if n_bg < bcfg["min_bg_frames"]:
        return ModuleResult(status=INSUFFICIENT,
                            reason=f"only {n_bg} low-energy frames (< {bcfg['min_bg_frames']})",
                            quality=quality, params=params)
    freqs = stft.freqs
    sel = (freqs >= 50) & (freqs <= max_hz)
    f = freqs[sel]
    p = stft.power[:, sel]
    spec = p[bg].mean(axis=0) + TINY
    spec_db = 10 * np.log10(spec)
    feats = {
        "bg.snr_db": (float(np.median(act.e_db[act.active]) - np.median(act.e_db[bg]))
                      if act.active.any() else np.nan),
        "bg.noise_flatness_db": float(10 * np.log10(np.exp(np.mean(np.log(spec))) / np.mean(spec))),
        "bg.noise_centroid_hz": float((f * spec).sum() / spec.sum()),
        "bg.noise_slope_db_per_khz": float(np.polyfit(f / 1000.0, spec_db, 1)[0]),
        "bg.noise_hf_ratio_db": float(10 * np.log10(spec[f >= 4000].sum() / spec[f < 4000].sum())),
        "bg.noise_tonality_db": float(np.max(spec_db - median_filter(spec_db, size=15, mode="nearest"))),
        "bg.frame_level_std_db": float(np.std(act.e_db[bg])),
    }
    # Temporal consistency over windows. Each window uses its own local floor (its
    # quietest non-silent frames), so a background level that changes mid-clip is
    # still visible; a global floor would label the louder part "active" throughout.
    # A window is used only if its quiet frames are plausibly background: either it
    # contains both quiet and loud frames (>= 10 dB spread) or it is entirely below
    # the global activity threshold. Speech-only windows are skipped.
    hop_s = act.hop_s
    w = max(1, int(round(bcfg["window_s"] / hop_s)))
    h = max(1, int(round(bcfg["hop_s"] / hop_s)))
    n_frames = act.e_db.size
    rms = float(np.sqrt(np.mean(x ** 2)) + 1e-12)
    edges = np.searchsorted(f, BG_BANDS_HZ)
    ns = ~act.digital_silence
    windows = []
    for s in range(0, max(n_frames - w, 0) + 1, h):
        cand = np.arange(s, min(s + w, n_frames))
        cand = cand[ns[cand]]
        if cand.size < bcfg["min_bg_frames_per_window"]:
            continue
        e = act.e_db[cand]
        p20, p90 = np.percentile(e, [20, 90])
        if p90 - p20 < 10.0 and p90 >= act.thr_active_db:
            continue
        idx = cand[e <= p20 + 3.0]
        if idx.size < bcfg["min_bg_frames_per_window"]:
            continue
        bands = np.array([p[idx, edges[i]:edges[i + 1]].mean() for i in range(len(edges) - 1)])
        bands_db = 10 * np.log10(bands + TINY)
        a0 = int(stft.starts[s])
        a1 = int(min(stft.starts[min(s + w, n_frames) - 1] + stft.win_len, x.size))
        windows.append({
            "t": float((stft.starts[s] + (a1 - a0) / 2.0) / stft.sr),
            "floor_db": float(np.median(act.e_db[idx])),
            "shape": bands_db - bands_db.mean(),
            "dc": float(np.mean(x[a0:a1]) / rms),
        })
    quality["q.bg.n_windows"] = len(windows)
    diagnostics = {}
    candidates = []
    if len(windows) >= 2:
        floors = np.array([wd["floor_db"] for wd in windows])
        fj = np.abs(np.diff(floors))
        sj = np.array([np.sqrt(np.mean((b["shape"] - a["shape"]) ** 2))
                       for a, b in zip(windows[:-1], windows[1:])])
        dj = np.abs(np.diff([wd["dc"] for wd in windows]))
        feats.update({
            "bg.floor_std_db": float(floors.std()),
            "bg.max_floor_jump_db": float(fj.max()),
            "bg.max_shape_jump_db": float(sj.max()),
            "bg.max_dc_jump": float(dj.max()),
        })
        f_thr = _robust_threshold(fj, bcfg["floor_jump_min_db"], bcfg["robust_k"])
        s_thr = _robust_threshold(sj, bcfg["shape_jump_min_db"], bcfg["robust_k"])
        raw = []
        for i in range(fj.size):
            if fj[i] >= f_thr or sj[i] >= s_thr:
                raw.append({"time_s": 0.5 * (windows[i]["t"] + windows[i + 1]["t"]),
                            "range_s": [windows[i]["t"], windows[i + 1]["t"]],
                            "floor_jump_db": float(fj[i]), "shape_jump_db": float(sj[i]),
                            "dc_jump": float(dj[i]), "label": "possible discontinuity"})
        for c in raw:  # merge candidates closer than merge_s, keeping the strongest
            if candidates and c["time_s"] - candidates[-1]["time_s"] < bcfg["merge_s"]:
                if c["floor_jump_db"] + c["shape_jump_db"] > (candidates[-1]["floor_jump_db"]
                                                               + candidates[-1]["shape_jump_db"]):
                    candidates[-1] = c
            else:
                candidates.append(c)
        dur = x.size / stft.sr
        feats["bg.candidates_per_10s"] = float(len(candidates) / dur * 10.0)
        diagnostics["diag.bg.thresholds"] = {"floor_jump_db": f_thr, "shape_jump_db": s_thr}
        quality["q.bg.temporal_status"] = OK
    else:
        for k in ("bg.floor_std_db", "bg.max_floor_jump_db", "bg.max_shape_jump_db",
                  "bg.max_dc_jump", "bg.candidates_per_10s"):
            feats[k] = np.nan
        quality["q.bg.temporal_status"] = INSUFFICIENT
    return ModuleResult(status=OK, features=feats, quality=quality, diagnostics=diagnostics,
                        params=params, candidates=candidates)
