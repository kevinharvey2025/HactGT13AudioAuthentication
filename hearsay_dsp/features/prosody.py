"""Classical prosody and periodicity measurements via Praat (parselmouth).

Pitch: Praat autocorrelation method (To Pitch (ac)), 10 ms step, floor/ceiling
75/500 Hz. F0 is expressed in semitones relative to the clip median, so the
classifier sees contour shape and variability, not the speaker's pitch level
(absolute median F0 is a speaker cue and is reported as a diagnostic only).

Jitter (local), shimmer (local) and HNR use Praat's standard voice-report
settings and are returned only with at least `min_voiced_s` of voicing and
`min_pulses` glottal pulses; otherwise they are missing, never invented.
Pauses are interior low-activity runs >= min_pause_s from the energy-based
activity estimate: a conservative proxy, not breath or word-level timing.
"""

from __future__ import annotations

import numpy as np
import parselmouth
from parselmouth.praat import call

from .common import INSUFFICIENT, OK, ModuleResult


def _runs(mask: np.ndarray):
    """(start, length) of True runs."""
    if mask.size == 0:
        return []
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0]
    return list(zip(starts, ends - starts))


def prosody_module(x: np.ndarray, sr: int, act, cfg: dict) -> ModuleResult:
    pc = cfg["features"]["prosody"]
    params = {**pc, "pitch_method": "Praat To Pitch (ac)", "time_step_s": 0.01,
              "jitter_args": [0, 0, 0.0001, 0.02, 1.3], "shimmer_args": [0, 0, 0.0001, 0.02, 1.3, 1.6],
              "hnr": "To Harmonicity (cc) 0.01 s, periods_per_window 1.0",
              "praat_version": parselmouth.PRAAT_VERSION}
    if act is None:
        return ModuleResult(status=INSUFFICIENT, reason="no activity estimate", params=params)
    snd = parselmouth.Sound(x, sampling_frequency=sr)
    pitch = snd.to_pitch_ac(time_step=0.01, pitch_floor=pc["pitch_floor"],
                            pitch_ceiling=pc["pitch_ceiling"])
    f0 = pitch.selected_array["frequency"]
    voiced = f0 > 0
    voiced_s = float(voiced.sum() * 0.01)
    active_s = float(act.active.sum() * act.hop_s)
    quality = {"q.pros.voiced_s": voiced_s, "q.pros.active_s": active_s}
    if voiced_s < pc["min_voiced_s"] or active_s <= 0:
        return ModuleResult(status=INSUFFICIENT,
                            reason=f"{voiced_s:.2f} s voiced (< {pc['min_voiced_s']} s)",
                            quality=quality, params=params)
    med = float(np.median(f0[voiced]))
    st = np.full(f0.shape, np.nan)
    st[voiced] = 12.0 * np.log2(f0[voiced] / med)
    sv = st[voiced]
    feats = {
        "pros.voiced_per_active": voiced_s / active_s,
        "pros.f0_st_std": float(np.std(sv)),
        "pros.f0_st_iqr": float(np.subtract(*np.percentile(sv, [75, 25]))),
        "pros.f0_st_range90": float(np.subtract(*np.percentile(sv, [95, 5]))),
    }
    d1 = np.diff(st)
    d1 = d1[np.isfinite(d1)]
    d2 = np.diff(st, n=2)
    d2 = d2[np.isfinite(d2)]
    feats["pros.f0_dst_absmed"] = float(np.median(np.abs(d1))) if d1.size else np.nan
    feats["pros.f0_dst_std"] = float(np.std(d1)) if d1.size else np.nan
    feats["pros.f0_d2st_std"] = float(np.std(d2)) if d2.size else np.nan
    feats["pros.octave_jump_rate"] = (float(np.mean(np.abs(d1) > pc["octave_jump_st"]))
                                      if d1.size else np.nan)
    runs = _runs(voiced)
    feats["pros.voiced_segments_per_s"] = len(runs) / active_s
    feats["pros.voiced_segment_med_s"] = float(np.median([n for _, n in runs]) * 0.01)
    # Interior pauses from the activity estimate.
    act_idx = np.where(act.active)[0]
    pauses = []
    if act_idx.size:
        inner = ~act.active[act_idx[0]: act_idx[-1] + 1]
        pauses = [n * act.hop_s for _, n in _runs(inner) if n * act.hop_s >= pc["min_pause_s"]]
    feats["pros.pauses_per_active_s"] = len(pauses) / active_s
    feats["pros.pause_med_s"] = float(np.median(pauses)) if pauses else np.nan
    # Voice-quality measures, only with enough voicing.
    pp = call(snd, "To PointProcess (periodic, cc)", pc["pitch_floor"], pc["pitch_ceiling"])
    n_pulses = int(call(pp, "Get number of points"))
    quality["q.pros.n_pulses"] = n_pulses
    if n_pulses >= pc["min_pulses"]:
        jit = call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        shim = call([snd, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        harm = snd.to_harmonicity_cc(time_step=0.01, minimum_pitch=pc["pitch_floor"],
                                     silence_threshold=0.1, periods_per_window=1.0)
        hv = harm.values[0]
        hv = hv[hv > -200]
        feats["pros.jitter_local"] = float(jit) if np.isfinite(jit) else np.nan
        feats["pros.shimmer_local"] = float(shim) if np.isfinite(shim) else np.nan
        feats["pros.hnr_db"] = float(np.mean(hv)) if hv.size else np.nan
    else:
        feats.update({"pros.jitter_local": np.nan, "pros.shimmer_local": np.nan,
                      "pros.hnr_db": np.nan})
        quality["q.pros.voice_quality_status"] = INSUFFICIENT
    diagnostics = {"diag.pros.f0_median_hz": med,
                   "diag.pros.f0_p5_p95_hz": [float(np.percentile(f0[voiced], 5)),
                                              float(np.percentile(f0[voiced], 95))]}
    return ModuleResult(status=OK, features=feats, quality=quality, diagnostics=diagnostics,
                        params=params)
