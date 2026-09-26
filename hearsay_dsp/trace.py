"""Per-file trace and template explanations.

Three kinds of statements are kept apart:
  observations    measured facts with units (e.g. a background-level jump of 9 dB at 2.4 s)
  contributions   how the model's calculation moved the score (logit units); these
                  explain the model, not physical causation, and are not proof of forgery
  interpretation  a cautious template sentence tied to the calibrated probability
No generator family, "vocoder fingerprint" or "confirmed splice" is ever asserted.
"""

from __future__ import annotations

import numpy as np

from .routing import DIAGNOSTIC_ONLY

GROUP_OF_MODULE = {"lfcc": ("cep", "gmm"), "spectral": ("spec",), "lpc": ("lpc",),
                   "background": ("bg",), "phase": ("phase",), "prosody": ("pros",)}


def _fmt(v, nd=2):
    return "n/a" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{nd}f}"


def module_findings(modules: dict) -> dict:
    out = {}
    for name, res in modules.items():
        st = res.get("status")
        f = {"status": st}
        if res.get("reason"):
            f["reason"] = res["reason"]
        d, q = res.get("diagnostics", {}) or {}, res.get("quality", {}) or {}
        cands = res.get("candidates") or []
        if name == "background" and st == "ok":
            if cands:
                f["finding"] = (f"{len(cands)} possible background discontinuit"
                                f"{'y' if len(cands) == 1 else 'ies'} (level/spectral-shape jump); "
                                "authenticity unresolved")
                f["candidate_times_seconds"] = [round(c["time_s"], 2) for c in cands]
                f["candidates"] = cands
            else:
                f["finding"] = "no background discontinuity above threshold"
        elif name == "phase" and st == "ok":
            f["finding"] = (f"{len(cands)} frame(s) with unusually large instantaneous-frequency jumps "
                            "(onsets and plosives can cause these)" if cands
                            else "no outlying instantaneous-frequency jumps")
            if cands:
                f["candidate_times_seconds"] = [round(c["time_s"], 2) for c in cands[:10]]
        elif name == "enf":
            f["finding"] = (res.get("reason") or "mains component tracked") + \
                           f" (best {_fmt(d.get('diag.enf.best_snr_db'), 1)} dB at " \
                           f"{d.get('diag.enf.best_nominal_hz')} Hz; presence/absence is neutral)"
        elif name == "container" and st == "ok":
            notes = []
            if d.get("diag.container.extension_mismatch"):
                notes.append(f"extension {d.get('diag.container.extension')} but content is "
                             f"{d.get('diag.container.sniffed_format')}")
            info = d.get("diag.container.riff_info") or {}
            if info:
                notes.append("RIFF INFO " + ", ".join(f"{k}={v}" for k, v in info.items()))
            enc = d.get("diag.container.mp3_encoder_strings")
            if enc:
                notes.append("MP3 encoder strings " + ", ".join(enc))
            f["finding"] = "; ".join(notes) if notes else "no container inconsistencies noted"
            f["codec"] = d.get("diag.container.codec")
            f["native_sr"] = d.get("diag.container.native_sr")
        elif name == "spectral" and st == "ok":
            f["finding"] = (f"eligible band 50-{_fmt(q.get('q.spec.eligible_max_hz'), 0)} Hz; "
                            f"native occupied bandwidth {_fmt(d.get('diag.spec.native_occupied_bw_hz'), 0)} Hz "
                            f"(channel/processing property, not scored)")
        elif name == "prosody" and st == "ok":
            f["finding"] = (f"median F0 {_fmt(d.get('diag.pros.f0_median_hz'), 0)} Hz (diagnostic), "
                            f"{_fmt(q.get('q.pros.voiced_s'), 1)} s voiced")
        f["used_in_classifier"] = name not in DIAGNOSTIC_ONLY and name != "activity" and st == "ok"
        out[name] = f
    return out


def explanation_text(p: float, route: str, gmm_term: float | None, lr_term: float | None,
                     top_features: list, findings: dict) -> str:
    level = ("strongly synthetic-leaning" if p >= 0.9 else "synthetic-leaning" if p >= 0.5
             else "real-leaning" if p > 0.1 else "strongly real-leaning")
    parts = [f"Synthetic probability {p:.3f} ({level}; calibrated for a 50% prior; {route} model)."]
    terms = []
    if gmm_term is not None and np.isfinite(gmm_term):
        terms.append(f"LFCC-GMM term {gmm_term:+.2f}")
    if lr_term is not None and np.isfinite(lr_term):
        terms.append(f"engineered-feature term {lr_term:+.2f}")
    if terms:
        parts.append("Model contributions (logit): " + "; ".join(terms) + ".")
    if top_features:
        feats = ", ".join(f"{c['feature']} {c['contribution_logit']:+.2f}" for c in top_features[:4])
        parts.append(f"Largest feature contributions inside the feature model: {feats}.")
    obs = [f"{m}: {v['finding']}" for m, v in findings.items()
           if v.get("finding") and m in ("background", "container", "enf")]
    if obs:
        parts.append("Observations: " + " | ".join(obs) + ".")
    parts.append("Contributions describe the model's calculation, not physical causation or proof "
                 "of forgery.")
    return " ".join(parts)
