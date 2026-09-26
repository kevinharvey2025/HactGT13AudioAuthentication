"""Deterministic eligibility and cost routing (no learned components, no LLM).

Stage 1 (core) runs on every decodable file. Stage 2 (extended) modules run
when their evidence requirements are met; otherwise they abstain with a
reason. In "confidence" mode, stage 2 is skipped when the calibrated stage-1
probability falls outside the configured band. A probability near 0.5 is not
an out-of-distribution detector, and a confident model can still be wrong, so
this is a cost policy, not a correctness guarantee.
"""

from __future__ import annotations

from .features.common import INSUFFICIENT, NOT_APPLICABLE

CORE_MODULES = ("container", "lfcc", "spectral")
EXTENDED_MODULES = ("lpc", "background", "phase", "prosody", "enf")
ALL_MODULES = CORE_MODULES + EXTENDED_MODULES + ("compression",)
DIAGNOSTIC_ONLY = ("container", "enf", "compression")


def eligibility(module: str, act, cfg: dict) -> tuple[bool, str, str]:
    """Returns (run, status_if_not_run, reason)."""
    f = cfg["features"]
    if module == "compression":
        return False, NOT_APPLICABLE, ("no validated compressed-domain (bitstream/MDCT) analysis is "
                                       "implemented; codec and band-limit observations are "
                                       "diagnostics only")
    if module in ("container", "enf", "lfcc"):
        return True, "", "always eligible for decoded audio"
    if act is None:
        return False, INSUFFICIENT, "no usable activity estimate (too little non-silent signal)"
    n_active = int(act.active.sum())
    if module == "spectral":
        return True, "", f"{n_active} active frames"
    if module == "lpc":
        need = f["lpc"]["min_active_frames"]
        ok = n_active >= need
        return ok, INSUFFICIENT, f"{n_active} active frames ({'>=' if ok else '<'} {need})"
    if module == "phase":
        need = f["phase"]["min_active_frames"]
        ok = n_active >= need
        return ok, INSUFFICIENT, f"{n_active} active frames ({'>=' if ok else '<'} {need})"
    if module == "background":
        need = f["background"]["min_bg_frames"]
        n_bg = int(act.background.sum())
        ok = n_bg >= need
        return ok, INSUFFICIENT, f"{n_bg} low-energy frames ({'>=' if ok else '<'} {need})"
    if module == "prosody":
        active_s = n_active * act.hop_s
        need = f["prosody"]["min_voiced_s"]
        ok = active_s >= need
        return ok, INSUFFICIENT, f"{active_s:.2f} s active ({'>=' if ok else '<'} {need} s)"
    raise KeyError(module)


def confidence_route(p_core: float, cfg: dict) -> tuple[bool, str]:
    lo, hi = cfg["routing"]["confidence_band"]
    if lo <= p_core <= hi:
        return True, f"core probability {p_core:.3f} inside [{lo}, {hi}]: run extended modules"
    return False, f"core probability {p_core:.3f} outside [{lo}, {hi}]: extended modules skipped"
