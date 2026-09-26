"""Configuration defaults, YAML loading and identity hashing.

Every learned artifact and every cache entry records the hash of the
configuration that produced it, so results can be traced to exact settings.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import yaml

# Bump a module's version whenever its measurements change. Versions are part
# of the per-module cache key, so stale cache entries are recomputed.
MODULE_VERSIONS = {
    "decode": "1",
    "activity": "1",
    "container": "1",
    "lfcc": "1",
    "spectral": "1",
    "lpc": "1",
    "background": "1",
    "phase": "1",
    "prosody": "1",
    "enf": "1",
    "compression": "1",
}

DEFAULTS: dict = {
    "run": {
        "seed": 20260926,
        "n_jobs": 8,
        "cache_dir": "cache/dsp",
    },
    "decode": {
        # "first": lowest-index audio stream; "most_channels": most channels, ties by index.
        "stream_policy": "first",
        # Samples are float64 in [-1, 1): integer PCM is divided by 2**(bits-1).
        "analysis_sr": 16000,
        "resampler": {"method": "scipy.signal.resample_poly", "window": ["kaiser", 5.0]},
        # Round the analysis signal to the 16-bit PCM grid (a no-op for 16-bit
        # 16 kHz inputs such as the test set; makes resampled training audio match).
        "quantize_bits": 16,
        "max_nonfinite_frac": 0.01,
        "clip_level": 0.999,
    },
    "features": {
        # Upper analysis frequency for classifier features. The HEARSAY test set is
        # low-passed near 7.5 kHz and the training real/fake resamplers differ above
        # 7 kHz, so energy above this limit is excluded from classification.
        "max_hz": 7000.0,
        "low_band_max_hz": 3400.0,
        "stft": {"win_s": 0.025, "hop_s": 0.010, "nfft": 512},
        "activity": {
            "min_contrast_db": 6.0,
            "active_rel_db": 9.0,
            "active_rel_frac": 0.35,
            "active_below_peak_db": 40.0,
            "bg_rel_db": 3.0,
            "bg_rel_frac": 0.10,
            "bg_guard_frames": 3,
            "median_frames": 5,
        },
        "lfcc": {
            # Reproduction of ASVspoof 2021 LA MATLAB baseline lfcc_bp.m (Todisco/Sahidullah).
            "official": {"window_ms": 30, "nfft": 1024, "n_filter": 70, "n_coeff": 19,
                         "low_hz": 0.0, "high_hz": 4000.0},
            # Our variant: same framing, 0-7 kHz, c0 dropped (gain invariance).
            "custom": {"window_ms": 30, "nfft": 1024, "n_filter": 70, "n_coeff": 20,
                       "low_hz": 0.0, "high_hz": 7000.0, "drop_c0": True},
        },
        "spectral": {"ltas_smooth_hz": 470.0, "ripple_lag_hz": [90.0, 1900.0]},
        "lpc": {"order": 18, "min_active_frames": 50},
        "background": {
            "min_bg_frames": 30,
            "window_s": 1.0,
            "hop_s": 0.5,
            "min_bg_frames_per_window": 10,
            "n_bands": 8,
            "floor_jump_min_db": 6.0,
            "shape_jump_min_db": 4.0,
            "robust_k": 4.0,
            "merge_s": 0.5,
        },
        "phase": {"alpha": 0.4, "gamma": 0.9, "lifter": 30, "n_ceps": 12,
                  "min_active_frames": 50, "rel_power_db": -40.0, "jump_z": 8.0},
        "prosody": {"pitch_floor": 75.0, "pitch_ceiling": 500.0, "min_voiced_s": 0.5,
                    "min_pulses": 40, "octave_jump_st": 7.0, "min_pause_s": 0.15},
        "enf": {"nominal_hz": [50.0, 60.0], "harmonics": 3, "presence_snr_db": 10.0,
                "min_duration_s": 10.0, "track_win_s": 2.0, "track_hop_s": 0.5},
    },
    "routing": {
        # "all_eligible": run every module whose eligibility check passes.
        # "confidence": run core modules, then extended modules only when the
        # calibrated core probability is inside [low, high].
        "mode": "all_eligible",
        "confidence_band": [0.1, 0.9],
    },
    "model": {
        "gmm": {"feature": "custom", "frames": "nonsilent", "n_components": 64,
                "max_frames_per_clip": 200, "max_iter": 100, "reg_covar": 1e-4,
                "n_init": 1},
        "classifier": "lr",
        "lr": {"C_grid": [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]},
        "hgb": {"learning_rate": 0.05, "max_iter": 300, "max_leaf_nodes": 15,
                "min_samples_leaf": 20, "l2_regularization": 1.0},
        "feature_groups": ["cep", "spec", "lpc", "bg", "phase", "pros"],
        "use_gmm_score": True,
        "inner_folds": 5,
        "calibration": {"method": "sigmoid", "prior": 0.5},
    },
    "export": {"float_format": "%.6f"},
}

# Feature-name prefixes per group. Only names matching an enabled group are
# classifier inputs (explicit allowlist); quality ("q.") and diagnostic ("diag.")
# namespaces are never matched by these prefixes.
FEATURE_GROUP_PREFIXES = {
    "cep": ("cep.",),
    "spec": ("spec.",),
    "lpc": ("lpc.",),
    "bg": ("bg.",),
    "phase": ("phase.",),
    "pros": ("pros.",),
}


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> dict:
    cfg = copy.deepcopy(DEFAULTS)
    if path is not None:
        with open(path, "r", encoding="utf-8") as fh:
            cfg = deep_merge(cfg, yaml.safe_load(fh) or {})
    if overrides:
        cfg = deep_merge(cfg, overrides)
    return cfg


def canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def sha256_hex(data: bytes | str) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def config_hash(obj, n: int = 16) -> str:
    return sha256_hex(canonical_json(obj))[:n]


def preprocessing_identity(cfg: dict) -> dict:
    """Everything that changes the decoded analysis signal."""
    return {"decode_version": MODULE_VERSIONS["decode"], "decode": cfg["decode"]}


def module_key(cfg: dict, module: str, params: dict) -> str:
    return config_hash({
        "module": module,
        "version": MODULE_VERSIONS[module],
        "params": params,
        "preprocessing": preprocessing_identity(cfg),
    })
