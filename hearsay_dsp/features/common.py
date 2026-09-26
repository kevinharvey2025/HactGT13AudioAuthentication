"""Shared DSP primitives: module result contract, framing, STFT, activity.

Every analysis module returns a ModuleResult whose status is one of
ok / not_applicable / insufficient_signal / error, with a reason. Module
outputs are split into namespaces:
  features     - candidate classifier inputs (still subject to the allowlist)
  quality      - applicability / signal-quality measurements
  diagnostics  - observations reported in traces, never classifier inputs
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import binary_dilation, median_filter
from scipy.signal import get_window

OK = "ok"
NOT_APPLICABLE = "not_applicable"
INSUFFICIENT = "insufficient_signal"
ERROR = "error"
STATUSES = (OK, NOT_APPLICABLE, INSUFFICIENT, ERROR)

# Frames whose RMS is below one 16-bit LSB (~ -90.3 dBFS) carry no measurable
# acoustic content ("digital silence"). They are excluded from analysis rather
# than treated as evidence.
DIGITAL_SILENCE_RMS = 2.0 ** -15
EPS = 1e-20


@dataclass
class ModuleResult:
    status: str
    reason: str = ""
    features: dict = field(default_factory=dict)
    quality: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)
    params: dict = field(default_factory=dict)
    candidates: list = field(default_factory=list)
    runtime_s: float = 0.0
    arrays: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return {
            "status": self.status,
            "reason": self.reason,
            "features": jsonable(self.features),
            "quality": jsonable(self.quality),
            "diagnostics": jsonable(self.diagnostics),
            "params": jsonable(self.params),
            "candidates": jsonable(self.candidates),
            "runtime_s": round(float(self.runtime_s), 6),
        }

    @classmethod
    def from_json(cls, d: dict) -> "ModuleResult":
        return cls(status=d["status"], reason=d.get("reason", ""),
                   features=unjson_floats(d.get("features", {})),
                   quality=d.get("quality", {}), diagnostics=d.get("diagnostics", {}),
                   params=d.get("params", {}), candidates=d.get("candidates", []),
                   runtime_s=d.get("runtime_s", 0.0))


def jsonable(obj):
    """Convert numpy scalars/arrays to JSON types; NaN/inf become None."""
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [jsonable(v) for v in obj.tolist()]
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        v = float(obj)
        return v if math.isfinite(v) else None
    return obj


def unjson_floats(d: dict) -> dict:
    return {k: (np.nan if v is None else v) for k, v in d.items()}


def run_module(fn, *args, **kwargs) -> ModuleResult:
    t0 = time.perf_counter()
    try:
        res = fn(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 - recorded as module error
        res = ModuleResult(status=ERROR, reason=f"{type(exc).__name__}: {exc}")
    res.runtime_s = time.perf_counter() - t0
    if res.status not in STATUSES:
        raise ValueError(f"invalid module status {res.status}")
    if res.status != OK:
        res.features = {}
    return res


def frame_signal(x: np.ndarray, frame_len: int, hop: int):
    """Frames covering the whole signal; the last frame is zero-padded.

    Returns (frames[n, frame_len], valid_len[n], starts[n]). Frame i covers
    samples [starts[i], starts[i] + frame_len).
    """
    n = x.size
    if n <= frame_len:
        n_frames = 1
    else:
        n_frames = 1 + int(math.ceil((n - frame_len) / hop))
    total = (n_frames - 1) * hop + frame_len
    padded = np.zeros(total, dtype=np.float64)
    padded[:n] = x
    starts = np.arange(n_frames) * hop
    idx = starts[:, None] + np.arange(frame_len)[None, :]
    frames = padded[idx]
    valid = np.clip(n - starts, 0, frame_len)
    return frames, valid, starts


def frame_energy_db(frames: np.ndarray, valid: np.ndarray) -> np.ndarray:
    ms = (frames ** 2).sum(axis=1) / np.maximum(valid, 1)
    return 10.0 * np.log10(ms + EPS)


@dataclass
class Stft:
    sr: int
    win_len: int
    hop: int
    nfft: int
    spec: np.ndarray        # complex, [frames, bins]
    power: np.ndarray       # |spec|^2
    freqs: np.ndarray
    starts: np.ndarray
    frames: np.ndarray      # unwindowed frames (for time-domain modules)
    valid: np.ndarray

    @property
    def times(self) -> np.ndarray:
        return (self.starts + self.win_len / 2.0) / self.sr


def compute_stft(x: np.ndarray, sr: int, cfg: dict) -> Stft:
    win_len = int(round(cfg["win_s"] * sr))
    hop = int(round(cfg["hop_s"] * sr))
    nfft = int(cfg["nfft"])
    frames, valid, starts = frame_signal(x, win_len, hop)
    window = get_window("hann", win_len, fftbins=True)
    spec = np.fft.rfft(frames * window, n=nfft, axis=1)
    return Stft(sr=sr, win_len=win_len, hop=hop, nfft=nfft, spec=spec,
                power=np.abs(spec) ** 2, freqs=np.fft.rfftfreq(nfft, 1.0 / sr),
                starts=starts, frames=frames, valid=valid)


@dataclass
class Activity:
    """Classical energy-based activity estimate on the STFT frame grid.

    'active' frames are clearly above the clip's low-energy floor; 'background'
    frames are near the floor and away from active frames. Neither is a claim
    about speech presence: quiet speech can look like background and loud
    noise like speech. Frames in between are left unassigned.
    """
    e_db: np.ndarray
    digital_silence: np.ndarray
    active: np.ndarray
    background: np.ndarray
    floor_db: float
    peak_db: float
    thr_active_db: float
    thr_bg_db: float
    low_contrast: bool
    times: np.ndarray
    hop_s: float


def detect_activity(stft: Stft, cfg: dict) -> tuple[Activity | None, ModuleResult]:
    e_db = frame_energy_db(stft.frames, stft.valid)
    rms = np.sqrt((stft.frames ** 2).sum(axis=1) / np.maximum(stft.valid, 1))
    silent = rms < DIGITAL_SILENCE_RMS
    ns = ~silent
    hop_s = stft.hop / stft.sr
    params = {"method": "percentile energy thresholds", **cfg, "frame_s": stft.win_len / stft.sr,
              "hop_s": hop_s}
    quality = {"q.act.n_frames": int(e_db.size),
               "q.act.digital_silence_frac": float(silent.mean())}
    if ns.sum() < 10:
        return None, ModuleResult(status=INSUFFICIENT, reason="fewer than 10 non-silent frames",
                                  quality=quality, params=params)
    floor = float(np.percentile(e_db[ns], 10))
    peak = float(np.percentile(e_db[ns], 95))
    rng = peak - floor
    low_contrast = rng < cfg["min_contrast_db"]
    thr_active = max(floor + max(cfg["active_rel_db"], cfg["active_rel_frac"] * rng),
                     peak - cfg["active_below_peak_db"])
    thr_bg = floor + max(cfg["bg_rel_db"], cfg["bg_rel_frac"] * rng)
    thr_bg = min(thr_bg, thr_active - 6.0)
    if low_contrast:
        active = ns.copy()
        background = np.zeros_like(ns)
    else:
        active = ns & (e_db >= thr_active)
        k = int(cfg["median_frames"])
        if k > 1 and active.size >= k:
            active = median_filter(active.astype(np.uint8), size=k, mode="nearest").astype(bool) & ns
        guard = binary_dilation(active, iterations=int(cfg["bg_guard_frames"])) if active.any() else active
        background = ns & (e_db <= thr_bg) & ~guard
    act = Activity(e_db=e_db, digital_silence=silent, active=active, background=background,
                   floor_db=floor, peak_db=peak, thr_active_db=thr_active, thr_bg_db=thr_bg,
                   low_contrast=bool(low_contrast), times=stft.times, hop_s=hop_s)
    quality.update({
        "q.act.active_frac": float(active.mean()),
        "q.act.background_frac": float(background.mean()),
        "q.act.active_s": float(active.sum() * hop_s),
        "q.act.dynamic_range_db": float(rng),
        "q.act.low_contrast": bool(low_contrast),
        "q.act.snr_est_db": (float(np.median(e_db[active]) - np.median(e_db[background]))
                             if active.any() and background.any() else None),
    })
    diagnostics = {"diag.act.floor_dbfs": floor, "diag.act.peak_dbfs": peak}
    return act, ModuleResult(status=OK, quality=quality, diagnostics=diagnostics, params=params)


def robust_stats(values: np.ndarray, prefix: str, which=("med", "iqr", "p10", "p90")) -> dict:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    out = {}
    if v.size == 0:
        return {f"{prefix}_{w}": np.nan for w in which}
    q10, q25, q50, q75, q90 = np.percentile(v, [10, 25, 50, 75, 90])
    table = {"med": q50, "iqr": q75 - q25, "p10": q10, "p90": q90,
             "mean": float(v.mean()), "std": float(v.std())}
    for w in which:
        out[f"{prefix}_{w}"] = float(table[w])
    return out


def robust_z(values: np.ndarray) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    med = np.nanmedian(v)
    mad = 1.4826 * np.nanmedian(np.abs(v - med))
    return (v - med) / (mad + 1e-12)
