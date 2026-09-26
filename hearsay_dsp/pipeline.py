"""Per-file analysis: decode -> activity -> routed modules -> cached record.

analyze_file() returns a JSON-able record with decode provenance, per-module
results (status, reason, measurements, parameters, runtime) and routing
decisions. Module results are cached per content hash and module key; results
with status "error" are not cached, so a rerun retries them.
"""

from __future__ import annotations

import os
import resource
import sys
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits

from .config import config_hash, module_key, preprocessing_identity
from .features.background import background_module
from .features.common import ERROR, OK, ModuleResult, compute_stft, detect_activity, run_module
from .features.container import container_module
from .features.enf import enf_module
from .features.lfcc import lfcc_module
from .features.lpc import lpc_module
from .features.phase import phase_module
from .features.prosody import prosody_module
from .features.spectral import spectral_module
from .io.cache import FeatureCache
from .io.decode import DecodeError, decode_audio, file_sha256
from .routing import ALL_MODULES, eligibility


def module_params(cfg: dict, module: str) -> dict:
    f = cfg["features"]
    base = {"stft": f["stft"], "activity": f["activity"]}
    return {
        "activity": base,
        "container": {},
        "compression": {},
        "lfcc": {**base, "lfcc": f["lfcc"]},
        "spectral": {**base, "spectral": f["spectral"], "max_hz": f["max_hz"],
                     "low_band_max_hz": f["low_band_max_hz"]},
        "lpc": {**base, "lpc": f["lpc"], "max_hz": f["max_hz"]},
        "background": {**base, "background": f["background"], "max_hz": f["max_hz"]},
        "phase": {**base, "phase": f["phase"], "max_hz": f["max_hz"]},
        "prosody": {**base, "prosody": f["prosody"]},
        "enf": {"enf": f["enf"]},
    }[module]


def _run(module: str, dec, stft, act, path: str, cfg: dict) -> ModuleResult:
    x, sr = dec.analysis, dec.analysis_sr
    if module == "container":
        return run_module(container_module, path, dec.provenance)
    if module == "lfcc":
        return run_module(lfcc_module, x, sr, act, cfg)
    if module == "spectral":
        return run_module(spectral_module, stft, act, x, dec.native, dec.native_sr, cfg)
    if module == "lpc":
        return run_module(lpc_module, x, stft, act, cfg)
    if module == "background":
        return run_module(background_module, x, stft, act, cfg)
    if module == "phase":
        return run_module(phase_module, stft, act, cfg)
    if module == "prosody":
        return run_module(prosody_module, x, sr, act, cfg)
    if module == "enf":
        return run_module(enf_module, dec.native, dec.native_sr, cfg)
    raise KeyError(module)


def file_diagnostics(path: str) -> dict:
    st = os.stat(path)
    out = {"diag.file.size_bytes": st.st_size, "diag.file.mtime": st.st_mtime,
           "diag.file.ctime": st.st_ctime}
    if hasattr(st, "st_birthtime"):
        out["diag.file.birthtime"] = st.st_birthtime
    return out


def analyze_file(path: str, cfg: dict, cache: FeatureCache | None = None,
                 modules=ALL_MODULES) -> dict:
    t_start = time.perf_counter()
    path = str(path)
    sha = file_sha256(path)
    record = cache.load_record(sha) if cache else {}
    record.setdefault("modules", {})
    record["sha256"] = sha
    wanted = ("activity",) + tuple(m for m in modules if m != "activity")
    keys = {m: module_key(cfg, m, module_params(cfg, m)) for m in wanted}
    need = [m for m in wanted if not (cache and cache.valid_entry(record, sha, m, keys[m]))]
    routing, timing, dirty = [], {}, False
    out = {"path": path, "filename": os.path.basename(path), "sha256": sha,
           "file": file_diagnostics(path)}
    if need:
        t0 = time.perf_counter()
        try:
            dec = decode_audio(path, cfg)
        except DecodeError as exc:
            out.update({"error": f"decode failed: {exc}", "modules": {}, "routing": [],
                        "timing": {"total_s": time.perf_counter() - t_start}})
            return out
        timing["decode_s"] = time.perf_counter() - t0
        record["decode"] = {"key": config_hash(preprocessing_identity(cfg)),
                            "provenance": dec.provenance, "quality": dec.quality}
        t0 = time.perf_counter()
        stft = compute_stft(dec.analysis, dec.analysis_sr, cfg["features"]["stft"])
        act, act_res = detect_activity(stft, cfg["features"]["activity"])
        act_res.runtime_s = time.perf_counter() - t0
        record["modules"]["activity"] = {"key": keys["activity"], "result": act_res.to_json()}
        dirty = True
        for m in need:
            if m == "activity":
                continue
            ok, status, reason = eligibility(m, act, cfg)
            if not ok:
                res = ModuleResult(status=status, reason=reason)
                routing.append({"module": m, "decision": "abstain", "reason": reason})
            else:
                res = _run(m, dec, stft, act, path, cfg)
                routing.append({"module": m, "decision": "run", "reason": reason})
            if res.status == ERROR:
                out.setdefault("module_errors", {})[m] = res.reason
                record["modules"].pop(m, None)
                out.setdefault("uncached", {})[m] = res.to_json()
                continue
            if res.arrays and cache:
                cache.save_arrays(sha, m, keys[m], res.arrays)
            record["modules"][m] = {"key": keys[m], "result": res.to_json(),
                                    "arrays": sorted(res.arrays)}
    for m in wanted:
        if m not in need:
            routing.append({"module": m, "decision": "cached"})
    if dirty and cache:
        cache.save_record(sha, record)
    out["decode"] = record.get("decode")
    out["modules"] = {m: record["modules"][m]["result"] for m in wanted if m in record["modules"]}
    for m, res in out.pop("uncached", {}).items():
        out["modules"][m] = res
    out["module_keys"] = keys
    out["routing"] = routing
    timing["total_s"] = time.perf_counter() - t_start
    out["timing"] = timing
    return out


def _extract_one(path: str, cfg: dict, modules) -> dict:
    with threadpool_limits(limits=1):
        try:
            rec = analyze_file(path, cfg, FeatureCache(cfg["run"]["cache_dir"]), modules)
        except Exception as exc:  # noqa: BLE001 - counted as failed file
            rec = {"path": str(path), "filename": os.path.basename(str(path)),
                   "error": f"{type(exc).__name__}: {exc}", "modules": {}, "timing": {}}
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rec.setdefault("timing", {})["worker_maxrss_mb"] = rss / (1024 * 1024 if sys.platform == "darwin"
                                                                else 1024)
    return rec


def extract_paths(paths, cfg: dict, modules=ALL_MODULES, n_jobs: int | None = None,
                  log_every: int = 500) -> list[dict]:
    n_jobs = int(n_jobs or cfg["run"]["n_jobs"])
    paths = [str(p) for p in paths]
    t0 = time.perf_counter()
    results = []
    gen = Parallel(n_jobs=n_jobs, backend="loky", return_as="generator",
                   batch_size=4)(delayed(_extract_one)(p, cfg, modules) for p in paths)
    for i, rec in enumerate(gen, 1):
        results.append(rec)
        if log_every and (i % log_every == 0 or i == len(paths)):
            el = time.perf_counter() - t0
            print(f"[extract] {i}/{len(paths)} files, {el:.0f} s elapsed, "
                  f"{i / max(el, 1e-9):.1f} files/s", file=sys.stderr, flush=True)
    return results


SCALAR_TYPES = (int, float, bool, str, type(None), np.floating, np.integer)


def records_to_table(records: list[dict]) -> pd.DataFrame:
    rows = []
    for rec in records:
        row = {"path": rec["path"], "sha256": rec.get("sha256"),
               "decode_status": "error" if rec.get("error") else OK,
               "decode_error": rec.get("error")}
        dec = rec.get("decode") or {}
        prov, q = dec.get("provenance", {}), dec.get("quality", {})
        for k in ("native_sr", "channels", "container", "codec", "decoder", "sniffed_format"):
            row[f"diag.decode.{k}"] = prov.get(k)
        for k in ("native_duration_s", "rms_dbfs", "peak_dbfs", "clipped_frac", "nonfinite_samples"):
            row[f"diag.decode.{k}"] = q.get(k)
        for m, res in rec.get("modules", {}).items():
            row[f"status.{m}"] = res.get("status")
            for ns in ("features", "quality", "diagnostics"):
                for k, v in (res.get(ns) or {}).items():
                    if isinstance(v, SCALAR_TYPES):
                        row[k] = np.nan if (v is None and ns == "features") else v
            row[f"n_candidates.{m}"] = len(res.get("candidates") or [])
        rows.append(row)
    return pd.DataFrame(rows)
