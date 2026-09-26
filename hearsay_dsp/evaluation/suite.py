"""Development ablation suite (dev rows only; holdout/test rows are never loaded).

Parts:
  gmm         LFCC-GMM variants: official reproduction vs custom front-end, frame
              policy, component count (5-fold grouped CV, OOF scores)
  pretrained  official ASVspoof 2021 LA pretrained LFCC-GMM, unchanged
  features    logistic regression per feature group, all groups, drop-one-group,
              histogram gradient boosting; paired deltas vs "all groups"
  fusion      late fusion of GMM + feature LR (cross-fitted weights)
  logo        leave-one-generator-out: each generator's fakes excluded from training
  audit       shortcut audit (diagnostic / quality / missingness features alone) and
              speaker-confound audit (train on DiffSSD-as-provided only, test on unseen
              real speakers / recording chains)
Outputs: <out>/suite_results.json, <out>/oof_scores.tsv, <out>/folds.tsv.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import FEATURE_GROUP_PREFIXES
from ..io.manifest import read_manifest, select_training_rows
from ..models.detector import DetectorSpec
from ..models.gmm import OfficialPretrainedGMM
from .experiments import (FrameStore, LateFusion, compact, crossfit_calibrate, crossfit_fusion,
                          detector_oof, gmm_oof, load_table, make_folds, save_json, summarize)
from .metrics import auc_w, binary_metrics, group_bootstrap

PRETRAINED = "cache/dsp/external/pre_trained_LA_LFCC-GMM.mat"
GROUPS = list(FEATURE_GROUP_PREFIXES)
MODULE_OF_GROUP = {"cep": "lfcc", "spec": "spectral", "lpc": "lpc", "bg": "background",
                   "phase": "phase", "pros": "prosody"}

SHORTCUT_SETS = {
    "stream_properties": ["diag.decode.native_sr", "diag.decode.channels", "diag.decode.native_duration_s"],
    "levels_and_silence": ["diag.decode.rms_dbfs", "diag.decode.peak_dbfs", "q.act.digital_silence_frac",
                           "diag.act.floor_dbfs"],
    "bandwidth": ["diag.spec.native_occupied_bw_hz", "diag.spec.native_edge_drop_db", "q.spec.occupied_bw_hz"],
    "container_flags": ["diag.container.extension_mismatch"],
}


def paired_delta(y, s_new, s_base, groups, n_boot=500, seed=0) -> dict:
    """AUC(new) - AUC(base) on the same rows, CI by paired group bootstrap."""
    ok = np.isfinite(s_new) & np.isfinite(s_base)
    y, a, b, g = y[ok], s_new[ok], s_base[ok], np.asarray(groups)[ok]
    d = auc_w(y, a, None) - auc_w(y, b, None)
    return {"delta_auc": float(d), "ci95": _paired_boot(y, a, b, g, n_boot, seed), "n_rows": int(ok.sum())}


def _paired_boot(y, a, b, g, n_boot, seed):
    rng = np.random.default_rng(seed)
    parts = []
    for cls in (0, 1):
        rows = np.where(y == cls)[0]
        uniq, inv = np.unique(g[rows].astype(str), return_inverse=True)
        parts.append((rows, uniq.size, inv))
    vals = []
    for _ in range(n_boot):
        w = np.zeros(y.size)
        for rows, n, inv in parts:
            w[rows] = np.bincount(rng.integers(0, n, n), minlength=n)[inv]
        vals.append(auc_w(y, a, w) - auc_w(y, b, w))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def _record_runtime(recs: list) -> dict:
    rt = {}
    for r in recs:
        for m, res in (r.get("modules") or {}).items():
            rt.setdefault(m, []).append(res.get("runtime_s", np.nan))
    return {m: float(np.nanmean(v)) for m, v in rt.items()}


def run_suite(manifest: str, cfg: dict, out_dir: str, parts=None) -> dict:
    parts = parts or ["gmm", "pretrained", "features", "fusion", "logo", "audit"]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    seed = cfg["run"]["seed"]
    t0 = time.time()
    df = read_manifest(manifest)
    dev = select_training_rows(df).reset_index(drop=True)
    table, recs = load_table_from_rows(dev, cfg)
    ok = table["decode_status"] != "error"
    table = table[ok].reset_index(drop=True)
    recs = [r for r, k in zip(recs, ok) if k]
    y = table["label"].astype(int).values
    groups = table["group_id"].values
    folds = make_folds(table, 5, seed)
    ids = table[["sha256", "path", "label", "attack_type", "source_id", "family", "dataset", "group_id"]].copy()
    ids["fold"] = folds
    ids.to_csv(out / "folds.tsv", sep="\t", index=False)
    oof = ids.copy()
    res: dict = {"n_dev": int(len(table)), "n_real": int((y == 0).sum()), "n_synthetic": int((y == 1).sum()),
                 "n_decode_failures": int((~ok).sum()), "module_runtime_s": _record_runtime(recs),
                 "module_status": {c[7:]: table[c].value_counts().to_dict() for c in table
                                   if c.startswith("status.")}}
    prior = cfg["model"]["calibration"]["prior"]

    def add(name, raw, extra=None):
        p = crossfit_calibrate(raw, y, folds, prior) if np.isfinite(raw).sum() > 10 else raw
        oof[name] = raw
        s = summarize(table, raw, p, n_boot=300, seed=seed)
        s["coverage"] = float(np.isfinite(raw).mean())
        if extra:
            s.update(extra)
        res.setdefault("configs", {})[name] = s
        print(f"[suite] {name}: {json.dumps(compact(s))}", flush=True)
        save_json(res, out / "suite_results.json")
        oof.to_csv(out / "oof_scores.tsv", sep="\t", index=False)

    stores = {}

    def arrays_for(feature):
        if feature not in stores:
            st = FrameStore(cfg, feature)
            stores[feature] = st.many(table["sha256"].tolist())
        return stores[feature]

    gcfg = cfg["model"]["gmm"]
    base = {k: gcfg[k] for k in ("max_frames_per_clip", "reg_covar", "n_init")}
    base["max_frames_total"] = gcfg.get("max_frames_total", 150000)
    if "gmm" in parts:
        variants = {
            "gmm:official_repro(0-4k,c0,all,512c,10it)": dict(feature="official", frames="all",
                                                             n_components=512, max_iter=10),
            "gmm:official(0-4k,c0,nonsilent,64c)": dict(feature="official", frames="nonsilent",
                                                        n_components=64, max_iter=100),
            "gmm:custom(all,64c)": dict(feature="custom", frames="all", n_components=64, max_iter=100),
            "gmm:custom(nonsilent,64c)": dict(feature="custom", frames="nonsilent", n_components=64, max_iter=100),
            "gmm:custom(active,64c)": dict(feature="custom", frames="active", n_components=64, max_iter=100),
            "gmm:custom(nonsilent,16c)": dict(feature="custom", frames="nonsilent", n_components=16, max_iter=100),
            "gmm:custom(nonsilent,32c)": dict(feature="custom", frames="nonsilent", n_components=32, max_iter=100),
            "gmm:custom(nonsilent,128c)": dict(feature="custom", frames="nonsilent", n_components=128, max_iter=100),
        }
        for name, v in variants.items():
            params = {**base, **v, "seed": seed}
            t = time.time()
            s, infos = gmm_oof(arrays_for(v["feature"]), y, folds, params)
            add(name, s, {"params": params, "fit_info": infos, "wall_s": time.time() - t})
        custom = {k: v for k, v in res["configs"].items() if k.startswith("gmm:custom")}
        best = max(custom, key=lambda k: custom[k]["overall"]["auc"])
        res["gmm_selected"] = {"name": best, "params": custom[best]["params"],
                               "rule": "highest dev OOF AUC among custom variants"}

    if "pretrained" in parts and Path(PRETRAINED).exists():
        pre = OfficialPretrainedGMM(PRETRAINED)
        arr = arrays_for("official")
        s = np.array([pre.score(a) for a in arr])
        add("gmm:official_pretrained_asvspoof2019LA", s,
            {"note": "fixed model trained on ASVspoof 2019 LA; only the sigmoid calibrator is cross-fitted",
             "train_info": pre.train_info})

    lr_grid = tuple(cfg["model"]["lr"]["C_grid"])
    feat_specs = {}
    if "features" in parts or "fusion" in parts or "logo" in parts:
        for g in GROUPS:
            feat_specs[f"lr:{g}"] = (g,)
        feat_specs["lr:all"] = tuple(GROUPS)
        for g in GROUPS:
            feat_specs[f"lr:all-minus-{g}"] = tuple(x for x in GROUPS if x != g)
        feat_specs["lr:core(cep+spec)"] = ("cep", "spec")
    selections = {}
    if "features" in parts:
        for name, gr in feat_specs.items():
            spec = DetectorSpec(name, use_gmm=False, feature_groups=gr, classifier="lr",
                                lr_C_grid=lr_grid, seed=seed)
            t = time.time()
            s, sel = detector_oof(spec, table, folds)
            selections[name] = sel
            add(name, s, {"groups": list(gr), "selection": sel, "wall_s": time.time() - t,
                          "runtime_modules_s": float(sum(res["module_runtime_s"].get(MODULE_OF_GROUP[x], 0)
                                                         for x in gr))})
        spec = DetectorSpec("hgb:all", use_gmm=False, feature_groups=tuple(GROUPS), classifier="hgb",
                            hgb=cfg["model"]["hgb"], seed=seed)
        t = time.time()
        s, _ = detector_oof(spec, table, folds)
        add("hgb:all", s, {"groups": GROUPS, "wall_s": time.time() - t})
        res["paired_vs_lr_all"] = {
            name: paired_delta(y, oof[name].values, oof["lr:all"].values, groups, 300, seed)
            for name in oof.columns if name.startswith(("lr:", "hgb:")) and name != "lr:all"}
        save_json(res, out / "suite_results.json")

    gsel = res.get("gmm_selected", {}).get("name")
    if "fusion" in parts and gsel and "lr:all" in oof:
        for lr_name in ("lr:all", "lr:core(cep+spec)"):
            Z = np.column_stack([oof[gsel].values, oof[lr_name].values])
            logit, _p = crossfit_fusion(Z, y, folds, [gsel, lr_name], prior)
            add(f"fusion:{gsel}+{lr_name}", logit, {"components": [gsel, lr_name],
                                                    "weights_all_dev": LateFusion([gsel, lr_name], prior)
                                                    .fit(Z, y).to_dict()})
        res["paired_fusion_vs_lr_all"] = paired_delta(y, oof[f"fusion:{gsel}+lr:all"].values,
                                                      oof["lr:all"].values, groups, 300, seed)
        res["paired_fusion_vs_gmm"] = paired_delta(y, oof[f"fusion:{gsel}+lr:all"].values,
                                                   oof[gsel].values, groups, 300, seed)
        save_json(res, out / "suite_results.json")

    if "logo" in parts and gsel:
        gparams = {**base, **res["configs"][gsel]["params"]}
        C_all = _modal_C(selections.get("lr:all"), lr_grid)
        spec = DetectorSpec("lr:all", use_gmm=False, feature_groups=tuple(GROUPS), classifier="lr",
                            lr_C_grid=(C_all,), seed=seed)
        logo = {}
        for gen in sorted(table.loc[y == 1, "attack_type"].unique()):
            keep = ~((y == 1) & (table["attack_type"].values == gen))
            m = (y == 0) | ~keep
            sg, _ = gmm_oof(arrays_for(gparams["feature"]), y, folds, gparams, train_mask=keep)
            sl, _ = detector_oof(spec, table, folds, train_mask=keep)
            Z = np.column_stack([sg, sl])
            fl, _ = crossfit_fusion(Z, y, folds, ["gmm", "lr"], prior, train_mask=keep)
            entry = {}
            for nm, sc in (("gmm", sg), ("lr_all", sl), ("fusion", fl)):
                mm = binary_metrics(y[m], 1 / (1 + np.exp(-np.nan_to_num(sc[m]))), raw=sc[m])
                entry[nm] = {"auc": mm["auc"], "eer": mm["eer"]}
                seen = res["configs"].get({"gmm": gsel, "lr_all": "lr:all",
                                           "fusion": f"fusion:{gsel}+lr:all"}[nm])
                if seen:
                    entry[nm]["auc_when_seen"] = seen["per_generator"][gen]["auc"]
            entry["n_fake"] = int((~keep).sum())
            logo[gen] = entry
            oof[f"logo_fusion:{gen}"] = np.where(m, fl, np.nan)
            print(f"[suite] LOGO {gen}: {json.dumps(entry)}", flush=True)
            res["logo"] = logo
            save_json(res, out / "suite_results.json")
        oof.to_csv(out / "oof_scores.tsv", sep="\t", index=False)

    if "audit" in parts:
        audit = {}
        for name, cols in SHORTCUT_SETS.items():
            spec = DetectorSpec(f"shortcut:{name}", use_gmm=False, feature_groups=(),
                                extra_features=tuple(c for c in cols if c in table.columns),
                                classifier="lr", lr_C_grid=(1.0,), seed=seed)
            tt = table.copy()
            for c in cols:
                if c in tt and tt[c].dtype == bool:
                    tt[c] = tt[c].astype(float)
            s, _ = detector_oof(spec, tt, folds)
            audit[name] = {"columns": cols, "oof_auc": binary_metrics(y, 1 / (1 + np.exp(-s)), raw=s)["auc"]}
        feat_cols = [c for c in table.columns if c.startswith(tuple(p for g in GROUPS
                                                                    for p in FEATURE_GROUP_PREFIXES[g]))]
        miss = table[feat_cols].isna()
        miss = miss.loc[:, miss.any()].astype(float)
        miss.columns = [f"missing::{c}" for c in miss.columns]
        if miss.shape[1]:
            tt = pd.concat([table[["label", "group_id"]], miss], axis=1)
            spec = DetectorSpec("shortcut:missingness", use_gmm=False, extra_features=tuple(miss.columns),
                                classifier="lr", lr_C_grid=(1.0,), seed=seed)
            s, _ = detector_oof(spec, tt, folds)
            audit["missingness_indicators"] = {"n_columns": int(miss.shape[1]),
                                               "oof_auc": binary_metrics(y, 1 / (1 + np.exp(-s)), raw=s)["auc"]}
        else:
            audit["missingness_indicators"] = {"n_columns": 0, "oof_auc": None,
                                               "note": "no allowlisted feature is ever missing on dev"}
        res["shortcut_audit"] = audit
        gparams = res.get("gmm_selected", {}).get("params")
        res["speaker_audit"] = speaker_audit(table, y, folds, cfg, oof, gsel, gparams, arrays_for,
                                             lr_grid, prior)
        save_json(res, out / "suite_results.json")
        oof.to_csv(out / "oof_scores.tsv", sep="\t", index=False)
    res["wall_s"] = time.time() - t0
    save_json(res, out / "suite_results.json")
    return res


def _modal_C(sel, grid):
    if not sel:
        return grid[len(grid) // 2]
    cs = [s.get("C") for s in sel if s.get("C") is not None]
    return max(set(cs), key=cs.count) if cs else grid[len(grid) // 2]


def speaker_audit(table, y, folds, cfg, oof, gsel, params, arrays_for, lr_grid, prior) -> dict:
    """Train on DiffSSD-as-provided dev rows only; score external reals it never saw.

    External reals = LibriSpeech utterances of the clone reference speakers (unseen real
    speakers/recording chain) and LJSpeech-1.1 originals (same speaker, our resampler).
    Compared with the extended model's OOF probabilities on the same rows.
    """
    from ..models.detector import Detector
    seed = cfg["run"]["seed"]
    is_ds = (table["dataset"] == "diffssd").values
    ext = ~is_ds
    tr_tab = table[is_ds].reset_index(drop=True)
    y_tr = y[is_ds]
    f_tr = folds[is_ds]
    if gsel is None or params is None:
        return {"skipped": "no GMM selected"}
    arr = arrays_for(params["feature"])
    arr_tr = [a for a, k in zip(arr, is_ds) if k]
    sg_oof, _ = gmm_oof(arr_tr, y_tr, f_tr, params)
    spec = DetectorSpec("lr:all", use_gmm=False, feature_groups=tuple(GROUPS), classifier="lr",
                        lr_C_grid=lr_grid, seed=seed)
    sl_oof, _ = detector_oof(spec, tr_tab, f_tr)
    fus = LateFusion(["gmm", "lr"], prior).fit(np.column_stack([sg_oof, sl_oof]), y_tr)
    det_g = Detector(DetectorSpec("gmm", gmm=params, seed=seed)).fit(tr_tab, y_tr, tr_tab["group_id"].values,
                                                                     arrays=arr_tr)
    det_l = Detector(spec).fit(tr_tab, y_tr, tr_tab["group_id"].values)
    ext_tab = table[ext].reset_index(drop=True)
    arr_ext = [a for a, k in zip(arr, ext) if k]
    p_ext = fus.predict(np.column_stack([[det_g.gmm.score(a) for a in arr_ext],
                                         det_l.decision_function(ext_tab)]))
    ext_model_p = 1 / (1 + np.exp(-oof[f"fusion:{gsel}+lr:all"].values[ext])) \
        if f"fusion:{gsel}+lr:all" in oof else None
    out = {"train_rows": int(is_ds.sum()), "train_real": int((y_tr == 0).sum()),
           "note": "DiffSSD-only model never saw LibriSpeech or LJSpeech-1.1 files; extended model "
                   "scores are its out-of-fold probabilities on the same rows"}
    for src in sorted(ext_tab["source_id"].unique()):
        mm = (ext_tab["source_id"] == src).values
        out[src] = {"n": int(mm.sum()),
                    "diffssd_only_model_genuine_fpr@0.5": float((p_ext[mm] >= 0.5).mean()),
                    "diffssd_only_model_median_p": float(np.median(p_ext[mm]))}
        if ext_model_p is not None:
            out[src]["extended_model_oof_genuine_fpr@0.5"] = float((ext_model_p[mm] >= 0.5).mean())
            out[src]["extended_model_oof_median_p"] = float(np.median(ext_model_p[mm]))
    return out


def load_table_from_rows(rows: pd.DataFrame, cfg: dict):
    from ..pipeline import extract_paths, records_to_table
    recs = extract_paths(rows["path"].tolist(), cfg, log_every=2000)
    ft = records_to_table(recs)
    return rows.merge(ft, on="path", how="left", validate="one_to_one"), recs
