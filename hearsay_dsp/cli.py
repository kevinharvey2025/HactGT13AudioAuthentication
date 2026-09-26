"""Command-line interface for the HEARSAY DSP track.

  python -m hearsay_dsp.cli validate --manifest runs/dsp/manifests/pool.tsv
  python -m hearsay_dsp.cli extract  --manifest runs/dsp/manifests/pool.tsv --config configs/dsp.yaml
  python -m hearsay_dsp.cli train    --manifest runs/dsp/manifests/pool.tsv --config configs/dsp.yaml --out artifacts/dsp/model
  python -m hearsay_dsp.cli evaluate --manifest runs/dsp/manifests/pool.tsv --model artifacts/dsp/model --splits holdout
  python -m hearsay_dsp.cli predict  --input data/hearsay_test --model artifacts/dsp/model --output predictions.tsv
  python -m hearsay_dsp.cli report   --run runs/dsp/evaluate_holdout

`train` never uses rows whose split is eval/evaluation/test/holdout.
`predict` never trains; it fails with an actionable error if the bundle is missing,
and refuses to write the TSV if any expected file lacks a valid score.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_config
from .evaluation.experiments import FrameStore, compact, save_json, summarize
from .io.export import ExportError, build_submission, read_reference, write_submission
from .io.manifest import load_inputs, read_manifest, select_training_rows, validate_manifest
from .pipeline import extract_paths, records_to_table
from .routing import ALL_MODULES, CORE_MODULES


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _cfg(args, base: dict | None = None) -> dict:
    cfg = base if base is not None else load_config(getattr(args, "config", None))
    if getattr(args, "cache_dir", None):
        cfg["run"]["cache_dir"] = args.cache_dir
    if getattr(args, "n_jobs", None):
        cfg["run"]["n_jobs"] = args.n_jobs
    return cfg


def _filter(df: pd.DataFrame, where: list[str] | None) -> pd.DataFrame:
    for cond in where or []:
        col, val = cond.split("=", 1)
        df = df[df[col].astype(str).isin(val.split(","))]
    return df


def _table(df: pd.DataFrame, cfg: dict, modules=ALL_MODULES) -> tuple[pd.DataFrame, list]:
    recs = extract_paths(df["path"].tolist(), cfg, modules, log_every=1000)
    ft = records_to_table(recs)
    return df.merge(ft, on="path", how="left", validate="one_to_one"), recs


# ---------------------------------------------------------------------------
def cmd_validate(args) -> int:
    from joblib import Parallel, delayed
    from .io.decode import probe_file
    df = load_inputs(args.manifest, args.input)
    df = _filter(df, args.where)
    rep = validate_manifest(df)
    probes = Parallel(n_jobs=args.n_jobs or 8)(delayed(probe_file)(p) for p in df["path"]
                                               if os.path.isfile(p))
    pr = pd.DataFrame(probes)
    m = df.merge(pr, on="path", how="left")
    rep["probe_errors"] = int((m["probe_status"] == "error").sum())
    dup = m[m["sha256"].duplicated(keep=False) & m["sha256"].notna()]
    rep["duplicate_content_groups"] = int(dup["sha256"].nunique())
    rep["duplicate_content_files"] = int(len(dup))
    if len(dup) and "group_id" in m and (m["group_id"] != "").any():
        split_conflicts = dup.groupby("sha256")["split"].nunique()
        rep["duplicate_content_across_splits"] = int((split_conflicts > 1).sum())
    d = m["duration_s"].astype(float)
    rep["duration_s"] = {k: float(v) for k, v in d.describe(percentiles=[.05, .5, .95]).items()}
    rep["n_shorter_than_2s"] = int((d < 2.0).sum())
    by = "label_raw" if "label_raw" in m else None
    prof = m.groupby([by] if by else lambda _: "all").agg(
        n=("path", "size"), sr=("sr", lambda x: dict(Counter(x))),
        codec=("codec", lambda x: dict(Counter(x))), sniffed=("sniffed_format", lambda x: dict(Counter(x))),
        ext=("extension", lambda x: dict(Counter(x))), dur_med=("duration_s", "median"))
    rep["technical_profile_by_label"] = json.loads(prof.to_json(orient="index"))
    if args.reference:
        ref = read_reference(args.reference)
        names = set(df["filename"])
        rep["reference"] = {"path": args.reference, "n": len(ref),
                            "missing_from_inputs": sorted(set(ref) - names)[:20],
                            "n_missing_from_inputs": len(set(ref) - names),
                            "n_inputs_not_in_reference": len(names - set(ref))}
    print(json.dumps(rep, indent=1, default=str))
    if args.out:
        save_json(rep, args.out)
        m.to_csv(Path(args.out).with_suffix(".probe.tsv"), sep="\t", index=False)
    return 1 if rep["errors"] else 0


def cmd_extract(args) -> int:
    cfg = _cfg(args)
    df = _filter(load_inputs(args.manifest, args.input), args.where)
    modules = CORE_MODULES if args.modules == "core" else ALL_MODULES
    t0 = time.time()
    table, recs = _table(df, cfg, modules)
    el = time.time() - t0
    n_err = int((table["decode_status"] == "error").sum())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    table.to_pickle(out / "features.pkl")
    table.to_csv(out / "features.tsv", sep="\t", index=False)
    summ = {"n": len(table), "decode_errors": n_err, "wall_s": el, "modules": list(modules),
            "module_status": {c[7:]: table[c].value_counts().to_dict() for c in table if c.startswith("status.")}}
    save_json(summ, out / "extract_summary.json")
    _log(json.dumps(summ, default=str))
    return 0 if n_err == 0 else 2


def cmd_train(args) -> int:
    from .models.bundle import ModelBundle
    cfg = _cfg(args)
    df = _filter(read_manifest(args.manifest), args.where)
    train = select_training_rows(df)
    blocked = len(df) - len(train)
    _log(f"[train] {len(train)} training rows; {blocked} rows excluded (unlabeled or eval/test/holdout)")
    table, _ = _table(train, cfg)
    bad = table["decode_status"] == "error"
    if bad.any():
        _log(f"[train] dropping {int(bad.sum())} undecodable rows")
        table = table[~bad]
    bundle = ModelBundle.train(table.reset_index(drop=True), cfg, log=_log)
    bundle.save(args.out)
    _log(json.dumps({k: compact({"overall": v}) for k, v in bundle.meta["oof_metrics"].items()}, indent=1))
    return 0


def _score(bundle, table: pd.DataFrame, mode: str) -> pd.DataFrame:
    cfg = bundle.cfg
    store = FrameStore(cfg, bundle.dets["gmm"].spec.gmm["feature"])
    comps = bundle.component_scores(table, store)
    probs = bundle.probabilities(comps)
    final, used_full = bundle.route(probs["p_core"].values, probs["p_full"].values, mode)
    out = pd.concat([comps, probs], axis=1)
    out["p_final"] = final
    out["used_extended"] = used_full
    return out


def cmd_evaluate(args) -> int:
    from .models.bundle import ModelBundle
    bundle = ModelBundle.load(args.model)
    cfg = _cfg(args, bundle.cfg)
    df = _filter(read_manifest(args.manifest), args.where)
    splits = set(args.splits.split(","))
    df = df[df["split"].astype(str).isin(splits) & df["label"].notna()].reset_index(drop=True)
    table, _ = _table(df, cfg)
    ok = table["decode_status"] != "error"
    table = table[ok].reset_index(drop=True)
    sc = _score(bundle, table, "all_eligible")
    lo, hi = bundle.meta["routing"]["band"]
    routed = np.where((sc["p_core"] >= lo) & (sc["p_core"] <= hi), sc["p_full"], sc["p_core"])
    res = {"model": str(args.model), "rows": len(table), "decode_failures": int((~ok).sum()),
           "splits": sorted(splits), "where": args.where,
           "full": summarize(table, sc["p_full"].values, sc["p_full"].values),
           "core": summarize(table, sc["p_core"].values, sc["p_core"].values),
           "routed": summarize(table, routed, routed),
           "routed_frac_extended": float(((sc["p_core"] >= lo) & (sc["p_core"] <= hi)).mean()),
           "components_raw_auc": {c: summarize(table, sc[c].values, sc[c].values, n_boot=0)["overall"]["auc"]
                                  for c in ("gmm", "lr_core", "lr_full")}}
    out = Path(args.out)
    save_json(res, out / "evaluation.json")
    pd.concat([table[["path", "label", "attack_type", "source_id", "group_id"]], sc], axis=1).to_csv(
        out / "scores.tsv", sep="\t", index=False)
    _log(json.dumps({k: compact(res[k]) for k in ("full", "core", "routed")}, indent=1))
    return 0


def cmd_predict(args) -> int:
    from .models.bundle import ModelBundle
    from .trace import explanation_text, module_findings
    try:
        bundle = ModelBundle.load(args.model)
    except FileNotFoundError as exc:
        _log(f"error: {exc}")
        return 3
    cfg = _cfg(args, bundle.cfg)
    mode = args.routing or cfg["routing"]["mode"]
    df = load_inputs(args.manifest, args.input)
    names = df["filename"]
    if names.duplicated().any():
        _log(f"error: duplicate basenames would collide in the TSV: {names[names.duplicated()].tolist()[:5]}")
        return 4
    reference = read_reference(args.reference) if args.reference else None
    if reference is not None:
        missing = sorted(set(reference) - set(names))
        if missing:
            _log(f"error: {len(missing)} reference files not found in inputs, e.g. {missing[:5]}")
            return 4
        df = df[df["filename"].isin(set(reference))].reset_index(drop=True)
    t0 = time.time()
    table, recs = _table(df, cfg, CORE_MODULES if mode == "confidence" else ALL_MODULES)
    t_core = time.time() - t0
    rec_by_path = {r["path"]: r for r in recs}
    failed = table["decode_status"] == "error"
    good = table[~failed].reset_index(drop=True)
    comps = bundle.component_scores(good, which=("gmm", "lr_core"))
    p_core = bundle.fusion["core"].predict(comps[["gmm", "lr_core"]].values)
    if mode == "confidence":
        lo, hi = bundle.meta["routing"]["band"]
        need = (p_core >= lo) & (p_core <= hi)
    else:
        need = np.ones(len(good), bool)
    p_full = np.full(len(good), np.nan)
    t1 = time.time()
    if need.any():
        ext_df = good.loc[need, df.columns.tolist()].reset_index(drop=True)
        ext_table, ext_recs = _table(ext_df, cfg, ALL_MODULES)
        rec_by_path.update({r["path"]: r for r in ext_recs})
        c_full = bundle.dets["lr_full"].decision_function(ext_table)
        comps.loc[need, "lr_full"] = c_full
        p_full[need] = bundle.fusion["full"].predict(comps.loc[need, ["gmm", "lr_full"]].values)
    t_ext = time.time() - t1
    final = np.where(need, p_full, p_core)
    good["cm-score"] = final
    bad_score = ~np.isfinite(final)
    failures = pd.concat([
        table.loc[failed, ["path", "filename"]].assign(reason=table.loc[failed, "decode_error"]),
        good.loc[bad_score, ["path", "filename"]].assign(reason="non-finite score")])
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if len(failures):
        failures.to_csv(out.with_suffix(".failures.tsv"), sep="\t", index=False)
    # traces
    trace_path = Path(args.trace or out.with_suffix(".traces.jsonl"))
    fus_c, fus_f = bundle.fusion["core"], bundle.fusion["full"]
    contrib = {}
    if need.any():
        ext_rows = good.loc[need].reset_index(drop=True)
        ext_tab = records_to_table([rec_by_path[p] for p in ext_rows["path"]])
        ext_tab = ext_rows[["path"]].merge(ext_tab, on="path", how="left")
        for p, cl in zip(ext_rows["path"], bundle.dets["lr_full"].contributions(ext_tab)):
            contrib[p] = cl
    with open(trace_path, "w", encoding="utf-8") as fh:
        for i, row in good.iterrows():
            rec = rec_by_path.get(row["path"], {})
            findings = module_findings(rec.get("modules", {}))
            use_full = bool(need[i])
            fus = fus_f if use_full else fus_c
            lr_name = "lr_full" if use_full else "lr_core"
            gmm_term = float(fus.coef[0] * comps.loc[i, "gmm"]) if np.isfinite(comps.loc[i, "gmm"]) else None
            lr_term = float(fus.coef[1] * comps.loc[i, lr_name])
            top = [dict(c, contribution_logit=c["contribution_logit"] * fus.coef[1])
                   for c in contrib.get(row["path"], [])]
            p = float(final[i])
            trace = {
                "filename": row["filename"], "path": row["path"], "sha256": row["sha256"],
                "model_config_hash": bundle.meta["config_hash"], "cm_score": p,
                "score_status": "ok" if np.isfinite(p) else "error",
                "route": "full" if use_full else "core",
                "components": {"gmm_llr": comps.loc[i, "gmm"], "lr_core_logit": comps.loc[i, "lr_core"],
                               "lr_full_logit": comps.loc[i, "lr_full"] if "lr_full" in comps and use_full else None,
                               "p_core": float(p_core[i]), "p_full": float(p_full[i]) if use_full else None},
                "routing": rec.get("routing", []) + [{"module": "extended_modules",
                                                      "decision": "run" if use_full else "skip",
                                                      "reason": ("mode all_eligible" if mode != "confidence"
                                                                 else f"p_core={p_core[i]:.3f} vs band "
                                                                      f"{bundle.meta['routing']['band']}")}],
                "analyses": findings,
                "model_contributions": top,
                "explanation": explanation_text(p, "full" if use_full else "core", gmm_term, lr_term,
                                                top, findings),
            }
            fh.write(json.dumps(trace, default=lambda o: None if isinstance(o, float) and not np.isfinite(o)
                                else (o.item() if hasattr(o, "item") else str(o))) + "\n")
    summary = {"n_inputs": len(df), "n_scored": int(np.isfinite(final).sum()), "n_failed": len(failures),
               "routing_mode": mode, "n_extended": int(need.sum()), "core_stage_s": t_core,
               "extended_stage_s": t_ext, "wall_s": time.time() - t0,
               "score_quantiles": {q: float(np.nanquantile(final, q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)},
               "frac_ge_0.5": float(np.nanmean(final >= 0.5)), "model": str(args.model),
               "model_meta": {k: bundle.meta[k] for k in ("config_hash", "created_utc",
                                                          "training_content_sha256_hash")}}
    save_json(summary, out.with_suffix(".summary.json"))
    _log(json.dumps(summary, indent=1))
    if len(failures):
        _log(f"error: {len(failures)} files lack a valid score (see {out.with_suffix('.failures.tsv')}); "
             f"TSV not written")
        return 2
    try:
        sub = build_submission(good["filename"], final, reference)
    except ExportError as exc:
        _log(f"error: {exc}")
        return 2
    write_submission(sub, out, cfg["export"]["float_format"])
    _log(f"wrote {out} ({len(sub)} rows)")
    return 0


def cmd_report(args) -> int:
    from .evaluation.report import render_run_report
    path = render_run_report(args.run)
    _log(f"wrote {path}")
    return 0


def cmd_experiment(args) -> int:
    from .evaluation.suite import run_suite
    cfg = _cfg(args)
    run_suite(args.manifest, cfg, args.out, args.parts.split(",") if args.parts else None, args.where)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="hearsay_dsp", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, config=True):
        if config:
            p.add_argument("--config", default=None)
        p.add_argument("--cache-dir", default=None)
        p.add_argument("--n-jobs", type=int, default=None)
        p.add_argument("--where", action="append", help="column=value[,value] row filter (repeatable)")

    p = sub.add_parser("validate")
    p.add_argument("--manifest")
    p.add_argument("--input")
    p.add_argument("--reference")
    p.add_argument("--out")
    common(p)
    p.set_defaults(fn=cmd_validate)

    p = sub.add_parser("extract")
    p.add_argument("--manifest")
    p.add_argument("--input")
    p.add_argument("--modules", choices=["core", "all"], default="all")
    p.add_argument("--out", default="runs/dsp/extract")
    common(p)
    p.set_defaults(fn=cmd_extract)

    p = sub.add_parser("train")
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", default="artifacts/dsp/model")
    common(p)
    p.set_defaults(fn=cmd_train)

    p = sub.add_parser("evaluate")
    p.add_argument("--manifest", required=True)
    p.add_argument("--model", default="artifacts/dsp/model")
    p.add_argument("--splits", default="holdout")
    p.add_argument("--out", default="runs/dsp/evaluate")
    common(p, config=False)
    p.set_defaults(fn=cmd_evaluate)

    p = sub.add_parser("predict")
    p.add_argument("--input")
    p.add_argument("--manifest")
    p.add_argument("--model", default="artifacts/dsp/model")
    p.add_argument("--output", required=True)
    p.add_argument("--reference", help="prefilled/template TSV defining filenames and order")
    p.add_argument("--routing", choices=["all_eligible", "confidence"])
    p.add_argument("--trace")
    common(p, config=False)
    p.set_defaults(fn=cmd_predict)

    p = sub.add_parser("report")
    p.add_argument("--run", required=True)
    p.set_defaults(fn=cmd_report)

    p = sub.add_parser("experiment")
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", default="runs/dsp/experiments")
    p.add_argument("--parts")
    common(p)
    p.set_defaults(fn=cmd_experiment)

    args = ap.parse_args(argv)
    return int(args.fn(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
