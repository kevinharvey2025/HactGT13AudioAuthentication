"""Render markdown reports from run directories (numbers come only from saved JSON)."""

from __future__ import annotations

import json
from pathlib import Path


def _f(v, nd=3):
    if v is None:
        return "n/a"
    if isinstance(v, (list, tuple)) and len(v) == 2:
        return f"[{_f(v[0], nd)}, {_f(v[1], nd)}]"
    try:
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return str(v)


def _ci(o, key):
    c = o.get(f"{key}_ci95")
    return f" [{_f(c['lo'])}, {_f(c['hi'])}]" if c else ""


def summary_row(name: str, s: dict, runtime=None) -> str:
    o = s["overall"]
    cov = s.get("coverage")
    return (f"| {name} | {_f(o.get('auc'))}{_ci(o, 'auc')} | {_f(o.get('eer'))}{_ci(o, 'eer')} | "
            f"{_f(o.get('log_loss_balanced'))} | {_f(o.get('genuine_fpr'))} | "
            f"{_f(cov, 3) if cov is not None else 'n/a'} | {_f(runtime, 3) if runtime is not None else 'n/a'} |")


HEADER = ("| Configuration | AUC [95% CI] | EER [95% CI] | Balanced log loss | Genuine FPR @0.5 | "
          "Coverage | Module runtime (s/file) |\n|---|---|---|---|---|---|---|")


def per_generator_table(configs: dict, names: list[str]) -> str:
    gens = sorted(next(iter(configs.values()))["per_generator"])
    lines = ["| Generator | " + " | ".join(names) + " |", "|---" * (len(names) + 1) + "|"]
    for g in gens:
        lines.append(f"| {g} | " + " | ".join(_f(configs[n]["per_generator"][g]["auc"]) for n in names) + " |")
    return "\n".join(lines)


def render_suite(res: dict) -> str:
    cfgs = res.get("configs", {})
    rt = res.get("module_runtime_s", {})
    out = [f"Dev rows: {res['n_dev']} ({res['n_real']} real / {res['n_synthetic']} synthetic); "
           f"decode failures excluded: {res['n_decode_failures']}. All numbers are 5-fold grouped "
           "out-of-fold (OOF) on dev rows only.\n", HEADER]
    for name, s in cfgs.items():
        out.append(summary_row(name, s, s.get("runtime_modules_s")))
    if "paired_vs_lr_all" in res:
        out += ["\n**Paired AUC deltas vs `lr:all` (same rows, group bootstrap 95% CI):**\n",
                "| Configuration | ΔAUC | 95% CI |", "|---|---|---|"]
        for k, v in res["paired_vs_lr_all"].items():
            out.append(f"| {k} | {v['delta_auc']:+.4f} | {_f(v['ci95'], 4)} |")
    for key in ("paired_fusion_vs_lr_all", "paired_fusion_vs_gmm"):
        if key in res:
            v = res[key]
            out.append(f"\n{key}: ΔAUC {v['delta_auc']:+.4f}, 95% CI {_f(v['ci95'], 4)}")
    main = [n for n in cfgs if n.startswith(("gmm:custom(nonsilent,64c)", "lr:all", "fusion:", "hgb:all",
                                             "gmm:official_pretrained"))]
    if main:
        out += ["\n**Per-generator AUC (all dev reals vs that generator's fakes):**\n",
                per_generator_table(cfgs, main)]
    if "logo" in res:
        out += ["\n**Leave-one-generator-out (generator's fakes excluded from all training):**\n",
                "| Held-out generator | n fake | GMM AUC (seen) | LR AUC (seen) | Fusion AUC (seen) |",
                "|---|---|---|---|---|"]
        for g, e in res["logo"].items():
            cells = [f"{_f(e[k]['auc'])} ({_f(e[k].get('auc_when_seen'))})" for k in ("gmm", "lr_all", "fusion")]
            out.append(f"| {g} | {e['n_fake']} | " + " | ".join(cells) + " |")
    if "shortcut_audit" in res:
        out += ["\n**Shortcut audit (OOF AUC of excluded properties used alone):**\n",
                "| Property set | Columns | OOF AUC |", "|---|---|---|"]
        for k, v in res["shortcut_audit"].items():
            cols = ", ".join(v.get("columns", [])) or f"{v.get('n_columns')} indicator columns"
            out.append(f"| {k} | {cols} | {_f(v.get('oof_auc'))} |")
    if "speaker_audit" in res:
        sa = res["speaker_audit"]
        out += ["\n**Speaker / recording-chain audit (reals never seen by the DiffSSD-only model):**\n",
                f"DiffSSD-only training rows: {sa.get('train_rows')} ({sa.get('train_real')} real).\n",
                "| Real source | n | DiffSSD-only model: genuine FPR@0.5 (median p) | "
                "Extended model OOF: genuine FPR@0.5 (median p) |", "|---|---|---|---|"]
        for k, v in sa.items():
            if isinstance(v, dict):
                out.append(f"| {k} | {v['n']} | {_f(v['diffssd_only_model_genuine_fpr@0.5'])} "
                           f"({_f(v['diffssd_only_model_median_p'])}) | "
                           f"{_f(v.get('extended_model_oof_genuine_fpr@0.5'))} "
                           f"({_f(v.get('extended_model_oof_median_p'))}) |")
    if rt:
        out += ["\n**Mean module runtime (s/file, measured on this machine under load):**\n",
                "| Module | s/file |", "|---|---|"] + [f"| {m} | {_f(v, 4)} |" for m, v in sorted(rt.items())]
    return "\n".join(out)


def render_evaluation(res: dict) -> str:
    out = [f"Rows: {res['rows']} (decode failures excluded: {res['decode_failures']}); splits "
           f"{res['splits']}{' where ' + str(res['where']) if res.get('where') else ''}.\n", HEADER]
    for k in ("full", "core", "routed"):
        out.append(summary_row(k, res[k]))
    out.append(f"\nRouted: {res['routed_frac_extended']:.3f} of files ran extended modules.")
    out.append(f"Component raw-score AUCs: {json.dumps({k: round(v, 4) for k, v in res['components_raw_auc'].items()})}")
    out += ["\n**Per generator (full model):**\n", "| Generator | n fake | AUC | EER | Sensitivity @0.5 |",
            "|---|---|---|---|---|"]
    for g, v in res["full"]["per_generator"].items():
        out.append(f"| {g} | {v['n_fake']} | {_f(v['auc'])} | {_f(v['eer'])} | {_f(v['sensitivity'])} |")
    out += ["\n**Genuine false-positive rate by real source (full model):**\n",
            "| Real source | n | FPR @0.5 | median p |", "|---|---|---|---|"]
    for s, v in res["full"]["per_real_source"].items():
        out.append(f"| {s} | {v['n']} | {_f(v['genuine_fpr@0.5'])} | {_f(v['median_p'])} |")
    return "\n".join(out)


def render_robustness(res: dict) -> str:
    out = ["| Condition | n | AUC | EER | Genuine FPR @0.5 | Sensitivity @0.5 | Balanced log loss |",
           "|---|---|---|---|---|---|---|"]
    for cond, s in res["conditions"].items():
        o = s["overall"]
        out.append(f"| {cond} | {o['n']} | {_f(o['auc'])} | {_f(o['eer'])} | {_f(o['genuine_fpr'])} | "
                   f"{_f(o['sensitivity'])} | {_f(o.get('log_loss_balanced'))} |")
    return "\n".join(out)


def render_run_report(run_dir: str) -> Path:
    d = Path(run_dir)
    parts = [f"# DSP run report: {d.name}\n"]
    for fname, fn, title in (("suite_results.json", render_suite, "Development ablations"),
                             ("evaluation.json", render_evaluation, "Evaluation"),
                             ("robustness.json", render_robustness, "Robustness")):
        p = d / fname
        if p.exists():
            parts += [f"## {title}\n", fn(json.loads(p.read_text())), ""]
    out = d / "report.md"
    out.write_text("\n".join(parts))
    return out
