"""Robustness of a frozen DSP bundle under matched channel transforms.

A seeded sample of labeled holdout rows (both classes) is transformed with each
EVAL_CONDITIONS chain (settings not used for training augmentation), analyzed
by the normal pipeline, and scored with the frozen bundle. Derivatives keep
the source row's labels and group ids. Nothing here is fitted.

  python scripts/dsp_robustness.py --manifest runs/dsp/manifests/pool.tsv \
      --model artifacts/dsp/model --out runs/dsp/robustness
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hearsay_dsp.evaluation.augment import EVAL_CONDITIONS, make_variant  # noqa: E402
from hearsay_dsp.evaluation.experiments import save_json, summarize  # noqa: E402
from hearsay_dsp.io.decode import file_sha256  # noqa: E402
from hearsay_dsp.io.manifest import read_manifest  # noqa: E402
from hearsay_dsp.models.bundle import ModelBundle  # noqa: E402
from hearsay_dsp.pipeline import extract_paths, records_to_table  # noqa: E402


def sample_rows(df: pd.DataFrame, n_real: int, n_fake: int, seed: int) -> pd.DataFrame:
    parts = []
    for src, g in df[df.label == 0].groupby("source_id"):
        parts.append(g.sample(min(n_real, len(g)), random_state=seed))
    for gen, g in df[df.label == 1].groupby("attack_type"):
        parts.append(g.sample(min(n_fake, len(g)), random_state=seed))
    return pd.concat(parts).reset_index(drop=True)


def _variant(path, cond, cfg, out_root, seed):
    sha = file_sha256(path)
    return make_variant(path, sha, cond, EVAL_CONDITIONS[cond], cfg, out_root, seed)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--model", default=str(ROOT / "artifacts/dsp/model"))
    ap.add_argument("--out", default=str(ROOT / "runs/dsp/robustness"))
    ap.add_argument("--splits", default="holdout")
    ap.add_argument("--n-real-per-source", type=int, default=150)
    ap.add_argument("--n-fake-per-generator", type=int, default=40)
    ap.add_argument("--conditions", default=",".join(EVAL_CONDITIONS))
    ap.add_argument("--n-jobs", type=int, default=7)
    args = ap.parse_args()
    bundle = ModelBundle.load(args.model)
    cfg = bundle.cfg
    cfg["run"]["n_jobs"] = args.n_jobs
    seed = cfg["run"]["seed"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = read_manifest(args.manifest)
    df = df[df["split"].isin(args.splits.split(",")) & df["label"].notna()]
    rows = sample_rows(df, args.n_real_per_source, args.n_fake_per_generator, seed)
    rows.to_csv(out / "sampled_rows.tsv", sep="\t", index=False)
    aug_root = ROOT / cfg["run"]["cache_dir"] / "augmented"
    res = {"model": args.model, "model_config_hash": bundle.meta["config_hash"],
           "n_rows": len(rows), "sample": {"n_real_per_source": args.n_real_per_source,
                                           "n_fake_per_generator": args.n_fake_per_generator,
                                           "seed": seed, "splits": args.splits},
           "conditions": {}, "chains": {}}
    for cond in args.conditions.split(","):
        t0 = time.time()
        if EVAL_CONDITIONS[cond]:
            paths = Parallel(n_jobs=args.n_jobs)(delayed(_variant)(p, cond, cfg, aug_root, seed)
                                                 for p in rows["path"])
        else:
            paths = rows["path"].tolist()
        recs = extract_paths(paths, cfg, log_every=0)
        tab = rows.assign(path=paths).merge(records_to_table(recs), on="path", how="left")
        ok = (tab["decode_status"] != "error").values
        tab = tab[ok].reset_index(drop=True)
        comps = bundle.component_scores(tab)
        probs = bundle.probabilities(comps)
        s = summarize(tab, probs["p_full"].values, probs["p_full"].values, n_boot=200, seed=seed)
        s["n_failed"] = int((~ok).sum())
        s["core_overall"] = summarize(tab, probs["p_core"].values, probs["p_core"].values, n_boot=0)["overall"]
        s["component_auc"] = {c: summarize(tab, comps[c].values, comps[c].values, n_boot=0)["overall"]["auc"]
                              for c in comps}
        s["wall_s"] = time.time() - t0
        res["conditions"][cond] = s
        res["chains"][cond] = EVAL_CONDITIONS[cond]
        o = s["overall"]
        print(f"[robustness] {cond}: n={o['n']} auc={o['auc']:.4f} eer={o['eer']:.4f} "
              f"genuine_fpr={o['genuine_fpr']:.3f} sens={o['sensitivity']:.3f} ({s['wall_s']:.0f}s)",
              flush=True)
        pd.concat([tab[["path", "label", "attack_type", "source_id"]], comps, probs], axis=1).to_csv(
            out / f"scores_{cond}.tsv", sep="\t", index=False)
        save_json(res, out / "robustness.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
