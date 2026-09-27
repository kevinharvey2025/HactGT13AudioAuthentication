"""Compare single systems and small ensembles on the shared evaluation sets (never the test labels, which we
do not have).

    python scripts/compare_systems.py [--runs name[@epoch] ...] [--max-ensemble 3] [--top 25]

Every run in runs/diffusion/ft/ (or the given ones; '@k' picks epoch k, default: every epoch) contributes one
system per epoch. Ensembles average z-normalized logits (statistics from the selection sets). Rows are
ranked by `select` = mean minDCF over val and In-the-Wild, clean and augmented views (the same criterion the
fine-tuning used); holdout is reported alongside and never used for ranking. Also reports each system's
test-score summary (share above the Bayes threshold after Platt calibration on the selection sets).
Writes runs/diffusion/compare.csv.
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hearsay import config, metrics, submission  # noqa: E402

FT = config.RUNS / "ft"
SETS = ("val", "itw", "holdout")


def systems(specs):
    out = {}
    names = specs or sorted(p.name for p in FT.iterdir() if (p / "log.jsonl").exists() and p.name != "smoke")
    for spec in names:
        name, _, ep = spec.partition("@")
        epochs = [int(ep)] if ep else sorted(int(p.stem.split("epoch")[1]) for p in (FT / name).glob("test_epoch*.parquet"))
        for e in epochs:
            files = {s: FT / name / f"{s}_epoch{e}.parquet" for s in SETS}
            if not all(f.exists() for f in files.values()) or not (FT / name / f"test_epoch{e}.parquet").exists():
                continue
            ev = {s: pd.read_parquet(f).set_index("uid") for s, f in files.items()}
            out[f"{name}@{e}"] = (ev, pd.read_parquet(FT / name / f"test_epoch{e}.parquet").set_index("filename").score)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*")
    ap.add_argument("--max-ensemble", type=int, default=3)
    ap.add_argument("--pool", type=int, default=8, help="ensembles only over the best N single systems")
    ap.add_argument("--top", type=int, default=25)
    a = ap.parse_args()
    S = systems(a.runs)
    common = {s: sorted(set.intersection(*[set(S[k][0][s].index) for k in S])) for s in SETS}
    ytab = {s: S[next(iter(S))][0][s].loc[common[s]].label.to_numpy() for s in SETS}

    z, zt = {}, {}
    for k, (ev, test) in S.items():
        ref = np.concatenate([ev[s].loc[common[s], v].to_numpy() for s in ("val", "itw") for v in ("score_clean", "score_aug")])
        mu, sd = ref.mean(), ref.std() + 1e-6
        z[k] = {s: {v: (ev[s].loc[common[s], v].to_numpy() - mu) / sd for v in ("score_clean", "score_aug")} for s in SETS}
        zt[k] = (test - mu) / sd

    def evaluate(keys):
        r = {"system": " + ".join(keys), "k": len(keys)}
        sel = []
        for s in SETS:
            for v, tag in (("score_clean", "clean"), ("score_aug", "aug")):
                sc = np.mean([z[k][s][v] for k in keys], 0)
                m = metrics.summary(ytab[s], sc)
                r[f"{s}_{tag}_mindcf"], r[f"{s}_{tag}_eer"] = round(m["min_dcf"], 4), round(m["eer"], 4)
                if s in ("val", "itw"):
                    sel.append(m["min_dcf"])
        r["select"] = round(float(np.mean(sel)), 4)
        # test: calibrated on the selection sets, share flagged at the Bayes threshold (P > 0.2)
        ycal = np.concatenate([np.tile(ytab[s], 2) for s in ("val", "itw")])
        scal = np.concatenate([np.concatenate([np.mean([z[k][s][v] for k in keys], 0) for v in ("score_clean", "score_aug")])
                               for s in ("val", "itw")])
        pl = submission.Platt(prior=metrics.DCF["p_spoof"]).fit(scal, ycal)
        pt = pl(pd.concat([zt[k] for k in keys], axis=1).mean(1).to_numpy())
        thr = metrics.bayes_threshold(calib_prior=metrics.DCF["p_spoof"])
        r["test_frac_flagged"], r["test_frac_gt_0_5"] = round(float((pt > thr).mean()), 3), round(float((pt > 0.5).mean()), 3)
        return r

    singles = sorted((evaluate([k]) for k in S), key=lambda r: r["select"])
    pool = [r["system"] for r in singles[: a.pool]]
    rows = list(singles)
    for n in range(2, a.max_ensemble + 1):
        for combo in itertools.combinations(pool, n):
            if len({c.split("@")[0] for c in combo}) < n:
                continue  # one epoch per run in an ensemble
            rows.append(evaluate(list(combo)))
    df = pd.DataFrame(rows).sort_values("select")
    df.to_csv(config.RUNS / "compare.csv", index=False)
    cols = ["system", "select", "val_clean_mindcf", "val_aug_mindcf", "itw_clean_mindcf", "itw_aug_mindcf", "itw_clean_eer",
            "holdout_clean_mindcf", "holdout_aug_mindcf", "test_frac_flagged", "test_frac_gt_0_5"]
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 70)
    print(df[cols].head(a.top).to_string(index=False))
    print(f"\n{len(S)} single systems, {len(df) - len(S)} ensembles; common eval clips:", {s: len(common[s]) for s in SETS})


if __name__ == "__main__":
    main()
