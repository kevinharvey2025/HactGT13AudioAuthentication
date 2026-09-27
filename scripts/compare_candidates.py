"""Compare candidate submissions (scripts/candidate.sh) with the shipped final, without test labels.

    python scripts/compare_candidates.py TAG [TAG ...]

For each candidate: In-the-Wild minDCF per view of its fused score, rebuilt from the members' stored per-epoch scores
with the candidate's own fusion.json (in-sample for calibration, as in the shipped recipe), and on the NSA test set
its flag rate at P > 0.2, decision agreement and Spearman correlation with the shipped final scored by the same path
(runs/diffusion/predict_final_v3). Writes runs/diffusion/candidates/comparison.csv.
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hearsay import config, metrics  # noqa: E402

FT, CAND = config.RUNS / "ft", config.RUNS / "candidates"
THR = metrics.bayes_threshold(calib_prior=metrics.DCF["p_spoof"])


def itw(fusion):
    parts = []
    for s in fusion["systems"]:
        d = pd.read_parquet(FT / s["name"] / f"itw_epoch{s['epoch']}.parquet").set_index("uid")
        parts.append(((d[["score_clean", "score_aug"]] - s["z_mean"]) / s["z_std"], d.label))
    u = sorted(set.intersection(*[set(p[0].index) for p in parts]))
    fused = sum(p[0].loc[u] for p in parts) / len(parts)
    y = parts[0][1].loc[u].to_numpy().astype(int)
    return {v: metrics.min_dcf(y, fused[f"score_{v}"].to_numpy()) for v in ("clean", "aug")}, len(u)


def main():
    read = lambda p: pd.read_csv(p, sep="\t").set_index("filename")["cm-score"]
    ref = read(config.RUNS / "predict_final_v3" / "SideQuests_predictions_final.tsv")
    ship = json.load(open(config.REPO / "submission" / "fusion.json"))
    rows = []
    for tag, fusion, tsv in [("A_verified", ship, config.RUNS / "predict_final_v3" / "SideQuests_predictions_final.tsv")] + \
            [(t, json.load(open(CAND / t / "fusion.json")), CAND / t / "predict" / "SideQuests_predictions_final.tsv") for t in sys.argv[1:]]:
        m, n = itw(fusion)
        r = dict(candidate=tag, members=" + ".join(f"{s['name']}@{s['epoch']}" for s in fusion["systems"]), itw_n=n,
                 itw_clean=m["clean"], itw_aug=m["aug"])
        if tsv.exists():
            p = read(tsv).loc[ref.index]
            r.update(test_flagged=float((p > THR).mean()), agree_with_A=float(((p > THR) == (ref > THR)).mean()),
                     spearman_with_A=float(p.corr(ref, method="spearman")))
        rows.append(r)
    out = pd.DataFrame(rows)
    out.to_csv(CAND / "comparison.csv", index=False, float_format="%.4f")
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
