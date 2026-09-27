"""Fuse selected fine-tuning runs and write the challenge TSV.

    python scripts/make_submission.py --runs xlsr1b_ft xlsr2b_ft mms1b_ft [--epoch best] \
        [--calib val,itw] [--label final] [--team teamName]

For each run (runs/diffusion/ft/<name>) the chosen epoch's scores are read (best = best.json's epoch). Each
system's logits are z-normalized with the mean/std of its own scores on the calibration sets (val and ITW;
never the test set), the fused score is the mean of the z-scores, and a class-balanced Platt fit on the
calibration sets maps it to P(synthetic). Writes runs/diffusion/submission/<team>_predictions[_<label>].tsv
(validated against the template), a per-file score table and a summary with metrics per system and fused.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hearsay import config, metrics, submission  # noqa: E402

FT = config.RUNS / "ft"


def load_run(name, epoch, sets):
    name, _, at = name.partition("@")           # run@epoch overrides --epoch for that run
    epoch = at or epoch
    d = FT / name
    ep = json.load(open(d / "best.json"))["epoch"] if epoch == "best" else int(epoch)
    ev = {s: pd.read_parquet(d / f"{s}_epoch{ep}.parquet") for s in sets if (d / f"{s}_epoch{ep}.parquet").exists()}
    test = pd.read_parquet(d / f"test_epoch{ep}.parquet")
    return ep, ev, test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--epoch", default="best")
    ap.add_argument("--calib", default="val,itw", help="eval sets used for z-norm statistics and calibration")
    ap.add_argument("--report", default="val,holdout,itw", help="eval sets to report metrics on")
    ap.add_argument("--view", default="both", choices=["clean", "aug", "both"], help="eval views used")
    ap.add_argument("--prior", type=float, default=metrics.DCF["p_spoof"], help="calibration prior (organizers' Pspoof)")
    ap.add_argument("--team", default=config.TEAM)
    ap.add_argument("--label", default="")
    ap.add_argument("--out", default=str(config.RUNS / "submission"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    calib, report = a.calib.split(","), a.report.split(",")
    views = ["score_clean", "score_aug"] if a.view == "both" else [f"score_{a.view}"]

    systems, summary = {}, {"runs": {}, "fused": {}}
    for name in a.runs:
        ep, ev, test = load_run(name, a.epoch, set(calib) | set(report))
        systems[name] = (ev, test)
        summary["runs"][name] = {"epoch": ep}
    # align rows across systems (every run scores the same subsampled eval clips and the full test set)
    key_sets = {s: None for s in set(calib) | set(report)}
    for s in key_sets:
        frames = [systems[n][0][s].set_index("uid") for n in systems if s in systems[n][0]]
        key_sets[s] = frames[0].index.intersection(frames[-1].index) if frames else None
        for f in frames[1:-1]:
            key_sets[s] = key_sets[s].intersection(f.index)

    def stacked(name, s):
        f = systems[name][0][s].set_index("uid").loc[key_sets[s]]
        return f.label.to_numpy(), np.concatenate([f[v].to_numpy() for v in views]), f

    z, zt, spec = {}, {}, []
    for name in systems:
        ref = np.concatenate([stacked(name, s)[1] for s in calib if key_sets.get(s) is not None])
        mu, sd = float(ref.mean()), float(ref.std() + 1e-6)
        run = name.split("@")[0]
        cfg = json.load(open(FT / run / "config.json"))
        best_ep = json.load(open(FT / run / "best.json"))["epoch"] if (FT / run / "best.json").exists() else None
        if summary["runs"][name]["epoch"] != best_ep:
            print(f"WARNING: {name}: epoch {summary['runs'][name]['epoch']} is not the saved best.pt (epoch {best_ep})")
        spec.append(dict(name=run, backbone=cfg["backbone"], epoch=summary["runs"][name]["epoch"],
                         checkpoint=f"runs/diffusion/ft/{run}/best.pt", z_mean=mu, z_std=sd))
        z[name] = {s: (stacked(name, s)[0], (stacked(name, s)[1] - mu) / sd) for s in key_sets if key_sets[s] is not None}
        zt[name] = (systems[name][1].set_index("filename").score - mu) / sd
        for s in report:
            if s in z[name]:
                y, sc = z[name][s]
                yy = np.tile(y, len(views))
                m = metrics.summary(yy, sc)
                summary["runs"][name][s] = {k: round(float(m[k]), 4) for k in ("auc", "eer", "min_dcf")}
    fused = {s: (np.tile(z[a.runs[0]][s][0], len(views)), np.mean([z[n][s][1] for n in systems], 0)) for s in z[a.runs[0]]}
    fused_test = pd.concat([zt[n] for n in systems], axis=1).mean(1)
    for s in report:
        if s in fused:
            m = metrics.summary(*fused[s])
            summary["fused"][s] = {k: round(float(m[k]), 4) for k in ("auc", "eer", "min_dcf")}
    ycal = np.concatenate([fused[s][0] for s in calib if s in fused])
    scal = np.concatenate([fused[s][1] for s in calib if s in fused])
    platt = submission.Platt(prior=a.prior).fit(scal, ycal)
    prob = pd.Series(np.clip(platt(fused_test.to_numpy()), 1e-6, 1 - 1e-6), index=fused_test.index)
    name = submission.output_name(a.team).replace(".tsv", f"_{a.label}.tsv" if a.label else ".tsv")
    submission.write(prob, out / name)
    pd.DataFrame({"filename": prob.index, "prob": prob.values, "fused_z": fused_test.values,
                  **{f"z_{n}": zt[n].reindex(prob.index).values for n in systems}}).to_csv(out / name.replace(".tsv", "_scores.csv"), index=False)
    summary.update(platt=dict(coef=platt.coef, intercept=platt.intercept, prior=a.prior),
                   dcf=metrics.DCF, test=submission.describe(prob.values, calib_prior=a.prior),
                   calib_sets=calib, views=views, file=str(out / name))
    # everything predict.py needs to reproduce these probabilities offline
    json.dump(dict(systems=spec, platt=summary["platt"], team=a.team, label=a.label,
                   note="fused = mean_k (logit_k - z_mean_k) / z_std_k ; p = sigmoid(coef * fused + intercept)"),
              open(out / "fusion.json", "w"), indent=1)
    json.dump(summary, open(out / name.replace(".tsv", "_summary.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
