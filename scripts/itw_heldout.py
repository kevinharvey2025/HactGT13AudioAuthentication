"""T1 (docs/ROADMAP.md): In-the-Wild, fully held out. The release has 31,779 clips; 4,000 of them (cache/manifest_pool)
were used to choose checkpoints and fit calibration. The other 27,779 were never used for any choice, so they give
an unbiased out-of-domain estimate.

    python scripts/itw_heldout.py --make [--parts 3]     # templates + labels -> runs/diffusion/itw_heldout/
    bash mpcdf/predict_cpu.sh <ITW dir> runs/diffusion/itw_heldout/template_part<i>.tsv runs/diffusion/itw_heldout/part<i> itw 6
    python scripts/itw_heldout.py --evaluate              # merge the parts -> metrics with bootstrap intervals

--evaluate scores the shipped ensemble (the TSVs) and each member detector (per-model logits in traces.jsonl):
AUC, EER, minDCF and actDCF at P > 0.2 with 95% stratified bootstrap intervals, plus a per-speaker and per-duration
breakdown. Writes runs/diffusion/itw_heldout/{metrics.json, per_speaker.csv, per_duration.csv}.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hearsay import config, metrics  # noqa: E402

ITW = config.EXTERNAL / "in_the_wild" / "release_in_the_wild"
OUT = config.RUNS / "itw_heldout"


def make(parts):
    OUT.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(ITW / "meta.csv")
    used = pd.read_parquet(config.CACHE / "manifest_pool.parquet").query("family == 'itw'").uid.str.replace("itw/", "", regex=False)
    held = meta[~meta.file.str.rsplit(".", n=1).str[0].isin(set(used))].sort_values("file").reset_index(drop=True)
    def write(df, name):  # atomic: jobs waiting for these files never see a partial one
        tmp = OUT / f".{name}.tmp"
        df.to_csv(tmp, sep="\t", index=False)
        tmp.replace(OUT / name)
    write(held.assign(label=(held.label == "spoof").astype(int)), "labels.tsv")
    for i in range(parts):
        write(held.iloc[i::parts][["file"]].rename(columns={"file": "filename"}).assign(**{"cm-score": 0.5}), f"template_part{i}.tsv")
    print(f"{len(meta)} clips in the release, {len(used)} used for selection/calibration, {len(held)} held out "
          f"({held.label.value_counts().to_dict()}), {held.speaker.nunique()} speakers -> {parts} parts")


def boot(y, s, prob=None, reps=1000, seed=0):
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    b = []
    for _ in range(reps):
        i = np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]
        dcf, e = metrics.min_dcf_eer(y[i], s[i])
        b.append((dcf, e, metrics.act_dcf(y[i], prob[i]) if prob is not None else np.nan))
    b = np.array(b)
    m = metrics.summary(y, s)
    r = dict(n=int(len(y)), n_fake=int(y.sum()), auc=m["auc"], eer=m["eer"], min_dcf=m["min_dcf"],
             min_dcf_ci=np.percentile(b[:, 0], [2.5, 97.5]).tolist(), eer_ci=np.percentile(b[:, 1], [2.5, 97.5]).tolist())
    if prob is not None:
        r.update(act_dcf=metrics.act_dcf(y, prob), act_dcf_ci=np.percentile(b[:, 2], [2.5, 97.5]).tolist(),
                 flagged=float((prob > metrics.bayes_threshold(calib_prior=metrics.DCF["p_spoof"])).mean()))
    return r


def evaluate():
    lab = pd.read_csv(OUT / "labels.tsv", sep="\t").set_index("file")
    tsv = pd.concat([pd.read_csv(p, sep="\t") for p in sorted(OUT.glob("part*/SideQuests_predictions_itw.tsv"))]).set_index("filename")
    tr = pd.concat([pd.read_json(p, lines=True) for p in sorted(OUT.glob("part*/traces.jsonl"))]).set_index("filename")
    d = lab.join(tsv, how="inner").join(pd.DataFrame(tr.detectors.tolist(), index=tr.index), how="left")
    d["duration"] = [t.get("decoded_duration", np.nan) if isinstance(t, dict) else np.nan for t in tr.triage.reindex(d.index)]
    y, p = d.label.to_numpy(), d["cm-score"].to_numpy()
    res = {"ensemble (submitted recipe)": boot(y, p, prob=p)}
    for col in [c for c in d.columns if c.endswith("_d6rall")]:
        res[col] = boot(y, d[col].to_numpy())
    res["n_missing"] = int(len(lab) - len(d))
    json.dump(res, open(OUT / "metrics.json", "w"), indent=1)
    rows = []
    for spk, g in d.groupby("speaker"):
        yy, pp = g.label.to_numpy(), g["cm-score"].to_numpy()
        r = dict(speaker=spk, n=len(g), n_fake=int(yy.sum()), fa_rate=float((pp[yy == 0] > 0.2).mean()) if (yy == 0).any() else np.nan,
                 miss_rate=float((pp[yy == 1] <= 0.2).mean()) if (yy == 1).any() else np.nan)
        if 0 < yy.sum() < len(yy):
            r.update(eer=metrics.eer(yy, pp), min_dcf=metrics.min_dcf(yy, pp))
        rows.append(r)
    pd.DataFrame(rows).sort_values("n", ascending=False).to_csv(OUT / "per_speaker.csv", index=False, float_format="%.4f")
    d["dur_bin"] = pd.cut(d.duration, [0, 2, 3, 4, 6, 1e9], labels=["<2 s", "2-3 s", "3-4 s", "4-6 s", ">6 s"])
    rows = []
    for b, g in d.groupby("dur_bin", observed=True):
        yy, pp = g.label.to_numpy(), g["cm-score"].to_numpy()
        if 0 < yy.sum() < len(yy):
            rows.append(dict(duration=b, n=len(g), min_dcf=metrics.min_dcf(yy, pp), eer=metrics.eer(yy, pp)))
    pd.DataFrame(rows).to_csv(OUT / "per_duration.csv", index=False, float_format="%.4f")
    from hearsay import provenance
    provenance.write(OUT, inputs=[OUT / "labels.tsv", *OUT.glob("part*/SideQuests_predictions_itw.tsv")])
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--make", action="store_true")
    ap.add_argument("--evaluate", action="store_true")
    ap.add_argument("--parts", type=int, default=3)
    a = ap.parse_args()
    if a.make:
        make(a.parts)
    if a.evaluate:
        evaluate()
