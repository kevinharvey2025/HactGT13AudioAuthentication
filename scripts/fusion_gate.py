"""WP5 fusion gate (metadata-analysis prompt F0/F1 and neural + DSP): does a branch add anything?

    python scripts/fusion_gate.py --neural xlsr2b_ft@1 [--neural-view clean]

Branch scores, each out-of-sample on its rows:
  N   neural detector: run@epoch 'val' and 'holdout' parquet scores (fine-tuning never saw them)
  D0  DSP bundle (artifacts/dsp/model): out-of-fold p_full on its dev rows, runs/dsp/eval_holdout on holdout
  M0/M1/M2/X0  metadata (runs/meta): out-of-fold on the fit pool, refit predictions on holdout
Rows are joined on the audio path relative to data/. Fusion models (logistic regression on the branch
logits, class weights for the organizers' effective spoof prior 0.8) are fitted on the fusion-training rows
(neural 'val' clips that also have DSP/metadata out-of-fold scores) and evaluated on the shared holdout.
Writes runs/fusion_gate.json.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hearsay import config, metrics  # noqa: E402


def key(p):
    p = str(p).replace("\\", "/")
    return p.split("/data/", 1)[1] if "/data/" in p else p


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--neural", required=True, help="run@epoch under runs/diffusion/ft")
    ap.add_argument("--neural-view", default="clean", choices=["clean", "aug"])
    ap.add_argument("--meta", default="M0:hgb,M1:hgb,X0-inputs:hgb")
    a = ap.parse_args()
    runs = config.REPO / "runs"
    man = pd.read_parquet(config.CACHE / "manifest_pool.parquet")[["uid", "path"]]
    man["key"] = man.path.map(key)
    name, ep = a.neural.split("@")
    ft = config.RUNS / "ft" / name
    parts = {}
    for split in ("val", "holdout"):
        n = pd.read_parquet(ft / f"{split}_epoch{ep}.parquet").merge(man, on="uid")
        parts[split] = n[["key", "label", "generator", f"score_{a.neural_view}"]].rename(columns={f"score_{a.neural_view}": "N"})

    # DSP: out-of-fold on dev rows (bundle training) and holdout predictions
    b = runs / ".." / "artifacts" / "dsp" / "model"
    b = config.REPO / "artifacts" / "dsp" / "model"
    oof = pd.read_csv(b / "training_oof_scores.tsv", sep="\t").merge(pd.read_csv(b / "training_rows.tsv", sep="\t"), on="sha256")
    oof = oof.assign(key=oof.path.map(key), D0=logit(oof.oof_p_full))[["key", "D0"]]
    hol = pd.read_csv(runs / "dsp" / "eval_holdout" / "scores.tsv", sep="\t")
    pcol = next(c for c in ("p_full", "score_full", "cm_score", "p") if c in hol.columns)
    hol = hol.assign(key=hol.path.map(key), D0=logit(hol[pcol]))[["key", "D0"]]
    dsp = {"val": oof, "holdout": hol}

    # metadata: out-of-fold (fit pool) and holdout
    mcols = [c for c in a.meta.split(",")]
    mo = pd.read_parquet(runs / "meta" / "oof.parquet").merge(man, on="uid")
    mh = pd.read_parquet(runs / "meta" / "holdout.parquet").merge(man, on="uid")
    meta = {"val": mo.assign(**{c: logit(mo[c]) for c in mcols})[["key"] + mcols],
            "holdout": mh.assign(**{c: logit(mh[c]) for c in mcols})[["key"] + mcols]}

    tab = {s: parts[s].merge(dsp[s], on="key").merge(meta[s], on="key") for s in ("val", "holdout")}
    print({s: (len(t), int(t.label.sum())) for s, t in tab.items()}, flush=True)
    systems = {"N": ["N"], "D0": ["D0"], **{m: [m] for m in mcols},
               "F0 (M0+D0)": [mcols[0], "D0"], "N+D0": ["N", "D0"], "N+D0+M0": ["N", "D0", mcols[0]],
               "N+X0": ["N", mcols[-1]]}
    res = {}
    for sname, cols in systems.items():
        tr, te = tab["val"], tab["holdout"]
        if len(cols) == 1:
            s_te = te[cols[0]].to_numpy()
        else:
            w = np.where(tr.label == 1, 0.8 / tr.label.mean(), 0.2 / (1 - tr.label.mean()))
            lr = LogisticRegression(C=1.0, max_iter=1000).fit(tr[cols], tr.label, sample_weight=w)
            s_te = lr.decision_function(te[cols])
        m = metrics.summary(te.label.to_numpy(), s_te)
        res[sname] = {k: round(float(m[k]), 4) for k in ("auc", "eer", "min_dcf")}
        print(f"{sname:16s} holdout AUC {m['auc']:.4f} EER {m['eer']:.4f} minDCF {m['min_dcf']:.4f}", flush=True)
    # out of domain: In-the-Wild (neural itw scores + the frozen DSP bundle's ITW predictions); fusion weights
    # as fitted above on val rows; metadata is left out (constant-format web audio, at chance there)
    itw_res = {}
    itw_path = runs / "dsp" / "itw" / "itw_predictions.tsv"
    if itw_path.exists():
        n = pd.read_parquet(ft / f"itw_epoch{ep}.parquet").merge(man, on="uid")
        n = n.assign(file=n.path.map(lambda p: Path(p).name))[["file", "label", f"score_{a.neural_view}"]].rename(
            columns={f"score_{a.neural_view}": "N"})
        dd = pd.read_csv(itw_path, sep="\t").rename(columns={"filename": "file"})
        dd["D0"] = logit(dd["cm-score"])
        it = n.merge(dd[["file", "D0"]], on="file")
        tr = tab["val"]
        w = np.where(tr.label == 1, 0.8 / tr.label.mean(), 0.2 / (1 - tr.label.mean()))
        lr = LogisticRegression(C=1.0, max_iter=1000).fit(tr[["N", "D0"]], tr.label, sample_weight=w)
        for sname, s in (("N", it.N.to_numpy()), ("D0", it.D0.to_numpy()), ("N+D0", lr.decision_function(it[["N", "D0"]]))):
            m = metrics.summary(it.label.to_numpy(), s)
            itw_res[sname] = {k: round(float(m[k]), 4) for k in ("auc", "eer", "min_dcf")}
            print(f"ITW {sname:12s} AUC {m['auc']:.4f} EER {m['eer']:.4f} minDCF {m['min_dcf']:.4f}  (n={len(it)})", flush=True)
    json.dump(dict(neural=a.neural, view=a.neural_view, n={s: len(t) for s, t in tab.items()}, results=res, itw=itw_res),
              open(runs / "fusion_gate.json", "w"), indent=1)


if __name__ == "__main__":
    main()
