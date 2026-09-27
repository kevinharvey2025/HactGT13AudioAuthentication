"""Final evaluation: every system under one protocol, from the per-clip scores already on disk (nothing is re-scored).
Replaces the earlier compare_systems.py and fusion_gate.py.

    python scripts/evaluate.py [--boot 1000] [--emb xlsr2b_d6rall_last] [--out results] [--workers 64]

Protocol. The organizers' metric (hearsay/metrics.py: minDCF with Pspoof 0.3, Cmiss 1, Cfa 4, and EER) on the shared
held-out sets (hearsay/splits.py): val (model selection, calibration), holdout (in domain; never used for any choice)
and In-the-Wild (ITW; out of domain). Each clip is scored on its clean canonical view and on one fixed random channel
chain ("aug" = views.ViewMaker view 1, the same audio for every system). Every system is compared with the final
ensemble on exactly the same clips: 95% intervals from a stratified bootstrap (reals and fakes resampled separately),
differences from a paired one (the same resamples for both systems).

Systems: AntiDeepfake detectors zero-shot (epoch 0), fine-tuned, fine-tuned with copy-synthesis fakes (D6-R, three or
four vocoders), WiSE-FT; the interim and final ensembles (mean of z-normalized logits); concept formation
(cobweb-private P(fake), TTCG prototype P(fake), and their fusion with the final score); the DSP detector; metadata.

Writes to --out: benchmark.csv, breakdown.csv (generator / real source / channel / duration), calibration.csv
(cross-fitted Platt: actual vs minimum DCF, Cllr, prior-weighted ECE), curves.csv (fine-tuning epochs),
fusion.csv (does DSP / metadata / concepts add to the detector?), test_agreement.csv (label-free, NSA test set) and
tables.md (the tables quoted in docs/). The augmented views' channel chains and crop lengths are replayed once
(cache/eval_views.parquet).
"""
import argparse
import json
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hearsay import audio, config, metrics, splits, submission, views  # noqa: E402

FT = config.RUNS / "ft"
SETS, VIEWS = ("val", "holdout", "itw"), ("clean", "aug")
P = metrics.DCF["p_spoof"]
PI_EFF = metrics.effective_prior()                 # 0.632: the prior the organizers' costs are equivalent to
THR = metrics.bayes_threshold(calib_prior=P)       # 0.2 on P(synthetic) calibrated at the 30% prior
fast = metrics.min_dcf_eer                         # (minDCF, EER) from one DET curve

# name -> run@epoch under runs/diffusion/ft/ (epoch 0 = the pretrained model before any update); epochs are the
# ones each run selected on val + ITW (best.json)
NEURAL = {
    "zero-shot/XLS-R-2B": "xlsr2b_ft@0", "zero-shot/XLS-R-1B": "xlsr1b_ft@0", "zero-shot/MMS-1B": "mms1b_ft@0",
    "zero-shot/MMS-300M": "mms300m_ft@0", "zero-shot/W2V-Large": "w2vlarge_ft@0", "zero-shot/HuBERT-XL": "hubertxl_zs@0",
    "fine-tuned/XLS-R-2B": "xlsr2b_ft@2", "fine-tuned/XLS-R-1B": "xlsr1b_ft@1", "fine-tuned/MMS-1B": "mms1b_ft@1",
    "fine-tuned/MMS-300M": "mms300m_ft@3", "fine-tuned/W2V-Large": "w2vlarge_ft@4",
    "D6-R 3 vocoders/XLS-R-2B": "xlsr2b_d6r@3", "D6-R 3 vocoders/XLS-R-1B": "xlsr1b_d6r@3",
    "D6-R/XLS-R-2B": "xlsr2b_d6rall@3", "D6-R/XLS-R-1B": "xlsr1b_d6rall@2", "D6-R/XLS-R-1B seed 1": "xlsr1b_d6rall_s1@3",
    "D6-R/MMS-1B": "mms1b_d6rall@3",
    "WiSE-FT/XLS-R-2B a=0.3": "xlsr2b_wise030@0", "WiSE-FT/XLS-R-2B a=0.5": "xlsr2b_wise050@0",
    "WiSE-FT/XLS-R-2B a=0.7": "xlsr2b_wise070@0", "WiSE-FT/XLS-R-1B a=0.3": "xlsr1b_wise030@0",
    "WiSE-FT/XLS-R-1B a=0.5": "xlsr1b_wise050@0",
}
ENSEMBLES = {"ensemble/v1 interim": ["D6-R 3 vocoders/XLS-R-2B", "D6-R 3 vocoders/XLS-R-1B", "WiSE-FT/XLS-R-2B a=0.3"],
             "ensemble/v2 FINAL": ["D6-R/XLS-R-2B", "D6-R/XLS-R-1B", "D6-R/MMS-1B"]}
FINAL = "ensemble/v2 FINAL"
KEY = ["zero-shot/XLS-R-2B", "fine-tuned/XLS-R-2B", "D6-R/XLS-R-2B", FINAL, "concepts/cobweb P(fake)",
       "concepts/TTCG P(fake)", "concepts/TTCG P(fake), closed-form basic naming", "DSP/D0"]
METADATA = {"metadata/M0 technical fields": "M0:hgb", "metadata/M1 tags": "M1:hgb", "metadata/X0 cross-checks": "X0-inputs:hgb"}
DUR_BINS, DUR_LABELS = [0, 2, 3, 4, 6, 1e9], ["<2 s", "2-3 s", "3-4 s", "4-6 s", ">6 s"]
OPS = ["codec", "telephony", "bandlimit", "resample", "tilt", "noise", "hum", "reverb", "clipping"]


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def fit_lr(X, y):
    """Logistic fusion with class weights for the organizers' effective prior."""
    w = np.where(y == 1, PI_EFF / y.mean(), (1 - PI_EFF) / (1 - y.mean()))
    return LogisticRegression(C=1.0, max_iter=2000).fit(X, y, sample_weight=w)


# ----------------------------------------------------------------------------------------------- loading

def load_neural():
    sc, test, info = {}, {}, {}
    for name, spec in NEURAL.items():
        run, ep = spec.split("@")
        files = {s: FT / run / f"{s}_epoch{ep}.parquet" for s in SETS}
        if not all(f.exists() for f in files.values()):
            print("missing, skipped:", name, spec, flush=True)
            continue
        ev = {s: pd.read_parquet(f).set_index("uid") for s, f in files.items()}
        sc[name] = {(s, v): ev[s][f"score_{v}"].astype(float) for s in SETS for v in VIEWS}
        test[name] = pd.read_parquet(FT / run / f"test_epoch{ep}.parquet").set_index("filename").score.astype(float)
        for s in SETS:
            info.setdefault(s, ev[s][["label", "generator", "family"]])
    common = {s: sorted(set.intersection(*[set(sc[k][(s, "clean")].index) for k in sc])) for s in SETS}
    info = {s: info[s].loc[common[s]] for s in SETS}
    sc = {k: {sv: x.loc[common[sv[0]]] for sv, x in d.items()} for k, d in sc.items()}
    return sc, test, info


def znorm(sc, test):
    """Per-system z statistics from val + ITW (both views), as in the submission; ensembles average them."""
    z, zt = {}, {}
    for k in sc:
        ref = pd.concat([sc[k][(s, v)] for s in ("val", "itw") for v in VIEWS])
        mu, sd = ref.mean(), ref.std(ddof=0) + 1e-6
        z[k] = {sv: (x - mu) / sd for sv, x in sc[k].items()}
        zt[k] = (test[k] - mu) / sd
    fus = json.load(open(config.REPO / "submission" / "fusion.json"))
    for f in fus["systems"]:
        k = next((n for n, spec in NEURAL.items() if spec == f"{f['name']}@{f['epoch']}"), None)
        if k in sc:
            ref = pd.concat([sc[k][(s, v)] for s in ("val", "itw") for v in VIEWS])
            print(f"z stats {k}: here ({ref.mean():.4f}, {ref.std(ddof=0):.4f}) vs fusion.json ({f['z_mean']:.4f}, {f['z_std']:.4f})")
    for name, members in ENSEMBLES.items():
        if all(m in z for m in members):
            sc[name] = {sv: sum(z[m][sv] for m in members) / len(members) for sv in z[members[0]]}
            test[name] = sum(zt[m] for m in members) / len(members)
            z[name], zt[name] = sc[name], test[name]
    return z, zt


def load_concepts(emb, sc, test, info):
    root = config.RUNS / "concepts" / emb
    need = [root / f"scores_{s}.parquet" for s in SETS] + [root / "test_explanations.jsonl"]
    if not all(f.exists() for f in need):
        print("concept run incomplete, skipped:", root, flush=True)
        return
    cols = (("p_fake_cobweb", "concepts/cobweb P(fake)"), ("p_fake_ttcg", "concepts/TTCG P(fake)"),
            ("p_fake_ttcg_basic", "concepts/TTCG P(fake), closed-form basic naming"))
    for col, name in cols:
        if col not in pd.read_parquet(root / "scores_val.parquet").columns:
            continue
        sc[name] = {}
        for s in SETS:
            d = pd.read_parquet(root / f"scores_{s}.parquet")
            d = d[d.uid.isin(info[s].index) & np.isfinite(d[col])]
            for v, m in (("clean", d.view == 0), ("aug", d.view > 0)):
                sc[name][(s, v)] = pd.Series(logit(d.loc[m, col]), index=d.loc[m, "uid"].to_numpy())
    ex = pd.read_json(root / "test_explanations.jsonl", lines=True).set_index("filename")
    test["concepts/cobweb P(fake)"] = pd.Series(logit(ex.cobweb_p_fake), index=ex.index)
    test["concepts/TTCG P(fake)"] = pd.Series(logit(ex.ttcg_p_fake.astype(float)), index=ex.index).dropna()
    # fusions, fitted on val (both views), applied everywhere (val rows are in-sample)
    for name, cols in (("concepts/cobweb + TTCG (LR)", ["concepts/cobweb P(fake)", "concepts/TTCG P(fake)"]),
                       ("concepts/FINAL + cobweb + TTCG (LR)", [FINAL, "concepts/cobweb P(fake)", "concepts/TTCG P(fake)"])):
        tabs = {sv: pd.concat([sc[c][sv].rename(c) for c in cols], axis=1, join="inner") for sv in sc[cols[-1]]}
        tr = pd.concat([tabs[("val", v)] for v in VIEWS])
        lr = fit_lr(tr.to_numpy(), info["val"].label.loc[tr.index].to_numpy())
        sc[name] = {sv: pd.Series(lr.decision_function(t.to_numpy()), index=t.index) for sv, t in tabs.items() if sv[0] != "val"}
        print(name, "weights", np.round(lr.coef_[0], 3).tolist(), flush=True)


def data_key(p):
    """An audio path relative to data/, whichever machine or snapshot wrote it."""
    p = str(p).replace("\\", "/")
    return p.split("/data/", 1)[1] if "/data/" in p else (p[5:] if p.startswith("data/") else p)


def path_keys():
    man = pd.read_parquet(config.CACHE / "manifest_pool.parquet")[["uid", "path", "family"]]
    return man.assign(key=man.path.map(data_key))


def load_dsp_meta(sc, test, info):
    man = path_keys()
    k2u = dict(zip(man.key, man.uid))
    itw_file = dict(zip(man.loc[man.family == "itw", "path"].map(lambda p: Path(p).name), man.loc[man.family == "itw", "uid"]))
    runs = config.REPO / "runs"
    d = {}
    b = config.REPO / "artifacts" / "dsp" / "model"
    if (b / "training_oof_scores.tsv").exists():
        oof = pd.read_csv(b / "training_oof_scores.tsv", sep="\t").merge(pd.read_csv(b / "training_rows.tsv", sep="\t"), on="sha256")
        oof["uid"] = oof.path.map(lambda p: k2u.get(data_key(p)))
        d["val"] = pd.Series(logit(oof.oof_p_full.to_numpy()), index=oof.uid).dropna()
    f = runs / "dsp" / "eval_holdout" / "scores.tsv"
    if f.exists():
        h = pd.read_csv(f, sep="\t")
        h["uid"] = h.path.map(lambda p: k2u.get(data_key(p)))
        d["holdout"] = pd.Series(logit(h.p_full.to_numpy()), index=h.uid).dropna()
    f = runs / "dsp" / "itw" / "itw_predictions.tsv"
    if f.exists():
        t = pd.read_csv(f, sep="\t")
        d["itw"] = pd.Series(logit(t["cm-score"].to_numpy()), index=t.filename.map(itw_file)).dropna()
    if d:
        sc["DSP/D0"] = {(s, "clean"): x[~x.index.duplicated()].loc[lambda x: x.index.isin(info[s].index)] for s, x in d.items()}
        f = runs / "dsp" / "submission" / "dsp_predictions.tsv"
        if f.exists():
            t = pd.read_csv(f, sep="\t")
            test["DSP/D0"] = pd.Series(logit(t["cm-score"].to_numpy()), index=t.filename)
    meta = runs / "meta"
    if (meta / "holdout.parquet").exists():
        parts = {"val": pd.read_parquet(meta / "oof.parquet"), "holdout": pd.read_parquet(meta / "holdout.parquet")}
        mt = pd.read_parquet(meta / "test.parquet").set_index("filename")
        for name, col in METADATA.items():
            sc[name] = {(s, "clean"): pd.Series(logit(x[col].to_numpy()), index=x.uid).loc[lambda q: q.index.isin(info[s].index)]
                        for s, x in parts.items()}
            test[name] = pd.Series(logit(mt[col].to_numpy()), index=mt.index)


# ----------------------------------------------------------------------------------------------- channel replay

def babble_uids():
    """The babble pool of scripts/finetune_ssl.py (150 train-split reals), so replayed views equal the scored ones."""
    man = pd.read_parquet(config.CACHE / "manifest_pool.parquet").merge(
        pd.read_parquet(config.CACHE / "triage_pool.parquet")[["uid", "decode_ok", "decoded_duration"]], on="uid", how="left")
    man = man[man.decode_ok.fillna(False)].reset_index(drop=True)
    lab = man[man.label >= 0].reset_index(drop=True)
    lab["split"] = splits.shared_split(lab)
    tr = lab[(lab.split == "train").to_numpy() & (lab.decoded_duration >= 3.3).to_numpy()].reset_index(drop=True)
    return tr[tr.label == 0].sample(min(150, int((tr.label == 0).sum())), random_state=0).uid.tolist()


def _replay(job):
    uids, babble = job
    maker = views.ViewMaker([audio.load_cached(u) for u in babble])
    rows = []
    for u in uids:
        for v in (0, 1):
            x, prm = maker(u, v)
            rows.append((u, v, prm["channel"], len(x) / config.SR))
    return rows


def replay(uids, workers):
    path = config.CACHE / "eval_views.parquet"
    if path.exists():
        d = pd.read_parquet(path)
        if set(uids) <= set(d.uid):
            return d
    bab = babble_uids()
    jobs = [(uids[i::workers * 4], bab) for i in range(workers * 4)]
    with ProcessPoolExecutor(workers) as ex:
        rows = [r for part in ex.map(_replay, jobs) for r in part]
    d = pd.DataFrame(rows, columns=["uid", "view", "channel", "dur"])
    d.to_parquet(path)
    return d


# ----------------------------------------------------------------------------------------------- evaluation

def _row(job):
    name, s, v, y, x, r, B, seed = job
    m = metrics.summary(y, x)
    dcf, e = m["min_dcf"], m["eer"]
    out = dict(system=name, set=s, view=v, n=len(y), n_fake=int(y.sum()), auc=m["auc"], eer=e, min_dcf=dcf,
               final_min_dcf=fast(y, r)[0])
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    bs = np.empty((B, 3))
    for b in range(B):
        i = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        (d1, e1), (d2, _) = fast(y[i], x[i]), fast(y[i], r[i])
        bs[b] = d1, e1, d1 - d2
    for j, k in enumerate(("min_dcf", "eer", "delta")):
        out[f"{k}_lo"], out[f"{k}_hi"] = np.percentile(bs[:, j], [2.5, 97.5])
    out["delta"] = dcf - out["final_min_dcf"]
    out["p_not_worse"] = float((bs[:, 2] <= 0).mean())
    return out


def benchmark(sc, info, B, workers):
    jobs = []
    for name, d in sc.items():
        for (s, v), x in d.items():
            ref = sc[FINAL][(s, v)]
            u = x.index.intersection(ref.index)
            if len(u) < 20 or info[s].label.loc[u].nunique() < 2:
                continue
            y = info[s].label.loc[u].to_numpy().astype(int)
            seed = zlib.crc32(f"{s}/{v}/{len(u)}/{int(y.sum())}".encode())
            jobs.append((name, s, v, y, x.loc[u].to_numpy(float), ref.loc[u].to_numpy(float), B, seed))
    with ProcessPoolExecutor(workers) as ex:
        rows = list(ex.map(_row, jobs, chunksize=1))
    return pd.DataFrame(rows)


def calibrate(sc, info, fit_sets, name):
    """Platt at the 30% prior fitted on fit_sets (both views) -> {(set, view): P(synthetic)}."""
    xs = pd.concat([sc[name][(s, v)] for s in fit_sets for v in VIEWS if (s, v) in sc[name]])
    ys = pd.concat([info[s].label.loc[sc[name][(s, v)].index] for s in fit_sets for v in VIEWS if (s, v) in sc[name]])
    pl = submission.Platt(prior=P).fit(xs.to_numpy(), ys.to_numpy())
    return pl, {sv: pd.Series(pl(x.to_numpy()), index=x.index) for sv, x in sc[name].items()}


def calibration_table(sc, info):
    rows = []
    for name in sc:
        if not all((s, "clean") in sc[name] for s in SETS) or name.startswith(("DSP", "metadata", "concepts/FINAL", "concepts/cobweb +")):
            continue
        for fit in (("val", "itw"), ("val",), ("itw",)):
            _, prob = calibrate(sc, info, fit, name)
            for (s, v), p in prob.items():
                if s in fit:
                    continue
                y = info[s].label.loc[p.index].to_numpy().astype(int)
                pv = np.clip(p.to_numpy(), 1e-6, 1 - 1e-6)
                llr = logit(pv) - logit(P)
                cllr = 0.5 * (np.log2(1 + np.exp(-llr[y == 1])).mean() + np.log2(1 + np.exp(llr[y == 0])).mean())
                w = np.where(y == 1, P / y.mean(), (1 - P) / (1 - y.mean()))        # reweight to the 30% prior
                bins = np.minimum((pv * 10).astype(int), 9)
                ece = sum(abs(np.average(y[bins == b], weights=w[bins == b]) - np.average(pv[bins == b], weights=w[bins == b]))
                          * w[bins == b].sum() for b in range(10) if (bins == b).any()) / w.sum()
                rows.append(dict(system=name, fit="+".join(fit), set=s, view=v, n=len(y), act_dcf=metrics.act_dcf(y, pv),
                                 min_dcf=fast(y, p.to_numpy())[0], cllr=cllr, ece_prior_weighted=ece,
                                 flagged=float((pv > THR).mean())))
    return pd.DataFrame(rows)


def breakdown(sc, info, ev):
    rows = []
    ch = ev.set_index(["uid", "view"])
    for name in [k for k in KEY if k in sc]:
        prob = calibrate(sc, info, ("val", "itw"), name)[1] if all((s, "clean") in sc[name] for s in ("val", "itw")) else {}
        for (s, v), x in sc[name].items():
            d = info[s].loc[x.index].assign(score=x.to_numpy())
            if (s, v) in prob:
                d["flag"] = prob[(s, v)].loc[d.index].to_numpy() > THR
            y, sco = d.label.to_numpy().astype(int), d.score.to_numpy()
            key = [(u, int(v == "aug")) for u in d.index]
            d["channel"] = ch.channel.reindex(key).to_numpy()
            d["dur"] = pd.cut(ch.dur.reindex(key).to_numpy(), DUR_BINS, labels=DUR_LABELS)

            flag = d.flag.to_numpy() if "flag" in d else None

            def add(kind, group, m):
                if m.sum() == 0 or len(np.unique(y[m])) < 2:
                    return
                dcf, e = fast(y[m], sco[m])
                r = dict(system=name, set=s, view=v, kind=kind, group=group, n_real=int((y[m] == 0).sum()),
                         n_fake=int((y[m] == 1).sum()), min_dcf=dcf, eer=e)
                if flag is not None:  # decisions at the Bayes threshold of the shipped calibration
                    r["fa_rate"], r["miss_rate"] = float(flag[m & (y == 0)].mean()), float((~flag[m & (y == 1)]).mean())
                rows.append(r)
            for g in sorted(d.generator[d.label == 1].unique()):              # a generator's fakes vs every real
                add("generator", g, (y == 0) | (d.generator == g).to_numpy())
            for f in sorted(d.family[d.label == 0].unique()):                 # a real source vs every fake
                add("real source", f, (y == 1) | ((d.family == f) & (d.label == 0)).to_numpy())
            if name.startswith(("DSP", "metadata")):                           # they score the original files
                continue
            if v == "aug":
                chains = d.channel.fillna("").str.split("+")
                for op in OPS:
                    add("channel", op, chains.map(lambda c: any(x.startswith(op[:5]) for x in c)).to_numpy())
                for sub in sorted({c for chain in chains for c in chain if c.startswith(("codec_", "noise_"))}):
                    add("channel", sub, chains.map(lambda c: sub in c).to_numpy())
            for b in DUR_LABELS:
                add("duration", b, (d.dur == b).to_numpy())
    return pd.DataFrame(rows)


def curves():
    """Every run and epoch, recomputed from the stored per-epoch scores under the official metric (the training
    logs of runs made before the metric correction hold Pspoof-0.5 values)."""
    rows = []
    for run in sorted(p for p in FT.iterdir() if p.is_dir() and p.name != "smoke"):
        for f in sorted(run.glob("val_epoch*.parquet")):
            ep = int(f.stem.split("epoch")[1])
            for s in SETS:
                g = run / f"{s}_epoch{ep}.parquet"
                if not g.exists():
                    continue
                d = pd.read_parquet(g)
                for v in VIEWS:
                    dcf, e = fast(d.label.to_numpy().astype(int), d[f"score_{v}"].to_numpy(float))
                    rows.append(dict(run=run.name, epoch=ep, set=s, view=v, n=len(d), min_dcf=dcf, eer=e))
    return pd.DataFrame(rows)


def fusion_gate(sc, info):
    """Can DSP or metadata improve the detector? Logistic fusion fitted on val (out-of-fold branch scores), evaluated
    on holdout and In-the-Wild (clean view; DSP and metadata score the original files)."""
    branches = {"FINAL": FINAL, "DSP": "DSP/D0", "M0": "metadata/M0 technical fields", "X0": "metadata/X0 cross-checks"}
    combos = [("FINAL",), ("DSP",), ("M0",), ("X0",), ("FINAL", "DSP"), ("FINAL", "M0"), ("FINAL", "X0"), ("FINAL", "DSP", "M0")]
    rows = []
    for combo in combos:
        cols = [branches[c] for c in combo]
        if not all(c in sc for c in cols):
            continue
        tab = {s: pd.concat([sc[c][(s, "clean")].rename(c) for c in cols], axis=1, join="inner")
               for s in SETS if all((s, "clean") in sc[c] for c in cols)}
        tab = {s: t for s, t in tab.items() if len(t)}
        if "val" not in tab:
            continue
        lr = fit_lr(tab["val"].to_numpy(), info["val"].label.loc[tab["val"].index].to_numpy()) if len(cols) > 1 else None
        for s in ("holdout", "itw"):
            if s not in tab:
                continue
            x = lr.decision_function(tab[s].to_numpy()) if lr is not None else tab[s].iloc[:, 0].to_numpy()
            y = info[s].label.loc[tab[s].index].to_numpy().astype(int)
            dcf, e = fast(y, x)
            rows.append(dict(fusion=" + ".join(combo), set=s, n=len(y), auc=roc_auc_score(y, x), eer=e, min_dcf=dcf,
                             weights=None if lr is None else np.round(lr.coef_[0], 3).tolist()))
    return pd.DataFrame(rows)


def test_agreement(sc, info, test):
    man = pd.read_parquet(config.CACHE / "manifest_pool.parquet")[["uid", "filename"]].dropna()
    tri = pd.read_parquet(config.CACHE / "triage_pool.parquet")[["uid", "decoded_duration"]]
    dur = man.merge(tri, on="uid").set_index("filename").decoded_duration
    sub = pd.read_csv(config.REPO / "submission" / "SideQuests_predictions_final.tsv", sep="\t").set_index("filename")["cm-score"]
    test = dict(test, **{"submitted TSV (predict.py)": pd.Series(logit(sub.to_numpy()), index=sub.index)})
    ref = test[FINAL]
    pref = calibrate(sc, info, ("val", "itw"), FINAL)[0](ref.to_numpy())
    q = float((pref > THR).mean())
    top_ref = ref.rank(ascending=False, method="first") <= q * len(ref)
    rows = []
    for name, t in test.items():
        u = t.index.intersection(ref.index)
        top = t.loc[u].rank(ascending=False, method="first") <= q * len(u)
        a, b = top.to_numpy(), top_ref.loc[u].to_numpy()
        po, pe = (a == b).mean(), a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
        r = dict(system=name, n=len(u), spearman_vs_final=spearmanr(t.loc[u], ref.loc[u])[0], kappa_at_final_rate=(po - pe) / (1 - pe),
                 spearman_vs_duration=spearmanr(t.loc[u], dur.reindex(u), nan_policy="omit")[0])
        if name == "submitted TSV (predict.py)":
            r["flagged"] = float((sub.loc[u] > THR).mean())
        elif name in sc and all((s, "clean") in sc[name] for s in ("val", "itw")) and not name.startswith(("DSP", "metadata")):
            r["flagged"] = float((calibrate(sc, info, ("val", "itw"), name)[0](t.to_numpy()) > THR).mean())
        rows.append(r)
    by_dur = pd.DataFrame({"p": pref, "dur": pd.cut(dur.reindex(ref.index).to_numpy(), DUR_BINS, labels=DUR_LABELS)})
    print("FINAL flagged share by test duration:", by_dur.groupby("dur", observed=True).p.apply(lambda p: round(float((p > THR).mean()), 3)).to_dict())
    return pd.DataFrame(rows), q


def docker_path():
    """The shipped code path (predict.py: whole clips, fp32, stored fusion + calibration; the newest predict_* run) on
    all 4,000 labeled In-the-Wild clips, and the router's decisions there and on the NSA test set."""
    out = {}
    newest = lambda stem: max(config.RUNS.glob(f"{stem}_v*"), key=lambda p: int(p.name.rsplit("_v", 1)[1]), default=config.RUNS / stem)
    itw_dir, test_dir = newest("predict_itw"), newest("predict_final")
    out["runs"] = dict(itw=itw_dir.name, test=test_dir.name)
    f, lab = itw_dir / "SideQuests_predictions_itw.tsv", config.RUNS / "itw_labels.tsv"
    if f.exists() and lab.exists():
        t = pd.read_csv(f, sep="\t").merge(pd.read_csv(lab, sep="\t"), on="filename")
        y, p = t.label.to_numpy().astype(int), t["cm-score"].to_numpy(float)
        m = metrics.summary(y, p)
        out["itw_all_clips"] = dict(n=len(t), auc=m["auc"], eer=m["eer"], min_dcf=m["min_dcf"],
                                    act_dcf_at_0_2=metrics.act_dcf(y, p), flagged=float((p > THR).mean()))
    for name, d in (("itw", itw_dir), ("test", test_dir)):
        tr = d / "traces.jsonl"
        if tr.exists():
            routes = pd.Series(["+".join(sorted(json.loads(line).get("router", {}))) for line in open(tr)])
            out[f"router_{name}"] = routes.value_counts(normalize=True).round(4).to_dict()
    return out


# ----------------------------------------------------------------------------------------------- tables

def fmt(x, lo=None, hi=None, nd=3):
    if x is None or not np.isfinite(x):
        return "-"
    return f"{x:.{nd}f}" + (f" [{lo:.{nd}f}, {hi:.{nd}f}]" if lo is not None else "")


def tables(bm, bd, cal, fu, ta, q, cu, dp):
    L = ["# Evaluation tables (generated by scripts/evaluate.py; do not edit by hand)", "",
         f"minDCF: organizers' metric (Pspoof {P}, Cmiss 1, Cfa 4); lower is better, 1.0 = a constant decision. "
         "[95% stratified bootstrap interval]. 'aug' = the same clips through one random channel chain.", "",
         "## 1. Every system (minDCF; EER on In-the-Wild clean)", "",
         "| system | val clean | val aug | holdout clean | holdout aug | ITW clean | ITW aug | ITW EER |", "|---|---|---|---|---|---|---|---|"]
    for name, g in bm.groupby("system", sort=False):
        c = {(r.set, r.view): r for r in g.itertuples()}
        cell = lambda s, v: (fmt(c[(s, v)].min_dcf) + ("" if c[(s, v)].n >= 2000 else f" (n={c[(s, v)].n})")) if (s, v) in c else "-"
        e = c.get(("itw", "clean"))
        L.append(f"| {name} | " + " | ".join(cell(s, v) for s in SETS for v in VIEWS) + f" | {fmt(100 * e.eer, nd=2) + '%' if e else '-'} |")
    L += ["", "## 2. Against the final ensemble on the same clips (minDCF [95% CI]; delta = system - final, paired)", "",
          "| system | set | view | n | minDCF | final, same clips | delta [95% CI] |", "|---|---|---|---|---|---|---|"]
    for r in bm[bm.system.isin(KEY + ["ensemble/v1 interim", "concepts/FINAL + cobweb + TTCG (LR)"]) & bm.set.isin(["holdout", "itw"])].itertuples():
        L.append(f"| {r.system} | {r.set} | {r.view} | {r.n} | {fmt(r.min_dcf, r.min_dcf_lo, r.min_dcf_hi)} | {fmt(r.final_min_dcf)} | "
                 + ("-" if r.system == FINAL else fmt(r.delta, r.delta_lo, r.delta_hi)) + " |")
    f = bd[(bd.system == FINAL) & (bd.set != "val")]                       # val chose the checkpoints: not reported here
    for kind, title in (("generator", "per fake generator (its fakes vs every real of the set)"),
                        ("real source", "per real source (its reals vs every fake)"),
                        ("channel", "per channel condition (augmented view)"), ("duration", "per crop duration")):
        g = f[f.kind == kind]
        if not len(g):
            continue
        L += ["", f"## Final ensemble {title}", "",
              "| set | view | group | reals | fakes | minDCF | EER | false alarms at P>0.2 | misses at P>0.2 |", "|---|---|---|---|---|---|---|---|---|"]
        L += [f"| {r.set} | {r.view} | {r.group} | {r.n_real} | {r.n_fake} | {fmt(r.min_dcf)} | {fmt(100 * r.eer, nd=1)}% | "
              f"{fmt(getattr(r, 'fa_rate', np.nan))} | {fmt(getattr(r, 'miss_rate', np.nan))} |" for r in g.itertuples()]
    L += ["", "## Calibration (Platt at the 30% prior; evaluated only on sets it was not fitted on)", "",
          "| system | fitted on | set | view | actDCF at P>0.2 | minDCF | Cllr | ECE (prior-weighted) |", "|---|---|---|---|---|---|---|---|"]
    for r in cal[cal.system.isin([FINAL, "zero-shot/XLS-R-2B", "D6-R/XLS-R-2B", "concepts/cobweb P(fake)", "concepts/TTCG P(fake)"])].itertuples():
        L.append(f"| {r.system} | {r.fit} | {r.set} | {r.view} | {fmt(r.act_dcf)} | {fmt(r.min_dcf)} | {fmt(r.cllr)} | {fmt(r.ece_prior_weighted)} |")
    L += ["", "## Fusion gate (logistic fusion fitted on val, clean view)", "", "| fusion | set | n | AUC | EER | minDCF |", "|---|---|---|---|---|---|"]
    L += [f"| {r.fusion} | {r.set} | {r.n} | {fmt(r.auc, nd=4)} | {fmt(100 * r.eer, nd=2)}% | {fmt(r.min_dcf)} |" for r in fu.itertuples()]
    L += ["", f"## NSA test set, label-free (final flags {q:.1%} at P>0.2; kappa at that flag rate)", "",
          "| system | n | Spearman vs final | kappa | flagged at P>0.2 | Spearman vs duration |", "|---|---|---|---|---|---|"]
    L += [f"| {r.system} | {r.n} | {fmt(r.spearman_vs_final)} | {fmt(r.kappa_at_final_rate)} | {fmt(getattr(r, 'flagged', np.nan))} | "
          f"{fmt(r.spearman_vs_duration)} |" for r in ta.itertuples()]
    L += ["", "## Fine-tuning curves: In-the-Wild minDCF by epoch (clean / aug; epoch 0 = zero-shot)", ""]
    cur = cu[cu.set == "itw"].pivot_table(index="run", columns=["epoch", "view"], values="min_dcf")
    eps = sorted({e for e, _ in cur.columns})
    L += ["| run | " + " | ".join(f"epoch {e}" for e in eps) + " |", "|---|" + "---|" * len(eps)]
    for run, r in cur.iterrows():
        L.append(f"| {run} | " + " | ".join(f"{fmt(r.get((e, 'clean')))} / {fmt(r.get((e, 'aug')))}" if (e, "clean") in r and np.isfinite(r[(e, "clean")])
                                            else "-" for e in eps) + " |")
    if dp:
        L += ["", "## The shipped code path (predict.py)", ""]
        if "itw_all_clips" in dp:
            d = dp["itw_all_clips"]
            L.append(f"All {d['n']} labeled In-the-Wild clips: AUC {d['auc']:.4f}, EER {100 * d['eer']:.2f}%, minDCF {d['min_dcf']:.3f}, "
                     f"actDCF at P>0.2 {d['act_dcf_at_0_2']:.3f}, flagged {d['flagged']:.1%}.")
        for k in ("router_itw", "router_test"):
            if k in dp:
                L.append(f"Router decisions ({k[7:]}): " + ", ".join(f"{r or 'none'} {v:.1%}" for r, v in dp[k].items()) + ".")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--emb", default="xlsr2b_d6rall_last", help="concept-formation run (runs/diffusion/concepts/<emb>)")
    ap.add_argument("--out", default="results")
    ap.add_argument("--workers", type=int, default=64)
    a = ap.parse_args()
    out = config.REPO / a.out
    out.mkdir(parents=True, exist_ok=True)

    sc, test, info = load_neural()
    print({s: (len(d), int(d.label.sum())) for s, d in info.items()}, flush=True)
    sc, test = znorm(sc, test)
    load_concepts(a.emb, sc, test, info)
    load_dsp_meta(sc, test, info)
    ev = replay(sorted(set().union(*[set(d.index) for d in info.values()])), a.workers)
    print("replayed views:", ev.channel.str.split("+").str[0].str.split("_").str[0].value_counts().to_dict(), flush=True)

    bm = benchmark(sc, info, a.boot, a.workers)
    bm.to_csv(out / "benchmark.csv", index=False, float_format="%.5f")
    bd = breakdown(sc, info, ev)
    bd.to_csv(out / "breakdown.csv", index=False, float_format="%.5f")
    cal = calibration_table(sc, info)
    cal.to_csv(out / "calibration.csv", index=False, float_format="%.5f")
    curves().to_csv(out / "curves.csv", index=False, float_format="%.5f")
    fu = fusion_gate(sc, info)
    fu.to_csv(out / "fusion.csv", index=False, float_format="%.5f")
    ta, q = test_agreement(sc, info, test)
    ta.to_csv(out / "test_agreement.csv", index=False, float_format="%.5f")
    dp = docker_path()
    json.dump(dp, open(out / "docker_path.json", "w"), indent=1)
    (out / "tables.md").write_text(tables(bm, bd, cal, fu, ta, q, pd.read_csv(out / "curves.csv"), dp))
    from hearsay import provenance
    provenance.write(out, inputs=[*FT.glob("*/*_epoch*.parquet"), *(config.RUNS / "concepts" / a.emb).glob("scores_*.parquet"),
                                  config.REPO / "submission" / "fusion.json", config.REPO / "submission" / "SideQuests_predictions_final.tsv"],
                     bootstrap_reps=a.boot)
    pd.set_option("display.width", 250)
    print(bm[bm.set.isin(["holdout", "itw"])].pivot_table(index="system", columns=["set", "view"], values="min_dcf", sort=False).round(3))
    print(fu.round(4).to_string(index=False))
    print(ta.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
