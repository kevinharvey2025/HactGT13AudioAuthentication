"""The RawBoost experiment from stored scores (nothing is re-scored) -> results/rawboost/.

    python scripts/rawboost_report.py [--boot 1000] [--out results/rawboost] [--workers 32]
    python scripts/rawboost_report.py --val-only     # the stage-decision table: val only

Runs (runs/diffusion/ft/): rb_<arm>_<backbone>_s<seed>, trained with scripts/finetune_ssl.py (--rawboost-algo,
--p-rawboost, --select val, --eval-views 0,1,100,101); views_<run> are the shipped members' checkpoints scored on all
four views (--epochs 0 --wise-from). Every arm is compared with R0 of the same backbone and seed on the same clips (or,
for a backbone without an R0, the stored run of that backbone and seed: the same code and data stream):
95% intervals from the stratified bootstrap, differences from the paired one (scripts/evaluate.py's _row, imported, so
the metric code is not forked). R0 is compared with the stored run of its seed (xlsr1b_d6rall, xlsr1b_d6rall_s1),
which is the reproduction check. The stored runs' epochs are re-selected here on val alone.

Operating points: "val" = the epoch with the lowest val minDCF (mean of the clean and aug views; the training script's
rule), "last" = the final epoch. Stage decisions use --val-only, which reads the val parquets and nothing else, and are
written into decisions.md before holdout, ITW or the unseen views of those arms are opened. The gates (gates.csv) are
the ones fixed in decisions.md before stage 1: primary at the val operating point, the last epoch as a check.

Writes benchmark.csv, curves.csv, seed_noise.csv, breakdown.csv, gates.csv, test_checks.csv, ensemble.csv, tables.md,
provenance.json.
"""
import argparse
import json
import re
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from hearsay import config, metrics, provenance, submission  # noqa: E402
import evaluate  # noqa: E402  (the paired bootstrap _row and the view-1 channel replay)

FT = config.RUNS / "ft"
SETS, VIEWS = ("val", "holdout", "itw"), ("clean", "aug", "unseen", "device")
RUN = re.compile(r"rb_(?P<arm>[A-Z]\d)_(?P<backbone>[a-z0-9]+)_s(?P<seed>\d+)$")
STORED = {("xlsr1b", 0): "xlsr1b_d6rall", ("xlsr1b", 1): "xlsr1b_d6rall_s1", ("xlsr2b", 0): "xlsr2b_d6rall",
          ("mms1b", 0): "mms1b_d6rall"}
P = metrics.DCF["p_spoof"]
THR = metrics.bayes_threshold(calib_prior=P)
SLICES = [("itw", "aug", "channel", "reverb"), ("itw", "aug", "channel", "noise_babble"),
          ("holdout", "clean", "generator", "elevenlabs"), ("holdout", "clean", "generator", "playht")]


# ------------------------------------------------------------------------------------------------ loading

def load_run(path, sets=SETS):
    log = {}
    for line in open(path / "log.jsonl"):   # appended across restarts: the last entry per epoch wins
        r = json.loads(line)
        log[r["epoch"]] = r
    epochs = sorted(int(f.stem.split("epoch")[1]) for f in path.glob("val_epoch*.parquet"))
    sc = {(e, s): pd.read_parquet(f).set_index("uid") for e in epochs for s in sets
          if (f := path / f"{s}_epoch{e}.parquet").exists()}
    return dict(name=path.name, cfg=json.load(open(path / "config.json")), log=log, epochs=epochs, sc=sc)


def discover():
    runs = [p for p in sorted(FT.glob("rb_*")) if RUN.match(p.name)] + sorted(FT.glob("views_*")) + \
           [FT / n for n in STORED.values() if (FT / n).exists()]
    return [p for p in runs if (p / "log.jsonl").exists()]


def op_points(run):
    """{"val": the val-best fine-tuned epoch (ties: the earliest), "last": the final epoch}; epoch 0 for an
    evaluation-only run."""
    vals = {}
    for e in run["epochs"]:
        d = run["sc"].get((e, "val"))
        if e > 0 and d is not None:
            y = d.label.to_numpy().astype(int)
            vals[e] = np.mean([metrics.min_dcf(y, d[f"score_{v}"].to_numpy(float)) for v in ("clean", "aug")])
    last = max(run["epochs"])
    return {"val": min(vals, key=lambda e: (vals[e], e)) if vals else last, "last": last}, vals


def baseline(name):
    """R0 of the same backbone and seed, else the stored run of that backbone and seed; for R0, the stored run."""
    m = RUN.match(name)
    if not m:
        return None
    stored = STORED.get((m["backbone"], int(m["seed"])))
    r0 = f"rb_R0_{m['backbone']}_s{m['seed']}"
    return stored if m["arm"] == "R0" or not (FT / r0).exists() else r0


def s_per_step(run):
    t = [r["s_per_step"] for e, r in run["log"].items() if e > 0 and "s_per_step" in r]
    return float(np.mean(t)) if t else np.nan


# ------------------------------------------------------------------------------------------------ tables

def calibrated(run, e):
    """Platt at the 30% prior fitted on this run's val scores at epoch e (clean + aug views)."""
    d = run["sc"][(e, "val")]
    x = np.r_[d.score_clean.to_numpy(float), d.score_aug.to_numpy(float)]
    y = np.r_[d.label.to_numpy(int), d.label.to_numpy(int)]
    return submission.Platt(prior=P).fit(x, y)


def jobs_for(runs, ops, B):
    jobs, meta = [], []
    for name, run in runs.items():
        ref = runs.get(baseline(name))
        for op, e in ops[name].items():
            pl = calibrated(run, e)
            for s in SETS:
                d = run["sc"].get((e, s))
                r = ref["sc"].get((ops[ref["name"]][op], s)) if ref else None
                if d is None:
                    continue
                for v in VIEWS:
                    if f"score_{v}" not in d:
                        continue
                    u = d.index if r is None or f"score_{v}" not in r else d.index.intersection(r.index)
                    y = d.label.loc[u].to_numpy().astype(int)
                    x = d[f"score_{v}"].loc[u].to_numpy(float)
                    xr = r[f"score_{v}"].loc[u].to_numpy(float) if r is not None and f"score_{v}" in r else x
                    seed = zlib.crc32(f"{s}/{v}/{len(u)}/{int(y.sum())}".encode())
                    jobs.append((name, s, v, y, x, xr, B, seed))
                    prob = pl(x)
                    meta.append(dict(op=op, epoch=e, reference=(ref["name"] if ref is not None and xr is not x else ""),
                                     ref_epoch=(ops[ref["name"]][op] if ref is not None and xr is not x else np.nan),
                                     act_dcf=metrics.act_dcf(y, prob) if s != "val" else np.nan,
                                     cllr=metrics.cllr(y, prob) if s != "val" else np.nan,
                                     flagged=float((prob > THR).mean()) if s != "val" else np.nan))
    return jobs, meta


def benchmark(runs, ops, B, workers):
    jobs, meta = jobs_for(runs, ops, B)
    with ProcessPoolExecutor(workers) as ex:
        rows = list(ex.map(evaluate._row, jobs, chunksize=1))
    bm = pd.concat([pd.DataFrame(rows), pd.DataFrame(meta)], axis=1).rename(columns={"final_min_dcf": "ref_min_dcf"})
    none = bm.reference == ""
    bm.loc[none, ["ref_min_dcf", "delta", "delta_lo", "delta_hi", "p_not_worse"]] = np.nan
    bm["s_per_step"] = bm.system.map({n: s_per_step(r) for n, r in runs.items()})
    return bm.rename(columns={"system": "run"})


def curves(runs):
    rows = []
    for name, run in runs.items():
        for (e, s), d in sorted(run["sc"].items()):
            y = d.label.to_numpy().astype(int)
            for v in VIEWS:
                if f"score_{v}" in d:
                    dcf, eer = metrics.min_dcf_eer(y, d[f"score_{v}"].to_numpy(float))
                    rows.append(dict(run=name, epoch=e, set=s, view=v, n=len(y), min_dcf=dcf, eer=eer))
    return pd.DataFrame(rows)


def seed_noise(bm):
    rows = []
    for label, (a, b) in {"R0 (val selection)": ("rb_R0_xlsr1b_s0", "rb_R0_xlsr1b_s1"),
                          "stored (val + ITW selection)": ("xlsr1b_d6rall", "xlsr1b_d6rall_s1")}.items():
        x, y = (bm[bm.run == n].set_index(["op", "set", "view"]).min_dcf for n in (a, b))
        for k in x.index.intersection(y.index):
            rows.append(dict(pair=label, op=k[0], set=k[1], view=k[2], seed0=x[k], seed1=y[k], abs_diff=abs(x[k] - y[k])))
    return pd.DataFrame(rows)


def gates(bm, noise):
    """The predeclared gates (decisions.md): 1 target, 2 clean non-inferiority, 3 seeds, 4 cost."""
    spread = noise.groupby(["op", "set", "view"]).abs_diff.max()
    rows = []
    arms = bm[bm.run.str.match(r"rb_") & ~bm.run.str.match(r"rb_R0_") & (bm.reference != "")]
    for (run, op), g in arms.groupby(["run", "op"]):
        g = g.set_index(["set", "view"])
        r = dict(run=run, op=op)
        for v in ("aug", "unseen"):
            if ("itw", v) in g.index:
                x = g.loc[("itw", v)]
                sp = spread.get((op, "itw", v), np.nan)
                r[f"itw_{v}_rel"] = x.delta / x.ref_min_dcf
                r[f"itw_{v}_pass"] = bool(x.delta / x.ref_min_dcf <= -0.10 and x.delta_hi < 0 and -x.delta > sp)
                r[f"itw_{v}_seed_spread"] = sp
        r["gate1_target"] = any(r.get(f"itw_{v}_pass", False) for v in ("aug", "unseen"))
        ok = []
        for s in ("itw", "holdout"):
            x = g.loc[(s, "clean")]
            ok.append(x.delta <= max(0.05 * x.ref_min_dcf, 0.005))
            r[f"{s}_clean_delta"] = x.delta
        r["gate2_clean"] = bool(all(ok))
        base = bm[(bm.run == baseline(run))].s_per_step.iloc[0] if (bm.run == baseline(run)).any() else np.nan
        r["s_per_step"], r["gate4_cost"] = g.s_per_step.iloc[0], bool(g.s_per_step.iloc[0] <= 1.25 * base)
        rows.append(r)
    out = pd.DataFrame(rows)
    if len(out):   # gate 3: every seed of the arm improves the view(s) that passed gate 1 (each against its own R0)
        key = out.run.str.extract(RUN.pattern)
        g3 = []
        for i in range(len(out)):
            same = out[(key.arm == key.arm[i]) & (key.backbone == key.backbone[i]) & (out.op == out.op[i])]
            passed = [v for v in ("aug", "unseen") if f"itw_{v}_pass" in out
                      and pd.notna(out[f"itw_{v}_pass"].iloc[i]) and bool(out[f"itw_{v}_pass"].iloc[i])]
            g3.append(bool(len(same) >= 2 and any((same[f"itw_{v}_rel"] < 0).all() for v in passed)))
        out["gate3_seeds"] = g3
        out["passes"] = out.gate1_target & out.gate2_clean & out.gate3_seeds & out.gate4_cost
    return out


def split_ops(ch):
    return [c for c in str(ch or "").split("+") if c]


def breakdown(runs, ops, B, workers):
    """Slices at the val operating point: generator, real source, view-1 channel op, unseen op, device; paired with
    the run's reference."""
    replay = config.CACHE / "eval_views.parquet"
    ev = pd.read_parquet(replay).set_index(["uid", "view"]) if replay.exists() else None
    jobs, meta = [], []
    for name, run in runs.items():
        ref = runs.get(baseline(name))
        e = ops[name]["val"]
        for s in ("holdout", "itw"):
            d = run["sc"].get((e, s))
            r = ref["sc"].get((ops[ref["name"]]["val"], s)) if ref else None
            if d is None:
                continue
            y = d.label.to_numpy().astype(int)
            for v in VIEWS:
                if f"score_{v}" not in d:
                    continue
                x = d[f"score_{v}"].to_numpy(float)
                xr = r[f"score_{v}"].loc[d.index].to_numpy(float) if r is not None and f"score_{v}" in r else x
                masks = {("generator", g): (y == 0) | (d.generator == g).to_numpy() for g in sorted(d.generator[y == 1].unique())}
                masks.update({("real source", f): (y == 1) | ((d.family == f).to_numpy() & (y == 0))
                              for f in sorted(d.family[y == 0].unique())})
                if v == "aug":
                    ch = d.channel_aug if "channel_aug" in d else (
                        ev.channel.reindex([(u, 1) for u in d.index]).set_axis(d.index) if ev is not None else None)
                    if ch is not None:
                        chains = ch.map(split_ops)
                        for op in evaluate.OPS:
                            masks[("channel", op)] = chains.map(lambda c: any(t.startswith(op[:5]) for t in c)).to_numpy()
                        for sub in sorted({t for c in chains for t in c if t.startswith(("codec_", "noise_"))}):
                            masks[("channel", sub)] = chains.map(lambda c: sub in c).to_numpy()
                if v == "unseen" and "channel_unseen" in d:
                    chains = d.channel_unseen.map(split_ops)
                    for op in ("room", "codec", "tandem", "packet_loss", "agc"):
                        masks[("unseen op", op)] = chains.map(lambda c: any(t.startswith(op) for t in c)).to_numpy()
                    for sub in sorted({t for c in chains for t in c if t.startswith("codec_")}):
                        masks[("unseen op", sub)] = chains.map(lambda c: sub in c).to_numpy()
                if ev is not None and v in ("clean", "aug"):
                    dur = pd.cut(ev.dur.reindex([(u, int(v == "aug")) for u in d.index]).to_numpy(), evaluate.DUR_BINS,
                                 labels=evaluate.DUR_LABELS)
                    masks.update({("duration", b): np.asarray(dur == b) for b in evaluate.DUR_LABELS})
                for (kind, group), m in masks.items():
                    if m.sum() < 20 or len(np.unique(y[m])) < 2:
                        continue
                    seed = zlib.crc32(f"{s}/{v}/{kind}/{group}".encode())
                    jobs.append((name, s, v, y[m], x[m], xr[m], B, seed))
                    meta.append(dict(epoch=e, kind=kind, group=group, n_real=int((y[m] == 0).sum()),
                                     n_fake=int((y[m] == 1).sum()), reference=ref["name"] if ref is not None and xr is not x else ""))
    with ProcessPoolExecutor(workers) as ex:
        rows = list(ex.map(evaluate._row, jobs, chunksize=4))
    bd = pd.concat([pd.DataFrame(rows).drop(columns=["n", "n_fake"]), pd.DataFrame(meta)], axis=1).rename(
        columns={"system": "run", "final_min_dcf": "ref_min_dcf"})
    bd.loc[bd.reference == "", ["ref_min_dcf", "delta", "delta_lo", "delta_hi", "p_not_worse"]] = np.nan
    return bd


def test_checks(runs, ops):
    """NSA test set, label-free (no labels exist here): agreement with the submitted final ensemble."""
    final = pd.read_csv(config.REPO / "submission" / "SideQuests_predictions_final.tsv", sep="\t").set_index("filename")["cm-score"]
    q = float((final > THR).mean())
    tri = pd.read_parquet(config.CACHE / "triage_pool.parquet").set_index("uid").decoded_duration
    rows = []
    for name, run in runs.items():
        for op, e in ops[name].items():
            f = FT / name / f"test_epoch{e}.parquet"
            if not f.exists():
                continue
            t = pd.read_parquet(f).set_index("filename")
            common = t.index.intersection(final.index)
            s, p = t.score.loc[common].to_numpy(float), final.loc[common].to_numpy(float)
            flag_final, flag_arm = p > THR, s >= np.quantile(s, 1 - q)
            prob = calibrated(run, e)(s)
            rows.append(dict(run=name, op=op, epoch=e, n=len(common), spearman_final=spearmanr(s, p)[0],
                             kappa_at_final_flag_rate=cohen_kappa_score(flag_final, flag_arm), final_flagged=q,
                             flagged_val_platt=float((prob > THR).mean()),
                             spearman_duration=spearmanr(s, tri.reindex(t.uid.loc[common]).to_numpy())[0]))
    return pd.DataFrame(rows)


def member_scores(runs, name, epoch):
    """{(set, view): scores} of one member; a shipped member's views_<run> evaluation (all four views) if present."""
    ev = runs.get(f"views_{name}")
    src, e = (ev, 0) if ev is not None else (runs[name], epoch)
    return {(s, v): d[f"score_{v}"].astype(float) for s in SETS if (d := src["sc"].get((e, s))) is not None
            for v in VIEWS if f"score_{v}" in d}


def fuse(parts, stats=None):
    """Mean of z-normalized logits; z statistics from val (clean + aug) unless given ({member: (mean, std)})."""
    out, zs = {}, {}
    for m, sc in parts.items():
        ref = pd.concat([sc[("val", "clean")], sc[("val", "aug")]])
        zs[m] = stats[m] if stats else (ref.mean(), ref.std(ddof=0) + 1e-6)
    keys = set.intersection(*[set(sc) for sc in parts.values()])
    for k in keys:
        u = sorted(set.intersection(*[set(sc[k].index) for sc in parts.values()]))
        out[k] = sum((sc[k].loc[u] - zs[m][0]) / zs[m][1] for m, sc in parts.items()) / len(parts)
    return out


def ensemble(runs, ops, selected, B, workers):
    """E1 (stage 3): the selected arm on all three backbones, fused like the submission (val z statistics), against the
    shipped members at their shipped epochs fused the same way, on the same clips (paired). The shipped ensemble with
    its own fusion.json statistics (val + ITW) is listed for reference."""
    arms = [f"rb_{selected}_xlsr1b_s0", "rb_C2_xlsr2b_s0", "rb_C3_mms1b_s0"]
    if not selected or not all(a in runs for a in arms):
        return pd.DataFrame([dict(status="not run: needs stage 3 (C2, C3), and --selected <arm>")])
    fus = json.load(open(config.REPO / "submission" / "fusion.json"))["systems"]
    shipped = {f["name"]: member_scores(runs, f["name"], f["epoch"]) for f in fus}
    systems = {"E1 (RawBoost, val z)": fuse({a: member_scores(runs, a, ops[a]["val"]) for a in arms}),
               "shipped (val z)": fuse(shipped),
               "shipped (fusion.json z)": fuse(shipped, {f["name"]: (f["z_mean"], f["z_std"]) for f in fus})}
    lab = {s: pd.concat([runs[a]["sc"][(ops[a]["val"], s)].label for a in arms]).groupby(level=0).first() for s in SETS}
    jobs = []
    for name, sc in systems.items():
        for (s, v), x in sc.items():
            ref = systems["shipped (val z)"].get((s, v))
            if ref is None:
                continue
            u = x.index.intersection(ref.index)
            y = lab[s].loc[u].to_numpy().astype(int)
            jobs.append((name, s, v, y, x.loc[u].to_numpy(float), ref.loc[u].to_numpy(float), B,
                         zlib.crc32(f"{s}/{v}/{len(u)}/{int(y.sum())}".encode())))
    with ProcessPoolExecutor(workers) as ex:
        rows = list(ex.map(evaluate._row, jobs, chunksize=1))
    return pd.DataFrame(rows).rename(columns={"final_min_dcf": "ref_min_dcf"}).assign(reference="shipped (val z)")


# ------------------------------------------------------------------------------------------------ report

def fmt(x, lo=None, hi=None):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "not run"
    return f"{x:.3f}" + (f" [{lo:.3f}, {hi:.3f}]" if lo is not None and not np.isnan(lo) else "")


def tables(bm, bd, noise, gt, tc):
    L = ["# RawBoost experiment: results", "",
         "Generated by `scripts/rawboost_report.py` from stored per-epoch scores (in-training path: batch-cropped, bf16).",
         "minDCF (organizers' costs) [95% stratified bootstrap]; Δ = run − reference on the same clips [paired 95%].",
         "Reference: R0 of the same seed; for R0, the stored run of its seed. The stored runs were selected on val + ITW.", ""]
    for op, title in (("val", "val-selected epoch"), ("last", "last epoch")):
        L += [f"## Comparison at the {title}", "",
              "| run | epoch | ITW clean | ITW aug | ITW unseen | ITW device | holdout clean | holdout aug | reverb (ITW aug) | babble (ITW aug) | ElevenLabs (holdout) | Δ ITW aug | Δ ITW unseen | s/step |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for run in bm.run.unique():
            g = bm[(bm.run == run) & (bm.op == op)].set_index(["set", "view"])
            if g.empty:
                continue
            c = lambda s, v: fmt(*(g.loc[(s, v), ["min_dcf", "min_dcf_lo", "min_dcf_hi"]])) if (s, v) in g.index else "not run"
            dl = lambda v: fmt(*(g.loc[("itw", v), ["delta", "delta_lo", "delta_hi"]])) if ("itw", v) in g.index and g.loc[("itw", v), "reference"] else "–"
            sl = []
            for s, v, kind, grp in SLICES:
                h = bd[(bd.run == run) & (bd.set == s) & (bd.view == v) & (bd.kind == kind) & (bd.group == grp)] if op == "val" else bd.iloc[:0]
                sl.append(f"{h.min_dcf.iloc[0]:.3f} (n={h.n_real.iloc[0] + h.n_fake.iloc[0]})" if len(h) else "–")
            L.append(f"| {run} | {int(g.epoch.iloc[0])} | {c('itw', 'clean')} | {c('itw', 'aug')} | {c('itw', 'unseen')} | "
                     f"{c('itw', 'device')} | {c('holdout', 'clean')} | {c('holdout', 'aug')} | {sl[0]} | {sl[1]} | {sl[2]} | "
                     f"{dl('aug')} | {dl('unseen')} | {g.s_per_step.iloc[0]:.2f} |")
        L.append("")
    if len(noise):
        L += ["## Seed noise (|seed 0 − seed 1| minDCF)", "",
              noise.pivot_table(index=["set", "view"], columns=["pair", "op"], values="abs_diff").round(4).to_markdown(), ""]
    if len(gt):
        L += ["## Gates (decisions.md)", "", gt.round(4).to_markdown(index=False), ""]
    if len(tc):
        L += ["## NSA test set, label-free", "", tc.round(4).to_markdown(index=False), ""]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--out", default=str(config.REPO / "results" / "rawboost"))
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--val-only", action="store_true", help="the stage-decision table from val only; writes nothing else")
    ap.add_argument("--selected", default="", help="the arm carried to stage 3 (decisions.md), e.g. R2; enables E1")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.val_only:
        rows = []
        for p in discover():
            run = load_run(p, sets=("val",))
            if max(run["epochs"]) == 0:
                continue
            op, vals = op_points(run)
            for e, sel in vals.items():
                d = run["sc"][(e, "val")]
                y = d.label.to_numpy().astype(int)
                rows.append(dict(run=p.name, epoch=e, val_clean=metrics.min_dcf(y, d.score_clean.to_numpy(float)),
                                 val_aug=metrics.min_dcf(y, d.score_aug.to_numpy(float)), val_select=sel,
                                 selected=e == op["val"]))
        t = pd.DataFrame(rows)
        t.to_csv(out / "stage_val.csv", index=False, float_format="%.5f")
        print(t.round(4).to_string(index=False))
        return
    runs = {p.name: load_run(p) for p in discover()}
    ops = {n: op_points(r)[0] for n, r in runs.items()}
    bm = benchmark(runs, ops, a.boot, a.workers)
    bm.to_csv(out / "benchmark.csv", index=False, float_format="%.5f")
    curves(runs).to_csv(out / "curves.csv", index=False, float_format="%.5f")
    noise = seed_noise(bm)
    noise.to_csv(out / "seed_noise.csv", index=False, float_format="%.5f")
    gt = gates(bm, noise)
    gt.to_csv(out / "gates.csv", index=False, float_format="%.5f")
    bd = breakdown(runs, ops, max(200, a.boot // 5), a.workers)
    bd.to_csv(out / "breakdown.csv", index=False, float_format="%.5f")
    tc = test_checks(runs, ops)
    tc.to_csv(out / "test_checks.csv", index=False, float_format="%.5f")
    ensemble(runs, ops, a.selected, a.boot, a.workers).to_csv(out / "ensemble.csv", index=False, float_format="%.5f")
    (out / "tables.md").write_text(tables(bm, bd, noise, gt, tc))
    provenance.write(out, inputs=[f for p in discover() for f in sorted(p.glob("*_epoch*.parquet"))] +
                     [config.REPO / "submission" / "SideQuests_predictions_final.tsv"], bootstrap_reps=a.boot)
    print((out / "tables.md").read_text())


if __name__ == "__main__":
    main()
