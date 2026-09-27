"""Cognitive-science tests of the concept layer (docs/ROADMAP.md, workstream C), on the stored concept embeddings.
CPU only; needs cobweb-private (mpcdf/run.sbatch -- concepts "python scripts/concept_experiments.py").

    python scripts/concept_experiments.py [--emb xlsr2b_d6rall_last] [--seeds 3] [--workers 30]

C1a  Learning to name a new family from k examples, with no retraining and no forgetting. For each DiffSSD generator
     G: fit the tree without G, then insert k of G's training clips one by one (k = 0, 1, 3, 10, 30, 100). Measure
     source attribution (is G named?) and P(fake) AUC on G's held-out clips, and the same for the other generators
     (forgetting). Baselines on the same data, refitted at every k: kNN and logistic regression. The embedding itself
     comes from a detector that saw G, so attribution, not detection, is the informative number here.
C1b  Adapting to a new domain from k labelled examples. Insert k labelled In-the-Wild clips (k = 0, 10, 30, 100, 300,
     balanced) from half of the speakers into the full tree; score P(fake) on the other speakers. In-the-Wild was
     never trained on, so this is a true adaptation test. Same baselines, plus the detector's own score.
C2   Novelty, from COBWEB's create-a-new-concept operator. Novelty = -max held-out pmi along a clip's path. AUROC for
     G's clips vs the other generators' (trees without G), and In-the-Wild vs in-domain holdout (full tree).
     Baselines: squared whitened norm (Mahalanobis) and mean distance to the 10 nearest training clips.
C3   Typicality predicts reliability (prototype theory). Error rate and confidence of the shipped ensemble by
     typicality quintile (typicality = max held-out pmi), on holdout and In-the-Wild, clean and perturbed.
C4   Analyst triage. Rank In-the-Wild clips by concept-detector disagreement, novelty, both, or detector uncertainty;
     precision@k and average precision for the shipped ensemble's errors at P > 0.2.
I1   Concept atlas (global interpretability). The top levels of the tree: each concept's size, synthetic share,
     sources, channels, D(c), and its most typical member clips (nearest to the concept mean).
I3   Surrogate fidelity. How often the readable concept layer (TTCG, cobweb) makes the shipped detector's decision:
     agreement and Cohen's kappa at the detector's flag rate, Spearman, bootstrap intervals; labelled sets and the
     NSA test set.
I5   Acoustic names for concept dimensions. Spearman correlation of each of the 32 concept dimensions with the DSP
     track's interpretable acoustic features (spectral shape, prosody, background, LPC, phase jumps), on clips that
     have both; the detector's own direction in concept space (logit change per unit of each dimension) and the
     acoustic correlates of its logit. Correlational: it names dimensions, it does not prove causes.
I7   Failure localization. The shipped ensemble's errors on holdout and In-the-Wild (clean and perturbed), grouped
     by each clip's concept: which concepts hold the errors, and what they are made of.
I8   Unsupervised concept formation. cobweb-private's category utility adds a label-entropy term when labels are
     given, so the reported tree is label-informed (new clips are still categorized from the embedding alone). I8 fits
     the same tree with no labels and names its concepts afterwards from their members: does the detector's
     representation organize itself into real vs synthetic, and by generator, on its own? Purity by depth, AMI with
     the source, and an atlas.
Writes runs/diffusion/concepts/<emb>/experiments/: c1a.csv, c1b.csv, c2.csv, c3.csv, c4.csv, atlas.json, atlas.md,
i3_fidelity.csv, i5_dims.csv, i5_detector_correlates.csv, i7_failures.csv, i8_unsupervised.json, atlas_unsupervised.md,
summary.json, i2_test_exemplars.jsonl. --parts i5, or any of i2,i7,i8, runs those parts alone.
I2   Case-based explanations: for every NSA test clip, the training clips of its concept closest to it.
"""
import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hearsay import concepts, config, metrics  # noqa: E402
from hearsay.concepts import Concepts  # noqa: E402

G = {}  # data shared with forked workers


def auc(y, s):
    y = np.asarray(y)
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else float("nan")


def tree_for(sources, channels, rows, seed):
    sid, cid = {s: i for i, s in enumerate(sources)}, {c: i for i, c in enumerate(channels)}
    return Concepts(G["dims"], sources, channels, seed=seed).fit(G["Z"](rows.row), rows.source.map(sid).to_numpy(), rows.chan.map(cid).to_numpy())


def insert(cw, rows):
    if not len(rows):
        return
    sid, cid = {s: i for i, s in enumerate(cw.sources)}, {c: i for i, c in enumerate(cw.channels)}
    Zr = G["Z"](rows.row)
    for z, s, c in zip(Zr, rows.source.map(sid), rows.chan.map(cid)):
        lab = np.zeros(cw.S + cw.C, np.float32)
        lab[s], lab[cw.S + c] = 1.0, 1.0
        cw.tree.ifit(concepts.f32(z), lab)


def source_dist(cw, Zx, max_nodes=300):
    empty = np.zeros(cw.S + cw.C, np.float32)
    out = np.zeros((len(Zx), cw.S))
    for i, z in enumerate(Zx):
        d = np.asarray(cw.tree.predict(concepts.f32(z), empty, max_nodes, False))[: cw.S]
        out[i] = d / max(d.sum(), 1e-12)
    return out


def novelty(cw, Zx):
    return np.array([-cw.informative(z)[2] for z in Zx])


def baselines(train_Z, train_src, train_fake, Zx, k=10):
    """kNN (P(fake) = fake share of the 10 nearest; source = majority) and logistic regression (fake; multinomial
    source), both refitted from scratch on the current training set."""
    nn = NearestNeighbors(n_neighbors=k).fit(train_Z)
    dist, ind = nn.kneighbors(Zx)
    knn_p = train_fake[ind].mean(1)
    knn_src = np.array([pd.Series(train_src[i]).mode().iloc[0] for i in ind])
    lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(train_Z, train_fake)
    lr_src = LogisticRegression(max_iter=2000).fit(train_Z, train_src)
    return dict(knn_p=knn_p, knn_src=knn_src, knn_dist=dist.mean(1), lr_p=lr.predict_proba(Zx)[:, 1], lr_src=lr_src.predict(Zx))


def c1a_worker(job):
    gen, seed = job
    lab, fit_n = G["lab"], G["fit_n"]
    tr = lab[lab.split == "train"]
    sources, channels = sorted(tr.source.unique()), sorted(tr.chan.unique())
    fit = tr[tr.source != gen]
    fit = fit.groupby("source", group_keys=False).sample(frac=min(1.0, fit_n / len(tr)), random_state=seed)
    cw = tree_for(sources, channels, fit, seed)
    pool = tr[(tr.source == gen) & (tr.view == 0)].sample(frac=1.0, random_state=seed)
    ev = lab[lab.split.isin(["val", "holdout"]) & (lab.view == 0)]
    ev_g, ev_o, ev_r = ev[ev.source == gen], ev[(ev.label == 1) & (ev.source != gen)], ev[ev.label == 0]
    Zg, Zo, Zr = G["Z"](ev_g.row), G["Z"](ev_o.row), G["Z"](ev_r.row)
    rows, done = [], 0
    nov = None
    for k in (0, 1, 3, 10, 30, 100):
        k = min(k, len(pool))
        insert(cw, pool.iloc[done:k])
        done = k
        dg, do, dr = source_dist(cw, Zg), source_dist(cw, Zo), source_dist(cw, Zr)
        pf = lambda d: d[:, cw.fake].sum(1)  # noqa: E731
        si = {s: i for i, s in enumerate(sources)}
        r = dict(generator=gen, seed=seed, k=k,
                 cobweb_attr_new=float(np.mean(dg.argmax(1) == si[gen])),
                 cobweb_auc_new=auc(np.r_[np.ones(len(dg)), np.zeros(len(dr))], np.r_[pf(dg), pf(dr)]),
                 cobweb_attr_old=float(np.mean(do.argmax(1) == ev_o.source.map(si).to_numpy())),
                 cobweb_auc_old=auc(np.r_[np.ones(len(do)), np.zeros(len(dr))], np.r_[pf(do), pf(dr)]))
        trn = pd.concat([fit, pool.iloc[:k]])
        b = baselines(G["Z"](trn.row), trn.source.to_numpy(), trn.label.to_numpy(), np.r_[Zg, Zo, Zr])
        n1, n2 = len(Zg), len(Zg) + len(Zo)
        y_new, y_old = np.r_[np.ones(n1), np.zeros(len(Zr))], np.r_[np.ones(len(Zo)), np.zeros(len(Zr))]
        for m in ("knn", "lr"):
            p = b[f"{m}_p"]
            r[f"{m}_attr_new"] = float(np.mean(b[f"{m}_src"][:n1] == gen))
            r[f"{m}_auc_new"] = auc(y_new, np.r_[p[:n1], p[n2:]])
            r[f"{m}_attr_old"] = float(np.mean(b[f"{m}_src"][n1:n2] == ev_o.source.to_numpy()))
            r[f"{m}_auc_old"] = auc(y_old, np.r_[p[n1:n2], p[n2:]])
        if k == 0:  # C2: novelty of the unseen generator vs the seen ones, before any insertion
            nv = novelty(cw, np.r_[Zg, Zo])
            y = np.r_[np.ones(len(Zg)), np.zeros(len(Zo))]
            nnd = baselines(G["Z"](fit.row), fit.source.to_numpy(), fit.label.to_numpy(), np.r_[Zg, Zo])["knn_dist"]
            maha = (np.r_[Zg, Zo] ** 2).sum(1)
            nov = dict(task="unseen generator vs seen", generator=gen, seed=seed, n_new=len(Zg), n_seen=len(Zo),
                       auroc_novelty=auc(y, nv), auroc_mahalanobis=auc(y, maha), auroc_knn_distance=auc(y, nnd))
        rows.append(r)
    return rows, nov


def c1b_worker(seed):
    lab, fit_n = G["lab"], G["fit_n"]
    tr = lab[lab.split == "train"]
    itw = lab[(lab.split == "itw")].copy()
    itw["source"] = np.where(itw.label == 1, "itw_fake", "real:itw")
    spk = sorted(itw.speaker.unique())
    rng = np.random.default_rng(seed)
    ins_spk = set(rng.permutation(spk)[: len(spk) // 2])
    pool = itw[itw.speaker.isin(ins_spk) & (itw.view == 0)]
    test = itw[~itw.speaker.isin(ins_spk)]
    sources = sorted(set(tr.source.unique()) | {"itw_fake", "real:itw"})
    channels = sorted(tr.chan.unique())
    fit = tr.groupby("source", group_keys=False).sample(frac=min(1.0, fit_n / len(tr)), random_state=seed)
    cw = tree_for(sources, channels, fit, seed)
    Zt = G["Z"](test.row)
    y = test.label.to_numpy()
    det = G["final_p"].reindex(list(zip(test.uid, test.view))).to_numpy()
    ok = np.isfinite(det)
    rows, done = [], 0
    order = pd.concat([pool[pool.label == 1].sample(frac=1.0, random_state=seed).assign(o=lambda d: np.arange(len(d)) * 2),
                       pool[pool.label == 0].sample(frac=1.0, random_state=seed).assign(o=lambda d: np.arange(len(d)) * 2 + 1)]).sort_values("o")
    for k in (0, 10, 30, 100, 300):
        k = min(k, len(order))
        insert(cw, order.iloc[done:k])
        done = k
        p = source_dist(cw, Zt)[:, cw.fake].sum(1)
        trn = pd.concat([fit, order.iloc[:k]])
        b = baselines(G["Z"](trn.row), trn.source.to_numpy(), trn.label.to_numpy(), Zt)
        r = dict(seed=seed, k=k, n_test=len(test), n_test_speakers=int(test.speaker.nunique()))
        for name, s in (("cobweb", p), ("knn", b["knn_p"]), ("lr", b["lr_p"])):
            for v, m in (("clean", (test.view == 0).to_numpy()), ("aug", (test.view > 0).to_numpy())):
                r[f"{name}_{v}_auc"] = auc(y[m], s[m])
                r[f"{name}_{v}_mindcf"] = metrics.min_dcf(y[m], s[m]) if 0 < y[m].sum() < m.sum() else float("nan")
        for v, m in (("clean", (test.view == 0).to_numpy()), ("aug", (test.view > 0).to_numpy())):
            mm = m & ok
            r[f"detector_{v}_auc"] = auc(y[mm], det[mm])
            r[f"detector_{v}_mindcf"] = metrics.min_dcf(y[mm], det[mm])
        rows.append(r)
    return rows


def atlas(cw, fit, Zfit, max_depth=4, min_share=0.01, n_examples=3):
    """I1: the top of the concept hierarchy with the most typical member clips of each concept."""
    members = {}
    empty = np.zeros(cw.S + cw.C, np.float32)
    for i, z in enumerate(Zfit):
        for n in Concepts.path(cw.tree.get_leaf(concepts.f32(z), empty)):
            members.setdefault(concepts.node_key(n), []).append(i)
    total = float(cw.tree.root.count)
    rows = []

    def walk(node, nid):
        if node.depth() > max_depth or node.count < min_share * total:
            return
        src, ch, pf = cw.makeup(node, k=3)
        idx = np.array(members.get(concepts.node_key(node), []), int)
        mean = np.asarray(node.mean, np.float32)
        near = idx[np.argsort(((Zfit[idx] - mean) ** 2).sum(1))[:n_examples]] if len(idx) else []
        rows.append(dict(id=nid, depth=node.depth(), size=int(node.count), share=round(node.count / total, 4), p_fake=round(pf, 3),
                         expected_pmi=round(node.expected_pmi(), 2), sources=src, channels=ch,
                         exemplars=[fit.uid.iloc[j] for j in near]))
        for j, c in enumerate(sorted(node.children, key=lambda c: -c.count)):
            walk(c, f"{nid}.{j + 1}")
    walk(cw.tree.root, "0")
    return rows


def render_atlas(rows):
    L = ["# Concept atlas (generated by scripts/concept_experiments.py)", "",
         "The top levels of the COBWEB concept tree over the final detector's representation: concepts holding at least 1% of the",
         "12,000 training clips, down to depth 4. Each line: concept id, size, synthetic share, top sources, top channels, D(c),",
         "and the most typical member clips (closest to the concept mean).", ""]
    for r in rows:
        src = ", ".join(f"{x['source']} {x['share']:.0%}" for x in r["sources"])
        ch = ", ".join(f"{x['channel']} {x['share']:.0%}" for x in r["channels"])
        L.append(f"{'  ' * r['depth']}- **{r['id']}** · {r['size']:,} clips ({r['share']:.1%}) · {r['p_fake']:.0%} synthetic · {src} · {ch}"
                 f" · D(c) {r['expected_pmi']} · e.g. {', '.join(r['exemplars'])}")
    return "\n".join(L) + "\n"


def kappa(a, b):
    po = (a == b).mean()
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")


def fidelity_rows(name, det, s, label=None, reps=1000, seed=0):
    """Agreement of a concept score s with the detector's decisions det, thresholding s at the detector's flag rate."""
    ok = np.isfinite(s)
    det, s = det[ok], s[ok]
    q = det.mean()
    con = pd.Series(s).rank(ascending=False, method="first").to_numpy() <= q * len(s)
    rng = np.random.default_rng(seed)
    ks = [kappa(det[i], con[i]) for i in (rng.integers(0, len(s), len(s)) for _ in range(reps))]
    r = dict(scorer=name, n=len(s), detector_flag_rate=float(q), agreement=float((det == con).mean()), kappa=kappa(det, con),
             kappa_lo=float(np.nanpercentile(ks, 2.5)), kappa_hi=float(np.nanpercentile(ks, 97.5)))
    if label is not None:
        y = np.asarray(label)[ok]
        r.update(detector_accuracy=float((det == y).mean()), concept_accuracy=float((con == y).mean()))
    return r


INTERPRETABLE = ("spec.", "pros.", "bg.", "lpc.", "phase.ifjump")


def data_key(p):
    p = str(p).replace("\\", "/")
    return p.split("/data/", 1)[1] if "/data/" in p else (p[5:] if p.startswith("data/") else p)


def acoustics(D, out, reps=200, seed=0):
    """I5 (see the module docstring)."""
    import torch
    feats = pd.concat([pd.read_csv(config.REPO / "runs" / "dsp" / d / "features.tsv", sep="\t", low_memory=False)
                       for d in ("extract_pool", "extract_hearsay_test")])
    cols = [c for c in feats.columns if c.startswith(INTERPRETABLE)]
    feats["key"] = feats.path.map(data_key)
    feats = feats.drop_duplicates("key").set_index("key")[cols].apply(pd.to_numeric, errors="coerce")
    rows = pd.concat([D.lab, D.test])
    rows = rows[rows.view == 0].copy()
    rows["key"] = rows.path.map(data_key)
    m = rows[rows.key.isin(feats.index)].drop_duplicates("key")
    Zm = D.Z(m.row)
    F = feats.loc[m.key].reset_index(drop=True)
    dims = [f"dim{d:02d}" for d in range(Zm.shape[1])]
    both = pd.concat([pd.DataFrame(Zm, columns=dims), F], axis=1)
    rho = both.corr(method="spearman").loc[dims, cols]
    sd = torch.load(config.REPO / D.meta["checkpoint"], map_location="cpu", mmap=True)
    w = (sd["head.weight"][0] - sd["head.weight"][1]).float().numpy()
    b = float(sd["head.bias"][0] - sd["head.bias"][1])
    sens = D.space.delta_h(np.eye(Zm.shape[1])) @ w                # logit change per unit of each concept dimension
    rng = np.random.default_rng(seed)
    out_rows = []
    for j, dname in enumerate(dims):
        top = rho.loc[dname].abs().sort_values(ascending=False).index[:3]
        f0 = top[0]
        pair = both[[dname, f0]].dropna().to_numpy()
        bs = [pd.DataFrame(pair[rng.integers(0, len(pair), len(pair))]).corr(method="spearman").iloc[0, 1] for _ in range(reps)]
        out_rows.append(dict(dim=dname, detector_sensitivity=float(sens[j]), n=len(pair),
                             top1=f0, rho1=float(rho.loc[dname, f0]), rho1_lo=float(np.percentile(bs, 2.5)), rho1_hi=float(np.percentile(bs, 97.5)),
                             top2=top[1], rho2=float(rho.loc[dname, top[1]]), top3=top[2], rho3=float(rho.loc[dname, top[2]])))
    dimtab = pd.DataFrame(out_rows)
    dimtab["importance_rank"] = dimtab.detector_sensitivity.abs().rank(ascending=False).astype(int)
    dimtab.sort_values("importance_rank").to_csv(out / "i5_dims.csv", index=False, float_format="%.4f")
    logit = D.feats(m.row) @ w + b
    det = F.apply(lambda c: pd.Series(logit).corr(c, method="spearman")).rename("rho_with_detector_logit")
    det = det.to_frame().assign(abs=lambda d: d.rho_with_detector_logit.abs()).sort_values("abs", ascending=False).drop(columns="abs")
    det.to_csv(out / "i5_detector_correlates.csv", float_format="%.4f")
    print(f"I5: {len(m)} clips with both the concept embedding and DSP features, {len(cols)} acoustic features", flush=True)
    return dict(n_clips=len(m), n_features=len(cols),
                top_dims=dimtab.sort_values("importance_rank").head(8)[["dim", "detector_sensitivity", "top1", "rho1", "rho1_lo", "rho1_hi"]].round(3).to_dict("records"),
                detector_correlates=det.head(10).round(3).reset_index().rename(columns={"index": "feature"}).to_dict("records"))


def failure_concepts(cw, ev, Ze, min_errors=3):
    """I7: errors of the shipped ensemble grouped by each clip's concept (held-out basic level)."""
    nodes = [cw.informative(z)[1] for z in Ze]
    ev = ev.assign(concept=[concepts.node_key(n) for n in nodes])
    info = {}
    for n in nodes:
        k = concepts.node_key(n)
        if k not in info:
            src, ch, pf = cw.makeup(n, k=2)
            info[k] = dict(depth=n.depth(), size=int(n.count), p_fake=round(pf, 3),
                           sources=", ".join(f"{x['source']} {x['share']:.0%}" for x in src),
                           channels=", ".join(f"{x['channel']} {x['share']:.0%}" for x in ch))
    g = ev.groupby("concept").agg(n=("error", "size"), errors=("error", "sum"), fa=("error", lambda e: int(((ev.loc[e.index, "label"] == 0) & (e == 1)).sum())))
    g = g[g.errors >= min_errors].sort_values("errors", ascending=False)
    rows = [dict(concept=str(k), **info[k], n=int(r.n), errors=int(r.errors), false_alarms=int(r.fa), misses=int(r.errors - r.fa),
                 error_rate=round(r.errors / r.n, 3), share_of_all_errors=round(r.errors / max(ev.error.sum(), 1), 3)) for k, r in g.iterrows()]
    return rows, int(ev.error.sum()), int(len(ev))


def unsupervised(fit, Zfit, dims, seed=0):
    """I8: the same COBWEB tree without labels; concepts named afterwards from their members."""
    from cobweb.cobweb_continuous import CobwebContinuousTree
    tree = CobwebContinuousTree(size=dims, num_labels=0)
    empty = np.zeros(0, np.float32)
    for i in np.random.default_rng(seed).permutation(len(Zfit)):
        tree.ifit(concepts.f32(Zfit[i]), empty)
    members, depth_of = {}, {}
    for i, z in enumerate(Zfit):
        for n in Concepts.path(tree.get_leaf(concepts.f32(z), empty)):
            k = concepts.node_key(n)
            members.setdefault(k, []).append(i)
            depth_of[k] = n.depth()
    lab, src = fit.label.to_numpy(), fit.source.to_numpy()
    by_depth = {}
    for k, idx in members.items():
        by_depth.setdefault(depth_of[k], []).append(idx)
    purity = {}
    for dpt in sorted(by_depth):
        parts = by_depth[dpt]
        covered = sum(len(i) for i in parts)
        purity[dpt] = dict(n_concepts=len(parts), covered=covered,
                           real_fake_purity=round(sum(max(lab[i].mean(), 1 - lab[i].mean()) * len(i) for i in parts) / covered, 4),
                           source_purity=round(sum(pd.Series(src[i]).value_counts().iloc[0] for i in parts) / covered, 4))
    from sklearn.metrics import adjusted_mutual_info_score
    leaf_ids = [concepts.node_key(tree.get_leaf(concepts.f32(z), empty)) for z in Zfit]
    rows = []
    total = float(tree.root.count)

    def walk(node, nid):
        if node.depth() > 4 or node.count < 0.01 * total:
            return
        idx = np.array(members.get(concepts.node_key(node), []), int)
        vc = pd.Series(src[idx]).value_counts(normalize=True)
        rows.append(dict(id=nid, depth=node.depth(), size=int(node.count), share=round(node.count / total, 4),
                         p_fake=round(float(lab[idx].mean()), 3) if len(idx) else float("nan"),
                         sources=[dict(source=k, share=round(float(v), 3)) for k, v in vc.head(3).items()], channels=[],
                         expected_pmi=round(node.expected_pmi(), 2), exemplars=[]))
        for j, c in enumerate(sorted(node.children, key=lambda c: -c.count)):
            walk(c, f"{nid}.{j + 1}")
    walk(tree.root, "0")
    return dict(purity_by_depth=purity, leaf_ami_source=round(float(adjusted_mutual_info_score(src, leaf_ids)), 4),
                leaf_ami_real_fake=round(float(adjusted_mutual_info_score(lab, leaf_ids)), 4)), rows


def exemplars(cw, fit, Zfit, test, Zt, k=3):
    """I2: case-based explanations. For each test clip, the training clips of its concept (held-out basic level) that
    are closest to it: real recordings or known generators' output an analyst can look up and listen to."""
    members = {}
    empty = np.zeros(cw.S + cw.C, np.float32)
    for i, z in enumerate(Zfit):
        for n in Concepts.path(cw.tree.get_leaf(concepts.f32(z), empty)):
            members.setdefault(concepts.node_key(n), []).append(i)
    rows = []
    for (fname, z) in zip(test.filename, Zt):
        _, node, pmi = cw.informative(z)
        idx = np.array(members.get(concepts.node_key(node), []), int)
        near = idx[np.argsort(((Zfit[idx] - z) ** 2).sum(1))[:k]] if len(idx) else []
        src, ch, pf = cw.makeup(node, k=2)
        rows.append(dict(filename=fname, concept_depth=node.depth(), concept_size=int(node.count), concept_p_fake=round(pf, 3),
                         concept_sources=src, exemplars=[dict(clip=fit.uid.iloc[j], source=fit.source.iloc[j],
                                                              synthetic=bool(fit.label.iloc[j]), channel=fit.chan.iloc[j]) for j in near]))
    return rows


def interpretability_extras(D, out, a, parts=("i2", "i7", "i8")):
    """I2, I7, I8 on the full label-informed tree (I8: plus an unsupervised one)."""
    lab = D.lab
    tr = lab[lab.split == "train"]
    sources, channels = sorted(tr.source.unique()), sorted(tr.chan.unique())
    fit = tr.groupby("source", group_keys=False).sample(frac=min(1.0, a.fit_n / len(tr)), random_state=0)
    cw = tree_for(sources, channels, fit, 0)
    Zfit = D.Z(fit.row)
    res = {}
    if "i2" in parts:
        test = D.test[D.test.view == 0]
        ex = exemplars(cw, fit, Zfit, test, D.Z(test.row))
        with open(out / "i2_test_exemplars.jsonl", "w") as fh:
            for r in ex:
                fh.write(json.dumps(r) + "\n")
        res["i2"] = dict(n_test=len(ex), example=ex[0] if ex else None)
    if "i7" in parts:
        ev = lab[lab.split.isin(["holdout", "itw"])].copy()
        ev["p_final"] = G["final_p"].reindex(list(zip(ev.uid, ev.view))).to_numpy()
        ev = ev[np.isfinite(ev.p_final)]
        ev["error"] = ((ev.p_final > 0.2).astype(int) != ev.label).astype(int)
        fails, n_err, n = failure_concepts(cw, ev, D.Z(ev.row))
        pd.DataFrame(fails).to_csv(out / "i7_failures.csv", index=False)
        top = sum(r["errors"] for r in fails[:5])
        res["i7"] = dict(n_clips=n, n_errors=n_err, top5_concepts_share_of_errors=round(top / max(n_err, 1), 3), top=fails[:8])
    if "i8" in parts:
        i8, rows = unsupervised(fit, Zfit, a.dims)
        json.dump(dict(i8, atlas=rows), open(out / "i8_unsupervised.json", "w"), indent=1, default=str)
        (out / "atlas_unsupervised.md").write_text(render_atlas(rows).replace(
            "# Concept atlas (generated by scripts/concept_experiments.py)",
            "# Concept atlas, unsupervised tree (no labels during concept formation; names from members afterwards)"))
        res["i8"] = i8
    return res


def final_scores():
    """The shipped ensemble's P(synthetic) per (uid, view) on val / holdout / In-the-Wild, from the stored per-epoch
    scores and submission/fusion.json (z statistics + Platt)."""
    fus = json.load(open(config.REPO / "submission" / "fusion.json"))
    parts = []
    for split in ("val", "holdout", "itw"):
        z = []
        for s in fus["systems"]:
            d = pd.read_parquet(config.RUNS / "ft" / s["name"] / f"{split}_epoch{s['epoch']}.parquet").set_index("uid")
            z.append(pd.concat([(d.score_clean - s["z_mean"]) / s["z_std"], (d.score_aug - s["z_mean"]) / s["z_std"]],
                               keys=[0, 1], names=["view", "uid"]))
        zz = sum(z) / len(z)
        p = 1 / (1 + np.exp(-(fus["platt"]["coef"] * zz + fus["platt"]["intercept"])))
        parts.append(p.swaplevel().rename("p"))
    s = pd.concat(parts)
    return s[~s.index.duplicated()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", default="xlsr2b_d6rall_last")
    ap.add_argument("--dims", type=int, default=32)
    ap.add_argument("--fit-n", type=int, default=12000)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--workers", type=int, default=30)
    ap.add_argument("--parts", default="all", help="all, or i5 alone")
    a = ap.parse_args()
    out = config.RUNS / "concepts" / a.emb / "experiments"
    out.mkdir(parents=True, exist_ok=True)
    D = concepts.load(a.emb, a.dims)
    if a.parts == "i5":
        res = acoustics(D, out)
        json.dump(res, open(out / "i5_summary.json", "w"), indent=1, default=str)
        print(json.dumps(res, indent=1, default=str))
        return
    G.update(lab=D.lab, Z=D.Z, dims=a.dims, fit_n=a.fit_n, final_p=final_scores())
    if set(a.parts.split(",")) <= {"i2", "i7", "i8"}:
        extra = interpretability_extras(D, out, a, set(a.parts.split(",")))
        print(json.dumps(extra, indent=1, default=str))
        return
    gens = sorted(D.lab[(D.lab.label == 1) & (D.lab.split == "train")].source.unique())
    print("generators:", gens, flush=True)

    with ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(c1a_worker, [(g, s) for g in gens for s in range(a.seeds)]))
        c1b = [r for rows in ex.map(c1b_worker, range(a.seeds)) for r in rows]
    c1a = pd.DataFrame([r for rows, _ in res for r in rows])
    c1a.to_csv(out / "c1a.csv", index=False, float_format="%.5f")
    pd.DataFrame(c1b).to_csv(out / "c1b.csv", index=False, float_format="%.5f")

    # C2 (domain) + C3 + C4 on the full tree
    lab = D.lab
    tr = lab[lab.split == "train"]
    sources, channels = sorted(tr.source.unique()), sorted(tr.chan.unique())
    fit = tr.groupby("source", group_keys=False).sample(frac=min(1.0, a.fit_n / len(tr)), random_state=0)
    cw = tree_for(sources, channels, fit, 0)
    ev = lab[lab.split.isin(["holdout", "itw"])].copy()
    Ze = D.Z(ev.row)
    ev["typicality"] = -novelty(cw, Ze)
    ev["p_final"] = G["final_p"].reindex(list(zip(ev.uid, ev.view))).to_numpy()
    nb = baselines(D.Z(fit.row), fit.source.to_numpy(), fit.label.to_numpy(), Ze)
    ev["knn_dist"], ev["maha"] = nb["knn_dist"], (Ze ** 2).sum(1)
    c2 = [r for _, r in res if r]
    dom = ev[ev.view == 0]
    y = (dom.split == "itw").to_numpy().astype(int)
    c2.append(dict(task="In-the-Wild vs in-domain holdout", n_new=int(y.sum()), n_seen=int((1 - y).sum()),
                   auroc_novelty=auc(y, -dom.typicality), auroc_mahalanobis=auc(y, dom.maha), auroc_knn_distance=auc(y, dom.knn_dist)))
    pd.DataFrame(c2).to_csv(out / "c2.csv", index=False, float_format="%.5f")

    ev = ev[np.isfinite(ev.p_final)]
    ev["error"] = ((ev.p_final > 0.2).astype(int) != ev.label).astype(int)
    ev["confidence"] = np.abs(np.log(np.clip(ev.p_final, 1e-6, 1 - 1e-6) / (1 - np.clip(ev.p_final, 1e-6, 1 - 1e-6))) - np.log(0.25))
    c3 = []
    for (split, view), g in ev.groupby(["split", ev.view.clip(upper=1)]):
        q = pd.qcut(g.typicality.rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
        for qi, gg in g.groupby(q, observed=True):
            c3.append(dict(set=split, view="clean" if view == 0 else "aug", quintile=int(qi), n=len(gg), error_rate=gg.error.mean(),
                           mean_confidence=gg.confidence.mean(), typicality_median=gg.typicality.median()))
        c3.append(dict(set=split, view="clean" if view == 0 else "aug", quintile=0, n=len(g), error_rate=g.error.mean(),
                       spearman_typicality_confidence=spearmanr(g.typicality, g.confidence)[0]))
    pd.DataFrame(c3).to_csv(out / "c3.csv", index=False, float_format="%.5f")

    # C4 triage on In-the-Wild clips that have TTCG scores
    tt = pd.read_parquet(config.RUNS / "concepts" / a.emb / "scores_itw.parquet")[["uid", "view", "p_fake_ttcg"]]
    it = ev[ev.split == "itw"].merge(tt.assign(view=tt.view.clip(upper=1)), on=["uid", "view"], how="inner")
    it = it[np.isfinite(it.p_fake_ttcg)]
    rng = np.random.default_rng(0)
    rank = lambda s: pd.Series(s).rank(pct=True).to_numpy()  # noqa: E731
    lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))  # noqa: E731
    scores = {"disagreement |P_final - P_TTCG|": np.abs(it.p_final - it.p_fake_ttcg).to_numpy(),
              "novelty": -it.typicality.to_numpy(),
              "disagreement + novelty": rank(np.abs(it.p_final - it.p_fake_ttcg)) + rank(-it.typicality),
              "detector uncertainty": -np.abs(lg(it.p_final) - np.log(0.25)).to_numpy(),
              "detector uncertainty + disagreement": rank(-np.abs(lg(it.p_final) - np.log(0.25))) + rank(np.abs(it.p_final - it.p_fake_ttcg)),
              "random": rng.random(len(it))}
    c4 = []
    for view, g_idx in (("clean", (it.view == 0).to_numpy()), ("aug", (it.view > 0).to_numpy()), ("both", np.ones(len(it), bool))):
        err = it.error.to_numpy()[g_idx]
        for name, s in scores.items():
            s = s[g_idx]
            o = np.argsort(-s)
            r = dict(view=view, ranking=name, n=int(g_idx.sum()), n_errors=int(err.sum()), error_rate=float(err.mean()),
                     average_precision=float(average_precision_score(err, s)) if err.sum() else float("nan"))
            for pct in (1, 2, 5, 10):
                kk = max(1, int(round(len(o) * pct / 100)))
                r[f"precision_at_{pct}pct"] = float(err[o[:kk]].mean())
                r[f"recall_at_{pct}pct"] = float(err[o[:kk]].sum() / max(err.sum(), 1))
            c4.append(r)
    pd.DataFrame(c4).to_csv(out / "c4.csv", index=False, float_format="%.5f")

    # I1: concept atlas
    Zfit = D.Z(fit.row)
    at = atlas(cw, fit, Zfit)
    json.dump(at, open(out / "atlas.json", "w"), indent=1)
    (out / "atlas.md").write_text(render_atlas(at))

    # I3: surrogate fidelity (labelled sets: the concept runs' scored clips; NSA test: explanations vs submitted TSV)
    fid = []
    for split in ("val", "holdout", "itw"):
        d = pd.read_parquet(config.RUNS / "concepts" / a.emb / f"scores_{split}.parquet")
        d["view"] = d.view.clip(upper=1)
        d["p_final"] = G["final_p"].reindex(list(zip(d.uid, d.view))).to_numpy()
        d = d[np.isfinite(d.p_final)]
        for v, g in (("clean", d[d.view == 0]), ("aug", d[d.view == 1])):
            det = (g.p_final > 0.2).to_numpy()
            for name, col in (("TTCG", "p_fake_ttcg"), ("cobweb", "p_fake_cobweb")):
                fid.append(dict(set=split, view=v, **fidelity_rows(name, det, g[col].to_numpy(), g.label.to_numpy())))
    ex = pd.read_json(config.RUNS / "concepts" / a.emb / "test_explanations.jsonl", lines=True).set_index("filename")
    sub = pd.read_csv(config.REPO / "submission" / "SideQuests_predictions_final.tsv", sep="\t").set_index("filename")["cm-score"]
    det = (sub.loc[ex.index] > 0.2).to_numpy()
    for name, col in (("TTCG", "ttcg_p_fake"), ("cobweb", "cobweb_p_fake")):
        fid.append(dict(set="NSA test", view="as delivered", **fidelity_rows(name, det, ex[col].astype(float).to_numpy())))
    pd.DataFrame(fid).to_csv(out / "i3_fidelity.csv", index=False, float_format="%.5f")

    summary = dict(
        c1a=c1a.groupby("k")[[c for c in c1a.columns if c.endswith(("attr_new", "auc_new", "attr_old", "auc_old"))]].mean().round(4).to_dict("index"),
        c1b=pd.DataFrame(c1b).groupby("k").mean(numeric_only=True).round(4).to_dict("index"),
        c2=pd.DataFrame(c2).groupby("task")[["auroc_novelty", "auroc_mahalanobis", "auroc_knn_distance"]].mean().round(4).to_dict("index"),
        c3=pd.DataFrame(c3).query("quintile > 0").groupby(["set", "view", "quintile"]).error_rate.mean().round(4).reset_index().to_dict("records"),
        c4=pd.DataFrame(c4).query("view == 'both'")[["ranking", "average_precision", "precision_at_2pct", "precision_at_5pct", "recall_at_10pct"]].round(4).to_dict("records"),
        i1_top_levels=[{k: r[k] for k in ("id", "size", "p_fake", "sources")} for r in at if r["depth"] <= 2],
        i3=pd.DataFrame(fid)[["set", "view", "scorer", "n", "agreement", "kappa", "kappa_lo", "kappa_hi"]].round(4).to_dict("records"))
    summary["i5"] = acoustics(D, out)
    summary.update(interpretability_extras(D, out, a))
    json.dump(summary, open(out / "summary.json", "w"), indent=1, default=str)
    from hearsay import provenance
    emb = config.CACHE / "emb" / a.emb
    provenance.write(out, inputs=[emb / "pooled.npy", emb / "index.parquet", *(config.RUNS / "concepts" / a.emb).glob("scores_*.parquet"),
                                  config.REPO / "submission" / "fusion.json"], seeds=a.seeds, fit_n=a.fit_n)
    print(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    main()
