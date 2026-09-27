"""Track D5: prototypes and concept formation over the fine-tuned detector's embedding space.

    python scripts/run_concepts.py --emb xlsr2b_ft_last [--dims 32] [--fit-n 12000] [--acuity 0.25]

Grounding (plans/diffusion_cf_prompt.md, Addendum E): prototype theory (Posner & Keele 1968; Rosch 1975), category
utility and the basic level (Rosch et al. 1976; Gluck & Corter 1985), COBWEB/CLASSIT concept formation (Fisher 1987;
Gennari, Langley & Fisher 1989), diffusion <-> concept hierarchies (Wang, Singaravadivelan & MacLellan 2609.13047) and
prototype composition (Wang, Gupta, Zhu & MacLellan 2605.07078).

Input: cache/emb/<emb>/ (scripts/extract_ssl.py on the pool manifest, views 0/1, last layer). Space: the time-mean
of the detector's last layer, standardized and PCA-whitened on train rows. Outputs (runs/diffusion/concepts/<emb>/):
  1. prototypes: one diagonal Gaussian per source (each generator, each copy-synthesis vocoder, each real corpus,
     clean view) and per channel condition (augmented view);
  2. a COBWEB/CLASSIT concept tree over a stratified train sample (labels = sources);
  3. basic level, both definitions, on held-out (val) clips: the depth with the highest held-out
     I(X;C) = mean pmi(x; c) (DMCF's D(c)); the path node maximizing P(c)*KL(c || root) (the lab code's get_basic);
     plus I(fake; C_d), I(source; C_d) and the category utility of each depth's partition (levels.json);
  4. two prototype scores, evaluated like every detector on val / holdout / In-the-Wild (clean + augmented views):
     the concept-tree posterior P(fake | deepest concept with >= min_n members) and the prototype-mixture LLR
     log sum_fake pi_j N(x; m_j, S_j) - log sum_real pi_j N(x; m_j, S_j) (metrics.json, scores_<split>.parquet);
  5. checks: confound leakage (NMI of basic-level concepts with speaker, source, native sample rate, duration) and
     stability over insertion orders (adjusted Rand index of the basic-level partition across 3 trees);
  6. per-clip test explanations: basic-level concept (make-up, typicality percentile), deepest concept, the composed
     prototype explanation (greedy facility-location selection against the root baseline, K <= 3) and a novelty flag
     when no prototype explains the clip better than 99% of training clips (test_explanations.jsonl).
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import chi2
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hearsay import config, metrics, splits  # noqa: E402
from hearsay.concepts import ConceptTree, _mutual_info  # noqa: E402
from hearsay.diffusion import prototypes as P  # noqa: E402

LOG2PI = math.log(2 * math.pi)


def source_class(r):
    if r.family in ("lj_voice", "clone", "resynth"):
        return r.generator
    return {"real_lj": "real:ljspeech", "real_libri": "real:librispeech_cloned_speakers",
            "real_extra": "real:librispeech_other_speakers", "itw": "itw"}.get(r.family, r.family)


def channel_class(ch):
    ch = ch or "clean"
    return ch.split("+")[0].split("_")[0] if ch != "clean" else "clean"


def node_var(n, acuity):
    return np.maximum(n.m2 / max(n.n, 1), acuity ** 2)


def node_logpdf(n, X, acuity):
    v = node_var(n, acuity)
    return -0.5 * (((X - n.mean) ** 2 / v) + np.log(v) + LOG2PI).sum(-1)


def frontier(tree, depth):
    nodes = [tree.root]
    for _ in range(depth):
        nxt = []
        for n in nodes:
            nxt.extend(n.children if n.children else [n])
        nodes = nxt
    return nodes


def kl_to_root(n, root, acuity):
    vc, vr = node_var(n, acuity), node_var(root, acuity)
    return 0.5 * float((vc / vr + (root.mean - n.mean) ** 2 / vr - 1 + np.log(vr / vc)).sum())


def proto_llr(Z, fake, real):
    def mix(ps):
        w = np.log(np.array([p.n for p in ps], float))
        w -= logsumexp(w)
        return logsumexp(np.stack([p.logpdf_dims(Z).sum(1) for p in ps], 1) + w[None], 1)
    return mix(fake) - mix(real)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", required=True)
    ap.add_argument("--dims", type=int, default=32)
    ap.add_argument("--fit-n", type=int, default=12000)
    ap.add_argument("--acuity", type=float, default=0.25)
    ap.add_argument("--max-depth", type=int, default=10)
    ap.add_argument("--min-n", type=int, default=10)
    ap.add_argument("--n-heldout", type=int, default=3000)
    a = ap.parse_args()
    root = config.CACHE / "emb" / a.emb
    out = config.RUNS / "concepts" / a.emb
    out.mkdir(parents=True, exist_ok=True)
    meta = json.load(open(root / "meta.json"))
    idx = pd.read_parquet(root / "index.parquet")
    idx["row"] = np.arange(len(idx))
    arr = np.load(root / "pooled.npy", mmap_mode="r")
    man = pd.read_parquet(config.CACHE / f"{meta['manifest']}.parquet")
    tri_name = "triage_pool.parquet" if meta["manifest"] == "manifest_pool" else "triage.parquet"
    tri = pd.read_parquet(config.CACHE / tri_name)[["uid", "native_sr", "decoded_duration"]]
    d = idx[idx.done].merge(man, on="uid", how="left").merge(tri, on="uid", how="left")
    lab = d[d.label >= 0].copy()
    lab["split"] = splits.shared_split(lab)
    lab["source"] = [source_class(r) for r in lab.itertuples()]
    test = d[d.label < 0].copy()

    def feats(rows):
        r = rows.to_numpy()
        return np.asarray(arr[np.sort(r)][:, -1, 0], np.float32)[np.argsort(np.argsort(r))]

    tr = lab[lab.split == "train"]
    scaler = StandardScaler().fit(feats(tr.row))
    pca = PCA(a.dims, whiten=True, random_state=0).fit(scaler.transform(feats(tr.row)))
    Z = lambda rows: pca.transform(scaler.transform(feats(rows)))
    classes = sorted(tr.source.unique())
    fake_mask = np.array([not c.startswith("real:") for c in classes])
    cid = {c: i for i, c in enumerate(classes)}

    # 1. prototypes
    ztr = Z(tr.row)
    clean, augv = (tr.view == 0).to_numpy(), (tr.view > 0).to_numpy()
    src_protos = P.fit_direct(ztr[clean], tr.source[clean].to_numpy(), "generator")
    for p in src_protos:
        if p.name.startswith("real:"):
            p.kind = "bonafide"
    ch_protos = [p for p in P.fit_direct(ztr[augv], [channel_class(c) for c in tr.channel[augv]], "channel") if p.name != "clean"]
    protos = src_protos + ch_protos
    fake_p = [p for p in src_protos if p.kind == "generator"]
    real_p = [p for p in src_protos if p.kind == "bonafide"]
    # novelty reference: best prototype pmi (vs the whitened root N(0, I)) of training clips
    root_lp = lambda Zx: (-0.5 * (Zx ** 2 + LOG2PI)).sum(1)
    best_pmi = lambda Zx: np.max(np.stack([p.logpdf_dims(Zx).sum(1) for p in protos], 1), 1) - root_lp(Zx)
    novelty_thr = float(np.quantile(best_pmi(ztr[clean]), 0.01))

    # 2. concept trees (3 insertion orders; tree 0 is the reported one)
    samp = tr.groupby("source", group_keys=False).sample(frac=min(1.0, a.fit_n / len(tr)), random_state=0)
    zs, ys = Z(samp.row), samp.source.map(cid).to_numpy()
    trees = [ConceptTree(a.dims, n_labels=len(classes), acuity=a.acuity, max_depth=a.max_depth, seed=s).fit(zs, ys)
             for s in range(3)]
    tree = trees[0]

    # 3. basic level on held-out val clips (clean view)
    hv = lab[(lab.split == "val") & (lab.view == 0)]
    hv = hv.groupby("source", group_keys=False).sample(frac=min(1.0, a.n_heldout / max(1, len(hv))), random_state=0)
    Zh = Z(hv.row)
    paths = [tree.path(x) for x in Zh]
    yf, yg = hv.label.to_numpy(), hv.source.map(lambda s: cid.get(s, -1)).to_numpy()
    rows, cu = [], {r["depth"]: r for r in tree.levels(a.max_depth)}
    for depth in range(1, a.max_depth + 1):
        nodes = [p[min(depth, len(p) - 1)] for p in paths]
        fr = frontier(tree, depth)
        ntot = sum(n.n for n in fr)
        logmix = logsumexp(np.stack([math.log(n.n / ntot) + node_logpdf(n, Zh, a.acuity) for n in fr], 1), 1)
        own = np.array([node_logpdf(n, x[None], a.acuity)[0] for n, x in zip(nodes, Zh)])
        pmi = own - logmix
        ids = np.array([n.id for n in nodes])
        per_c = pd.Series(pmi).groupby(ids).mean()
        _, inv = np.unique(ids, return_inverse=True)
        tab = lambda y: np.array([np.bincount(inv[y == v], minlength=inv.max() + 1) for v in np.unique(y)]).T
        rows.append(dict(depth=depth, n_concepts_frontier=len(fr), heldout_I=float(pmi.mean()), mean_Dc=float(per_c.mean()),
                         I_fake=_mutual_info(tab(yf)), I_source=_mutual_info(tab(yg)),
                         category_utility=cu.get(depth, {}).get("category_utility")))
    basic_dmcf = max(rows, key=lambda r: r["heldout_I"])["depth"]
    basic_cu = max(rows, key=lambda r: r["category_utility"] or -1)["depth"]
    get_basic = [int(np.argmax([n.n / tree.root.n * kl_to_root(n, tree.root, a.acuity) for n in p])) for p in paths]

    # 5. checks: confound leakage at the basic level, stability over insertion orders
    bnodes = np.array([p[min(basic_dmcf, len(p) - 1)].id for p in paths])
    dur_bucket = pd.cut(hv.decoded_duration, [0, 3, 5, 8, 12, 1e9], labels=False).fillna(-1).to_numpy()
    leak = {k: round(float(normalized_mutual_info_score(v, bnodes)), 4) for k, v in (
        ("speaker", hv.speaker.astype(str).to_numpy()), ("source", hv.source.to_numpy()),
        ("native_sr", hv.native_sr.fillna(0).astype(int).to_numpy()), ("duration_bucket", dur_bucket),
        ("fake", yf))}
    other = [np.array([t.path(x)[min(basic_dmcf, len(t.path(x)) - 1)].id for x in Zh]) for t in trees[1:]]
    stability = [round(float(adjusted_rand_score(bnodes, o)), 4) for o in other]
    json.dump(dict(levels=rows, basic_level_heldout_I=basic_dmcf, basic_level_category_utility=basic_cu,
                   get_basic_depth_hist=np.bincount(get_basic).tolist(), confound_nmi_at_basic=leak,
                   stability_ari_insertion_orders=stability, classes=classes, n_fit=len(samp), dims=a.dims,
                   acuity=a.acuity, n_heldout=len(hv)), open(out / "levels.json", "w"), indent=1)
    print("basic level: held-out I(X;C) depth", basic_dmcf, "| category utility depth", basic_cu,
          "| get_basic hist", np.bincount(get_basic).tolist(), "| leakage", leak, "| stability", stability, flush=True)

    # 4. scores on val / holdout / ITW
    def tree_score(Zx):
        pf, ids = np.zeros(len(Zx)), np.zeros(len(Zx), int)
        for i, x in enumerate(Zx):
            deep = [n for n in tree.path(x) if n.n >= a.min_n][-1]
            dist = (deep.labels + 0.5) / (deep.labels.sum() + 0.5 * len(classes))
            pf[i], ids[i] = dist[fake_mask].sum(), deep.id
        return pf, ids

    res = {"basic_level_heldout_I": basic_dmcf, "basic_level_category_utility": basic_cu, "n_prototypes": len(protos),
           "confound_nmi_at_basic": leak, "stability_ari": stability}
    for split in ("val", "holdout", "itw"):
        rows_ = lab[lab.split == split]
        if not len(rows_):
            continue
        rows_ = rows_.groupby(["label", "view"], group_keys=False).sample(frac=min(1.0, 2 * a.n_heldout / len(rows_)), random_state=0)
        Zx = Z(rows_.row)
        pf, ids = tree_score(Zx)
        llr = proto_llr(Zx, fake_p, real_p)
        y = rows_.label.to_numpy()
        for vname, m in (("clean", (rows_.view == 0).to_numpy()), ("aug", (rows_.view > 0).to_numpy())):
            for sname, s in (("tree", pf), ("proto_llr", llr)):
                r = metrics.summary(y[m], s[m])
                res[f"{split}_{vname}_{sname}"] = {k: round(float(r[k]), 4) for k in ("auc", "eer", "min_dcf")}
        pd.DataFrame(dict(uid=rows_.uid, view=rows_.view, label=y, source=rows_.source, p_fake_tree=pf,
                          proto_llr=llr, concept=ids)).to_parquet(out / f"scores_{split}.parquet")
    Zt = Z(test.row)
    pf_t, ids_t = tree_score(Zt)
    llr_t = proto_llr(Zt, fake_p, real_p)
    pd.DataFrame(dict(uid=test.uid, filename=test.filename, p_fake_tree=pf_t, proto_llr=llr_t, concept=ids_t)).to_parquet(out / "scores_test.parquet")
    res.update(test_frac_tree_gt_0_5=float((pf_t > 0.5).mean()), test_frac_llr_gt_0=float((llr_t > 0).mean()))
    json.dump(res, open(out / "metrics.json", "w"), indent=1)
    print(json.dumps(res, indent=1))

    # 6. explanations for the test clips
    def makeup(n, k=3):
        dist = n.labels / max(1, n.labels.sum())
        return [dict(source=classes[j], share=round(float(dist[j]), 3)) for j in np.argsort(-dist)[:k] if dist[j] > 0]

    bp = best_pmi(Zt)
    with open(out / "test_explanations.jsonl", "w") as f:
        for i, r in enumerate(test.itertuples()):
            path = tree.path(Zt[i])
            b = path[min(basic_dmcf, len(path) - 1)]
            deep = [n for n in path if n.n >= a.min_n][-1]
            maha = float((((Zt[i] - b.mean) ** 2) / node_var(b, a.acuity)).sum())
            e = P.explain(Zt[i], protos, k=3)
            f.write(json.dumps(dict(
                filename=r.filename, concept_p_fake=round(float(pf_t[i]), 4), prototype_llr=round(float(llr_t[i]), 3),
                basic_level_concept=dict(id=int(b.id), depth=basic_dmcf, size=int(b.n), makeup=makeup(b),
                                         typicality=round(float(1 - chi2.cdf(maha, a.dims)), 3)),
                deepest_concept=dict(id=int(deep.id), size=int(deep.n), makeup=makeup(deep)),
                prototypes=[dict(name=s["name"], kind=s["kind"], share=round(s["share"], 3)) for s in e["selected"]],
                novel=bool(bp[i] < novelty_thr), closest_generator=e.get("closest_generator"),
                bonafide_margin=e.get("bonafide_margin"), summary=e["summary"])) + "\n")
    print("wrote", out / "test_explanations.jsonl")


if __name__ == "__main__":
    main()
