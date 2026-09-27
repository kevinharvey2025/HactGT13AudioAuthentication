"""Track D5: prototypes and concept formation over the fine-tuned detector's embedding space.

    python scripts/run_concepts.py --emb xlsr1b_ft_last [--dims 32] [--fit-n 12000] [--acuity 0.25]

Input: cache/emb/<emb>/ from scripts/extract_ssl.py on the pool manifest (views 0 and 1; last layer).
Space: time-mean of the last encoder layer (the vector the detector head reads), standardized and
PCA-whitened on train rows. Then:
  1. prototypes (prototype theory): one diagonal Gaussian per source class (each DiffSSD generator, each
     copy-synthesis vocoder, each bona fide corpus) and per channel condition (augmentation label);
  2. a COBWEB/CLASSIT concept hierarchy (hearsay/concepts.py) over a stratified train sample, labels =
     source classes, so every concept knows which generators/corpora it summarizes;
  3. the basic level: category utility and concept-label mutual information per depth;
  4. a prototype/concept classifier: P(fake) of the deepest concept with >= min_n members (a score for the
     fusion gate, evaluated on val / holdout / In-the-Wild like every other system);
  5. per-clip explanations for the test set: concept path, the basic-level concept's make-up, and the
     composed prototype explanation (greedy submodular selection, Wang et al. 2605.07078).
Writes runs/diffusion/concepts/<emb>/{metrics.json, levels.json, scores_<split>.parquet, test_explanations.jsonl}.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hearsay import config, metrics, splits  # noqa: E402
from hearsay.concepts import ConceptTree  # noqa: E402
from hearsay.diffusion import prototypes as P  # noqa: E402


def source_class(r):
    if r.family in ("lj_voice", "clone"):
        return r.generator
    if r.family == "resynth":
        return r.generator                      # resynth_<vocoder>
    return {"real_lj": "real:ljspeech", "real_libri": "real:librispeech_cloned_speakers",
            "real_extra": "real:librispeech_other_speakers", "itw": "itw"}.get(r.family, r.family)


def channel_class(ch):
    ch = ch or "clean"
    return ch.split("+")[0].split("_")[0] if ch != "clean" else "clean"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", required=True)
    ap.add_argument("--dims", type=int, default=32)
    ap.add_argument("--fit-n", type=int, default=12000)
    ap.add_argument("--acuity", type=float, default=0.25)
    ap.add_argument("--max-depth", type=int, default=10)
    ap.add_argument("--min-n", type=int, default=10)
    a = ap.parse_args()
    root = config.CACHE / "emb" / a.emb
    out = config.RUNS / "concepts" / a.emb
    out.mkdir(parents=True, exist_ok=True)
    meta = json.load(open(root / "meta.json"))
    idx = pd.read_parquet(root / "index.parquet")
    arr = np.load(root / "pooled.npy", mmap_mode="r")
    man = pd.read_parquet(config.CACHE / f"{meta['manifest']}.parquet")
    idx["row"] = np.arange(len(idx))                                   # row in pooled.npy
    d = idx[idx.done].merge(man, on="uid", how="left")
    lab = d[d.label >= 0].copy()
    lab["split"] = splits.shared_split(lab)
    test = d[d.label < 0].copy()
    feats = lambda rows: np.asarray(arr[np.sort(rows.to_numpy())][:, -1, 0], np.float32)[np.argsort(np.argsort(rows.to_numpy()))]

    tr = lab[lab.split == "train"]
    scaler = StandardScaler().fit(feats(tr["row"]))
    pca = PCA(a.dims, whiten=True, random_state=0).fit(scaler.transform(feats(tr["row"])))
    Z = lambda rows: pca.transform(scaler.transform(feats(rows)))  # rows: pooled.npy row numbers
    lab["source"] = [source_class(r) for r in lab.itertuples()]
    classes = sorted(lab.loc[lab.split == "train", "source"].unique())
    fake_mask = np.array([not (c.startswith("real:") or c == "itw") for c in classes])
    cid = {c: i for i, c in enumerate(classes)}

    # 1. prototypes (train rows): sources on the clean view, channels on the augmented view
    ztr = Z(tr["row"])
    trs = lab.loc[tr.index]
    clean, augv = (trs.view == 0).to_numpy(), (trs.view > 0).to_numpy()
    protos = (P.fit_direct(ztr[clean], trs.source[clean].to_numpy(), "generator") +
              P.fit_direct(ztr[augv], [channel_class(c) for c in trs.channel[augv]], "channel"))
    for p in protos:
        if p.name.startswith("real:"):
            p.kind = "bonafide"
    protos = [p for p in protos if not (p.kind == "channel" and p.name == "clean")]

    # 2. concept hierarchy on a stratified train sample (both views)
    samp = trs.groupby("source", group_keys=False).sample(frac=min(1.0, a.fit_n / len(trs)), random_state=0)
    zs = Z(samp["row"])
    tree = ConceptTree(a.dims, n_labels=len(classes), acuity=a.acuity, max_depth=a.max_depth, seed=0)
    tree.fit(zs, samp.source.map(cid).to_numpy())
    levels = tree.levels()
    json.dump(dict(levels=levels, classes=classes, n_fit=len(samp), dims=a.dims, acuity=a.acuity), open(out / "levels.json", "w"), indent=1)
    basic = max(levels, key=lambda r: r["category_utility"])["depth"] if levels else 1

    def score(rows):
        z = Z(rows["row"])
        pf, ids, basic_ids = np.zeros(len(z)), np.zeros(len(z), int), np.zeros(len(z), int)
        for i, x in enumerate(z):
            path = tree.path(x)
            deep = [n for n in path if n.n >= a.min_n][-1]
            dist = (deep.labels + 0.5) / (deep.labels.sum() + 0.5 * len(classes))
            pf[i], ids[i] = dist[fake_mask].sum(), deep.id
            basic_ids[i] = path[min(basic, len(path) - 1)].id
        return z, pf, ids, basic_ids

    res = {"basic_level_depth": basic, "n_prototypes": len(protos)}
    for split in ("val", "holdout", "itw"):
        rows = lab[lab.split == split]
        if not len(rows):
            continue
        _, pf, ids, _ = score(rows)
        y = rows.label.to_numpy()
        for vname, m in (("clean", (rows.view == 0).to_numpy()), ("aug", (rows.view > 0).to_numpy())):
            s = metrics.summary(y[m], pf[m])
            res[f"{split}_{vname}"] = {k: round(float(s[k]), 4) for k in ("auc", "eer", "min_dcf")}
        pd.DataFrame(dict(uid=rows.uid, view=rows.view, label=y, source=rows.source, p_fake=pf, concept=ids)
                     ).to_parquet(out / f"scores_{split}.parquet")
    z, pf, ids, bids = score(test)
    pd.DataFrame(dict(uid=test.uid, filename=test.filename, p_fake=pf, concept=ids, basic_concept=bids)).to_parquet(out / "scores_test.parquet")
    res["test_frac_gt_0_5"] = float((pf > 0.5).mean())
    json.dump(res, open(out / "metrics.json", "w"), indent=1)
    print(json.dumps(res, indent=1))

    # 5. explanations for the test clips
    nodes = {}
    stack = [tree.root]
    while stack:
        n = stack.pop()
        nodes[n.id] = n
        stack.extend(n.children)

    def makeup(n, k=3):
        dist = n.labels / max(1, n.labels.sum())
        top = np.argsort(-dist)[:k]
        return [dict(source=classes[j], share=round(float(dist[j]), 3)) for j in top if dist[j] > 0]

    with open(out / "test_explanations.jsonl", "w") as f:
        for i, r in enumerate(test.itertuples()):
            e = P.explain(z[i], protos, k=3)
            f.write(json.dumps(dict(
                filename=r.filename, concept_p_fake=round(float(pf[i]), 4),
                basic_level_concept=dict(id=int(bids[i]), size=int(nodes[bids[i]].n), makeup=makeup(nodes[bids[i]])),
                deepest_concept=dict(id=int(ids[i]), size=int(nodes[ids[i]].n), makeup=makeup(nodes[ids[i]])),
                prototypes=[dict(name=s["name"], kind=s["kind"], share=round(s["share"], 3)) for s in e["selected"]],
                closest_generator=e.get("closest_generator"), bonafide_margin=e.get("bonafide_margin"),
                summary=e["summary"])) + "\n")
    print("wrote", out / "test_explanations.jsonl")


if __name__ == "__main__":
    main()
