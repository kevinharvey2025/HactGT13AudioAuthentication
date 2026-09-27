"""Track D5: concept formation (cobweb-private) + diffusion prototypes (TTCG) over the detector's embedding space.

    python scripts/run_concepts.py --emb xlsr2b_d6rall_last [--dims 32] [--fit-n 12000] [--n-eval 1500]

Ground truth for concepts is the lab's COBWEB implementation (github.com/Teachable-AI-Lab/cobweb-private,
`cobweb.cobweb_continuous.CobwebContinuousTree`, built from the revision in /ptmp/.../ext/cobweb-private/GIT_REVISION):
incremental concept formation with diagonal-Gaussian concepts, prediction by best-first expansion (`predict`),
categorization (`get_leaf`), and the basic level as the node on a clip's path with the highest closed-form expected
PMI against the root (`get_basic` / `expected_pmi`, i.e. D(c) of arXiv 2609.13047). Diffusion prototypes follow Zekun
Wang et al., arXiv 2605.07078 (hearsay/diffusion/ttcg.py): per-query mode ascent on an unconditional DDPM over the same
space at t = 50..400, Tweedie means, Hutchinson covariances (2609.13047 Eq. 9), facility-location selection (K <= 3)
and product-of-experts composition; each selected prototype is interpreted by categorizing its mean in the tree.

Space: time-mean of the detector's last layer, standardized + PCA-whitened on train rows. Labels given to the tree:
multi-hot [source one-hot | channel one-hot] (source = generator / copy-synthesis vocoder / real corpus; channel =
the augmentation applied to that view), so every concept reports both its sources and its channel conditions.
Outputs (runs/diffusion/concepts/<emb>/): metrics.json (gate scores: cobweb predict P(fake), TTCG P(fake), on val /
holdout / In-the-Wild, clean and augmented views), levels.json (basic-level depths, expected PMI by depth, confound
leakage, insertion-order stability, noise-level <-> depth correspondence), scores_<split>.parquet,
test_explanations.jsonl, tree.json (the fitted tree, dump_json).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hearsay import config, metrics, splits  # noqa: E402
from hearsay.diffusion import ttcg  # noqa: E402
from hearsay.diffusion.ddpm import DDPMConfig, EpsMLP, Schedule, train_ddpm  # noqa: E402

from cobweb.cobweb_continuous import CobwebContinuousTree  # noqa: E402  (cobweb-private)


def source_class(r):
    if r.family in ("lj_voice", "clone", "resynth"):
        return r.generator
    return {"real_lj": "real:ljspeech", "real_libri": "real:librispeech_cloned_speakers",
            "real_extra": "real:librispeech_other_speakers", "itw": "itw"}.get(r.family, r.family)


def channel_class(ch):
    ch = ch or "clean"
    return ch.split("+")[0].split("_")[0] if ch != "clean" else "clean"


def f32(x):
    return np.ascontiguousarray(x, dtype=np.float32)


def node_key(n):
    """Stable identity of a cobweb-private node (nanobind may return a fresh wrapper for the same C++ node)."""
    return hash((n.depth(), round(float(n.count), 3), np.asarray(n.mean, np.float32).round(5).tobytes()))


class Concepts:
    """The fitted cobweb-private tree plus the label vocabulary."""

    def __init__(self, dims, sources, channels, seed=0):
        self.sources, self.channels = sources, channels
        self.S, self.C = len(sources), len(channels)
        self.fake = np.array([not s.startswith("real:") for s in sources])
        self.tree = CobwebContinuousTree(size=dims, num_labels=self.S + self.C)   # library defaults
        self.seed = seed

    def fit(self, Z, src, ch):
        lab = np.zeros((len(Z), self.S + self.C), np.float32)
        lab[np.arange(len(Z)), src] = 1.0
        lab[np.arange(len(Z)), self.S + ch] = 1.0
        for i in np.random.default_rng(self.seed).permutation(len(Z)):
            self.tree.ifit(f32(Z[i]), lab[i])
        return self

    def p_fake(self, Z, max_nodes=300):
        empty = np.zeros(self.S + self.C, np.float32)
        out = np.zeros(len(Z))
        for i, z in enumerate(Z):
            dist = np.asarray(self.tree.predict(f32(z), empty, max_nodes, False))[: self.S]
            out[i] = dist[self.fake].sum() / max(dist.sum(), 1e-12)
        return out

    def basic(self, z):
        leaf = self.tree.get_leaf(f32(z), np.zeros(self.S + self.C, np.float32))
        return leaf, leaf.get_basic()

    def makeup(self, node, k=3):
        lc = np.asarray(node.label_counts, float)
        src, ch = lc[: self.S], lc[self.S:]
        s = [dict(source=self.sources[j], share=round(float(src[j] / max(src.sum(), 1e-12)), 3)) for j in np.argsort(-src)[:k] if src[j] > 0]
        c = [dict(channel=self.channels[j], share=round(float(ch[j] / max(ch.sum(), 1e-12)), 3)) for j in np.argsort(-ch)[:2] if ch[j] > 0]
        pf = float(src[self.fake].sum() / max(src.sum(), 1e-12))
        return s, c, pf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", required=True)
    ap.add_argument("--dims", type=int, default=32)
    ap.add_argument("--fit-n", type=int, default=12000)
    ap.add_argument("--n-eval", type=int, default=1000, help="clips per split (and view) for scores")
    ap.add_argument("--ddpm-steps", type=int, default=8000)
    ap.add_argument("--ttcg-starts", type=int, default=32)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = config.CACHE / "emb" / a.emb
    out = config.RUNS / "concepts" / a.emb
    out.mkdir(parents=True, exist_ok=True)
    meta = json.load(open(root / "meta.json"))
    idx = pd.read_parquet(root / "index.parquet")
    idx["row"] = np.arange(len(idx))
    arr = np.load(root / "pooled.npy", mmap_mode="r")
    man = pd.read_parquet(config.CACHE / f"{meta['manifest']}.parquet")
    tri = pd.read_parquet(config.CACHE / "triage_pool.parquet")[["uid", "native_sr", "decoded_duration"]]
    d = idx[idx.done].merge(man, on="uid", how="left").merge(tri, on="uid", how="left")
    lab = d[d.label >= 0].copy()
    lab["split"] = splits.shared_split(lab)
    lab["source"] = [source_class(r) for r in lab.itertuples()]
    lab["chan"] = [channel_class(c) for c in lab.channel]
    test = d[d.label < 0].copy()

    def feats(rows):
        r = rows.to_numpy()
        return np.asarray(arr[np.sort(r)][:, -1, 0], np.float32)[np.argsort(np.argsort(r))]

    tr = lab[lab.split == "train"]
    scaler = StandardScaler().fit(feats(tr.row))
    pca = PCA(a.dims, whiten=True, random_state=0).fit(scaler.transform(feats(tr.row)))
    Z = lambda rows: pca.transform(scaler.transform(feats(rows))).astype(np.float32)
    sources = sorted(tr.source.unique())
    channels = sorted(tr.chan.unique())
    sid, cidx = {s: i for i, s in enumerate(sources)}, {c: i for i, c in enumerate(channels)}

    # 1. concept trees (cobweb-private), three insertion orders; tree 0 is reported
    samp = tr.groupby("source", group_keys=False).sample(frac=min(1.0, a.fit_n / len(tr)), random_state=0)
    zs = Z(samp.row)
    trees = [Concepts(a.dims, sources, channels, seed=s).fit(zs, samp.source.map(sid).to_numpy(), samp.chan.map(cidx).to_numpy())
             for s in range(3)]
    cw = trees[0]
    cw.tree.dump_json(str(out / "tree.json"))

    # 2. basic level on held-out val clips (clean view): depth, expected PMI, leakage, stability
    hv = lab[(lab.split == "val") & (lab.view == 0)]
    hv = hv.groupby("source", group_keys=False).sample(frac=min(1.0, a.n_eval / max(1, len(hv))), random_state=0)
    Zh = Z(hv.row)
    info = [cw.basic(z) for z in Zh]
    bid = np.array([node_key(b) for _, b in info])
    depths = np.array([b.depth() for _, b in info])
    leaf_depths = np.array([l.depth() for l, _ in info])
    epmi = np.array([b.expected_pmi() for _, b in info])
    dur_bucket = pd.cut(hv.decoded_duration, [0, 3, 5, 8, 12, 1e9], labels=False).fillna(-1).to_numpy()
    leak = {k: round(float(normalized_mutual_info_score(v, bid)), 4) for k, v in (
        ("speaker", hv.speaker.astype(str).to_numpy()), ("source", hv.source.to_numpy()),
        ("native_sr", hv.native_sr.fillna(0).astype(int).to_numpy()), ("duration_bucket", dur_bucket),
        ("fake", hv.label.to_numpy()))}
    stab = [round(float(adjusted_rand_score(bid, np.array([node_key(t.basic(z)[1]) for z in Zh]))), 4) for t in trees[1:]]
    purity = np.mean([cw.makeup(b)[2] if l == 1 else 1 - cw.makeup(b)[2] for (_, b), l in zip(info, hv.label)])
    rev = next((p / "GIT_REVISION" for p in Path(__import__("cobweb").__file__).resolve().parents if (p / "GIT_REVISION").exists()), None)
    levels = dict(n_heldout=len(hv), basic_depth_hist=np.bincount(depths).tolist(), leaf_depth_median=float(np.median(leaf_depths)),
                  basic_expected_pmi_mean=float(epmi.mean()), basic_label_agreement=float(purity),
                  confound_nmi_at_basic=leak, stability_ari_insertion_orders=stab, sources=sources, channels=channels,
                  n_fit=len(samp), dims=a.dims, cobweb_revision=rev.read_text().strip() if rev else "unknown")
    print("basic level:", {k: v for k, v in levels.items() if k not in ("sources", "channels")}, flush=True)

    # 3. unconditional DDPM over the same space (both classes, all views), for TTCG
    ddpm_path = out / "ddpm_uncond.pt"
    cfg = DDPMConfig(steps=a.ddpm_steps)
    if ddpm_path.exists():
        net = EpsMLP(a.dims, cfg.width, cfg.depth, dropout=0.0).to(dev)
        net.load_state_dict(torch.load(ddpm_path, map_location=dev))
        net.eval()
        sched = Schedule(cfg.T)
    else:
        net, sched, hist = train_ddpm(Z(tr.row), cfg, device=dev, log_every=2000)
        torch.save(net.state_dict(), ddpm_path)
        print("DDPM loss:", [(s, round(l, 4)) for s, l in hist], flush=True)
    tcfg = ttcg.TTCGConfig(starts=a.ttcg_starts)

    def ttcg_run(Zx):
        """-> per query: composition + interpretation of each selected prototype by the concept tree."""
        res = []
        for k in range(0, len(Zx), 64):
            cands = ttcg.discover(net, sched, Zx[k: k + 64], tcfg, device=dev)
            for z, c in zip(Zx[k: k + 64], cands):
                comp = ttcg.select_and_compose(z, c, tcfg)
                for s in comp["selected"]:
                    leaf, b = cw.basic(s["mean"])
                    s["basic_depth"], s["leaf_depth"] = b.depth(), leaf.depth()
                    s["sources"], s["channels"], s["concept_p_fake"] = cw.makeup(b)
                comp["p_fake"] = (float(sum(s["share"] * s["concept_p_fake"] for s in comp["selected"]) /
                                        max(sum(s["share"] for s in comp["selected"]), 1e-12)) if comp["selected"] else float("nan"))
                res.append(comp)
        return res

    # 4. scores for the gate: cobweb predict and TTCG composition, per split and view
    res = {"levels": {k: v for k, v in levels.items() if k not in ("sources", "channels")}}
    corr_rows = []
    for split in ("val", "holdout", "itw"):
        rows = lab[lab.split == split]
        if not len(rows):
            continue
        rows = rows.groupby(["label", "view"], group_keys=False).sample(frac=min(1.0, 2 * a.n_eval / len(rows)), random_state=0)
        Zx = Z(rows.row)
        p_cw = cw.p_fake(Zx)
        comps = ttcg_run(Zx)
        p_tt = np.array([c["p_fake"] for c in comps])
        for c in comps:
            corr_rows += [(s["t"], s["basic_depth"], s["leaf_depth"]) for s in c["selected"]]
        y = rows.label.to_numpy()
        for vname, m in (("clean", (rows.view == 0).to_numpy()), ("aug", (rows.view > 0).to_numpy())):
            for sname, s in (("cobweb", p_cw), ("ttcg", p_tt)):
                ok = m & np.isfinite(s)
                r = metrics.summary(y[ok], s[ok])
                res[f"{split}_{vname}_{sname}"] = {k: round(float(r[k]), 4) for k in ("auc", "eer", "min_dcf")}
        pd.DataFrame(dict(uid=rows.uid, view=rows.view, label=y, source=rows.source, p_fake_cobweb=p_cw,
                          p_fake_ttcg=p_tt)).to_parquet(out / f"scores_{split}.parquet")
        print(split, {k: v for k, v in res.items() if k.startswith(split)}, flush=True)
    # noise level <-> concept depth (2609.13047: higher noise ~ shallower concepts)
    cr = pd.DataFrame(corr_rows, columns=["t", "basic_depth", "leaf_depth"])
    levels["noise_level_vs_depth"] = dict(
        spearman_t_basic_depth=round(float(cr.t.corr(cr.basic_depth, method="spearman")), 4),
        spearman_t_leaf_depth=round(float(cr.t.corr(cr.leaf_depth, method="spearman")), 4),
        mean_leaf_depth_by_t=cr.groupby("t").leaf_depth.mean().round(3).to_dict(),
        mean_basic_depth_by_t=cr.groupby("t").basic_depth.mean().round(3).to_dict(), n_prototypes=len(cr))
    res["levels"]["noise_level_vs_depth"] = levels["noise_level_vs_depth"]
    json.dump(levels, open(out / "levels.json", "w"), indent=1, default=str)

    # 5. test clips: scores + explanations
    Zt = Z(test.row)
    p_cw_t = cw.p_fake(Zt)
    comps_t = ttcg_run(Zt)
    with open(out / "test_explanations.jsonl", "w") as f:
        for i, r in enumerate(test.itertuples()):
            leaf, b = cw.basic(Zt[i])
            src, ch, pf = cw.makeup(b)
            c = comps_t[i]
            protos = [dict(noise_level=s["t"], share=round(s["share"], 3), concept_depth=s["basic_depth"],
                           concept_sources=s["sources"], concept_channels=s["channels"],
                           concept_p_fake=round(s["concept_p_fake"], 3)) for s in c["selected"]]
            summary = "; ".join(f"t={p['noise_level']} prototype ({p['share']:.0%} of dims) ~ concept of "
                                f"{', '.join(x['source'] for x in p['concept_sources'][:2])}"
                                f"{' under ' + p['concept_channels'][0]['channel'] if p['concept_channels'] and p['concept_channels'][0]['channel'] != 'clean' else ''}"
                                for p in protos)
            f.write(json.dumps(dict(filename=r.filename, cobweb_p_fake=round(float(p_cw_t[i]), 4),
                                    ttcg_p_fake=round(float(c["p_fake"]), 4) if np.isfinite(c["p_fake"]) else None,
                                    basic_level_concept=dict(depth=b.depth(), size=int(b.count), expected_pmi=round(b.expected_pmi(), 3),
                                                             sources=src, channels=ch),
                                    diffusion_prototypes=protos, summary=summary)) + "\n")
    res.update(test_frac_cobweb_gt_0_5=float((p_cw_t > 0.5).mean()),
               test_frac_ttcg_gt_0_5=float(np.nanmean([c["p_fake"] > 0.5 for c in comps_t])))
    json.dump(res, open(out / "metrics.json", "w"), indent=1, default=str)
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
