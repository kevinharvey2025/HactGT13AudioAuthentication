"""Concept formation (cobweb-private) and diffusion prototypes (TTCG) over the final detector's embedding space, and
the tests that say whether the resulting explanations can be trusted.

    python scripts/run_concepts.py --emb xlsr2b_d6rall_last [--dims 32] [--fit-n 12000] [--n-eval 1000]

Space (hearsay/concepts.py): the detector's time-mean last layer (the input of its linear head), standardized and
PCA-whitened on train rows. Concepts: cobweb-private trees fitted on a source-stratified train sample in three
insertion orders, labels = multi-hot [source | channel]. Prototypes: an unconditional DDPM over the same space and
TTCG (Wang et al., arXiv 2605.07078; hearsay/diffusion/ttcg.py) per query; each selected prototype is interpreted by
categorizing its mean in the tree.

Tests (all on held-out clips):
  scores        cobweb P(fake) and TTCG P(fake) on val / holdout / In-the-Wild, clean and augmented views
  basic level   two definitions: the path node with the largest closed-form D(c) (get_basic; also under stronger
                variance smoothing) and the depth where held-out pmi = log p_c(x) - log p_root(x) peaks; label
                agreement; closed-form vs Monte-Carlo expected PMI
  leakage       NMI and chance-adjusted MI between basic-level concepts and nuisance variables (speaker, sample rate,
                duration) vs the label
  stability     across insertion orders: adjusted Rand index of the basic-level partition, agreement of the basic
                concept's top source, Spearman of P(fake)
  noise-depth   Spearman(noise level of a selected prototype, depth of the concept it falls in); 2609.13047 predicts
                higher noise ~ shallower (more general) concepts
  faithfulness  for explanations that mix synthetic and real concepts: deleting the dimensions attributed to synthetic
                concepts must lower the detector's own logit more than deleting the real-concept dimensions, and each
                group must beat deleting as many random dimensions (paired, per clip)
Writes runs/diffusion/concepts/<emb>/: metrics.json, levels.json, faithfulness.json, scores_<split>.parquet,
test_explanations.jsonl, tree.json, ddpm_uncond.pt.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, normalized_mutual_info_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hearsay import config, metrics  # noqa: E402
from hearsay import concepts  # noqa: E402
from hearsay.concepts import Concepts, f32, node_key  # noqa: E402
from hearsay.diffusion import ttcg  # noqa: E402
from hearsay.diffusion.ddpm import DDPMConfig, EpsMLP, Schedule, train_ddpm  # noqa: E402


def levels_tests(cws, Zh, hv):
    """Basic level of held-out clips under both definitions, then leakage, stability and the closed-form check on
    each partition: 'closed_form' = get_basic (largest D(c) on the path), 'heldout' = informative (largest held-out
    pmi on the path)."""
    cw = cws[0]
    empty = np.zeros(cw.S + cw.C, np.float32)
    closed = [cw.basic(z) for z in Zh]                                   # (leaf, node)
    held = [cw.informative(z) for z in Zh]                               # (leaf, node, pmi)
    parts = {"closed_form": [b for _, b in closed], "heldout": [n for _, n, _ in held]}
    ids = {k: np.array([node_key(n) for n in v]) for k, v in parts.items()}
    epmi_by_depth, pmi_by_depth = {}, {}
    for (leaf, _), z in zip(closed, Zh):
        zr = cw.tree.root.log_prob(f32(z), empty)
        for n in Concepts.path(leaf):
            epmi_by_depth.setdefault(n.depth(), []).append(n.expected_pmi())
            pmi_by_depth.setdefault(n.depth(), []).append(n.log_prob(f32(z), empty) - zr)
    dur = pd.cut(hv.decoded_duration, [0, 3, 5, 8, 12, 1e9], labels=False).fillna(-1).to_numpy()
    nuis = {"fake": hv.label.to_numpy(), "source": hv.source.to_numpy(), "speaker": hv.speaker.astype(str).to_numpy(),
            "native_sr": hv.native_sr.fillna(0).astype(int).to_numpy(), "duration_bucket": dur}
    other = {"closed_form": lambda t, z: t.basic(z)[1], "heldout": lambda t, z: t.informative(z)[1]}
    out = {}
    for k, nodes in parts.items():
        agree = np.mean([cw.makeup(n)[2] if y == 1 else 1 - cw.makeup(n)[2] for n, y in zip(nodes, hv.label)])
        out[k] = dict(
            depth_hist=np.bincount([n.depth() for n in nodes]).tolist(), n_concepts=int(len(set(ids[k]))),
            size_median=float(np.median([n.count for n in nodes])), label_agreement=round(float(agree), 4),
            leakage_nmi={v: round(float(normalized_mutual_info_score(x, ids[k])), 4) for v, x in nuis.items()},
            leakage_ami={v: round(float(adjusted_mutual_info_score(x, ids[k])), 4) for v, x in nuis.items()},
            stability_ari=[round(float(adjusted_rand_score(ids[k], [node_key(other[k](t, z)) for z in Zh])), 4) for t in cws[1:]],
            stability_top_source_agreement=[round(float(np.mean([cw.makeup(n)[0][0]["source"] == t.makeup(other[k](t, z))[0][0]["source"]
                                                                 for n, z in zip(nodes, Zh)])), 4) for t in cws[1:]])
    out["closed_form"]["expected_pmi_by_depth"] = {int(d): round(float(np.mean(v)), 4) for d, v in sorted(epmi_by_depth.items())}
    out["closed_form"]["depth_mean_by_eval_prior_var"] = {str(pv): round(float(np.mean([l.get_basic(False, pv).depth() for l, _ in closed])), 3)
                                                           for pv in (-1.0, 0.25, 1.0, 4.0)}
    out["heldout"]["pmi_by_depth"] = {int(d): round(float(np.mean(v)), 4) for d, v in sorted(pmi_by_depth.items())}
    from cobweb import cobweb_continuous
    cobweb_continuous.set_random_seed(0)
    multi = [n for n in parts["heldout"] if n.count > 1]                # singletons make the check trivially exact
    check = [(n.expected_pmi(), n.expected_pmi_sampled(20000)) for n in multi[:: max(1, len(multi) // 20)]]
    rev = next((p / "GIT_REVISION" for p in Path(__import__("cobweb").__file__).resolve().parents
                if (p / "GIT_REVISION").exists()), None)
    return dict(n_heldout=len(hv), basic_level=out,
                p_fake_spearman_across_insertion_orders=[round(float(pd.Series(cw.p_fake(Zh)).corr(pd.Series(t.p_fake(Zh)), method="spearman")), 4)
                                                          for t in cws[1:]],
                expected_pmi_closed_vs_sampled=dict(n=len(check), max_rel_err=round(float(max(abs(a - b) / max(abs(a), 1e-9) for a, b in check)), 4)),
                cobweb_revision=rev.read_text().strip() if rev else "unknown")


def faithfulness(space, comps, H, w, b, n_rand=20, seed=0):
    """Does an explanation point at what the detector uses? Per clip (clean view), each dimension belongs to its
    dominant selected prototype, and that prototype's concept is synthetic (share > 0.5) or real. Deleting a group of
    dimensions (set to the train mean, 0 in whitened units) changes the detector's own logit w.h + b by w.dh.
    Only explanations that mix synthetic and real concepts can be tested: when every prototype is synthetic, the
    evidence is every dimension and any deletion of that size is the same deletion. For mixed explanations:
      contrast    deleting the synthetic evidence lowers the logit more than deleting the real evidence
      vs random   each group moves the logit in its direction more than deleting as many random dimensions."""
    rng = np.random.default_rng(seed)
    rows, full, recon = [], [], []
    for comp, h in zip(comps, H):
        z = space(h[None])[0]
        full.append(float(h @ w + b))
        recon.append(float(space.reconstruct(z[None])[0] @ w + b))
        if not comp["selected"]:
            continue
        W = np.stack([s["weight"] for s in comp["selected"]])
        fake_dim = np.array([s["concept_p_fake"] > 0.5 for s in comp["selected"]])[W.argmax(0)]
        effect = lambda m: float(space.delta_h(np.where(m, -z, 0.0))[0] @ w)  # noqa: E731
        r = dict(logit=full[-1], n_fake_dims=int(fake_dim.sum()), mixed=bool(0 < fake_dim.sum() < len(z)))
        if r["mixed"]:
            r["del_fake"], r["del_real"] = effect(fake_dim), effect(~fake_dim)
            for name, k in (("rand_fake", int(fake_dim.sum())), ("rand_real", int((~fake_dim).sum()))):
                r[name] = float(np.mean([effect(np.isin(np.arange(len(z)), rng.choice(len(z), k, replace=False))) for _ in range(n_rand)]))
        rows.append(r)
    d = pd.DataFrame(rows)
    sd = float(np.std(full)) + 1e-9
    out = dict(n_queries=len(d), n_mixed=int(d.mixed.sum()), logit_sd=sd,
               head_r2_in_concept_space=float(np.corrcoef(full, recon)[0, 1] ** 2))
    m = d[d.mixed]

    def summarize(x):
        boots = [rng.choice(x, len(x)).mean() for _ in range(2000)]
        return dict(mean=float(x.mean()), ci=[float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                    share_positive=float((x > 0).mean()))
    if len(m) > 10:
        out["contrast_synthetic_vs_real_evidence"] = summarize(((m.del_real - m.del_fake) / sd).to_numpy())
        out["synthetic_evidence_vs_random"] = summarize(((m.rand_fake - m.del_fake) / sd).to_numpy())
        out["real_evidence_vs_random"] = summarize(((m.del_real - m.rand_real) / sd).to_numpy())
    return out, d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", required=True)
    ap.add_argument("--dims", type=int, default=32)
    ap.add_argument("--fit-n", type=int, default=12000)
    ap.add_argument("--n-eval", type=int, default=1000, help="clips per split (and view) for scores")
    ap.add_argument("--ddpm-steps", type=int, default=8000)
    ap.add_argument("--ttcg-starts", type=int, default=32)
    ap.add_argument("--levels-only", action="store_true", help="trees + basic-level tests only (CPU); updates levels.json")
    ap.add_argument("--faithfulness-only", action="store_true",
                    help="tree + cached DDPM + TTCG on the clean evaluation queries (CPU is fine); updates faithfulness.json")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = config.RUNS / "concepts" / a.emb
    out.mkdir(parents=True, exist_ok=True)
    D = concepts.load(a.emb, a.dims)
    meta, lab, test, feats, space, Z = D.meta, D.lab, D.test, D.feats, D.space, D.Z
    tr = lab[lab.split == "train"]
    sources, channels = sorted(tr.source.unique()), sorted(tr.chan.unique())
    sid, cidx = {s: i for i, s in enumerate(sources)}, {c: i for i, c in enumerate(channels)}

    # 1. concept trees, three insertion orders; tree 0 is the reported one
    samp = tr.groupby("source", group_keys=False).sample(frac=min(1.0, a.fit_n / len(tr)), random_state=0)
    zs = Z(samp.row)
    cws = [Concepts(a.dims, sources, channels, seed=s).fit(zs, samp.source.map(sid).to_numpy(), samp.chan.map(cidx).to_numpy())
           for s in range(1 if a.faithfulness_only else 3)]
    cw = cws[0]
    if not a.faithfulness_only:
        cw.tree.dump_json(str(out / "tree.json"))

    # 2. basic level on held-out val clips (clean view)
    hv = lab[(lab.split == "val") & (lab.view == 0)]
    hv = hv.groupby("source", group_keys=False).sample(frac=min(1.0, a.n_eval / max(1, len(hv))), random_state=0)
    levels = {} if a.faithfulness_only else dict(levels_tests(cws, Z(hv.row), hv), sources=sources, channels=channels,
                                                  n_fit=len(samp), dims=a.dims)
    print("basic level:", json.dumps({k: v for k, v in levels.items() if k not in ("sources", "channels")}), flush=True)
    if a.levels_only:
        prev = json.load(open(out / "levels.json")) if (out / "levels.json").exists() else {}
        levels["noise_level_vs_depth"] = prev.get("noise_level_vs_depth")
        json.dump(levels, open(out / "levels.json", "w"), indent=1, default=str)
        return

    # 3. unconditional DDPM over the same space (both classes, all views)
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
            for z, c in zip(Zx[k: k + 64], ttcg.discover(net, sched, Zx[k: k + 64], tcfg, device=dev)):
                comp = ttcg.select_and_compose(z, c, tcfg)
                for s in comp["selected"]:
                    leaf, b = cw.basic(s["mean"])
                    _, node, pmi = cw.informative(s["mean"])
                    s["basic_depth"], s["leaf_depth"], s["concept_depth"], s["concept_size"] = b.depth(), leaf.depth(), node.depth(), int(node.count)
                    s["sources"], s["channels"], s["concept_p_fake"] = cw.makeup(node)
                    s["basic_p_fake"] = cw.makeup(b)[2]
                tot = sum(s["share"] for s in comp["selected"])
                for key, col in (("p_fake", "concept_p_fake"), ("p_fake_basic", "basic_p_fake")):
                    comp[key] = (float(sum(s["share"] * s[col] for s in comp["selected"]) / tot)
                                 if comp["selected"] else float("nan"))
                res.append(comp)
        return res

    # 4. gate scores per split and view; clean-view compositions kept for the faithfulness test
    res, corr_rows, faith_comps, faith_rows = {}, [], [], []
    for split in ("val", "holdout", "itw"):
        rows = lab[lab.split == split]
        if not len(rows):
            continue
        rows = rows.groupby(["label", "view"], group_keys=False).sample(frac=min(1.0, 2 * a.n_eval / len(rows)), random_state=0)
        if a.faithfulness_only:
            clean = rows[rows.view == 0]
            faith_comps += ttcg_run(Z(clean.row))
            faith_rows.append(clean.row)
            continue
        Zx = Z(rows.row)
        p_cw = cw.p_fake(Zx)
        comps = ttcg_run(Zx)
        p_tt = np.array([c["p_fake"] for c in comps])
        p_tb = np.array([c["p_fake_basic"] for c in comps])
        corr_rows += [(s["t"], s["concept_depth"], s["basic_depth"], s["leaf_depth"]) for c in comps for s in c["selected"]]
        clean = (rows.view == 0).to_numpy()
        faith_comps += [c for c, m in zip(comps, clean) if m]
        faith_rows.append(rows.row[clean])
        y = rows.label.to_numpy()
        for vname, m in (("clean", clean), ("aug", ~clean)):
            for sname, s in (("cobweb", p_cw), ("ttcg", p_tt), ("ttcg_closed_form_basic", p_tb)):
                ok = m & np.isfinite(s)
                r = metrics.summary(y[ok], s[ok])
                res[f"{split}_{vname}_{sname}"] = {k: round(float(r[k]), 4) for k in ("n", "auc", "eer", "min_dcf")}
        pd.DataFrame(dict(uid=rows.uid, view=rows.view, label=y, source=rows.source, p_fake_cobweb=p_cw,
                          p_fake_ttcg=p_tt, p_fake_ttcg_basic=p_tb)).to_parquet(out / f"scores_{split}.parquet")
        print(split, {k: v for k, v in res.items() if k.startswith(split)}, flush=True)

    # 5. noise level <-> concept depth (full runs only)
    cr = pd.DataFrame(corr_rows, columns=["t", "concept_depth", "basic_depth", "leaf_depth"])
    levels["noise_level_vs_depth"] = dict(
        spearman_t_concept_depth=round(float(cr.t.corr(cr.concept_depth, method="spearman")), 4),
        spearman_t_basic_depth=round(float(cr.t.corr(cr.basic_depth, method="spearman")), 4),
        spearman_t_leaf_depth=round(float(cr.t.corr(cr.leaf_depth, method="spearman")), 4),
        mean_concept_depth_by_t=cr.groupby("t").concept_depth.mean().round(3).to_dict(),
        mean_leaf_depth_by_t=cr.groupby("t").leaf_depth.mean().round(3).to_dict(), n_prototypes=len(cr)) if len(cr) else None
    if not a.faithfulness_only:
        json.dump(levels, open(out / "levels.json", "w"), indent=1, default=str)

    # 6. faithfulness against the detector's own linear head (logit = (w_fake - w_real) . h + (b_fake - b_real))
    sd = torch.load(config.REPO / meta["checkpoint"], map_location="cpu")
    w = (sd["head.weight"][0] - sd["head.weight"][1]).float().numpy()
    b = float(sd["head.bias"][0] - sd["head.bias"][1])
    H = feats(pd.concat(faith_rows))
    faith, faith_df = faithfulness(space, faith_comps, H, w, b)
    faith_df.to_parquet(out / "faithfulness_rows.parquet")
    ft = Path(meta["checkpoint"]).parent
    ep = json.load(open(config.REPO / ft / "best.json"))["epoch"]
    q = lab.loc[pd.concat(faith_rows).index]
    ref = pd.concat([pd.read_parquet(config.REPO / ft / f"{s}_epoch{ep}.parquet") for s in ("val", "holdout", "itw")]).set_index("uid").score_clean
    m = q.uid.isin(ref.index).to_numpy()
    faith["head_vs_detector_spearman"] = float(pd.Series(H[m] @ w + b).corr(pd.Series(ref.loc[q.uid[m]].to_numpy()), method="spearman"))
    json.dump(faith, open(out / "faithfulness.json", "w"), indent=1)
    print("faithfulness:", json.dumps(faith), flush=True)
    if a.faithfulness_only:
        if (out / "metrics.json").exists():
            prev = json.load(open(out / "metrics.json"))
            prev["faithfulness"] = faith
            json.dump(prev, open(out / "metrics.json", "w"), indent=1, default=str)
        return

    # 7. test clips: scores + explanations
    Zt = Z(test.row)
    p_cw_t = cw.p_fake(Zt)
    comps_t = ttcg_run(Zt)
    with open(out / "test_explanations.jsonl", "w") as f:
        for i, r in enumerate(test.itertuples()):
            leaf, bnode = cw.basic(Zt[i])
            _, node, pmi = cw.informative(Zt[i])
            src, ch, pf = cw.makeup(node)
            c = comps_t[i]
            protos = [dict(noise_level=s["t"], share=round(s["share"], 3), concept_depth=s["concept_depth"],
                           concept_size=s["concept_size"], concept_sources=s["sources"], concept_channels=s["channels"],
                           concept_p_fake=round(s["concept_p_fake"], 3)) for s in c["selected"]]
            summary = "; ".join(f"t={p['noise_level']} prototype ({p['share']:.0%} of dims) ~ {p['concept_size']}-clip concept of "
                                f"{', '.join(x['source'] for x in p['concept_sources'][:2])}"
                                f"{' under ' + p['concept_channels'][0]['channel'] if p['concept_channels'] and p['concept_channels'][0]['channel'] != 'clean' else ''}"
                                for p in protos)
            f.write(json.dumps(dict(filename=r.filename, cobweb_p_fake=round(float(p_cw_t[i]), 4),
                                    ttcg_p_fake=round(float(c["p_fake"]), 4) if np.isfinite(c["p_fake"]) else None,
                                    concept=dict(depth=node.depth(), size=int(node.count), pmi=round(pmi, 3), p_fake=round(pf, 3),
                                                 sources=src, channels=ch),
                                    closed_form_basic=dict(depth=bnode.depth(), size=int(bnode.count),
                                                           expected_pmi=round(bnode.expected_pmi(), 3)),
                                    diffusion_prototypes=protos, summary=summary)) + "\n")
    res.update(levels={k: v for k, v in levels.items() if k not in ("sources", "channels")}, faithfulness=faith,
               test_frac_cobweb_gt_0_5=float((p_cw_t > 0.5).mean()),
               test_frac_ttcg_gt_0_5=float(np.nanmean([c["p_fake"] > 0.5 for c in comps_t])))
    json.dump(res, open(out / "metrics.json", "w"), indent=1, default=str)
    from hearsay import provenance
    emb = config.CACHE / "emb" / a.emb
    provenance.write(out, inputs=[emb / "pooled.npy", emb / "index.parquet", config.REPO / meta["checkpoint"]], dims=a.dims, fit_n=a.fit_n)
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
