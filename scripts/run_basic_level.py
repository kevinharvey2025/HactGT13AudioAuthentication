"""Track D5/D2: the basic level of real vs synthetic speech, found with a diffusion model (Wang, Singaravadivelan &
MacLellan, arXiv 2609.13047) and compared with the concept tree's basic level (scripts/run_concepts.py).

    python scripts/run_basic_level.py --emb xlsr1b_ft_last [--dims 32] [--steps 8000] [--n-eval 3000]

1. The same space as run_concepts.py: last-layer time-mean of the detector, standardized + PCA-whitened on
   train rows (both views).
2. An unconditional DDPM (hearsay/diffusion/ddpm.py, EpsMLP, T = 1000) trained on all train embeddings, both
   classes: its marginals at noise level t define "concepts at level t" (the mode that score ascent reaches
   from a noised point is the level-t prototype of that point).
3. For a stratified sample of val + holdout clips: I(label; level-t concept) per t (k-means of the Tweedie
   modes + a classifier estimate) for the real/fake label and for the source label (generator / corpus).
   The peak over t is the basic level: small t = fine detail, large t = coarse structure.
Writes runs/diffusion/concepts/<emb>/basic_level.json (and the DDPM checkpoint).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hearsay import config, splits  # noqa: E402
from hearsay.diffusion import prototypes as P  # noqa: E402
from hearsay.diffusion.ddpm import D2Model, DDPMConfig, ModePathConfig, train_ddpm  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", required=True)
    ap.add_argument("--dims", type=int, default=32)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--n-eval", type=int, default=3000)
    ap.add_argument("--clusters", type=int, default=32)
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
    d = idx[idx.done].merge(man, on="uid", how="left")
    lab = d[d.label >= 0].copy()
    lab["split"] = splits.shared_split(lab)

    def feats(rows):
        r = rows.to_numpy()
        return np.asarray(arr[np.sort(r)][:, -1, 0], np.float32)[np.argsort(np.argsort(r))]

    tr = lab[lab.split == "train"]
    scaler = StandardScaler().fit(feats(tr.row))
    pca = PCA(a.dims, whiten=True, random_state=0).fit(scaler.transform(feats(tr.row)))
    Z = lambda rows: pca.transform(scaler.transform(feats(rows))).astype(np.float32)

    cfg = DDPMConfig(steps=a.steps)
    net, sched, hist = train_ddpm(Z(tr.row), cfg, device=dev, log_every=1000)
    torch.save(net.state_dict(), out / "ddpm_uncond.pt")
    print("DDPM loss:", [(s, round(l, 4)) for s, l in hist], flush=True)
    d2 = D2Model(None, net, sched, ModePathConfig(), device=dev)

    ev = lab[lab.split.isin(["val", "holdout"]) & (lab.view == 0)]
    ev = ev.groupby("generator", group_keys=False).sample(frac=min(1.0, a.n_eval / len(ev)), random_state=0)
    z = Z(ev.row)
    t_grid = (10, 25, 50, 100, 150, 200, 300, 400, 600, 800)
    res = {"n_eval": len(ev), "dims": a.dims, "ddpm_steps": a.steps, "t_grid": t_grid, "curves": {}}
    res["curves"]["real_vs_fake"] = P.distinctiveness_curve(d2, z, ev.label.to_numpy(), t_grid=t_grid, n_clusters=a.clusters)
    best = max(res["curves"]["real_vs_fake"], key=lambda r: r["mi_clusters"])
    res["basic_level_t"] = best["t"]
    json.dump(res, open(out / "basic_level.json", "w"), indent=1, default=float)
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
