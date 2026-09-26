"""Build an augmented training manifest: dev rows + one transformed copy per dev row.

Each dev row gets one TRAIN_CONDITIONS chain chosen deterministically from its
content hash, so both classes receive the same transform distribution. Copies
inherit label, group_id and split ("dev"), so derivatives never cross into
holdout. Holdout rows are passed through untouched (never augmented here).
Transform settings differ from the robustness-evaluation settings.

  python scripts/dsp_augment_train.py --manifest runs/dsp/manifests/pool.tsv \
      --out runs/dsp/manifests/pool_aug.tsv
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hearsay_dsp.config import load_config  # noqa: E402
from hearsay_dsp.evaluation.augment import TRAIN_CONDITIONS, make_variant  # noqa: E402
from hearsay_dsp.io.decode import file_sha256  # noqa: E402
from hearsay_dsp.io.manifest import read_manifest  # noqa: E402


def _one(path: str, cfg: dict, out_root: Path, seed: int):
    sha = file_sha256(path)
    names = sorted(TRAIN_CONDITIONS)
    cond = names[int(hashlib.sha256(f"{seed}:cond:{sha}".encode()).hexdigest(), 16) % len(names)]
    return cond, make_variant(path, sha, cond, TRAIN_CONDITIONS[cond], cfg, out_root, seed)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=str(ROOT / "configs/dsp.yaml"))
    ap.add_argument("--n-jobs", type=int, default=7)
    args = ap.parse_args()
    cfg = load_config(args.config)
    seed = cfg["run"]["seed"]
    df = read_manifest(args.manifest)
    dev = df[df["split"] == "dev"]
    out_root = ROOT / cfg["run"]["cache_dir"] / "augmented"
    res = Parallel(n_jobs=args.n_jobs)(delayed(_one)(p, cfg, out_root, seed) for p in dev["path"])
    aug = dev.copy()
    aug["augmentation"] = [c for c, _ in res]
    aug["path"] = [p for _, p in res]
    aug["item_id"] = aug["item_id"] + "|" + aug["augmentation"]
    base = df.copy()
    base["augmentation"] = "none"
    full = pd.concat([base, aug], ignore_index=True)
    out = Path(args.out)
    full["path"] = [os.path.relpath(p, out.parent) for p in full["path"]]
    keep = [c for c in full.columns if c not in ("filename", "label", "label_raw")]
    full = full[keep].assign(label=pd.concat([base, aug])["label_raw"].values)
    full.to_csv(out, sep="\t", index=False)
    print(full.groupby(["split", "augmentation"]).size().to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
