"""Subset of the fine-tuning pool for the concept analysis (scripts/run_concepts.py): a source-stratified train sample,
equal-size val / holdout / In-the-Wild samples and every test clip -> cache/manifest_concepts.parquet.

    python scripts/concept_subset.py [--train 12000] [--eval 1500]
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hearsay import config, splits  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=12000)
    ap.add_argument("--eval", type=int, default=1500)
    a = ap.parse_args()
    man = pd.read_parquet(config.CACHE / "manifest_pool.parquet")
    lab = man[man.label >= 0].copy()
    lab["split"] = splits.shared_split(lab)
    parts = [lab[lab.split == "train"].groupby("generator", group_keys=False).sample(
        frac=min(1.0, a.train / (lab.split == "train").sum()), random_state=0)]
    for s in ("val", "holdout", "itw"):
        d = lab[lab.split == s]
        parts.append(d.groupby("label", group_keys=False).sample(frac=min(1.0, a.eval / len(d)), random_state=0))
    out = pd.concat(parts + [man[man.label < 0]]).drop(columns=["split"])
    out.to_parquet(config.CACHE / "manifest_concepts.parquet")
    print(len(out), out.groupby("family").size().to_dict())


if __name__ == "__main__":
    main()
