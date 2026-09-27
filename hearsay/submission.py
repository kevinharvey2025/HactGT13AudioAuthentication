"""Challenge TSV: calibrate detector scores to probabilities, write the file, validate it.

The organizers' template (data/hearsay_test/HGT_Hearsay_score_template.csv, tab-separated despite
the extension) defines the filenames and their order. The output has the header
`filename<TAB>cm-score`, one row per template file in template order, and finite floats in [0, 1]
where 1.0 = synthetic. Writing is refused, never padded, when any template file lacks a score.
"""
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from . import config

TEMPLATE = config.TEST_DIR / "HGT_Hearsay_score_template.csv"


class Platt:
    """P(synthetic) from a raw score by class-balanced logistic regression (prior 0.5: test
    prevalence is unknown). Monotone, so AUC/EER/minDCF of the calibrated score are unchanged."""

    def fit(self, s, y):
        self.lr = LogisticRegression(C=1e4, class_weight="balanced").fit(np.asarray(s, float).reshape(-1, 1), y)
        self.coef, self.intercept = float(self.lr.coef_[0, 0]), float(self.lr.intercept_[0])
        return self

    def __call__(self, s):
        return 1.0 / (1.0 + np.exp(-(self.coef * np.asarray(s, float) + self.intercept)))


def read_template(path=TEMPLATE):
    return pd.read_csv(path, sep="\t")


def output_name(team=None):
    return f"{team or config.TEAM}_predictions.tsv"


def write(scores, out_path, template=TEMPLATE):
    """scores: mapping or Series filename -> probability. Returns the written frame."""
    ref = read_template(template)
    s = pd.Series(scores, dtype=float)
    vals = s.reindex(ref.filename)
    bad = ~np.isfinite(vals.to_numpy()) | (vals.to_numpy() < 0) | (vals.to_numpy() > 1)
    if bad.any():
        raise ValueError(f"{int(bad.sum())} template files lack a valid score, e.g. "
                         f"{list(ref.filename[bad][:5])}; refusing to write {out_path}")
    out = pd.DataFrame({"filename": ref.filename, "cm-score": vals.to_numpy()})
    out.to_csv(out_path, sep="\t", index=False, float_format="%.6f", lineterminator="\n")
    validate(out_path, template)
    return out


def validate(pred_path, template=TEMPLATE):
    """The plan's section 12.1 checks, plus template order."""
    pred = pd.read_csv(pred_path, sep="\t")
    ref = read_template(template)
    assert list(pred.columns) == ["filename", "cm-score"], f"header {list(pred.columns)}"
    assert len(pred) == len(ref), f"{len(pred)} rows, template has {len(ref)}"
    assert pred.filename.is_unique, "duplicate filenames"
    assert (pred.filename.to_numpy() == ref.filename.to_numpy()).all(), "filenames differ from the template (set or order)"
    s = pred["cm-score"].to_numpy(dtype=float)
    assert np.isfinite(s).all() and ((s >= 0) & (s <= 1)).all(), "scores must be finite floats in [0, 1]"
    with open(pred_path, "rb") as f:
        assert b"\r" not in f.read(), "CRLF line endings"
    return True


def describe(prob):
    """Test-score distribution. An extreme flagged fraction usually means domain shift, not prevalence."""
    p = np.asarray(prob, float)
    q = np.quantile(p, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    # Bayes decision for the organizers' costs at prior 0.5 (Cfa 4, Cmiss 1): flag when p > 0.2
    return dict(n=len(p), mean=float(p.mean()), quantiles=dict(zip(["q01", "q05", "q25", "q50", "q75", "q95", "q99"], q.round(4).tolist())),
                frac_ge_0_5=float((p >= 0.5).mean()), frac_gt_0_2=float((p > 0.2).mean()))


def save_summary(path, **kw):
    with open(path, "w") as f:
        json.dump(kw, f, indent=1, default=float)
