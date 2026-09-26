"""Evaluation metrics with explicit handling of undefined cases.

EER convention: from the ROC curve, the point where FPR = FNR (FNR = 1 - TPR),
linearly interpolated between the two adjacent ROC points bracketing the
crossing. EER is descriptive; its threshold is never used for deployment.

Thresholded metrics use a prespecified threshold (default 0.5 on the
calibrated probability), not one chosen on the evaluation data.

"Balanced" log loss / Brier weight both classes equally, because our
evaluation sets are imbalanced and the test prevalence is unknown.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def eer(y, s, sample_weight=None) -> float:
    y = np.asarray(y, dtype=int)
    s = np.asarray(s, dtype=float)
    fpr, tpr, _ = roc_curve(y, s, sample_weight=sample_weight)
    fnr = 1.0 - tpr
    d = fpr - fnr
    j = int(np.argmax(d >= 0))
    if d[j] == 0 or j == 0:
        return float(fpr[j])
    t = d[j - 1] / (d[j - 1] - d[j])
    return float(fpr[j - 1] + t * (fpr[j] - fpr[j - 1]))


def _weighted(values: np.ndarray, y: np.ndarray, balanced: bool) -> float:
    if not balanced:
        return float(np.mean(values))
    return float(0.5 * values[y == 1].mean() + 0.5 * values[y == 0].mean())


def binary_metrics(y, p, raw=None, threshold: float = 0.5) -> dict:
    """y in {0,1} (1 = synthetic); p = calibrated probability; raw = uncalibrated score."""
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    out: dict = {"n": int(y.size), "n_real": int((y == 0).sum()), "n_synthetic": int((y == 1).sum())}
    finite = np.isfinite(p)
    out["n_nonfinite"] = int((~finite).sum())
    y, p = y[finite], p[finite]
    rank = np.asarray(raw, dtype=float)[finite] if raw is not None else p
    two_class = (y == 0).any() and (y == 1).any()
    if two_class:
        out["auc"] = float(roc_auc_score(y, rank))
        out["ap"] = float(average_precision_score(y, rank))
        out["eer"] = eer(y, rank)
    else:
        out.update({"auc": None, "ap": None, "eer": None,
                    "undefined_reason": "only one class present"})
    pc = np.clip(p, 1e-6, 1 - 1e-6)
    ll = -(y * np.log(pc) + (1 - y) * np.log(1 - pc))
    br = (p - y) ** 2
    out["log_loss"] = float(ll.mean())
    out["brier"] = float(br.mean())
    if two_class:
        out["log_loss_balanced"] = _weighted(ll, y, True)
        out["brier_balanced"] = _weighted(br, y, True)
    pred = (p >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    out["threshold"] = threshold
    out["confusion"] = {"tp": tp, "tn": tn, "fp": fp, "fn": fn}
    out["sensitivity"] = tp / (tp + fn) if (tp + fn) else None     # synthetic caught
    out["specificity"] = tn / (tn + fp) if (tn + fp) else None     # real passed
    out["genuine_fpr"] = fp / (tn + fp) if (tn + fp) else None     # real flagged synthetic
    out["precision"] = tp / (tp + fp) if (tp + fp) else None
    prec, rec = out["precision"], out["sensitivity"]
    out["f1"] = (2 * prec * rec / (prec + rec)) if prec and rec else (0.0 if prec == 0 or rec == 0 else None)
    out["accuracy"] = (tp + tn) / y.size if y.size else None
    if out["sensitivity"] is not None and out["specificity"] is not None:
        out["balanced_accuracy"] = 0.5 * (out["sensitivity"] + out["specificity"])
    return out


def reliability(y, p, n_bins: int = 10) -> list[dict]:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    edges = np.linspace(0, 1, n_bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & ((p < hi) if hi < 1 else (p <= hi))
        rows.append({"bin": [float(lo), float(hi)], "n": int(m.sum()),
                     "mean_p": float(p[m].mean()) if m.any() else None,
                     "frac_synthetic": float(y[m].mean()) if m.any() else None})
    return rows


def auc_w(y, s, w):
    return roc_auc_score(y, s, sample_weight=w)


def eer_w(y, s, w):
    return eer(y, s, sample_weight=w)


def group_bootstrap(y, s, groups, fn=auc_w, n_boot: int = 1000, seed: int = 0) -> dict:
    """Percentile CI from resampling groups with replacement, separately per class.

    Resampling a group k times is implemented as weighting its rows by k, so
    fn(y, s, weights) must accept sample weights. Groups are class-pure in our
    manifests; related clips of one recording share a group and move together.
    """
    y = np.asarray(y, dtype=int)
    s = np.asarray(s, dtype=float)
    groups = np.asarray(groups).astype(str)
    rng = np.random.default_rng(seed)
    parts = []
    for cls in (0, 1):
        rows = np.where(y == cls)[0]
        uniq, inv = np.unique(groups[rows], return_inverse=True)
        parts.append((rows, uniq.size, inv))
    vals = []
    for _ in range(n_boot):
        w = np.zeros(y.size)
        for rows, n_groups, inv in parts:
            if n_groups:
                counts = np.bincount(rng.integers(0, n_groups, n_groups), minlength=n_groups)
                w[rows] = counts[inv]
        try:
            vals.append(fn(y, s, w))
        except ValueError:
            continue
    vals = np.array(vals, dtype=float)
    return {"lo": float(np.percentile(vals, 2.5)), "hi": float(np.percentile(vals, 97.5)),
            "n_boot": int(vals.size), "n_groups_real": int(parts[0][1]),
            "n_groups_synthetic": int(parts[1][1])}
