"""Detection metrics. Scores follow the submission polarity: higher = more likely synthetic."""
import numpy as np
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score, roc_curve


def eer(y, s):
    fpr, tpr, _ = roc_curve(y, s)
    fnr = 1 - tpr
    i = np.nanargmin(np.abs(fnr - fpr))
    return float((fpr[i] + fnr[i]) / 2)


def summary(y, s, prob=None, n_boot=0, seed=0):
    """AUC/EER always; log loss/Brier when `prob` (calibrated probabilities) is given."""
    y, s = np.asarray(y), np.asarray(s)
    if len(np.unique(y)) < 2:
        return dict(n=len(y), n_fake=int(y.sum()), auc=np.nan, eer=np.nan)
    out = dict(n=len(y), n_fake=int(y.sum()), auc=roc_auc_score(y, s), eer=eer(y, s))
    if prob is not None:
        p = np.clip(np.asarray(prob), 1e-6, 1 - 1e-6)
        out.update(logloss=log_loss(y, p, labels=[0, 1]), brier=brier_score_loss(y, p))
    if n_boot:
        rng = np.random.default_rng(seed)
        pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
        aucs = []
        for _ in range(n_boot):  # stratified bootstrap keeps both classes present
            i = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
            aucs.append(roc_auc_score(y[i], s[i]))
        out.update(auc_lo=float(np.percentile(aucs, 2.5)), auc_hi=float(np.percentile(aucs, 97.5)))
    return out


def check_polarity(y, s, name="score"):
    """Plan section 3: spoof must score higher than bona fide on dev."""
    auc = roc_auc_score(y, s)
    if auc < 0.5:
        raise ValueError(f"{name}: AUC {auc:.3f} < 0.5 on dev, polarity looks flipped (1.0 must mean synthetic)")
    return auc
