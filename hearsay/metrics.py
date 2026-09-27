"""Detection metrics. Scores follow the submission polarity: higher = more likely synthetic.

The organizers rank by the ASVspoof 5 track-1 evaluation package (calculate_metrics.py) with Pspoof = 0.3, Cmiss = 1,
Cfa = 4 (package defaults 0.05 / 10; the copy in data/HackGTMinDCF still shows 0.5 — the organizers confirmed 0.3).
The package treats bona fide as the target class, so in our polarity a "miss" is a real clip flagged as synthetic and
a "false alarm" is a synthetic clip passed as real:  minDCF = min over thresholds of [FPR + (4 * 0.3) / (1 * 0.7) * FNR]
= FPR + 1.714 * FNR (normalized: 1.0 = a constant decision). tests/test_metrics.py checks parity with the package.

Ties: the curve is evaluated only at thresholds a detector can realize (between distinct scores). The package walks
every sorted position, which is identical without ties and slightly optimistic with them.
"""
import numpy as np
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

DCF = dict(p_spoof=0.3, c_miss=1.0, c_fa=4.0)


def effective_prior(p_spoof=DCF["p_spoof"], c_miss=DCF["c_miss"], c_fa=DCF["c_fa"]):
    """The spoof prior under unit costs that ranks systems exactly as the organizers' costs do (0.632)."""
    return c_fa * p_spoof / (c_fa * p_spoof + c_miss * (1 - p_spoof))


def bayes_threshold(p_spoof=DCF["p_spoof"], c_miss=DCF["c_miss"], c_fa=DCF["c_fa"], calib_prior=0.5):
    """Posterior threshold on P(synthetic) calibrated at `calib_prior` for the organizers' costs: flag when
    c_fa * pi * LR > c_miss * (1 - pi), i.e. LR > c_miss (1 - pi) / (c_fa pi). 0.2 at calib_prior 0.3."""
    lr = c_miss * (1 - p_spoof) / (c_fa * p_spoof)
    odds = lr * calib_prior / (1 - calib_prior)
    return odds / (1 + odds)


def det_curve(y, s):
    """(FPR, FNR) at every realizable threshold, from 'flag nothing' to 'flag everything': FPR = share of real clips
    flagged as synthetic, FNR = share of synthetic clips passed as real."""
    y, s = np.asarray(y), np.asarray(s, float)
    o = np.argsort(-s, kind="mergesort")
    ys, ss = y[o], s[o]
    last = np.r_[ss[1:] != ss[:-1], True]                  # the end of each run of tied scores
    tp, fp = np.r_[0, np.cumsum(ys)[last]], np.r_[0, np.cumsum(1 - ys)[last]]
    return fp / fp[-1], 1 - tp / tp[-1]


def _dcf(fpr, fnr, p_spoof, c_miss, c_fa):
    return float((c_miss * (1 - p_spoof) * fpr + c_fa * p_spoof * fnr).min() / min(c_miss * (1 - p_spoof), c_fa * p_spoof))


def _eer(fpr, fnr):
    i = np.argmin(np.abs(fnr - fpr))
    return float((fpr[i] + fnr[i]) / 2)


def eer(y, s):
    return _eer(*det_curve(y, s))


def min_dcf(y, s, p_spoof=DCF["p_spoof"], c_miss=DCF["c_miss"], c_fa=DCF["c_fa"]):
    """Normalized minimum detection cost over all thresholds (0 = perfect, 1 = no better than a constant decision)."""
    return _dcf(*det_curve(y, s), p_spoof, c_miss, c_fa)


def min_dcf_eer(y, s):
    """(minDCF, EER) from one curve, for bootstrap loops."""
    fpr, fnr = det_curve(y, s)
    return _dcf(fpr, fnr, **DCF), _eer(fpr, fnr)


def act_dcf(y, prob, threshold=None):
    """Normalized DCF of the decisions prob > threshold (default: the Bayes threshold at the 30% prior)."""
    thr = bayes_threshold(calib_prior=DCF["p_spoof"]) if threshold is None else threshold
    y, d = np.asarray(y), np.asarray(prob) > thr
    p, cm, cfa = DCF["p_spoof"], DCF["c_miss"], DCF["c_fa"]
    return float((cm * (1 - p) * d[y == 0].mean() + cfa * p * (~d[y == 1]).mean()) / min(cm * (1 - p), cfa * p))


def summary(y, s, prob=None):
    """AUC/EER/minDCF always; log loss/Brier when `prob` (calibrated probabilities) is given."""
    y, s = np.asarray(y), np.asarray(s)
    if len(np.unique(y)) < 2:
        return dict(n=len(y), n_fake=int(y.sum()), auc=np.nan, eer=np.nan, min_dcf=np.nan)
    dcf, e = min_dcf_eer(y, s)
    out = dict(n=len(y), n_fake=int(y.sum()), auc=roc_auc_score(y, s), eer=e, min_dcf=dcf)
    if prob is not None:
        p = np.clip(np.asarray(prob), 1e-6, 1 - 1e-6)
        out.update(logloss=log_loss(y, p, labels=[0, 1]), brier=brier_score_loss(y, p))
    return out


def check_polarity(y, s, name="score"):
    """Spoof must score higher than bona fide on dev."""
    auc = roc_auc_score(y, s)
    if auc < 0.5:
        raise ValueError(f"{name}: AUC {auc:.3f} < 0.5 on dev, polarity looks flipped (1.0 must mean synthetic)")
    return auc
