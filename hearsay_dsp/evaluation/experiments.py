"""Experiment machinery: data loading, grouped CV, late fusion, LOGO, summaries.

Leakage rules implemented here:
  * folds are StratifiedGroupKFold on group_id (all clips of a sentence id,
    LJSpeech chapter or LibriSpeech chapter stay together);
  * every learned step (GMMs, imputation, scaling, classifier, fusion weights,
    calibration) is fitted on training folds only; out-of-fold (OOF) scores
    are produced by models that never saw the scored rows;
  * fusion weights and calibrators are cross-fitted on OOF component scores,
    never on in-sample scores;
  * rows designated holdout/test are never passed to these functions for fitting.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits

from ..config import module_key
from ..io.cache import FeatureCache
from ..io.manifest import read_manifest
from ..models.calibration import SigmoidCalibrator
from ..models.detector import Detector, DetectorSpec
from ..models.gmm import GMMPair
from ..pipeline import extract_paths, module_params, records_to_table
from ..routing import ALL_MODULES
from .metrics import binary_metrics, eer_w, group_bootstrap, auc_w


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def load_table(manifest: str, cfg: dict, modules=ALL_MODULES, n_jobs=None,
               log_every: int = 1000) -> tuple[pd.DataFrame, list]:
    df = read_manifest(manifest)
    recs = extract_paths(df["path"].tolist(), cfg, modules, n_jobs, log_every=log_every)
    ft = records_to_table(recs)
    table = df.merge(ft, on="path", how="left", validate="one_to_one")
    return table, recs


class FrameStore:
    """Loads cached LFCC frame arrays for one feature variant, keeping them in memory."""

    def __init__(self, cfg: dict, feature: str):
        self.cache = FeatureCache(cfg["run"]["cache_dir"])
        self.key = module_key(cfg, "lfcc", module_params(cfg, "lfcc"))
        self.feature = feature
        self._mem: dict = {}

    def get(self, sha: str) -> dict | None:
        if sha not in self._mem:
            arr = self.cache.load_arrays(sha, "lfcc", self.key)
            self._mem[sha] = None if arr is None else {
                self.feature: arr[self.feature].astype(np.float32),
                "nonsilent": arr["nonsilent"], "active": arr["active"]}
        return self._mem[sha]

    def many(self, shas, n_threads: int = 8) -> list:
        missing = [s for s in dict.fromkeys(shas) if s not in self._mem]
        with ThreadPoolExecutor(n_threads) as ex:
            list(ex.map(self.get, missing))
        return [self._mem[s] for s in shas]


def make_folds(table: pd.DataFrame, n_folds: int, seed: int) -> np.ndarray:
    y = table["label"].astype(int).values
    cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = np.full(len(table), -1)
    for k, (_, te) in enumerate(cv.split(np.zeros(len(y)), y, table["group_id"].values)):
        folds[te] = k
    return folds


# ---------------------------------------------------------------------------
# out-of-fold scoring
# ---------------------------------------------------------------------------
def _fit_score_gmm(params, tr_arrays, tr_y, te_arrays, blas_threads=2):
    with threadpool_limits(limits=blas_threads):
        g = GMMPair(**params).fit([a for a, t in zip(tr_arrays, tr_y) if t == 0],
                                  [a for a, t in zip(tr_arrays, tr_y) if t == 1])
        return np.array([g.score(a) if a is not None else np.nan for a in te_arrays]), g.train_info


def gmm_oof(arrays: list, y: np.ndarray, folds: np.ndarray, params: dict,
            train_mask: np.ndarray | None = None, n_threads: int = 5) -> tuple[np.ndarray, list]:
    """OOF GMM scores. train_mask restricts which rows may be used for fitting (LOGO)."""
    n = len(arrays)
    train_mask = np.ones(n, bool) if train_mask is None else train_mask
    out = np.full(n, np.nan)
    jobs = []
    for k in np.unique(folds):
        tr = np.where((folds != k) & train_mask)[0]
        te = np.where(folds == k)[0]
        jobs.append((te, [arrays[i] for i in tr], y[tr], [arrays[i] for i in te]))
    infos = []
    with ThreadPoolExecutor(n_threads) as ex:
        futs = [ex.submit(_fit_score_gmm, params, a, b, c) for _, a, b, c in jobs]
        for (te, *_), fut in zip(jobs, futs):
            s, info = fut.result()
            out[te] = s
            infos.append(info)
    return out, infos


def _fit_fold(spec, X, y, g, tr, te):
    with threadpool_limits(limits=1):
        det = Detector(spec).fit(X.iloc[tr], y[tr], g[tr])
        return te, det.decision_function(X.iloc[te]), det.selection


def detector_oof(spec: DetectorSpec, table: pd.DataFrame, folds: np.ndarray,
                 train_mask: np.ndarray | None = None, n_jobs: int = 5) -> tuple[np.ndarray, list]:
    """OOF raw scores for a feature-only detector (no GMM); outer folds run in parallel."""
    from joblib import Parallel, delayed
    from ..models.detector import allowlisted_columns
    y = table["label"].astype(int).values
    g = table["group_id"].values
    cols = allowlisted_columns(table.columns, spec.feature_groups, spec.extra_features)
    X = table[cols]  # only the columns the detector can use are shipped to workers
    train_mask = np.ones(len(table), bool) if train_mask is None else train_mask
    jobs = [(np.where((folds != k) & train_mask)[0], np.where(folds == k)[0]) for k in np.unique(folds)]
    res = Parallel(n_jobs=min(n_jobs, len(jobs)), backend="loky")(
        delayed(_fit_fold)(spec, X, y, g, tr, te) for tr, te in jobs)
    out = np.full(len(table), np.nan)
    sel = []
    for te, s, selection in res:
        out[te] = s
        sel.append(selection)
    return out, sel


def crossfit_calibrate(raw: np.ndarray, y: np.ndarray, folds: np.ndarray, prior: float = 0.5,
                       train_mask: np.ndarray | None = None) -> np.ndarray:
    train_mask = np.ones(len(raw), bool) if train_mask is None else train_mask
    p = np.full(len(raw), np.nan)
    for k in np.unique(folds):
        tr = (folds != k) & train_mask & np.isfinite(raw)
        te = folds == k
        cal = SigmoidCalibrator(prior).fit(raw[tr], y[tr])
        p[te] = cal.predict(raw[te])
    return p


class LateFusion:
    """Logistic regression on component raw scores (class-balanced to `prior`).

    Its output logit is already a calibrated log-odds for the stated prior.
    Non-finite component scores are replaced by the training median.
    """

    def __init__(self, names: list[str], prior: float = 0.5):
        self.names = list(names)
        self.prior = prior
        self.coef = None
        self.intercept = 0.0
        self.medians = None
        self.scale = None

    def fit(self, Z: np.ndarray, y: np.ndarray) -> "LateFusion":
        Z = np.asarray(Z, dtype=float)
        self.medians = np.nanmedian(Z, axis=0)
        Z = np.where(np.isfinite(Z), Z, self.medians)
        self.scale = Z.std(axis=0) + 1e-9
        n_pos, n_neg = (y == 1).sum(), (y == 0).sum()
        w = np.where(y == 1, self.prior / n_pos, (1 - self.prior) / n_neg) * len(y)
        lr = LogisticRegression(C=100.0, max_iter=1000).fit(Z / self.scale, y, sample_weight=w)
        self.coef = lr.coef_[0] / self.scale
        self.intercept = float(lr.intercept_[0])
        return self

    def logit(self, Z: np.ndarray) -> np.ndarray:
        Z = np.where(np.isfinite(Z), Z, self.medians)
        return Z @ self.coef + self.intercept

    def predict(self, Z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-self.logit(Z)))

    def to_dict(self) -> dict:
        return {"names": self.names, "prior": self.prior, "coef": list(map(float, self.coef)),
                "intercept": self.intercept, "medians": list(map(float, self.medians))}

    @classmethod
    def from_dict(cls, d: dict) -> "LateFusion":
        obj = cls(d["names"], d["prior"])
        obj.coef = np.array(d["coef"])
        obj.intercept = d["intercept"]
        obj.medians = np.array(d["medians"])
        return obj


def crossfit_fusion(Z: np.ndarray, y: np.ndarray, folds: np.ndarray, names, prior=0.5,
                    train_mask=None) -> tuple[np.ndarray, np.ndarray]:
    train_mask = np.ones(len(y), bool) if train_mask is None else train_mask
    logit = np.full(len(y), np.nan)
    for k in np.unique(folds):
        tr = (folds != k) & train_mask
        te = folds == k
        f = LateFusion(names, prior).fit(Z[tr], y[tr])
        logit[te] = f.logit(Z[te])
    return logit, 1.0 / (1.0 + np.exp(-logit))


# ---------------------------------------------------------------------------
# summaries
# ---------------------------------------------------------------------------
def summarize(table: pd.DataFrame, raw: np.ndarray, p: np.ndarray, mask: np.ndarray | None = None,
              n_boot: int = 500, seed: int = 0) -> dict:
    mask = np.ones(len(table), bool) if mask is None else mask
    t = table[mask]
    y = t["label"].astype(int).values
    r, pp = raw[mask], p[mask]
    ok = np.isfinite(r) & np.isfinite(pp)
    out = {"overall": binary_metrics(y[ok], pp[ok], raw=r[ok]), "n_excluded_nonfinite": int((~ok).sum())}
    if (y[ok] == 0).any() and (y[ok] == 1).any() and n_boot:
        g = t["group_id"].values[ok]
        out["overall"]["auc_ci95"] = group_bootstrap(y[ok], r[ok], g, auc_w, n_boot, seed)
        out["overall"]["eer_ci95"] = group_bootstrap(y[ok], r[ok], g, eer_w, n_boot, seed)
    real = ok & (y == 0)
    per_gen = {}
    for gen in sorted(t.loc[t["label"] == 1, "attack_type"].unique()):
        m = ok & ((y == 0) | (t["attack_type"].values == gen))
        mm = binary_metrics(y[m], pp[m], raw=r[m])
        per_gen[gen] = {k: mm[k] for k in ("auc", "eer", "sensitivity")} | {"n_fake": int(((y == 1) & m).sum())}
    out["per_generator"] = per_gen
    per_src = {}
    for src in sorted(t.loc[t["label"] == 0, "source_id"].unique()):
        m = real & (t["source_id"].values == src)
        per_src[src] = {"n": int(m.sum()), "genuine_fpr@0.5": float((pp[m] >= 0.5).mean()) if m.any() else None,
                        "median_p": float(np.median(pp[m])) if m.any() else None}
    out["per_real_source"] = per_src
    if "family" in t.columns:
        fam = {}
        for f in sorted(t.loc[t["label"] == 1, "family"].unique()):
            m = ok & ((y == 0) | (t["family"].values == f))
            mm = binary_metrics(y[m], pp[m], raw=r[m])
            fam[f] = {k: mm[k] for k in ("auc", "eer")}
        out["per_fake_family"] = fam
    return out


def compact(summary: dict) -> dict:
    o = summary["overall"]
    row = {k: o.get(k) for k in ("n", "auc", "eer", "ap", "log_loss_balanced", "brier_balanced",
                                 "genuine_fpr", "sensitivity", "balanced_accuracy")}
    if "auc_ci95" in o:
        row["auc_ci95"] = [round(o["auc_ci95"]["lo"], 4), round(o["auc_ci95"]["hi"], 4)]
        row["eer_ci95"] = [round(o["eer_ci95"]["lo"], 4), round(o["eer_ci95"]["hi"], 4)]
    return row


def save_json(obj, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=1, default=lambda o: o.tolist()
                                     if hasattr(o, "tolist") else str(o)))
