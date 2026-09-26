"""Detector = optional LFCC-GMM score + allowlisted engineered features + classifier.

Configurations compared in the ablations:
  GMM only                 raw score = GMM log-likelihood ratio (synthetic - real)
  features + LR / HGB      regularized logistic regression or histogram boosting
  GMM score + features     "gmm.llr" enters the classifier as a feature. For
                           training rows it must be out-of-fold (callers pass
                           precomputed OOF scores); in-sample GMM scores are never
                           used to fit the fusion classifier.

Imputation (median) and scaling are fitted inside the classifier pipeline on
training rows only. Missingness indicators are added deliberately so that a
module's abstention is visible to the model; the shortcut audit checks whether
they carry label information by themselves.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..config import FEATURE_GROUP_PREFIXES
from .gmm import GMMPair


@dataclass
class DetectorSpec:
    name: str
    use_gmm: bool = True
    gmm: dict = field(default_factory=dict)
    feature_groups: tuple = ()
    extra_features: tuple = ()          # explicit column names (shortcut audits only)
    classifier: str | None = None       # None (GMM only), "lr" or "hgb"
    lr_C_grid: tuple = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
    hgb: dict = field(default_factory=dict)
    inner_folds: int = 5
    seed: int = 0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["feature_groups"] = list(self.feature_groups)
        d["extra_features"] = list(self.extra_features)
        d["lr_C_grid"] = list(self.lr_C_grid)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "DetectorSpec":
        d = dict(d)
        for k in ("feature_groups", "extra_features", "lr_C_grid"):
            d[k] = tuple(d.get(k, ()))
        return cls(**d)


def allowlisted_columns(columns, groups, extra=()) -> list[str]:
    prefixes = tuple(p for g in groups for p in FEATURE_GROUP_PREFIXES[g])
    cols = [c for c in columns if prefixes and c.startswith(prefixes)]
    cols += [c for c in extra if c in columns]
    return sorted(set(cols), key=list(columns).index)


def make_lr(C: float, seed: int) -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(C=C, class_weight="balanced", max_iter=5000, random_state=seed)),
    ])


def make_hgb(params: dict, seed: int) -> HistGradientBoostingClassifier:
    base = dict(learning_rate=0.05, max_iter=300, max_leaf_nodes=15, min_samples_leaf=20,
                l2_regularization=1.0)
    base.update(params or {})
    return HistGradientBoostingClassifier(class_weight="balanced", random_state=seed, **base)


class Detector:
    def __init__(self, spec: DetectorSpec):
        self.spec = spec
        self.gmm: GMMPair | None = None
        self.clf = None
        self.columns: list[str] = []
        self.selection: dict = {}

    # -- GMM ---------------------------------------------------------------
    def fit_gmm(self, arrays: list, y: np.ndarray) -> GMMPair:
        params = {**self.spec.gmm, "seed": self.spec.seed}
        real = [a for a, t in zip(arrays, y) if t == 0]
        fake = [a for a, t in zip(arrays, y) if t == 1]
        return GMMPair(**params).fit(real, fake)

    # -- design matrix -------------------------------------------------------
    def design(self, table: pd.DataFrame, gmm_scores: np.ndarray | None) -> pd.DataFrame:
        X = table.reindex(columns=self.columns[:-1] if self.spec.use_gmm else self.columns)
        X = X.apply(pd.to_numeric, errors="coerce").astype(float)
        if self.spec.use_gmm:
            X["gmm.llr"] = gmm_scores
        return X

    def fit(self, table: pd.DataFrame, y: np.ndarray, groups: np.ndarray, arrays: list | None = None,
            gmm: GMMPair | None = None, gmm_oof: np.ndarray | None = None) -> "Detector":
        """table/y/groups/arrays are training rows only.

        gmm: an already-fitted GMM on exactly these rows (optional).
        gmm_oof: out-of-fold GMM scores for these rows (required for fusion).
        """
        spec = self.spec
        y = np.asarray(y, dtype=int)
        if spec.use_gmm:
            self.gmm = gmm if gmm is not None else self.fit_gmm(arrays, y)
        if spec.classifier is None:
            if not spec.use_gmm:
                raise ValueError("a detector without a classifier needs the GMM")
            return self
        cols = allowlisted_columns(table.columns, spec.feature_groups, spec.extra_features)
        self.columns = cols + (["gmm.llr"] if spec.use_gmm else [])
        if spec.use_gmm and gmm_oof is None:
            raise ValueError("fusion requires out-of-fold GMM scores for the training rows")
        X = self.design(table, gmm_oof)
        if spec.classifier == "lr":
            C = self._select_C(X, y, groups)
            self.clf = make_lr(C, spec.seed).fit(X, y)
        elif spec.classifier == "hgb":
            self.clf = make_hgb(spec.hgb, spec.seed).fit(X, y)
            self.selection = {"hgb": spec.hgb}
        else:
            raise ValueError(spec.classifier)
        return self

    def _select_C(self, X: pd.DataFrame, y: np.ndarray, groups: np.ndarray) -> float:
        grid = list(self.spec.lr_C_grid)
        if len(grid) == 1:
            self.selection = {"C": grid[0], "cv": None}
            return grid[0]
        cv = StratifiedGroupKFold(n_splits=self.spec.inner_folds, shuffle=True,
                                  random_state=self.spec.seed)
        splits = list(cv.split(X, y, groups))
        scores = {}
        for C in grid:
            aucs = []
            for tr, te in splits:
                m = make_lr(C, self.spec.seed).fit(X.iloc[tr], y[tr])
                aucs.append(roc_auc_score(y[te], m.decision_function(X.iloc[te])))
            scores[C] = float(np.mean(aucs))
        best = max(scores.values())
        C = min(c for c, s in scores.items() if s >= best - 1e-4)  # most regularized near-best
        self.selection = {"C": C, "inner_cv_auc": scores}
        return C

    # -- inference -------------------------------------------------------------
    def gmm_scores(self, arrays: list) -> np.ndarray:
        return np.array([self.gmm.score(a) for a in arrays]) if self.gmm is not None else None

    def decision_function(self, table: pd.DataFrame, arrays: list | None = None,
                          gmm_scores: np.ndarray | None = None) -> np.ndarray:
        if self.spec.use_gmm and gmm_scores is None:
            gmm_scores = self.gmm_scores(arrays)
        if self.spec.classifier is None:
            return np.asarray(gmm_scores, dtype=float)
        X = self.design(table, gmm_scores)
        if self.spec.classifier == "lr":
            return self.clf.decision_function(X)
        p = np.clip(self.clf.predict_proba(X)[:, 1], 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))

    def contributions(self, table: pd.DataFrame, gmm_scores: np.ndarray | None = None,
                      top: int = 5) -> list[list[dict]]:
        """Per-row linear contributions (coef x standardized value) for LR detectors."""
        if self.spec.classifier != "lr":
            return [[] for _ in range(len(table))]
        X = self.design(table, gmm_scores)
        imp, sc, lr = (self.clf.named_steps[k] for k in ("impute", "scale", "clf"))
        Z = sc.transform(imp.transform(X))
        names = imp.get_feature_names_out(X.columns)
        contrib = Z * lr.coef_[0][None, :]
        out = []
        for row, z in zip(contrib, Z):
            order = np.argsort(-np.abs(row))[:top]
            out.append([{"feature": str(names[j]), "contribution_logit": float(row[j]),
                         "standardized_value": float(z[j])} for j in order])
        return out

    # -- persistence -------------------------------------------------------------
    def save(self, directory: str | Path) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        meta = {"spec": self.spec.to_dict(), "columns": self.columns, "selection": self.selection,
                "sklearn_version": sklearn.__version__}
        if self.gmm is not None:
            np.savez(d / "gmm.npz", **self.gmm.to_arrays())
            meta["gmm"] = {"params": self.gmm.params, "train_info": self.gmm.train_info}
        if self.clf is not None:
            joblib.dump(self.clf, d / "classifier.joblib")
        (d / "detector.json").write_text(json.dumps(meta, indent=1, default=str))

    @classmethod
    def load(cls, directory: str | Path) -> "Detector":
        d = Path(directory)
        meta = json.loads((d / "detector.json").read_text())
        if meta.get("sklearn_version") != sklearn.__version__:
            raise RuntimeError(f"model saved with scikit-learn {meta.get('sklearn_version')}, "
                               f"running {sklearn.__version__}; use the pinned environment")
        det = cls(DetectorSpec.from_dict(meta["spec"]))
        det.columns = meta["columns"]
        det.selection = meta.get("selection", {})
        if (d / "gmm.npz").exists():
            with np.load(d / "gmm.npz") as z:
                det.gmm = GMMPair.from_arrays(meta["gmm"]["params"], {k: z[k] for k in z.files},
                                              meta["gmm"].get("train_info"))
        if (d / "classifier.joblib").exists():
            det.clf = joblib.load(d / "classifier.joblib")
        return det
