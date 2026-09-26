"""Sigmoid (Platt) calibration fitted on out-of-fold development scores.

Class weights are set so the fitted probabilities correspond to a stated
synthetic prior (default 0.5: the test-set prevalence is unknown). If the
deployment prevalence pi is known, convert with the usual prior shift:
logit(p') = logit(p) + logit(pi) - logit(prior).
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


class SigmoidCalibrator:
    def __init__(self, prior: float = 0.5):
        self.prior = float(prior)
        self.a = 1.0
        self.b = 0.0
        self.info: dict = {}

    def fit(self, raw: np.ndarray, y: np.ndarray) -> "SigmoidCalibrator":
        raw = np.asarray(raw, dtype=float)
        y = np.asarray(y, dtype=int)
        ok = np.isfinite(raw)
        raw, y = raw[ok], y[ok]
        n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
        if n_pos == 0 or n_neg == 0:
            raise ValueError("calibration needs both classes")
        w = np.where(y == 1, self.prior / n_pos, (1 - self.prior) / n_neg) * y.size
        lr = LogisticRegression(C=1e6, max_iter=1000)
        lr.fit(raw.reshape(-1, 1), y, sample_weight=w)
        self.a, self.b = float(lr.coef_[0, 0]), float(lr.intercept_[0])
        self.info = {"n_pos": n_pos, "n_neg": n_neg, "n_dropped_nonfinite": int((~ok).sum())}
        return self

    def logit(self, raw) -> np.ndarray:
        return self.a * np.asarray(raw, dtype=float) + self.b

    def predict(self, raw) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-self.logit(raw)))

    def to_dict(self) -> dict:
        return {"method": "sigmoid", "prior": self.prior, "a": self.a, "b": self.b, "info": self.info}

    @classmethod
    def from_dict(cls, d: dict) -> "SigmoidCalibrator":
        obj = cls(d["prior"])
        obj.a, obj.b, obj.info = d["a"], d["b"], d.get("info", {})
        return obj
