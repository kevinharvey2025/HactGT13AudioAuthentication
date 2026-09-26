"""Deployable model bundle: components, fusion, routing band, provenance.

Components (all fitted on training rows only):
  gmm      LFCC-GMM log-likelihood ratio (core module: lfcc)
  lr_core  logistic regression on core feature groups (lfcc summaries, spectral)
  lr_full  logistic regression on all validated feature groups (core + extended)
Late fusion:
  core = LateFusion(gmm, lr_core), full = LateFusion(gmm, lr_full); weights are
  fitted on 5-fold out-of-fold component scores of the training rows, so the
  fused logit is a calibrated log-odds (prior 0.5).
Routing:
  all_eligible -> full probability for every file;
  confidence   -> core probability when it lies outside the band chosen on
                  out-of-fold training scores, else full probability.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd

from .. import __version__
from ..config import config_hash
from ..evaluation.experiments import (FrameStore, LateFusion, crossfit_calibrate, crossfit_fusion,
                                      detector_oof, gmm_oof, make_folds)
from ..evaluation.metrics import binary_metrics
from .detector import Detector, DetectorSpec

CORE_GROUPS = ("cep", "spec")


def package_versions() -> dict:
    import av
    import joblib
    import parselmouth
    import scipy
    import sklearn
    import soundfile
    return {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
            "scikit-learn": sklearn.__version__, "pandas": pd.__version__,
            "soundfile": soundfile.__version__, "av": av.__version__, "joblib": joblib.__version__,
            "praat-parselmouth": parselmouth.__version__, "hearsay_dsp": __version__}


def specs_from_config(cfg: dict) -> dict[str, DetectorSpec]:
    m = cfg["model"]
    seed = cfg["run"]["seed"]
    groups = tuple(m["feature_groups"])
    gmm_params = {k: v for k, v in m["gmm"].items()}
    return {
        "gmm": DetectorSpec("gmm", use_gmm=True, gmm=gmm_params, classifier=None, seed=seed),
        "lr_core": DetectorSpec("lr_core", use_gmm=False, feature_groups=tuple(g for g in CORE_GROUPS
                                                                                 if g in groups),
                                classifier=m["classifier"], lr_C_grid=tuple(m["lr"]["C_grid"]),
                                hgb=m["hgb"], inner_folds=m["inner_folds"], seed=seed),
        "lr_full": DetectorSpec("lr_full", use_gmm=False, feature_groups=groups,
                                classifier=m["classifier"], lr_C_grid=tuple(m["lr"]["C_grid"]),
                                hgb=m["hgb"], inner_folds=m["inner_folds"], seed=seed),
    }


def choose_band(y, p_core, p_full, max_auc_drop=0.002, max_ll_increase=0.01) -> dict:
    """Widest confident region (smallest band) whose routed OOF quality stays near 'full'."""
    base = binary_metrics(y, p_full)
    candidates = [(0.02, 0.98), (0.05, 0.95), (0.1, 0.9), (0.2, 0.8), (0.3, 0.7), (0.4, 0.6)]
    table = []
    chosen = (0.0, 1.0)
    for lo, hi in candidates:
        inside = (p_core >= lo) & (p_core <= hi)
        routed = np.where(inside, p_full, p_core)
        m = binary_metrics(y, routed)
        ok = (m["auc"] >= base["auc"] - max_auc_drop
              and m["log_loss_balanced"] <= base["log_loss_balanced"] + max_ll_increase)
        table.append({"band": [lo, hi], "frac_extended": float(inside.mean()), "auc": m["auc"],
                      "log_loss_balanced": m["log_loss_balanced"], "acceptable": bool(ok)})
        if ok:
            chosen = (lo, hi)
    return {"band": list(chosen), "full_oof_auc": base["auc"],
            "full_oof_log_loss_balanced": base["log_loss_balanced"], "candidates": table,
            "rule": f"narrowest band with AUC drop <= {max_auc_drop} and balanced log-loss increase "
                    f"<= {max_ll_increase} vs routing everything to the full model"}


class ModelBundle:
    def __init__(self):
        self.cfg: dict = {}
        self.dets: dict[str, Detector] = {}
        self.fusion: dict[str, LateFusion] = {}
        self.meta: dict = {}

    # ------------------------------------------------------------------ train
    @classmethod
    def train(cls, table: pd.DataFrame, cfg: dict, n_folds: int = 5, log=print) -> "ModelBundle":
        obj = cls()
        obj.cfg = cfg
        seed = cfg["run"]["seed"]
        y = table["label"].astype(int).values
        groups = table["group_id"].values
        specs = specs_from_config(cfg)
        store = FrameStore(cfg, specs["gmm"].gmm["feature"])
        arrays = store.many(table["sha256"].tolist())
        if any(a is None for a in arrays):
            raise RuntimeError("missing LFCC arrays for some training rows; run extract first")
        folds = make_folds(table, n_folds, seed)
        log(f"[train] {len(table)} rows ({int((y == 0).sum())} real / {int((y == 1).sum())} synthetic), "
            f"{len(np.unique(groups))} groups, {n_folds} folds")
        oof = {}
        oof["gmm"], _ = gmm_oof(arrays, y, folds, {**specs["gmm"].gmm, "seed": seed})
        log("[train] GMM out-of-fold scores done")
        for name in ("lr_core", "lr_full"):
            oof[name], _ = detector_oof(specs[name], table, folds)
            log(f"[train] {name} out-of-fold scores done")
        comps = {"core": ["gmm", "lr_core"], "full": ["gmm", "lr_full"]}
        oof_p = {}
        for fname, names in comps.items():
            Z = np.column_stack([oof[n] for n in names])
            _, oof_p[fname] = crossfit_fusion(Z, y, folds, names, cfg["model"]["calibration"]["prior"])
            obj.fusion[fname] = LateFusion(names, cfg["model"]["calibration"]["prior"]).fit(Z, y)
        band = choose_band(y, oof_p["core"], oof_p["full"])
        log(f"[train] routing band {band['band']}")
        for name, spec in specs.items():
            det = Detector(spec)
            if name == "gmm":
                det.fit(table, y, groups, arrays=arrays)
            else:
                det.fit(table, y, groups)
            obj.dets[name] = det
            log(f"[train] final {name} fitted {det.selection or ''}")
        prior = cfg["model"]["calibration"]["prior"]
        oof_metrics = {n: binary_metrics(y, crossfit_calibrate(oof[n], y, folds, prior), raw=oof[n])
                       for n in oof}
        oof_metrics.update({f"fusion_{k}": binary_metrics(y, v) for k, v in oof_p.items()})
        ids = sorted(table["sha256"])
        obj.meta = {
            "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "config_hash": config_hash(cfg), "package_versions": package_versions(),
            "n_train": int(len(table)), "n_real": int((y == 0).sum()), "n_synthetic": int((y == 1).sum()),
            "training_content_sha256_hash": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
            "splits_used": sorted(table["split"].astype(str).unique().tolist()),
            "n_folds": n_folds, "routing": band, "oof_metrics": oof_metrics,
            "score_polarity": "higher = more synthetic; probabilities calibrated for prior 0.5",
        }
        obj._train_rows = table[["sha256", "path", "label", "group_id", "source_id", "split"]].copy()
        obj._oof = pd.DataFrame({"sha256": table["sha256"].values, "fold": folds, "label": y,
                                 **{f"oof_{k}": v for k, v in oof.items()},
                                 **{f"oof_p_{k}": v for k, v in oof_p.items()}})
        return obj

    # ------------------------------------------------------------------ score
    def component_scores(self, table: pd.DataFrame, store: FrameStore | None = None,
                         which=("gmm", "lr_core", "lr_full")) -> pd.DataFrame:
        out = pd.DataFrame(index=table.index)
        if "gmm" in which:
            store = store or FrameStore(self.cfg, self.dets["gmm"].spec.gmm["feature"])
            arrays = store.many(table["sha256"].tolist())
            out["gmm"] = [self.dets["gmm"].gmm.score(a) if a is not None else np.nan for a in arrays]
        for name in which:
            if name != "gmm":
                out[name] = self.dets[name].decision_function(table)
        return out

    def probabilities(self, comps: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=comps.index)
        for fname, fus in self.fusion.items():
            if all(n in comps for n in fus.names):
                out[f"p_{fname}"] = fus.predict(comps[fus.names].values)
        return out

    def route(self, p_core: np.ndarray, p_full: np.ndarray | None, mode: str) -> tuple[np.ndarray, np.ndarray]:
        if mode == "all_eligible":
            return p_full, np.ones(len(p_full), bool)
        lo, hi = self.meta["routing"]["band"]
        inside = (p_core >= lo) & (p_core <= hi)
        final = p_core.copy()
        if p_full is not None:
            final[inside] = p_full[inside]
        return final, inside

    # ---------------------------------------------------------------- persist
    def save(self, directory: str | Path) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        for name, det in self.dets.items():
            det.save(d / name)
        payload = {"meta": self.meta, "fusion": {k: v.to_dict() for k, v in self.fusion.items()},
                   "config": self.cfg}
        (d / "bundle.json").write_text(json.dumps(payload, indent=1, default=str))
        if hasattr(self, "_train_rows"):
            self._train_rows.to_csv(d / "training_rows.tsv", sep="\t", index=False)
            self._oof.to_csv(d / "training_oof_scores.tsv", sep="\t", index=False)

    @classmethod
    def load(cls, directory: str | Path) -> "ModelBundle":
        d = Path(directory)
        if not (d / "bundle.json").exists():
            raise FileNotFoundError(f"no model bundle at {d} (expected bundle.json); train one with "
                                    f"`python -m hearsay_dsp.cli train` - prediction never retrains")
        payload = json.loads((d / "bundle.json").read_text())
        obj = cls()
        obj.meta, obj.cfg = payload["meta"], payload["config"]
        obj.fusion = {k: LateFusion.from_dict(v) for k, v in payload["fusion"].items()}
        for name in ("gmm", "lr_core", "lr_full"):
            obj.dets[name] = Detector.load(d / name)
        return obj
