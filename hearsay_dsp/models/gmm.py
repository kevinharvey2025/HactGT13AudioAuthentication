"""LFCC-GMM back-end.

Two diagonal-covariance GMMs are fitted on training frames only, one per
class. Our score convention is explicit:

    raw_spoof_score = mean_t log p(x_t | synthetic GMM) - mean_t log p(x_t | real GMM)

so higher means more synthetic. The ASVspoof baselines report the opposite
(llk_genuine - llk_spoof); `OfficialPretrainedGMM` negates it.

Each recording contributes at most `max_frames_per_clip` frames (seeded
sampling), so long clips do not dominate training.
"""

from __future__ import annotations

import numpy as np
import scipy.io as sio
from sklearn.mixture import GaussianMixture


def select_frames(arrays: dict, feature: str, frames: str) -> np.ndarray:
    feats = arrays[feature]
    if frames == "all":
        return feats
    if frames in ("nonsilent", "active"):
        return feats[arrays[frames].astype(bool)]
    raise ValueError(f"unknown frame policy {frames}")


def sample_rows(x: np.ndarray, max_n: int, rng: np.random.Generator) -> np.ndarray:
    if x.shape[0] <= max_n:
        return x
    idx = np.sort(rng.choice(x.shape[0], size=max_n, replace=False))
    return x[idx]


def gmm_to_arrays(g: GaussianMixture) -> dict:
    return {"means": g.means_, "covariances": g.covariances_, "weights": g.weights_}


def gmm_from_arrays(means, covariances, weights) -> GaussianMixture:
    g = GaussianMixture(n_components=weights.size, covariance_type="diag")
    g.means_ = np.asarray(means, dtype=float)
    g.covariances_ = np.asarray(covariances, dtype=float)
    g.weights_ = np.asarray(weights, dtype=float) / np.sum(weights)
    g.precisions_cholesky_ = 1.0 / np.sqrt(g.covariances_)
    g.converged_ = True
    g.n_features_in_ = g.means_.shape[1]
    return g


class GMMPair:
    def __init__(self, feature="custom", frames="nonsilent", n_components=64,
                 max_frames_per_clip=200, max_frames_total=150000, max_iter=100, reg_covar=1e-4,
                 n_init=1, seed=0):
        self.params = dict(feature=feature, frames=frames, n_components=n_components,
                           max_frames_per_clip=max_frames_per_clip,
                           max_frames_total=max_frames_total, max_iter=max_iter,
                           reg_covar=reg_covar, n_init=n_init, seed=seed)
        self.real: GaussianMixture | None = None
        self.fake: GaussianMixture | None = None
        self.train_info: dict = {}

    def _stack(self, arrays_list, rng) -> tuple[np.ndarray, int]:
        p = self.params
        chunks = []
        for arrays in arrays_list:
            x = select_frames(arrays, p["feature"], p["frames"])
            if x.shape[0]:
                chunks.append(sample_rows(x, p["max_frames_per_clip"], rng))
        if not chunks:
            raise ValueError("no training frames")
        x = sample_rows(np.vstack(chunks), p["max_frames_total"], rng)
        return x.astype(np.float64), len(chunks)

    def _fit_one(self, x: np.ndarray, seed: int) -> GaussianMixture:
        p = self.params
        k = int(min(p["n_components"], max(1, x.shape[0] // 20)))
        return GaussianMixture(n_components=k, covariance_type="diag", max_iter=p["max_iter"],
                               reg_covar=p["reg_covar"], n_init=p["n_init"],
                               init_params="k-means++", random_state=seed).fit(x)

    def fit(self, real_arrays: list, fake_arrays: list) -> "GMMPair":
        rng = np.random.default_rng(self.params["seed"])
        xr, nr = self._stack(real_arrays, rng)
        xf, nf = self._stack(fake_arrays, rng)
        self.real = self._fit_one(xr, self.params["seed"])
        self.fake = self._fit_one(xf, self.params["seed"] + 1)
        self.train_info = {"n_clips_real": nr, "n_clips_fake": nf, "n_frames_real": int(xr.shape[0]),
                           "n_frames_fake": int(xf.shape[0]),
                           "converged": [bool(self.real.converged_), bool(self.fake.converged_)]}
        return self

    def score_frames(self, x: np.ndarray) -> float:
        if x.shape[0] == 0:
            return float("nan")
        x = x.astype(np.float64)
        return float(self.fake.score(x) - self.real.score(x))

    def score(self, arrays: dict) -> float:
        return self.score_frames(select_frames(arrays, self.params["feature"], self.params["frames"]))

    def to_arrays(self) -> dict:
        out = {}
        for name, g in (("real", self.real), ("fake", self.fake)):
            for k, v in gmm_to_arrays(g).items():
                out[f"{name}_{k}"] = v
        return out

    @classmethod
    def from_arrays(cls, params: dict, arrays: dict, train_info: dict | None = None) -> "GMMPair":
        obj = cls(**params)
        obj.real = gmm_from_arrays(arrays["real_means"], arrays["real_covariances"], arrays["real_weights"])
        obj.fake = gmm_from_arrays(arrays["fake_means"], arrays["fake_covariances"], arrays["fake_weights"])
        obj.train_info = train_info or {}
        return obj


class OfficialPretrainedGMM(GMMPair):
    """ASVspoof 2021 LA pretrained LFCC-GMM (MATLAB .mat), evaluated unchanged.

    Trained by the organizers on ASVspoof 2019 LA train (bona fide VCTK speech and
    TTS/VC attacks A01-A06). Uses the official LFCC (57 dims) on all frames, as in
    the reference scoring script. Returns our polarity (higher = synthetic).
    """

    def __init__(self, mat_path: str):
        super().__init__(feature="official", frames="all", n_components=512)
        m = sio.loadmat(mat_path, squeeze_me=False, struct_as_record=False)
        g, s = m["genuineGMM"][0, 0], m["spoofGMM"][0, 0]
        self.real = gmm_from_arrays(g.m.T, g.s.T, g.w.ravel())
        self.fake = gmm_from_arrays(s.m.T, s.s.T, s.w.ravel())
        self.train_info = {"source": "https://www.asvspoof.org/asvspoof2021/pre_trained_LA_LFCC-GMM.zip",
                           "training_data": "ASVspoof 2019 LA train"}
