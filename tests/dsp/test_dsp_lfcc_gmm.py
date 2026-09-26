from pathlib import Path

import numpy as np
import pytest
from scipy.stats import norm

from hearsay_dsp.features.lfcc import (MATLAB_EPS, lfcc_bp, matlab_buffer_nodelay, matlab_deltas,
                                       trimf)
from hearsay_dsp.models.gmm import GMMPair, OfficialPretrainedGMM, gmm_from_arrays

ROOT = Path(__file__).resolve().parents[2]
PRETRAINED = ROOT / "cache/dsp/external/pre_trained_LA_LFCC-GMM.mat"


def test_buffer_nodelay_matches_matlab_semantics():
    f = matlab_buffer_nodelay(np.arange(1, 11, dtype=float), 4, 2)
    assert f.tolist() == [[1, 2, 3, 4], [3, 4, 5, 6], [5, 6, 7, 8], [7, 8, 9, 10]]
    f = matlab_buffer_nodelay(np.arange(1, 12, dtype=float), 4, 2)
    assert f.shape == (5, 4) and f[-1].tolist() == [9, 10, 11, 0]


def test_trimf_matches_matlab_definition():
    x = np.array([0.0, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0])
    assert trimf(x, 1, 2, 3).tolist() == [0, 0, 0.5, 1, 0.5, 0, 0]
    assert trimf(np.array([0.0, 0.5, 1.0]), 0, 0, 1).tolist() == [1, 0.5, 0]


def test_deltas_are_halved_central_differences_with_replicated_edges():
    x = (np.arange(6, dtype=float) ** 2)[:, None]
    d = matlab_deltas(x, 1)[:, 0]
    # interior: (x[t+1] - x[t-1]) / 2; edges replicate x[0] and x[-1]
    assert d.tolist() == [0.5, 2.0, 4.0, 6.0, 8.0, 4.5]


def test_lfcc_shapes_frame_count_and_silence_value():
    sr = 16000
    x = np.zeros(sr)
    s, d, dd, frames = lfcc_bp(x, sr)
    assert s.shape == (66, 19) and d.shape == s.shape and dd.shape == s.shape  # ceil((16000-240)/240)
    # an all-zero frame gives c0 = log10(eps) * sqrt(70): the value the official spoof GMM models
    assert s[0, 0] == pytest.approx(np.log10(MATLAB_EPS) * np.sqrt(70))
    assert np.allclose(s[:, 1:], 0, atol=1e-9)


def test_custom_lfcc_is_gain_invariant_without_c0():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(16000) * 0.05
    a, *_ = lfcc_bp(x, 16000, high_hz=7000, n_coeff=20)
    b, *_ = lfcc_bp(4.0 * x, 16000, high_hz=7000, n_coeff=20)
    assert not np.allclose(a[:, 0], b[:, 0])            # c0 moves with gain
    assert np.allclose(a[:, 1:], b[:, 1:], atol=1e-8)   # c1.. do not


def _two_class_arrays(rng, n_clips=40, n_frames=50, dim=3):
    real = [{"custom": rng.normal(-1.0, 1.0, (n_frames, dim)), "nonsilent": np.ones(n_frames, bool),
             "active": np.ones(n_frames, bool)} for _ in range(n_clips)]
    fake = [{"custom": rng.normal(+1.0, 1.0, (n_frames, dim)), "nonsilent": np.ones(n_frames, bool),
             "active": np.ones(n_frames, bool)} for _ in range(n_clips)]
    return real, fake


def test_gmm_score_polarity_higher_is_synthetic():
    rng = np.random.default_rng(1)
    real, fake = _two_class_arrays(rng)
    g = GMMPair(feature="custom", frames="all", n_components=2, seed=0).fit(real, fake)
    assert g.score({"custom": rng.normal(1.0, 1.0, (60, 3))}) > 0
    assert g.score({"custom": rng.normal(-1.0, 1.0, (60, 3))}) < 0


def test_gmm_score_equals_mean_loglik_difference():
    g = GMMPair(feature="custom", frames="all")
    g.real = gmm_from_arrays(np.array([[0.0]]), np.array([[1.0]]), np.array([1.0]))
    g.fake = gmm_from_arrays(np.array([[2.0]]), np.array([[4.0]]), np.array([1.0]))
    x = np.array([[0.5], [1.0], [3.0]])
    expected = np.mean(norm.logpdf(x[:, 0], 2.0, 2.0)) - np.mean(norm.logpdf(x[:, 0], 0.0, 1.0))
    assert g.score_frames(x) == pytest.approx(expected)


def test_frame_policy_excludes_digital_silence():
    arrays = {"custom": np.array([[0.0], [5.0], [5.0]]), "nonsilent": np.array([False, True, True]),
              "active": np.array([False, True, False])}
    from hearsay_dsp.models.gmm import select_frames
    assert select_frames(arrays, "custom", "nonsilent").ravel().tolist() == [5.0, 5.0]
    assert select_frames(arrays, "custom", "active").ravel().tolist() == [5.0]
    assert select_frames(arrays, "custom", "all").shape[0] == 3


@pytest.mark.skipif(not PRETRAINED.exists(), reason="pretrained .mat not downloaded")
def test_official_pretrained_polarity_matches_matlab_llk():
    pre = OfficialPretrainedGMM(str(PRETRAINED))
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, (30, 57))

    def matlab_llk(data, mu, sigma, w):  # GMM/compute_llk.m + lgmmprob.m, data is dim x frames
        ndim = data.shape[0]
        C = np.sum(mu * mu / sigma, 0) + np.sum(np.log(sigma), 0)
        D = (1.0 / sigma).T @ (data * data) - 2 * (mu / sigma).T @ data + ndim * np.log(2 * np.pi)
        lp = -0.5 * (C[:, None] + D) + np.log(w)[:, None]
        mx = lp.max(0)
        return mx + np.log(np.exp(lp - mx).sum(0))

    import scipy.io as sio
    m = sio.loadmat(str(PRETRAINED), struct_as_record=False)
    gen, spo = m["genuineGMM"][0, 0], m["spoofGMM"][0, 0]
    official = (np.mean(matlab_llk(x.T, gen.m, gen.s, gen.w.ravel()))
                - np.mean(matlab_llk(x.T, spo.m, spo.s, spo.w.ravel())))
    assert pre.score({"official": x}) == pytest.approx(-official, rel=1e-6)
