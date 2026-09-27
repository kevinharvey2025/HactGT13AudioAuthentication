"""The organizers' metric: parity with their evaluation package, tie handling, decision thresholds."""
import io
import tarfile
import types
import zipfile
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import roc_curve

from hearsay import metrics

ROOT = Path(__file__).resolve().parents[1]
MODULE = "HackGTMinDCF/asvspoof5/evaluation-package/calculate_modules.py"


def organizers():
    """calculate_modules.py of the organizers' package (data/HackGTMinDCF, or the zip as distributed)."""
    f = ROOT / "data" / MODULE
    if f.exists():
        src = f.read_text()
    else:
        zips = sorted(ROOT.glob("data/*HackGTMinDCF*.zip"))
        if not zips:
            pytest.skip("organizers' evaluation package not under data/")
        with zipfile.ZipFile(zips[0]) as z:
            tar = tarfile.open(fileobj=io.BytesIO(z.read("HackGTMinDCF.tar")))
            src = tar.extractfile("./" + MODULE).read().decode()
    mod = types.ModuleType("calculate_modules")
    exec(compile(src, "calculate_modules.py", "exec"), mod.__dict__)
    return mod


def gaussian_scores(n_real, n_fake, shift, seed):
    rng = np.random.default_rng(seed)
    y = np.r_[np.zeros(n_real), np.ones(n_fake)].astype(int)
    return y, np.r_[rng.normal(0, 1, n_real), rng.normal(shift, 1, n_fake)]


@pytest.mark.parametrize("n_real,n_fake,shift", [(500, 300, 1.0), (1000, 200, 2.5), (50, 400, 0.3)])
def test_parity_with_organizers_package(n_real, n_fake, shift):
    cm = organizers()
    y, s = gaussian_scores(n_real, n_fake, shift, seed=n_real)
    # the package: bona fide is the target class and scores higher; ours: 1 = synthetic scores higher
    eer, frr, far, thr, _ = cm.compute_eer(-s[y == 0], -s[y == 1])
    for p, c_miss, c_fa in ((0.3, 1, 4), (0.05, 1, 10), (0.5, 1, 4)):
        ref, _ = cm.compute_mindcf(frr, far, thr, p, c_miss, c_fa)
        assert metrics.min_dcf(y, s, p, c_miss, c_fa) == pytest.approx(ref, abs=1e-12)
    assert metrics.eer(y, s) == pytest.approx(eer, abs=1e-12)


def test_ties_are_never_optimistic():
    """The package walks every sorted position; with tied scores that includes unrealizable operating points."""
    cm = organizers()
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 3000)
    s = np.round(rng.normal(1.5 * y, 1), 1)
    _, frr, far, thr, _ = cm.compute_eer(-s[y == 0], -s[y == 1])
    ref, _ = cm.compute_mindcf(frr, far, thr, 0.3, 1, 4)
    assert metrics.min_dcf(y, s) >= ref - 1e-12


def test_matches_sklearn_curve_with_ties():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 2000)
    s = np.round(rng.normal(y, 1), 2)
    fpr, tpr, _ = roc_curve(y, s, drop_intermediate=False)
    assert metrics.min_dcf(y, s) == pytest.approx((0.7 * fpr + 1.2 * (1 - tpr)).min() / 0.7, abs=1e-12)
    dcf, eer = metrics.min_dcf_eer(y, s)
    assert (dcf, eer) == (metrics.min_dcf(y, s), metrics.eer(y, s))


def test_reference_points():
    y = np.array([0, 0, 1, 1])
    assert metrics.min_dcf(y, np.array([0, 0, 1, 1.0])) == 0 and metrics.eer(y, np.array([0, 0, 1, 1.0])) == 0
    assert metrics.min_dcf(y, np.full(4, 0.5)) == pytest.approx(1.0)       # a constant decision
    assert metrics.min_dcf(y, np.array([1, 1, 0, 0.0])) == pytest.approx(1.0)  # inverted: never better than constant
    assert metrics.eer(y, np.array([1, 1, 0, 0.0])) == 1.0


def test_thresholds_and_priors():
    assert metrics.bayes_threshold(calib_prior=0.3) == pytest.approx(0.2)
    assert metrics.bayes_threshold(calib_prior=0.5) == pytest.approx(0.3684, abs=1e-4)
    assert metrics.effective_prior() == pytest.approx(1.2 / 1.9)


def test_bayes_decisions_of_calibrated_scores_reach_min_dcf():
    """For exact posteriors at the 30% prior, flagging P > 0.2 is optimal: actDCF ~ minDCF."""
    rng = np.random.default_rng(3)
    n = 40000
    y = (rng.random(n) < 0.5).astype(int)
    x = rng.normal(np.where(y == 1, 1.0, -1.0), 1.0)
    llr = 2.0 * x                                             # log N(x; 1, 1) / N(x; -1, 1)
    p = 1 / (1 + np.exp(-(llr + np.log(0.3 / 0.7))))
    assert metrics.act_dcf(y, p) == pytest.approx(metrics.min_dcf(y, p), abs=0.01)
