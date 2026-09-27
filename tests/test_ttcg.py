"""TTCG (hearsay/diffusion/ttcg.py) on a Gaussian mixture, where the noised score, the modes and the posterior
variances are known in closed form: mode ascent + Tweedie must recover the component means, the Hutchinson
covariance (arXiv 2609.13047 Eq. 9) must equal Var[x0 | x_t] = (1 - a) s^2 / (a s^2 + 1 - a), and selection +
composition must explain a query that mixes two concepts across dimensions."""
import numpy as np
import pytest
import torch

from hearsay.diffusion import ttcg
from hearsay.diffusion.ddpm import Schedule


class MixtureEps(torch.nn.Module):
    """Exact eps(x, t) = -sqrt(1 - abar_t) grad log p_t(x) for p_0 = mean_k N(mu_k, s_k^2 I)."""

    def __init__(self, mus, stds, sched):
        super().__init__()
        self.mu, self.s2, self.sched = torch.tensor(mus).float(), torch.tensor(stds).float() ** 2, sched

    def forward(self, x, tt):
        a = self.sched.ab(tt.cpu()).to(x)[:, None]                     # [B, 1]
        v = a * self.s2[None] + (1 - a)                                # [B, K] per-component variance at t
        diff = x[:, None, :] - a.sqrt()[:, :, None] * self.mu[None]    # [B, K, d]
        r = torch.softmax(-0.5 * (diff ** 2).sum(-1) / v - 0.5 * x.shape[1] * torch.log(v), 1)
        return (1 - a).sqrt() * (r[:, :, None] * diff / v[:, :, None]).sum(1)


@pytest.fixture(scope="module")
def sched():
    return Schedule(1000)


def test_modes_and_posterior_variances(sched):
    mus, stds = [[2.0] * 4, [-2.0] * 4], [0.5, 0.8]
    cfg = ttcg.TTCGConfig(t_grid=(50, 150, 300), starts=8, steps=300, probes=8, fd_eps=1e-3)
    cand = ttcg.discover(MixtureEps(mus, stds, sched), sched, np.array([mus[0]], np.float32), cfg, device="cpu")[0]
    for t in cfg.t_grid:
        m = cand["t"] == t
        j = np.argmin(((cand["mean"][m] - mus[0]) ** 2).sum(1))
        np.testing.assert_allclose(cand["mean"][m][j], mus[0], atol=0.05)
        a = float(sched.ab(t))
        np.testing.assert_allclose(cand["var"][m][j], (1 - a) * 0.25 / (a * 0.25 + 1 - a), rtol=0.1)


def test_composition_across_dimensions(sched):
    mus = [[2.0] * 6, [-2.0] * 6]
    z = np.array([2.0] * 3 + [-2.0] * 3, np.float32)                   # dims 0-2 from concept A, 3-5 from concept B
    cfg = ttcg.TTCGConfig(t_grid=(400,), starts=32, steps=300)
    cand = ttcg.discover(MixtureEps(mus, [0.5, 0.5], sched), sched, z[None], cfg, device="cpu")[0]
    comp = ttcg.select_and_compose(z, cand, cfg)
    assert len(comp["selected"]) >= 2
    W = np.stack([s["weight"] for s in comp["selected"]])
    owner_mean = np.stack([s["mean"] for s in comp["selected"]])[W.argmax(0), np.arange(6)]
    np.testing.assert_allclose(owner_mean, z, atol=0.1)                # each dimension explained by the right concept
    np.testing.assert_allclose(comp["composed_mean"], z, atol=0.1)
    np.testing.assert_allclose(W.sum(0), 1.0, atol=1e-6)


def test_root_explains_a_typical_query():
    """Prototypes far from the query never beat the whitened root N(0, 1): nothing is selected."""
    z = np.zeros(4, np.float32)
    cand = dict(t=np.array([50, 50]), mean=np.array([[6.0] * 4, [-6.0] * 4]), var=np.full((2, 4), 0.05))
    assert ttcg.select_and_compose(z, cand)["selected"] == []
