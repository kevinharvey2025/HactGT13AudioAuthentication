"""Test-time concept discovery and composition with a diffusion model (Wang, Gupta, Zhu & MacLellan,
"Test-Time Compositional Generalization in Diffusion Models via Concept Discovery", arXiv 2605.07078), applied to
detector embeddings. Prototype covariances follow Wang, Singaravadivelan & MacLellan, arXiv 2609.13047, Eq. 9.

For a query z_q (a whitened detector embedding) and an unconditional DDPM over the same space:
  1. discovery: for every noise level t in the grid (default 50..400 step 25) and S random starts,
       x_t = sqrt(abar_t) z_q + sqrt(1 - abar_t) eps,  then Adam ascent on the score  grad log p_t  (150 steps);
     the mode x* gives a prototype mean  m = x* / sqrt(abar_t)  (Tweedie's denoiser at zero score) and a diagonal
     covariance  S = ((1 - abar_t) / sqrt(abar_t)) * diag(d xhat0 / d x)  estimated with 4 Rademacher probes of
     central finite differences of the denoiser xhat0(x) = (x - sqrt(1 - abar_t) eps_hat(x, t)) / sqrt(abar_t);
  2. selection: greedy maximization of the facility-location objective
       F(S) = sum_r max(baseline_r, max_{j in S} log N(z_{q,r}; m_{j,r}, S_{j,r})),
     baseline = the whitened root N(0, 1) per dimension (monotone submodular, so greedy is within 1 - 1/e), K <= 3;
  3. composition: per-dimension weights w_j(r) ∝ exp(l_{j,r} / tau), tau = 0.5, and the product of experts
       Sigma^-1 = sum_j diag(w_j) S_j^-1,   mu = Sigma sum_j diag(w_j) S_j^-1 m_j.
Returned per query: the selected prototypes (noise level, mean, variance, share of dimensions, gain) and the composed
Gaussian; callers interpret each prototype by categorizing its mean in a concept tree (scripts/run_concepts.py).
"""
import math
from dataclasses import dataclass, field

import numpy as np
import torch

from .ddpm import score_fn

LOG2PI = math.log(2 * math.pi)


@dataclass
class TTCGConfig:
    t_grid: tuple = tuple(range(50, 401, 25))     # the paper's grid (ColorMNIST): t = 50 ... 400, step 25
    starts: int = 32                              # random starts per (query, t); the paper uses 128
    steps: int = 150                              # Adam ascent iterations
    lr: float = 0.05                              # base step, scaled by sigma_t
    probes: int = 4                               # Hutchinson probes for the covariance diagonal
    fd_eps: float = 1e-2                          # finite-difference step
    var_floor: float = 1e-3
    k: int = 3                                    # prototypes per query
    tau: float = 0.5                              # composition temperature
    dedupe: float = 0.05                          # merge modes closer than this (whitened units, per dim RMS)
    seed: int = 0


def _denoise(net, sched, x, t):
    a = sched.ab(t).to(x.device)
    tt = torch.full((len(x),), t, device=x.device, dtype=torch.long)
    return (x - (1 - a).sqrt() * net(x, tt)) / a.sqrt()


@torch.no_grad()
def _jac_diag(net, sched, x, t, probes, eps, gen):
    """Hutchinson estimate of diag(d xhat0 / dx) with central finite differences."""
    acc = torch.zeros_like(x)
    for _ in range(probes):
        v = (torch.randint(0, 2, x.shape, generator=gen) * 2 - 1).to(x.device, x.dtype)
        acc += v * (_denoise(net, sched, x + eps * v, t) - _denoise(net, sched, x - eps * v, t)) / (2 * eps)
    return acc / probes


def discover(net, sched, zq, cfg=TTCGConfig(), device="cuda"):
    """zq [n, d] queries -> list (per query) of candidate prototypes as arrays (t [P], mean [P, d], var [P, d])."""
    gen = torch.Generator(device="cpu").manual_seed(cfg.seed)
    zq_t = torch.as_tensor(zq, dtype=torch.float32, device=device)
    n, d = zq_t.shape
    per = [dict(t=[], mean=[], var=[]) for _ in range(n)]
    for t in cfg.t_grid:
        a = sched.ab(t).to(device)
        sigma = float((1 - a).sqrt())
        x0 = a.sqrt() * zq_t.repeat_interleave(cfg.starts, 0) + (1 - a).sqrt() * torch.randn(
            n * cfg.starts, d, generator=gen).to(device)
        x = x0.clone().requires_grad_(True)
        opt = torch.optim.Adam([x], lr=cfg.lr * max(sigma, 0.02))
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.steps)
        for _ in range(cfg.steps):
            opt.zero_grad()
            with torch.no_grad():
                x.grad = -score_fn(net, sched, x, t)          # ascend log p_t
            opt.step()
            sch.step()
        xs = x.detach()
        mean = xs / a.sqrt()                                  # Tweedie at zero score
        jd = _jac_diag(net, sched, xs, t, cfg.probes, cfg.fd_eps, gen)
        var = torch.clamp((1 - a) / a.sqrt() * jd, min=cfg.var_floor)   # DMCF Eq. 9 (diagonal)
        mean, var = mean.view(n, cfg.starts, d).cpu().numpy(), var.view(n, cfg.starts, d).cpu().numpy()
        for i in range(n):
            m, v = mean[i], var[i]
            keep = [0]                                         # dedupe converged modes at this t
            for j in range(1, len(m)):
                if np.min(np.sqrt(((m[keep] - m[j]) ** 2).mean(1))) > cfg.dedupe:
                    keep.append(j)
            per[i]["t"] += [t] * len(keep)
            per[i]["mean"].append(m[keep])
            per[i]["var"].append(v[keep])
    return [dict(t=np.array(p["t"]), mean=np.concatenate(p["mean"]), var=np.concatenate(p["var"])) for p in per]


def select_and_compose(z, cand, cfg=TTCGConfig()):
    """Greedy facility-location selection (root baseline) and product-of-experts composition for one query."""
    L = -0.5 * (((z[None] - cand["mean"]) ** 2) / cand["var"] + np.log(cand["var"]) + LOG2PI)   # [P, d]
    best = -0.5 * (z ** 2 + LOG2PI)                                                             # root N(0, 1)
    chosen, gains = [], []
    for _ in range(min(cfg.k, len(L))):
        gain = np.maximum(L, best[None]).sum(1) - best.sum()
        gain[chosen] = -np.inf
        j = int(np.argmax(gain))
        if gain[j] <= 1e-9:
            break
        chosen.append(j)
        gains.append(float(gain[j]))
        best = np.maximum(best, L[j])
    if not chosen:
        return dict(selected=[], coverage_gain=0.0)
    Lc = L[chosen]
    w = np.exp((Lc - Lc.max(0)) / cfg.tau)
    w /= w.sum(0, keepdims=True)
    prec = (w / cand["var"][chosen]).sum(0)
    mu = (w * cand["mean"][chosen] / cand["var"][chosen]).sum(0) / prec
    sel = [dict(idx=j, t=int(cand["t"][j]), share=float(w[i].mean()), gain=g,
                mean=cand["mean"][j], var=cand["var"][j]) for i, (j, g) in enumerate(zip(chosen, gains))]
    return dict(selected=sel, coverage_gain=float(sum(gains)), composed_mean=mu, composed_var=1.0 / prec,
                composed_loglik=float((-0.5 * ((z - mu) ** 2 * prec - np.log(prec) + LOG2PI)).mean()))
