"""An unconditional DDPM over vectors: the detector's whitened embeddings (hearsay/concepts.py Space), both classes.

Standard epsilon-prediction objective with a linear beta schedule (T = 1000) and a small residual MLP with FiLM time
conditioning; the EMA weights are returned. TTCG (hearsay/diffusion/ttcg.py) reads the model through `score_fn`,
grad_x log p_t(x) = -eps_hat(x, t) / sqrt(1 - abar_t): its modes at noise level t are the level-t prototypes.
"""
import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .. import config


class Schedule:
    def __init__(self, T=1000, beta_1=1e-4, beta_T=0.02):
        self.T = T
        self.betas = torch.linspace(beta_1, beta_T, T, dtype=torch.float64)
        self.abar = torch.cumprod(1 - self.betas, 0)              # abar[t-1] for t = 1..T

    def ab(self, t):
        """alpha-bar at integer timestep(s) t in 1..T (python int or long tensor)."""
        return self.abar[torch.as_tensor(t) - 1].float()


def timestep_embedding(t, dim=128):
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    a = t.float()[:, None] * freqs[None]
    return torch.cat([torch.sin(a), torch.cos(a)], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, width, temb, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.film = nn.Linear(temb, 2 * width)
        self.lin1, self.lin2 = nn.Linear(width, width), nn.Linear(width, width)
        self.drop = nn.Dropout(dropout)

    def forward(self, h, te):
        scale, shift = self.film(te).chunk(2, dim=-1)
        z = self.norm(h) * (1 + scale) + shift
        return h + self.lin2(self.drop(nn.functional.silu(self.lin1(nn.functional.silu(z)))))


class EpsMLP(nn.Module):
    """eps-prediction network for vectors: x_t [B, d], t [B] -> eps_hat [B, d]."""

    def __init__(self, d, width=512, depth=4, temb=128, dropout=0.1):
        super().__init__()
        self.temb = temb
        self.t_mlp = nn.Sequential(nn.Linear(temb, temb), nn.SiLU(), nn.Linear(temb, temb))
        self.inp = nn.Linear(d, width)
        self.blocks = nn.ModuleList([ResBlock(width, temb, dropout) for _ in range(depth)])
        self.out = nn.Sequential(nn.LayerNorm(width), nn.SiLU(), nn.Linear(width, d))
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    def forward(self, x, t):
        te = self.t_mlp(timestep_embedding(t, self.temb))
        h = self.inp(x)
        for b in self.blocks:
            h = b(h, te)
        return self.out(h)


@dataclass
class DDPMConfig:
    width: int = 512
    depth: int = 4
    dropout: float = 0.1
    steps: int = 8000
    batch: int = 512
    lr: float = 1e-3
    weight_decay: float = 1e-4
    ema: float = 0.999
    T: int = 1000
    seed: int = 0


def train_ddpm(z, cfg=DDPMConfig(), device=None, log_every=1000):
    """Standard DDPM objective E||eps - eps_hat(sqrt(abar) z + sqrt(1-abar) eps, t)||^2 on vectors z."""
    device = device or config.device()
    torch.manual_seed(cfg.seed)
    sched = Schedule(cfg.T)
    net = EpsMLP(z.shape[1], cfg.width, cfg.depth, dropout=cfg.dropout).to(device)
    ema = EpsMLP(z.shape[1], cfg.width, cfg.depth, dropout=0.0).to(device)
    ema.load_state_dict(net.state_dict())
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.steps)
    data = torch.from_numpy(np.asarray(z, np.float32)).to(device)
    abar = sched.abar.float().to(device)
    g = torch.Generator(device="cpu").manual_seed(cfg.seed)
    history = []
    for step in range(cfg.steps):
        i = torch.randint(0, len(data), (cfg.batch,), generator=g).to(device)
        t = torch.randint(1, cfg.T + 1, (cfg.batch,), generator=g).to(device)
        x0, eps = data[i], torch.randn(cfg.batch, data.shape[1], generator=g).to(device)
        a = abar[t - 1][:, None]
        loss = ((net(a.sqrt() * x0 + (1 - a).sqrt() * eps, t) - eps) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        lr_sched.step()
        with torch.no_grad():
            for pe, pn in zip(ema.parameters(), net.parameters()):
                pe.mul_(cfg.ema).add_(pn, alpha=1 - cfg.ema)
        if step % log_every == 0 or step == cfg.steps - 1:
            history.append((step, loss.item()))
    return ema.eval(), sched, history



def score_fn(net, sched, x, t):
    """grad_x log p_t(x) = -eps_hat(x, t) / sqrt(1 - abar_t)."""
    tt = torch.full((len(x),), t, device=x.device, dtype=torch.long)
    return -net(x, tt) / (1 - sched.ab(t).to(x.device)).sqrt()
