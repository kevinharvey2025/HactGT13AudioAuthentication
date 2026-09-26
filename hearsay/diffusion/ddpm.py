"""D2: a small DDPM on SSL embeddings of bona fide speech, read out as mode-path features (H2).

Hypothesis H2 ("coarse right, fine wrong"): relative to a model of real speech, synthetic clips
look normal at high noise levels (content, speaker) and diverge at low noise levels (phase,
micro-prosody, breath). Following Wang, Gupta, Zhu & MacLellan (arXiv 2605.07078), a datum's
modes across noise levels are clean-space Gaussian prototypes: ascend the learned score of p_t
from the noised datum to a mode x*_t, map it to clean space with Tweedie (m = x*_t / sqrt(abar_t)),
and measure how well the datum fits that prototype. Stacked over t this gives a curve.

Pipeline (plan section 7.2, D2):
  pooled SSL embedding -> standardize -> [optional speaker-subspace removal] -> PCA-whiten (64-256)
  -> DDPM eps-prediction MLP (linear beta, T=1000), trained on bona fide incl. augmented views
  -> for t in T_GRID: noise to t, Adam ascent on log p_t to x*_t, prototype m_t, diag covariance
  -> per-t features: diag-Gaussian log-lik of z, ||z - m||, ascent displacement/path, DSM loss
  -> small classifier on the curve (fit on held-out bona fide + fakes, never on DDPM training data)

Controls in the same whitened space (one-class Gaussian, GMM, kNN) make the go/no-go gate ask
whether diffusion adds anything beyond a plain density model.
"""
import math
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
from torch import nn

from .. import config

T_GRID = (10, 20, 30, 50, 75, 100, 150, 200, 300, 400)  # dense at low t, as in the plan


# ----------------------------------------------------------------------------- embedding space
class EmbeddingSpace:
    """standardize -> optional nuisance projection -> PCA whitening. Fit on bona fide only."""

    def __init__(self, dim=64, remove_speaker_dims=0):
        self.dim, self.remove_speaker_dims = dim, remove_speaker_dims

    def fit(self, x, speakers=None):
        x = np.asarray(x, np.float64)
        self.mu, self.sd = x.mean(0), x.std(0) + 1e-6
        xs = (x - self.mu) / self.sd
        self.nuisance = None
        if self.remove_speaker_dims and speakers is not None and len(np.unique(speakers)) > 1:
            # nuisance attribute projection: drop the top speaker-discriminant (LDA) directions so
            # "unfamiliar voice" is not read as "off the bona fide manifold"
            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
            k = min(self.remove_speaker_dims, len(np.unique(speakers)) - 1)
            lda = LinearDiscriminantAnalysis(n_components=k, solver="svd").fit(xs, speakers)
            q, _ = np.linalg.qr(lda.scalings_[:, :k])
            self.nuisance = q                                     # [D, k] orthonormal
            xs = xs - xs @ q @ q.T
        u, s, vt = np.linalg.svd(xs - xs.mean(0), full_matrices=False)
        self.center = xs.mean(0)
        self.components = vt[: self.dim]                          # [dim, D]
        self.scale = s[: self.dim] / math.sqrt(len(xs) - 1)       # component std
        self.explained = float((s[: self.dim] ** 2).sum() / (s ** 2).sum())
        return self

    def transform(self, x):
        xs = (np.asarray(x, np.float64) - self.mu) / self.sd
        if self.nuisance is not None:
            xs = xs - xs @ self.nuisance @ self.nuisance.T
        return (((xs - self.center) @ self.components.T) / self.scale).astype(np.float32)

    def state(self):
        return {k: v for k, v in self.__dict__.items()}

    @classmethod
    def from_state(cls, st):
        obj = cls.__new__(cls)
        obj.__dict__.update(st)
        return obj


# ----------------------------------------------------------------------------- DDPM
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
            history.append((step, float(loss)))
    return ema.eval(), sched, history


# ----------------------------------------------------------------------------- mode paths
@dataclass
class ModePathConfig:
    t_grid: tuple = T_GRID
    n_steps: int = 100          # Adam steps of score ascent per t
    lr: float = 0.05            # base step; scaled by sigma_t so small-t ascents don't overshoot
    n_draws: int = 2            # noise draws per (clip, t), averaged
    covariance: str = "fixed"   # "fixed" (per-t, per-dim, calibrated on bona fide) or "hutchinson"
    n_probes: int = 8           # Rademacher probes for the Hutchinson Hessian diagonal
    var_floor: float = 1e-3
    batch: int = 4096


def score_fn(net, sched, x, t):
    """grad_x log p_t(x) = -eps_hat(x, t) / sqrt(1 - abar_t)."""
    tt = torch.full((len(x),), t, device=x.device, dtype=torch.long)
    return -net(x, tt) / (1 - sched.ab(t).to(x.device)).sqrt()


def mode_ascent(net, sched, x0, t, cfg):
    """Adam ascent on log p_t from x0. Returns (x*, path length) per row."""
    sigma = float((1 - sched.ab(t)).sqrt())
    x = x0.clone().requires_grad_(True)
    opt = torch.optim.Adam([x], lr=cfg.lr * max(sigma, 0.02))
    lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.n_steps)
    path = torch.zeros(len(x0), device=x0.device)
    prev = x0.detach().clone()
    for _ in range(cfg.n_steps):
        opt.zero_grad()
        with torch.no_grad():
            x.grad = -score_fn(net, sched, x, t)                   # descend -log p_t == ascend log p_t
        opt.step()
        lr_sched.step()
        with torch.no_grad():
            path += (x - prev).norm(dim=1)
            prev = x.detach().clone()
    return x.detach(), path


def hutchinson_diag_hessian(net, sched, x, t, n_probes=8, seed=0):
    """diag of H = d score / dx at x via E[v * (H v)], v Rademacher. H is the Hessian of log p_t."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    diag = torch.zeros_like(x)
    for _ in range(n_probes):
        v = (torch.randint(0, 2, x.shape, generator=g) * 2 - 1).to(x.device, x.dtype)
        xi = x.detach().clone().requires_grad_(True)
        s = score_fn(net, sched, xi, t)
        (hv,) = torch.autograd.grad((s * v).sum(), xi)
        diag += v * hv
    return diag / n_probes


def prototype_variance_from_hessian(h_diag, abar, floor):
    """Gaussian component N(m, S) in clean space has p_t curvature -(abar S + (1-abar) I)^-1."""
    marginal = -1.0 / torch.clamp(h_diag, max=-1e-6)             # abar S + (1 - abar)
    return torch.clamp((marginal - (1 - abar)) / abar, min=floor)


def diag_gauss_loglik(z, m, var):
    """Per-dimension average log N(z; m, diag var) (comparable across embedding sizes)."""
    return -0.5 * (((z - m) ** 2) / var + torch.log(2 * math.pi * var)).mean(dim=1)


class D2Model:
    """A fitted D2 detector: embedding space + DDPM + per-t variances + curve classifier."""

    FEATURES = ("loglik", "dist", "disp", "path", "dsm")

    def __init__(self, space, net, sched, mp_cfg=ModePathConfig(), device=None):
        self.space, self.net, self.sched, self.cfg = space, net, sched, mp_cfg
        self.device = device or config.device()
        self.fixed_var = None        # {t: [d]} from bona fide calibration clips
        self.classifier = None

    # --- raw per-t quantities ----------------------------------------------------------------
    def _paths(self, z, seed=0):
        """z [n, d] whitened -> dict t -> (m [draws, n, d], x0, x*, path, dsm)."""
        z = torch.from_numpy(np.asarray(z, np.float32)).to(self.device)
        g = torch.Generator(device="cpu").manual_seed(seed)
        out = {}
        for t in self.cfg.t_grid:
            a = self.sched.ab(t).to(self.device)
            per_draw = []
            for _ in range(self.cfg.n_draws):
                eps = torch.randn(z.shape, generator=g).to(self.device)
                x0 = a.sqrt() * z + (1 - a).sqrt() * eps
                with torch.no_grad():   # denoising score-matching loss at this t: a likelihood proxy
                    tt = torch.full((len(z),), t, device=self.device, dtype=torch.long)
                    dsm = ((self.net(x0, tt) - eps) ** 2).mean(dim=1)
                xs, path = mode_ascent(self.net, self.sched, x0, t, self.cfg)
                per_draw.append((xs / a.sqrt(), x0, xs, path, dsm))
            out[t] = per_draw
        return z, out

    def calibrate_fixed_variance(self, z_real):
        """Per-t, per-dimension variance of (z - m_t(z)) over bona fide calibration clips."""
        z, paths = self._paths(z_real, seed=123)
        self.fixed_var = {t: torch.stack([((z - d[0]) ** 2).mean(0) for d in draws]).mean(0)
                          .clamp(min=self.cfg.var_floor).cpu() for t, draws in paths.items()}
        return self

    def curve(self, z_all, seed=0):
        """-> float32 [n, len(t_grid), len(FEATURES)] mode-path curve for each row of z_all."""
        feats = []
        for i in range(0, len(z_all), self.cfg.batch):
            z, paths = self._paths(z_all[i: i + self.cfg.batch], seed=seed + i)
            rows = []
            for t, draws in paths.items():
                a = self.sched.ab(t).to(self.device)
                acc = []
                for m, x0, xs, path, dsm in draws:
                    if self.cfg.covariance == "hutchinson":
                        h = hutchinson_diag_hessian(self.net, self.sched, xs, t, self.cfg.n_probes)
                        var = prototype_variance_from_hessian(h, a, self.cfg.var_floor)
                    else:
                        var = self.fixed_var[t].to(self.device)[None].expand_as(z)
                    d = z.shape[1]
                    acc.append(torch.stack([
                        diag_gauss_loglik(z, m, var),
                        (z - m).norm(dim=1) / math.sqrt(d),
                        (xs - x0).norm(dim=1) / math.sqrt(d),
                        path / math.sqrt(d),
                        dsm,
                    ], dim=1))
                rows.append(torch.stack(acc).mean(0))              # average over noise draws
            feats.append(torch.stack(rows, dim=1).cpu().numpy())    # [b, n_t, n_feat]
        return np.concatenate(feats).astype(np.float32)

    # --- curve -> flat features / scores -----------------------------------------------------
    @classmethod
    def flat(cls, curve, t_grid=T_GRID):
        """[n, n_t, n_f] -> (matrix, names) plus the H2 contrast: fine-level minus coarse-level fit."""
        n, nt, nf = curve.shape
        names = [f"d2_{f}_t{t}" for t in t_grid for f in cls.FEATURES]
        X = curve.reshape(n, nt * nf)
        t = np.asarray(t_grid)
        fine, coarse = t <= 50, t >= 200
        ll = curve[:, :, cls.FEATURES.index("loglik")]
        contrast = ll[:, fine].mean(1) - ll[:, coarse].mean(1)
        return np.column_stack([X, contrast, ll.mean(1)]), names + ["d2_fine_minus_coarse", "d2_loglik_mean"]

    def fit_classifier(self, curve, y):
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        X, _ = self.flat(curve, self.cfg.t_grid)
        self.classifier = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=2000,
                                                                             class_weight="balanced")).fit(X, y)
        return self

    def score(self, curve):
        """Supervised curve score (higher = synthetic); falls back to the one-class anomaly score."""
        X, _ = self.flat(curve, self.cfg.t_grid)
        if self.classifier is not None:
            return self.classifier.decision_function(X)
        return -X[:, -1]                                           # -mean log-lik: unsupervised

    # --- persistence -------------------------------------------------------------------------
    def save(self, path, ddpm_cfg=None):
        torch.save(dict(space=self.space.state(), net=self.net.state_dict(), net_cfg=asdict(ddpm_cfg or DDPMConfig()),
                        d=self.space.dim, mp_cfg=asdict(self.cfg), fixed_var=self.fixed_var,
                        classifier=self.classifier), path)

    @classmethod
    def load(cls, path, device=None):
        st = torch.load(path, map_location="cpu", weights_only=False)
        nc = DDPMConfig(**st["net_cfg"])
        net = EpsMLP(st["d"], nc.width, nc.depth, dropout=0.0)
        net.load_state_dict(st["net"])
        obj = cls(EmbeddingSpace.from_state(st["space"]), net.eval().to(device or config.device()), Schedule(nc.T),
                  ModePathConfig(**st["mp_cfg"]), device)
        obj.fixed_var, obj.classifier = st["fixed_var"], st["classifier"]
        return obj


# ----------------------------------------------------------------------------- controls
@dataclass
class OneClassControls:
    """Density baselines in the same whitened space, fit on the same bona fide clips as the DDPM."""
    n_gmm: int = 16
    k_nn: int = 10
    models: dict = field(default_factory=dict)

    def fit(self, z_real, seed=0):
        from sklearn.mixture import GaussianMixture
        from sklearn.neighbors import NearestNeighbors
        z = np.asarray(z_real, np.float64)
        self.models["gauss_mu"], cov = z.mean(0), np.cov(z, rowvar=False)
        self.models["gauss_prec"] = np.linalg.pinv(cov + 1e-3 * np.eye(len(cov)))
        self.models["gmm"] = GaussianMixture(self.n_gmm, covariance_type="diag", random_state=seed).fit(z)
        self.models["knn"] = NearestNeighbors(n_neighbors=self.k_nn).fit(z)
        return self

    def scores(self, z):
        """Anomaly scores (higher = less like bona fide) for each control."""
        z = np.asarray(z, np.float64)
        d = z - self.models["gauss_mu"]
        maha = np.einsum("ij,jk,ik->i", d, self.models["gauss_prec"], d)
        gmm = -self.models["gmm"].score_samples(z)
        knn = self.models["knn"].kneighbors(z)[0].mean(1)
        return dict(ctrl_mahalanobis=maha, ctrl_gmm_nll=gmm, ctrl_knn_dist=knn)
