"""Track A heads on pooled SSL features: a per-layer linear probe and a layer-weighted MLP.

LayerMLP learns a softmax weighting over encoder layers (the weights are reported, since which
layers carry artifact information is a finding), then a small MLP. Scores are logits; higher
means more likely synthetic.
"""
import numpy as np
import torch
from torch import nn

from . import config


class Standardizer:
    def fit(self, x):
        self.mu = x.mean(0, keepdims=True)
        self.sd = x.std(0, keepdims=True) + 1e-5
        return self

    def __call__(self, x):
        return (x - self.mu) / self.sd


class LayerMLP(nn.Module):
    def __init__(self, n_layers, dim, hidden=256, dropout=0.3):
        super().__init__()
        self.layer_logits = nn.Parameter(torch.zeros(n_layers))
        self.mlp = nn.Sequential(nn.Dropout(0.1), nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, 1))

    def layer_weights(self):
        return torch.softmax(self.layer_logits, 0)

    def forward(self, x):                                   # x [B, L, dim]
        h = (self.layer_weights()[None, :, None] * x).sum(1)
        return self.mlp(h).squeeze(-1)


class Scaled(nn.Module):
    """Standardization baked into the module so a saved head is self-contained."""

    def __init__(self, head, mu, sd):
        super().__init__()
        self.head = head
        self.register_buffer("mu", torch.as_tensor(mu, dtype=torch.float32))
        self.register_buffer("sd", torch.as_tensor(sd, dtype=torch.float32))

    def forward(self, x):
        return self.head((x.float() - self.mu) / self.sd)


def fit_standardizer(x, n=4000, seed=0):
    """Per-feature mean/sd from a row sample (x may be a float16 array too large to upcast)."""
    i = np.random.default_rng(seed).choice(len(x), min(n, len(x)), replace=False)
    s = np.asarray(x[np.sort(i)], dtype=np.float32)
    return s.mean(0), s.std(0) + 1e-5


def train_layer_mlp(x, y, epochs=12, lr=1e-3, wd=1e-2, batch=256, seed=0, device=None, verbose=False):
    """x [n, L, dim] (float16 ok), y {0,1}. Class-balanced BCE. Returns a Scaled(LayerMLP)."""
    device = device or config.device()
    torch.manual_seed(seed)
    mu, sd = fit_standardizer(x, seed=seed)
    model = Scaled(LayerMLP(x.shape[1], x.shape[2]), mu, sd).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * int(np.ceil(len(x) / batch)))
    pos_weight = torch.tensor((y == 0).sum() / max(1, (y == 1).sum()), device=device, dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    xt, yt = torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(y.astype(np.float32))
    g = torch.Generator().manual_seed(seed)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(x), generator=g)
        tot = 0.0
        for i in range(0, len(x), batch):
            j = perm[i: i + batch]
            loss = loss_fn(model(xt[j].to(device)), yt[j].to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(j)
        if verbose:
            print(f"epoch {ep}: loss {tot / len(x):.4f}")
    return model.eval()


@torch.inference_mode()
def predict(model, x, batch=2048, device=None):
    device = device or config.device()
    return np.concatenate([model(torch.from_numpy(np.ascontiguousarray(x[i: i + batch])).to(device)).float().cpu().numpy()
                           for i in range(0, len(x), batch)])


def layer_weights(model):
    head = model.head if isinstance(model, Scaled) else model
    return head.layer_weights().detach().cpu().numpy()
