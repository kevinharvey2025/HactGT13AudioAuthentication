"""D5: Gaussian prototypes, per-dimension submodular composition, and the basic level.

Prototypes (plan 7.2 D5). Diagonal Gaussians in the D2 whitened embedding space for
  - channel conditions (augmentation labels: codec_mp3, telephony, noise_pink, reverb, ...),
  - generator families (DiffSSD generators, D6 copy-synthesis vocoders),
  - bona fide sources (LJSpeech, LibriSpeech speakers).
Fit directly per subset (fast; the plan's starting point) or by mode ascent with an unconditional
DDPM trained on all embeddings (Tweedie means of the subset's modes at one noise level).

Composition (Wang, Gupta, Zhu & MacLellan, arXiv 2605.07078). For a query z choose prototypes S
greedily to maximize the monotone submodular per-dimension coverage
    F(S) = sum_r max_{j in S} log N(z_r; m_{j,r}, s2_{j,r}),
with F(empty) = the log-density of a broad background N(0, 1) (the whitened data scale), K = 2-4.
Per-dimension softmax weights (temperature tau) give each selected prototype's share of the
dimensions; the product of experts of the selection is the composed explanation. Channel
prototypes that claim dimensions take those dimensions out of the authenticity judgment, which
then rests on the dimensions owned by bona fide vs generator prototypes.

Basic level (Wang, Singaravadivelan & MacLellan, arXiv 2609.13047). At each noise level t, every
datum's Tweedie prototype m_t is its "concept at level t". The distinctiveness of the bona fide
vs spoof partition at level t is the mean pointwise mutual information between category and
level-t concept, i.e. I(C; K_t) with K_t a k-means clustering of the m_t (plus a classifier-based
estimate that needs no clustering). The peak over t is the abstraction level that carries the
real/fake signal: a peak at small t supports H2 (fakes differ in fine detail).
"""
import math
from dataclasses import dataclass

import numpy as np

LOG2PI = math.log(2 * math.pi)


@dataclass
class Prototype:
    name: str
    kind: str          # "channel" | "generator" | "bonafide"
    mean: np.ndarray   # [d]
    var: np.ndarray    # [d]
    n: int

    def logpdf_dims(self, z):
        """Per-dimension log N(z_r; m_r, s2_r). z [n, d] -> [n, d]."""
        return -0.5 * ((z - self.mean) ** 2 / self.var + np.log(self.var) + LOG2PI)


def fit_direct(z, labels, kind, min_n=20, var_floor=1e-3, shrink=0.1):
    """One diagonal Gaussian per label value (variance shrunk toward 1, the whitened scale)."""
    out = []
    for lab in sorted(set(labels)):
        m = np.asarray(labels) == lab
        if m.sum() < min_n:
            continue
        v = (1 - shrink) * z[m].var(0) + shrink
        out.append(Prototype(str(lab), kind, z[m].mean(0), np.maximum(v, var_floor), int(m.sum())))
    return out


def fit_by_mode_ascent(d2model, z, labels, kind, t=100, min_n=20, var_floor=1e-3):
    """Prototype = mean Tweedie mode of the subset at noise level t (unconditional DDPM in D2 space);
    variance = per-dimension spread of members around their own modes."""
    import torch
    from .ddpm import mode_ascent
    out = []
    for lab in sorted(set(labels)):
        m = np.asarray(labels) == lab
        if m.sum() < min_n:
            continue
        zz = torch.from_numpy(z[m].astype(np.float32)).to(d2model.device)
        a = d2model.sched.ab(t).to(d2model.device)
        x0 = a.sqrt() * zz + (1 - a).sqrt() * torch.randn_like(zz)
        xs, _ = mode_ascent(d2model.net, d2model.sched, x0, t, d2model.cfg)
        modes = (xs / a.sqrt()).cpu().numpy()
        var = np.maximum(((z[m] - modes) ** 2).mean(0), var_floor)
        out.append(Prototype(str(lab), kind, modes.mean(0), var, int(m.sum())))
    return out


# ----------------------------------------------------------------------------- composition
def greedy_select(z, prototypes, k=3, background_var=1.0):
    """Greedy maximization of F(S) for one query z [d]. Returns (selected indices, gains)."""
    L = np.stack([p.logpdf_dims(z[None])[0] for p in prototypes])        # [P, d]
    best = -0.5 * (z ** 2 / background_var + math.log(background_var) + LOG2PI)   # F(empty) per dim
    chosen, gains = [], []
    for _ in range(min(k, len(prototypes))):
        cand = np.maximum(L, best[None]).sum(1) - best.sum()            # marginal gain of each prototype
        cand[chosen] = -np.inf
        j = int(np.argmax(cand))
        if cand[j] <= 0:
            break
        chosen.append(j)
        gains.append(float(cand[j]))
        best = np.maximum(best, L[j])
    return chosen, gains


def compose(z, prototypes, chosen, tau=0.5):
    """Per-dimension softmax weights over the selection and the product-of-experts Gaussian."""
    L = np.stack([prototypes[j].logpdf_dims(z[None])[0] for j in chosen])   # [K, d]
    w = np.exp((L - L.max(0)) / tau)
    w /= w.sum(0, keepdims=True)                                            # [K, d]
    prec = sum(w[i] / prototypes[j].var for i, j in enumerate(chosen))
    mean = sum(w[i] * prototypes[j].mean / prototypes[j].var for i, j in enumerate(chosen)) / prec
    loglik = float((-0.5 * ((z - mean) ** 2 * prec - np.log(prec) + LOG2PI)).mean())
    return w, mean, 1.0 / prec, loglik


def explain(z, prototypes, k=3, tau=0.5):
    """Composed explanation of one clip: which channel / generator / bona fide prototypes cover it,
    their share of the dimensions, and an authenticity margin on the non-channel dimensions."""
    chosen, gains = greedy_select(z, prototypes, k)
    if not chosen:
        return dict(selected=[], summary="no prototype explains this clip better than the background")
    w, _, _, loglik = compose(z, prototypes, chosen, tau)
    sel = [dict(name=prototypes[j].name, kind=prototypes[j].kind, share=float(w[i].mean()), gain=g)
           for i, (j, g) in enumerate(zip(chosen, gains))]
    channel_owned = np.zeros(len(z), bool)
    for i, j in enumerate(chosen):
        if prototypes[j].kind == "channel":
            channel_owned |= w[i] >= w.max(0)
    rest = ~channel_owned
    bona = [p for p in prototypes if p.kind == "bonafide"]
    gens = [p for p in prototypes if p.kind == "generator"]
    margin = np.nan
    closest_gen = None
    if bona and gens and rest.any():
        lb = np.max([p.logpdf_dims(z[None])[0][rest].sum() for p in bona])
        lg_all = [p.logpdf_dims(z[None])[0][rest].sum() for p in gens]
        closest_gen = gens[int(np.argmax(lg_all))].name
        margin = float((lb - max(lg_all)) / rest.sum())                   # > 0: closer to bona fide
    ch = [s["name"] for s in sel if s["kind"] == "channel"]
    summary = (f"channel = {' + '.join(ch) if ch else 'clean'}; closest generator = {closest_gen}; "
               f"bona fide margin on non-channel dims = {margin:+.3f}")
    return dict(selected=sel, composed_loglik=loglik, channel_dim_share=float(channel_owned.mean()),
                closest_generator=closest_gen, bonafide_margin=margin, summary=summary)


# ----------------------------------------------------------------------------- basic level
def distinctiveness_curve(d2model_uncond, z, y, t_grid=None, n_clusters=32, seed=0):
    """I(C; K_t) (nats) between the bona fide/spoof label and the level-t concept, for each t.

    d2model_uncond: a D2Model whose DDPM was trained on ALL embeddings (both classes).
    Returns rows with the clustering MI, the classifier-based MI estimate, and the normalized MI.
    """
    import torch
    from sklearn.cluster import KMeans
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import mutual_info_score
    from sklearn.model_selection import cross_val_predict
    from .ddpm import mode_ascent
    t_grid = t_grid or d2model_uncond.cfg.t_grid
    y = np.asarray(y)
    p1 = y.mean()
    h_c = -(p1 * math.log(p1) + (1 - p1) * math.log(1 - p1))
    zz = torch.from_numpy(np.asarray(z, np.float32)).to(d2model_uncond.device)
    g = torch.Generator(device="cpu").manual_seed(seed)
    rows = []
    for t in t_grid:
        a = d2model_uncond.sched.ab(t).to(d2model_uncond.device)
        x0 = a.sqrt() * zz + (1 - a).sqrt() * torch.randn(zz.shape, generator=g).to(zz.device)
        xs, _ = mode_ascent(d2model_uncond.net, d2model_uncond.sched, x0, t, d2model_uncond.cfg)
        m = (xs / a.sqrt()).cpu().numpy()
        k = KMeans(n_clusters, n_init=4, random_state=seed).fit_predict(m)
        mi = mutual_info_score(y, k)
        p = cross_val_predict(LogisticRegression(max_iter=2000), m, y, cv=5, method="predict_proba")[:, 1]
        ce = -np.mean(y * np.log(np.clip(p, 1e-6, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-6, 1)))
        rows.append(dict(t=t, mi_clusters=mi, mi_classifier=max(0.0, h_c - ce), nmi=mi / h_c))
    return rows
