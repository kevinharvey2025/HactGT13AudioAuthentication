"""Concept formation over the detector's embedding space, with the lab's COBWEB as the ground truth for concepts.

cobweb-private (github.com/Teachable-AI-Lab/cobweb-private; `pip install -e .`; revision in ext/.../GIT_REVISION) builds
a hierarchy of diagonal-Gaussian concepts incrementally (Fisher 1987; continuous form: Gennari, Langley & Fisher 1989)
and gives each node its closed-form expected pointwise mutual information with the root, D(c) of Wang,
Singaravadivelan & MacLellan (arXiv 2609.13047); the basic level of a clip is the node on its path with the largest
D(c) (`get_basic`). Every inserted clip carries a multi-hot label [source one-hot | channel one-hot], so each concept
reports which generators / real corpora and which channel conditions it summarizes, and `predict` gives P(source).

`Space` is the representation both the tree and the diffusion model (hearsay/diffusion/ttcg.py) work in: the
detector's time-mean last layer, standardized and PCA-whitened on train rows. It also maps a change in concept space
back to the detector's embedding, so explanations can be tested against the detector's own linear head.
"""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

REAL_SOURCES = {"real_lj": "real:ljspeech", "real_libri": "real:librispeech_cloned_speakers",
                "real_extra": "real:librispeech_other_speakers"}


def source_class(r):
    """Generator (DiffSSD or copy-synthesis vocoder) for fakes, corpus for reals, 'itw' for In-the-Wild."""
    if r.family in ("lj_voice", "clone", "resynth"):
        return r.generator
    return REAL_SOURCES.get(r.family, "itw" if r.family == "itw" else r.family)


def channel_class(ch):
    """'codec_mp3+noise_pink' -> 'codec' (the first operation of the view's channel chain); None -> 'clean'."""
    ch = ch or "clean"
    return ch.split("+")[0].split("_")[0] if ch != "clean" else "clean"


def f32(x):
    return np.ascontiguousarray(x, dtype=np.float32)


def node_key(n):
    """Stable identity of a cobweb-private node (nanobind may return a fresh wrapper for the same C++ node)."""
    return hash((n.depth(), round(float(n.count), 3), np.asarray(n.mean, np.float32).round(5).tobytes()))


class Space:
    """Standardize + PCA-whiten (fit on train rows). z = ((h - mu) / sd - m) V^T / sqrt(ev)."""

    def __init__(self, dims=32, seed=0):
        self.scaler, self.pca = StandardScaler(), PCA(dims, whiten=True, random_state=seed)

    def fit(self, H):
        self.pca.fit(self.scaler.fit_transform(H))
        return self

    def __call__(self, H):
        return self.pca.transform(self.scaler.transform(H)).astype(np.float32)

    def delta_h(self, dZ):
        """The change of the detector embedding that a change dZ in concept space corresponds to (linear)."""
        return (np.atleast_2d(dZ) * np.sqrt(self.pca.explained_variance_)) @ self.pca.components_ * self.scaler.scale_

    def reconstruct(self, Z):
        return self.scaler.inverse_transform(self.pca.inverse_transform(Z))


class Concepts:
    """A fitted cobweb-private tree plus the label vocabulary (library defaults for every tree parameter)."""

    def __init__(self, dims, sources, channels, seed=0):
        from cobweb.cobweb_continuous import CobwebContinuousTree   # cobweb-private
        self.sources, self.channels = list(sources), list(channels)
        self.S, self.C = len(self.sources), len(self.channels)
        self.fake = np.array([not (s.startswith("real:") or s == "itw") for s in self.sources])
        self.tree = CobwebContinuousTree(size=dims, num_labels=self.S + self.C)
        self.seed = seed

    def fit(self, Z, src, ch):
        """Incremental insertion in a seeded random order (insertion order is the stability test's variable)."""
        lab = np.zeros((len(Z), self.S + self.C), np.float32)
        lab[np.arange(len(Z)), src] = 1.0
        lab[np.arange(len(Z)), self.S + ch] = 1.0
        for i in np.random.default_rng(self.seed).permutation(len(Z)):
            self.tree.ifit(f32(Z[i]), lab[i])
        return self

    def p_fake(self, Z, max_nodes=300):
        """P(synthetic source | z): the tree's label prediction (best-first expansion over max_nodes concepts)."""
        empty = np.zeros(self.S + self.C, np.float32)
        out = np.zeros(len(Z))
        for i, z in enumerate(Z):
            dist = np.asarray(self.tree.predict(f32(z), empty, max_nodes, False))[: self.S]
            out[i] = dist[self.fake].sum() / max(dist.sum(), 1e-12)
        return out

    def basic(self, z):
        """-> (leaf, basic-level node) of z's path."""
        leaf = self.tree.get_leaf(f32(z), np.zeros(self.S + self.C, np.float32))
        return leaf, leaf.get_basic()

    def informative(self, z):
        """-> (leaf, concept, pmi): the node on z's path with the largest pmi(z; c) = log p_c(z) - log p_root(z) under
        the tree's own (regularized) Gaussians. For an instance the tree has not seen this is the held-out definition
        of the basic level: a concept too narrow to cover z pays for it, unlike in the closed-form D(c)."""
        empty = np.zeros(self.S + self.C, np.float32)
        x = f32(z)
        leaf = self.tree.get_leaf(x, empty)
        path = self.path(leaf)
        pm = np.array([n.log_prob(x, empty) for n in path]) - self.tree.root.log_prob(x, empty)
        j = int(np.argmax(pm))
        return leaf, path[j], float(pm[j])

    @staticmethod
    def path(node):
        """node, parent, ..., root."""
        out = []
        while node is not None:
            out.append(node)
            node = node.parent
        return out

    def makeup(self, node, k=3):
        """Top-k sources, top-2 channels and the synthetic share of the clips a concept summarizes."""
        lc = np.asarray(node.label_counts, float)
        src, ch = lc[: self.S], lc[self.S:]
        s = [dict(source=self.sources[j], share=round(float(src[j] / max(src.sum(), 1e-12)), 3))
             for j in np.argsort(-src)[:k] if src[j] > 0]
        c = [dict(channel=self.channels[j], share=round(float(ch[j] / max(ch.sum(), 1e-12)), 3))
             for j in np.argsort(-ch)[:2] if ch[j] > 0]
        return s, c, float(src[self.fake].sum() / max(src.sum(), 1e-12))
