"""The cobweb-private wrapper (hearsay/concepts.py) and the concept space. Needs cobweb-private for the tree tests."""
import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from hearsay.concepts import Space, channel_class, node_key


def blobs(n, seed):
    rng = np.random.default_rng(seed)
    y = np.r_[np.zeros(n), np.ones(n)].astype(int)
    return rng.normal(np.where(y[:, None] == 1, 2.0, -2.0), 0.5, (2 * n, 4)).astype(np.float32), y


def test_tree_separates_and_describes_sources():
    cc = pytest.importorskip("cobweb.cobweb_continuous")
    if not hasattr(cc.CobwebContinuousNode, "get_basic"):
        pytest.skip("installed cobweb-private predates the continuous basic-level API (need revision 5012d51b or later)")
    from hearsay.concepts import Concepts
    Z, y = blobs(300, 0)
    c = Concepts(4, ["gen_a", "real:x"], ["clean"], seed=0).fit(Z, np.where(y == 1, 0, 1), np.zeros(len(Z), int))
    Zh, yh = blobs(100, 1)
    assert roc_auc_score(yh, c.p_fake(Zh)) > 0.99
    leaf, basic = c.basic(Zh[-1])                                      # a synthetic clip
    src, ch, pf = c.makeup(basic)
    assert src[0]["source"] == "gen_a" and pf > 0.9 and ch[0]["channel"] == "clean"
    path = c.path(leaf)
    assert path[-1].parent is None and [n.depth() for n in path] == sorted([n.depth() for n in path], reverse=True)
    assert basic.depth() <= leaf.depth()
    _, node, pmi = c.informative(Zh[-1])                             # held-out basic level: a node of the same path
    assert pmi >= 0 and node_key(node) in {node_key(n) for n in path} and c.makeup(node)[2] > 0.9


def test_space_inverse_is_consistent():
    rng = np.random.default_rng(0)
    H = rng.normal(size=(500, 40)) @ rng.normal(size=(40, 40)) + 3.0
    sp = Space(dims=8).fit(H)
    z = sp(H[:5])
    dz = np.zeros_like(z)
    dz[:, 2] = 1.0
    np.testing.assert_allclose(sp.reconstruct(z + dz) - sp.reconstruct(z), sp.delta_h(dz), atol=1e-6)


def test_channel_class():
    assert channel_class(None) == "clean" and channel_class("codec_mp3+noise_pink") == "codec" and channel_class("telephony") == "telephony"
