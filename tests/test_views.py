"""The canonical view contract: deterministic per (clip, view), band-limited, level-normalized, same for both classes."""
import numpy as np
import pytest

pytest.importorskip("soundfile")
from hearsay import audio, views  # noqa: E402


@pytest.fixture
def fake_cache(monkeypatch):
    rng = np.random.default_rng(0)
    x = (rng.normal(0, 3000, 16000 * 6)).astype(np.int16)                # broadband, 6 s
    monkeypatch.setattr(audio, "load_cached", lambda uid: x)


def test_views_are_deterministic(fake_cache):
    vm = views.ViewMaker()
    for view in (0, 1, 2):
        a, pa = vm("clip/a", view)
        b, pb = vm("clip/a", view)
        np.testing.assert_array_equal(a, b)
        assert pa == pb
    assert vm("clip/a", 0)[1]["channel"] == "clean" and vm("clip/a", 1)[1]["channel"] != "clean"
    assert not np.array_equal(vm("clip/a", 0)[0], vm("clip/b", 0)[0])  # different clips, different crops / dither


def test_views_are_band_limited_and_normalized(fake_cache):
    vm = views.ViewMaker()
    for view in (0, 1):
        x, _ = vm("clip/c", view)
        assert 0.96 <= np.abs(x).max() <= 1.001
        X = np.abs(np.fft.rfft(x)) ** 2
        f = np.fft.rfftfreq(len(x), 1 / 16000)
        assert X[f > 7600].sum() / X.sum() < 1e-4                        # the 7-7.4 kHz low-pass of the canonical view
