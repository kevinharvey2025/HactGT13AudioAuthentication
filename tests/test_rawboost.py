"""RawBoost (hearsay/rawboost.py), its training hook (scripts/finetune_ssl.py), the unseen-channel views (hearsay/unseen.py,
views 100 and 101) and the experiment report (scripts/rawboost_report.py).

The parity test needs the reference implementation, which is not committed: a clone of
github.com/TakHemlata/SSL_Anti-spoofing (MIT) at $RAWBOOST_REF, or at cache/rawboost_ref/.
"""
import ast
import hashlib
import importlib.util
import json
import os
import subprocess
import types
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.signal import freqz

pytest.importorskip("soundfile")
from hearsay import audio, augment, config, metrics, rawboost, views  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REF = Path(os.environ.get("RAWBOOST_REF", ROOT / "cache" / "rawboost_ref"))
SR = config.SR


def speech(n=56000, seed=0):
    """A speech-like test signal: a harmonic tone with vibrato and a syllable envelope, a little noise, peak 0.9."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / SR
    ph = 2 * np.pi * np.cumsum(140 + 20 * np.sin(2 * np.pi * 0.7 * t)) / SR
    x = sum(np.sin(k * ph) / k for k in range(1, 12)) * (0.6 + 0.4 * np.sin(2 * np.pi * 4 * t)) ** 2
    x = x + 0.02 * rng.standard_normal(n)
    return (0.9 * x / np.abs(x).max()).astype(np.float32)


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ----------------------------------------------------------------------------- the port

@pytest.fixture(scope="module")
def reference():
    if not ((REF / "RawBoost.py").exists() and (REF / "data_utils_SSL.py").exists()):
        pytest.skip(f"no reference RawBoost at {REF}: clone github.com/TakHemlata/SSL_Anti-spoofing there or set RAWBOOST_REF")
    spec = importlib.util.spec_from_file_location("rawboost_reference", REF / "RawBoost.py")
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)

    def randRange(x1, x2, integer):   # numpy 2 refuses int() of a 1-element array: the same draw, its element returned
        y = np.random.uniform(low=x1, high=x2, size=(1,))[0]
        return int(y) if integer else y
    ref.randRange = randRange
    tree = ast.parse((REF / "data_utils_SSL.py").read_text())   # the dispatcher, without the file's torch/librosa imports
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "process_Rawboost_feature")
    ns = dict(vars(ref))
    exec(compile(ast.Module([fn], []), str(REF / "data_utils_SSL.py"), "exec"), ns)
    return ns["process_Rawboost_feature"]


@pytest.mark.parametrize("algo", range(9))
def test_parity_with_the_reference(reference, algo):
    args = types.SimpleNamespace(**rawboost.RawBoostParams().to_json())
    for seed, x in [(0, speech()), (1, speech(seed=1)), (7, speech(8000, 2)), (3, np.clip(4 * speech(), -1, 1))]:
        np.random.seed(seed)
        want = np.asarray(reference(x.copy(), SR, args, algo), np.float32)
        got, _ = rawboost.apply(x.copy(), np.random.RandomState(seed), algo)
        np.testing.assert_allclose(got, want, rtol=0, atol=1e-6)


def test_same_seed_same_output_different_seed_different_output():
    x = speech()
    a, pa = rawboost.apply(x, np.random.default_rng(3), 4)
    b, pb = rawboost.apply(x, np.random.default_rng(3), 4)
    np.testing.assert_array_equal(a, b)
    assert pa == pb
    c, pc = rawboost.apply(x, np.random.default_rng(4), 4)
    assert not np.array_equal(a, c) and pa != pc


@pytest.mark.parametrize("algo", range(9))
def test_output_contract(algo):
    for x in (speech(), np.zeros(SR, np.float32), np.clip(4 * speech(), -1, 1), speech(4000), speech(100)):
        y, p = rawboost.apply(x, np.random.default_rng(1), algo)
        assert y.dtype == np.float32 and len(y) == len(x) and np.isfinite(y).all()
        if not x.any():
            assert not y.any()                                    # silence stays silent
        json.dumps(p)
        assert p["channel"] == f"rawboost_a{algo}" and p["algo"] == algo
    assert rawboost.apply(np.zeros(0, np.float32), np.random.default_rng(0), algo)[0].shape == (0,)


def test_ssi_adds_noise_at_the_drawn_snr():
    x = speech()
    for seed in range(5):
        y, p = rawboost.apply(x, np.random.default_rng(seed), 3)
        n = y.astype(float) - x.astype(float)
        snr = 20 * np.log10(np.linalg.norm(x) / np.linalg.norm(n))
        assert 10 <= p["ssi"]["snr_db"] <= 40 and abs(snr - p["ssi"]["snr_db"]) < 0.01


def test_isd_touches_at_most_p_percent_of_the_samples():
    x = speech() / 0.9 * 0.25                                     # peak 0.25: ISD at most triples a sample, no rescaling
    for seed in range(5):
        y, p = rawboost.apply(x, np.random.default_rng(seed), 2)
        assert np.count_nonzero(y != x) <= p["isd"]["samples"] <= 0.10 * len(x)


def test_notch_filters_are_band_stop_with_odd_tap_counts():
    prm, log = rawboost.RawBoostParams(), []
    rawboost.notch_filter(np.random.default_rng(0), prm, prm.minG, prm.maxG, SR, log)
    assert len(log[0]["bands"]) == prm.nBands
    assert all(c % 2 == 1 and prm.minCoeff <= c <= prm.maxCoeff for _, _, c in log[0]["bands"])
    one, log = rawboost.RawBoostParams(nBands=1, minF=3000, maxF=3001, minBW=900, maxBW=901, minCoeff=99, maxCoeff=100), []
    b = rawboost.notch_filter(np.random.default_rng(0), one, 0, 0, SR, log)
    w, h = freqz(b, 1, worN=4096, fs=SR)
    assert len(b) == 99 and 0.99 < np.abs(h).max() < 1.02       # gain 0 dB: unit peak
    assert np.abs(h[np.argmin(np.abs(w - 3000))]) < 0.01          # a deep notch at the centre


def test_parameters_parse_and_serialize():
    p = rawboost.RawBoostParams.parse("maxF=7000, SNRmin=5")
    assert p.maxF == 7000 and p.SNRmin == 5 and p.nBands == 5 and rawboost.RawBoostParams(**p.to_json()) == p
    with pytest.raises(ValueError):
        rawboost.RawBoostParams.parse("maxf=7000")


@pytest.mark.parametrize("algo", range(1, 9))
def test_canonical_view_contract_holds_after_rawboost(algo):
    x = np.random.default_rng(0).normal(0, 3000, SR * 6).astype(np.int16)
    y = audio.canonical(x, np.random.default_rng(algo), durations=audio.TestLikeDurations([3.5]),
                        aug=lambda z, r: rawboost.apply(z, np.random.default_rng(99), algo))
    assert 0.96 <= np.abs(y).max() <= 1.001
    X = np.abs(np.fft.rfft(y)) ** 2
    assert X[np.fft.rfftfreq(len(y), 1 / SR) > 7600].sum() / X.sum() < 1e-4


# ----------------------------------------------------------------------------- the training hook

@pytest.fixture(scope="module")
def ft():
    pytest.importorskip("torch")
    try:
        return load_script("finetune_ssl")
    except ImportError as e:
        pytest.skip(f"scripts/finetune_ssl.py needs the neural environment: {e}")


@pytest.fixture
def clips(monkeypatch):
    def load(uid):
        h = zlib.crc32(uid.encode())
        return (speech(int((4 + h % 300 / 100) * SR), seed=h % 1000) * 20000).astype(np.int16)
    monkeypatch.setattr(audio, "load_cached", load)


ITEMS = [(i, n) for i in range(12) for n in (48000, 64000)]


def trainset(ft, labels=None, **kw):
    labels = np.arange(12) % 2 if labels is None else labels
    return ft.TrainSet([f"clip/{k}" for k in range(12)], labels, 0.6, 0, [f"babble/{k}" for k in range(4)], **kw)


def legacy_item(ds, item):
    """TrainSet.__getitem__ as it was before RawBoost (commit 61a3fc8)."""
    i, n = item
    rng = np.random.default_rng([ds.seed, ds.epoch, i])
    x = audio.load_cached(ds.uids[i])
    aug = None
    if rng.random() < ds.p_aug:
        def aug(y, r):
            return augment.random_chain(y, r, babble_pool=ds.babble)
    y = audio.canonical(x, rng, durations=audio.TestLikeDurations([n / SR]), aug=aug)
    if len(y) < n:
        y = np.pad(y, (0, n - len(y)))
    return y[:n].astype(np.float32), int(1 - ds.labels[i])


@pytest.mark.parametrize("kw", [{}, dict(rawboost_algo=5, p_rawboost=0.0), dict(rawboost_algo=0, p_rawboost=0.5)])
def test_rawboost_off_gives_the_old_batches_bit_for_bit(ft, clips, kw):
    ds = trainset(ft, **kw)
    for epoch in (1, 2):
        ds.epoch = epoch
        for it in ITEMS:
            x, y, i, boosted = ds[it]
            want, wy = legacy_item(ds, it)
            assert not boosted and i == it[0] and y == wy
            np.testing.assert_array_equal(x.numpy(), want)


def test_rawboost_changes_only_its_own_stage(ft, clips, monkeypatch):
    """With RawBoost replaced by the identity, RawBoost-on items equal RawBoost-off items: same crop, same chain draws,
    same dither (bit for bit when the chain ran; to rounding otherwise, from the hook's input scaling)."""
    calls, chains = [], []
    monkeypatch.setattr(rawboost, "apply", lambda y, r, algo, prm: (calls.append(algo), (y, {}))[1])
    real_chain = augment.random_chain

    def recording(y, r, **kw):
        out = real_chain(y, r, **kw)
        chains.append(out[1])
        return out
    monkeypatch.setattr(augment, "random_chain", recording)
    on, off = trainset(ft, rawboost_algo=5, p_rawboost=1.0), trainset(ft)
    for epoch in (1, 2):
        on.epoch = off.epoch = epoch
        for it in ITEMS:
            chains.clear()
            a, _, _, boosted = on[it]
            ca = list(chains)
            chains.clear()
            b, _, _, _ = off[it]
            assert boosted and ca == chains
            if ca:
                np.testing.assert_array_equal(a.numpy(), b.numpy())
            else:
                np.testing.assert_allclose(a.numpy(), b.numpy(), atol=1e-6)
    assert calls == [5] * (2 * len(ITEMS))


def test_rawboost_never_depends_on_the_label(ft, clips):
    a = trainset(ft, rawboost_algo=4, p_rawboost=0.5)
    b = trainset(ft, labels=1 - np.arange(12) % 2, rawboost_algo=4, p_rawboost=0.5)
    a.epoch = b.epoch = 1
    for it in ITEMS[:10]:
        xa, ya, _, ba = a[it]
        xb, yb, _, bb = b[it]
        np.testing.assert_array_equal(xa.numpy(), xb.numpy())
        assert ba == bb and ya != yb


def test_applied_rate_is_p_for_every_class_and_family(ft):
    rng, n, p = np.random.default_rng(0), 20000, 0.3
    ds = ft.TrainSet([f"u{k}" for k in range(n)], rng.integers(0, 2, n), 0.6, 0, [], rawboost_algo=5, p_rawboost=p)
    ds.epoch = 1
    tr = pd.DataFrame(dict(label=ds.labels, family=rng.choice(["lj_voice", "clone", "resynth", "real_lj", "real_libri"], n)))
    r = ft.rawboost_rates(tr, np.arange(n), np.array([ds.rawboost_rng(i) is not None for i in range(n)]))
    for v in [r["all"], r["fake"], r["real"], *r["family"].values()]:
        assert abs(v - p) < 4 * np.sqrt(p * (1 - p) / (n / 5))


# ----------------------------------------------------------------------------- views

class LegacyViewMaker:
    """hearsay/views.py ViewMaker as it was before views 100 and 101 (commit 61a3fc8)."""

    def __init__(self, babble_pool=None):
        self.durations, self.babble_pool = audio.TestLikeDurations.from_cache(), babble_pool

    def __call__(self, uid, view=0, is_test=False):
        params, aug = {"channel": "clean"}, None
        if view > 0:
            def aug(y, r):
                y2, p = augment.random_chain(y, r, babble_pool=self.babble_pool)
                params.update(p)
                return y2, p
        x = audio.canonical(audio.load_cached(uid), audio.uid_rng(uid, view), None if is_test else self.durations, aug=aug)
        return x, params


def test_views_0_and_1_are_frozen(clips):
    babble = [audio.load_cached(f"babble/{k}") for k in range(4)]
    new, old = views.ViewMaker(babble), LegacyViewMaker(babble)
    h_new, h_old = hashlib.sha256(), hashlib.sha256()
    for k in range(10):
        for v in (0, 1, 2):
            for is_test in (False, True):
                a, pa = new(f"clip/{k}", v, is_test)
                b, pb = old(f"clip/{k}", v, is_test)
                assert pa == pb
                h_new.update(a.tobytes())
                h_old.update(b.tobytes())
    assert h_new.hexdigest() == h_old.hexdigest()


@pytest.fixture(scope="module")
def encoders():
    pytest.importorskip("pyroomacoustics")
    from hearsay import unseen
    have = subprocess.run([config.ffmpeg_bin(), "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
    need = {c: args(s)[1] for c, (args, _, _, (s, *_)) in unseen.CODECS.items()}
    missing = [e for e in need.values() if f" {e} " not in have]
    if missing:
        pytest.skip(f"ffmpeg lacks {missing}, which the unseen view needs (Raven: module load ffmpeg/7.1)")
    return unseen


def test_every_unseen_operation_keeps_length_and_is_finite(encoders):
    x = speech()
    for name in encoders.CODECS:
        y, p = encoders.codec(x, np.random.default_rng(0), name)
        assert len(y) == len(x) and np.isfinite(y).all() and p["codec"] == name
    for name, op in encoders.OPS.items():
        for seed in range(3):
            y, p = op(x, np.random.default_rng(seed))
            assert y.dtype == np.float32 and len(y) == len(x) and np.isfinite(y).all() and p["channel"]
    y, p = encoders.device(x, np.random.default_rng(0))
    assert len(y) == len(x) and np.isfinite(y).all() and p["channel"] == "device"


def test_unseen_views_are_deterministic_unseen_and_canonical(clips, encoders):
    vm, drawn = views.ViewMaker(), set()
    chain = {f"codec_{c}" for c in augment.CODECS} | {"telephony", "bandlimit", "resample", "tilt", "hum", "reverb", "clipping"}
    for k in range(12):
        for v in (views.UNSEEN, views.DEVICE):
            a, pa = vm(f"clip/{k}", v)
            b, pb = vm(f"clip/{k}", v)
            np.testing.assert_array_equal(a, b)
            assert pa == pb and pa["channel"] != "clean"
            assert 0.96 <= np.abs(a).max() <= 1.001
            X = np.abs(np.fft.rfft(a)) ** 2
            assert X[np.fft.rfftfreq(len(a), 1 / SR) > 7600].sum() / X.sum() < 1e-4
            if v == views.UNSEEN:
                drawn.add(pa["channel"])
                assert not set(pa["channel"].split("+")) & chain and "rawboost" not in pa["channel"]
    assert len(drawn) > 3


# ----------------------------------------------------------------------------- the report

def make_run(ft_dir, name, sep, s_per_step=0.4, extra=True):
    """A run with 3 epochs: scores = clip effect + separation sep[(set, view)] for fakes + noise shared by all runs, so
    runs differ exactly where their separations do."""
    d = ft_dir / name
    d.mkdir(parents=True)
    json.dump(dict(name=name), open(d / "config.json", "w"))
    with open(d / "log.jsonl", "w") as f:
        for e in range(4):
            f.write(json.dumps(dict(epoch=e, **({"s_per_step": s_per_step} if e else {}))) + "\n")
            for s in ("val", "holdout", "itw"):
                n = 400
                y = np.r_[np.zeros(n // 2, int), np.ones(n // 2, int)]
                clip = np.random.default_rng(zlib.crc32(s.encode())).standard_normal(n)
                cols = dict(uid=[f"{s}/{k}" for k in range(n)], label=y, split=s,
                            generator=np.where(y == 1, np.resize(["elevenlabs", "playht", "grad_tts"], n), "bonafide"),
                            family=np.where(y == 1, "clone", np.resize(["real_lj", "real_libri"], n)),
                            channel_aug=np.resize(["reverb", "noise_babble", "codec_mp3", "tilt"], n))
                for v in (("clean", "aug", "unseen") if extra else ("clean", "aug")):
                    noise = np.random.default_rng(zlib.crc32(f"{s}/{v}/{e}".encode())).standard_normal(n)
                    cols[f"score_{v}"] = clip + y * sep.get((s, v), 3.0) * (0.5 + e / 6) + 0.3 * noise
                if extra:
                    cols["channel_unseen"] = np.resize(["room", "codec_vorbis", "agc+packet_loss"], n)
                pd.DataFrame(cols).to_parquet(d / f"{s}_epoch{e}.parquet")


def test_report_end_to_end_on_synthetic_runs(tmp_path, monkeypatch):
    pytest.importorskip("sklearn")
    rep = load_script("rawboost_report")
    ft_dir = tmp_path / "ft"
    monkeypatch.setattr(rep, "FT", ft_dir)
    hard = {("itw", "aug"): 1.0, ("itw", "unseen"): 1.0}
    for seed in (0, 1):
        make_run(ft_dir, f"rb_R0_xlsr1b_s{seed}", hard)
        make_run(ft_dir, f"rb_R2_xlsr1b_s{seed}", {("itw", "aug"): 2.0, ("itw", "unseen"): 2.0}, s_per_step=0.42)
        make_run(ft_dir, f"rb_R1_xlsr1b_s{seed}", hard, s_per_step=0.9)
        make_run(ft_dir, ["xlsr1b_d6rall", "xlsr1b_d6rall_s1"][seed], hard, extra=False)
    runs = {p.name: rep.load_run(p) for p in rep.discover()}
    ops = {n: rep.op_points(r)[0] for n, r in runs.items()}
    assert all(o == {"val": 3, "last": 3} for o in ops.values())   # separation grows with the epoch
    bm = rep.benchmark(runs, ops, 50, 2)
    row = bm[(bm.run == "rb_R2_xlsr1b_s0") & (bm.op == "val") & (bm.set == "itw") & (bm.view == "aug")].iloc[0]
    d = runs["rb_R2_xlsr1b_s0"]["sc"][(3, "itw")]
    assert row.min_dcf == pytest.approx(metrics.min_dcf(d.label, d.score_aug))           # known answer
    assert row.reference == "rb_R0_xlsr1b_s0" and row.delta < 0 and row.delta_hi < 0
    r0 = bm[bm.run == "rb_R0_xlsr1b_s0"]                                                 # the reproduction check
    assert (r0.reference == np.where(r0.view == "unseen", "", "xlsr1b_d6rall")).all()   # (stored runs lack view 100)
    noise = rep.seed_noise(bm)
    assert set(noise.pair) == {"R0 (val selection)", "stored (val + ITW selection)"}
    gt = rep.gates(bm, noise).set_index(["run", "op"])
    assert gt.loc[("rb_R2_xlsr1b_s0", "val"), "passes"] and gt.loc[("rb_R2_xlsr1b_s1", "val"), "gate3_seeds"]
    assert not gt.loc[("rb_R1_xlsr1b_s0", "val"), "gate1_target"] and not gt.loc[("rb_R1_xlsr1b_s0", "val"), "gate4_cost"]
    bd = rep.breakdown(runs, ops, 20, 2)
    assert {"generator", "real source", "channel", "unseen op"} <= set(bd.kind)
    assert "not run" in rep.ensemble(runs, ops, "", 20, 2).status.iloc[0]
    assert "rb_R2_xlsr1b_s0" in rep.tables(bm, bd, noise, rep.gates(bm, noise), pd.DataFrame())


def test_cllr_known_answers():
    y = np.r_[np.zeros(50, int), np.ones(50, int)]
    assert metrics.cllr(y, np.full(100, 0.3)) == pytest.approx(1.0)                      # the prior itself: no information
    assert metrics.cllr(y, np.r_[np.full(50, 1e-6), np.full(50, 1 - 1e-6)]) < 0.01
