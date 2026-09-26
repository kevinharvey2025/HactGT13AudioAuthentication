import numpy as np
import pytest
from scipy.linalg import solve_toeplitz
from scipy.signal import butter, lfilter, sosfiltfilt

from hearsay_dsp.features.background import background_module
from hearsay_dsp.features.common import (INSUFFICIENT, OK, compute_stft, detect_activity,
                                         frame_signal)
from hearsay_dsp.features.enf import enf_module
from hearsay_dsp.features.lpc import autocorr, levinson_durbin, lpc_module
from hearsay_dsp.features.phase import if_jump_series, instantaneous_frequency, princarg
from hearsay_dsp.features.prosody import prosody_module
from hearsay_dsp.features.spectral import occupied_bandwidth, spectral_module
from hearsay_dsp.io.decode import resample

from dsp_testutils import SR, speechlike


def _ctx(x, cfg):
    stft = compute_stft(x, SR, cfg["features"]["stft"])
    act, _ = detect_activity(stft, cfg["features"]["activity"])
    return stft, act


def test_frames_cover_tail_and_time_coordinates():
    x = np.arange(1000, dtype=float)
    frames, valid, starts = frame_signal(x, 400, 160)
    assert starts[-1] + 400 >= x.size and valid[-1] == x.size - starts[-1]
    assert frames[1, 0] == 160.0
    assert frame_signal(np.ones(100), 400, 160)[0].shape == (1, 400)


def test_activity_separates_bursts_and_flags_digital_silence(cfg):
    x = speechlike(3.0)
    x[: SR // 2] = 0.0  # 0.5 s of exact digital silence
    stft, act = _ctx(x, cfg)
    assert act.digital_silence[: 40].all() and not act.digital_silence[60:].all()
    assert 0.2 < act.active.mean() < 0.8
    assert not (act.active & act.background).any()


def test_silent_clip_is_insufficient_not_error(cfg):
    stft = compute_stft(np.zeros(SR * 2), SR, cfg["features"]["stft"])
    act, res = detect_activity(stft, cfg["features"]["activity"])
    assert act is None and res.status == INSUFFICIENT


def test_spectral_tone_vs_noise_and_centroid(cfg):
    rng = np.random.default_rng(0)
    t = np.arange(SR * 2) / SR
    tone = 0.3 * np.sin(2 * np.pi * 1000 * t) * (np.sin(2 * np.pi * 2 * t) > 0)
    noise = 0.1 * rng.standard_normal(t.size) * (np.sin(2 * np.pi * 2 * t) > 0)
    rt = spectral_module(*_ctx(tone, cfg), tone, tone, SR, cfg)
    rn = spectral_module(*_ctx(noise, cfg), noise, noise, SR, cfg)
    assert rt.status == OK and rn.status == OK
    assert rt.features["spec.fb_centroid_hz_med"] == pytest.approx(1000, abs=40)
    assert rt.features["spec.fb_flatness_db_med"] < rn.features["spec.fb_flatness_db_med"] - 20


def test_band_limited_audio_marks_full_band_unavailable(cfg):
    rng = np.random.default_rng(1)
    t = np.arange(SR * 2) / SR
    x = sosfiltfilt(butter(10, 3400, fs=SR, output="sos"), rng.standard_normal(t.size))
    x = 0.1 * x * (np.sin(2 * np.pi * 2 * t) > 0) + 1e-5 * rng.standard_normal(t.size)
    r = spectral_module(*_ctx(x, cfg), x, x, SR, cfg)
    assert r.quality["q.spec.eligible_max_hz"] < 4500
    assert np.isnan(r.features["spec.fb_centroid_hz_med"])        # unavailable, not zero energy
    assert np.isnan(r.features["spec.band_5000_6000_db_med"])
    assert np.isfinite(r.features["spec.band_1000_2000_db_med"])


def test_occupied_bandwidth_and_resampling_rate_correctness():
    sr = 22050
    t = np.arange(sr * 2) / sr
    y, info = resample(np.sin(2 * np.pi * 1000 * t), sr, 16000)
    assert info["up"] == 320 and info["down"] == 441
    spec = np.abs(np.fft.rfft(y * np.hanning(y.size)))
    f = np.fft.rfftfreq(y.size, 1 / 16000)
    assert f[np.argmax(spec)] == pytest.approx(1000, abs=2)
    rng = np.random.default_rng(0)
    lp = sosfiltfilt(butter(12, 3000, fs=16000, output="sos"), rng.standard_normal(32000))
    assert 2800 < occupied_bandwidth(lp, 16000)["bw_hz"] < 4500


def test_levinson_matches_toeplitz_solution_and_lpc_gain():
    rng = np.random.default_rng(0)
    e = rng.standard_normal(4000)
    ar = lfilter([1.0], [1.0, -1.6, 0.8], e)  # stable AR(2)
    r = autocorr(ar, 4)
    a, err, ks = levinson_durbin(r, 4)
    ref = solve_toeplitz(r[:4], -r[1:5])
    assert np.allclose(a[1:], ref, atol=1e-8)
    assert np.all(np.abs(ks) < 1) and err > 0


def test_lpc_module_on_voiced_signal(cfg):
    x = speechlike(3.0)
    stft, act = _ctx(x, cfg)
    r = lpc_module(x, stft, act, cfg)
    assert r.status == OK
    assert r.features["lpc.pred_gain_db_med"] > 10
    assert r.quality["q.lpc.unstable_frac"] < 0.05


def test_princarg_and_instantaneous_frequency_of_sinusoid():
    assert princarg(np.array([3 * np.pi, -3 * np.pi, 0.5]))[2] == pytest.approx(0.5)
    assert np.all(np.abs(princarg(np.linspace(-20, 20, 101))) <= np.pi)
    f0 = 1015.625  # half-way between bins 32 and 33
    t = np.arange(SR) / SR
    x = np.sin(2 * np.pi * f0 * t)
    st = compute_stft(x, SR, {"win_s": 0.025, "hop_s": 0.010, "nfft": 512})
    inst = instantaneous_frequency(st.spec, st.hop, st.nfft, SR)
    assert np.median(inst[5:-5, 32]) == pytest.approx(f0, abs=1.0)
    assert np.median(inst[5:-5, 33]) == pytest.approx(f0, abs=1.0)


def test_if_jump_spikes_at_controlled_phase_discontinuity():
    t = np.arange(2 * SR) / SR
    x = np.sin(2 * np.pi * 440 * t)
    cut = SR  # splice: phase jump of pi/2 at 1.0 s
    x[cut:] = np.sin(2 * np.pi * 440 * t[cut:] + np.pi / 2)
    st = compute_stft(x, SR, {"win_s": 0.025, "hop_s": 0.010, "nfft": 512})
    j = if_jump_series(st.spec, st.power, st.freqs, st.hop, st.nfft, SR, 7000.0, -40.0)
    peak_t = st.times[np.nanargmax(j)]
    assert abs(peak_t - 1.0) < 0.05
    cont = np.sin(2 * np.pi * 440 * t)
    st2 = compute_stft(cont, SR, {"win_s": 0.025, "hop_s": 0.010, "nfft": 512})
    j2 = if_jump_series(st2.spec, st2.power, st2.freqs, st2.hop, st2.nfft, SR, 7000.0, -40.0)
    assert np.nanmax(j) > 20 * np.nanmedian(j2[5:-5]) + 1.0


def test_background_step_is_a_candidate_and_continuous_is_not(cfg):
    rng = np.random.default_rng(3)
    base = speechlike(6.0, noise_db=-200, seed=3)
    noise = rng.standard_normal(base.size)
    lvl = np.where(np.arange(base.size) < 3 * SR, 10 ** (-55 / 20), 10 ** (-35 / 20))
    stepped = base + lvl * noise
    steady = base + 10 ** (-45 / 20) * noise
    r1 = background_module(stepped, *_ctx(stepped, cfg), cfg)
    r2 = background_module(steady, *_ctx(steady, cfg), cfg)
    assert r1.status == OK and r2.status == OK
    times = [c["time_s"] for c in r1.candidates]
    assert any(abs(tc - 3.0) < 0.8 for tc in times)
    assert all(c["label"] == "possible discontinuity" for c in r1.candidates)
    assert r2.candidates == []


def test_prosody_on_periodic_and_unvoiced_input(cfg):
    x = speechlike(3.0, f0=150.0)
    stft, act = _ctx(x, cfg)
    r = prosody_module(x, SR, act, cfg)
    assert r.status == OK
    assert r.diagnostics["diag.pros.f0_median_hz"] == pytest.approx(150, rel=0.08)
    assert "pros.f0_median_hz" not in r.features  # speaker level never a classifier input
    rng = np.random.default_rng(0)
    n = 0.2 * rng.standard_normal(SR * 3) * (np.sin(2 * np.pi * 2 * np.arange(SR * 3) / SR) > 0)
    st, a = _ctx(n, cfg)
    assert prosody_module(n, SR, a, cfg).status == INSUFFICIENT  # no invented pitch


def test_enf_presence_duration_gate_and_absence(cfg):
    rng = np.random.default_rng(0)
    long = 0.01 * rng.standard_normal(SR * 12) + 0.02 * np.sin(2 * np.pi * 60 * np.arange(SR * 12) / SR)
    r = enf_module(long, SR, cfg)
    assert r.status == OK and r.diagnostics["diag.enf.best_nominal_hz"] == 60.0
    assert r.diagnostics["diag.enf.track_std_hz"] < 0.1
    short = long[: SR * 3]
    rs = enf_module(short, SR, cfg)
    assert rs.status == INSUFFICIENT and "for continuity analysis" in rs.reason
    assert enf_module(0.01 * rng.standard_normal(SR * 12), SR, cfg).status == INSUFFICIENT
