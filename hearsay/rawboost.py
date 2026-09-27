"""RawBoost (Tak, Kamble, Patino, Todisco & Evans, ICASSP 2022): raw-waveform data boosting with three signal models,
as a switchable training augmentation (scripts/finetune_ssl.py --rawboost-algo, --p-rawboost).

  1  LnL  linear and non-linear convolutive noise: the sum over k = 1..N_f of band-stop FIR filtering of x^k, the
          higher orders attenuated (channel and device frequency responses, amplifier non-linearity)
  2  ISD  impulsive signal-dependent additive noise on a random P% of the samples
  3  SSI  stationary signal-independent additive noise: white noise through random band-stop filters, 10-40 dB SNR
  combinations, numbered as in the reference: 0 none, 1-3 alone, 4 series 1->2->3, 5 series 1->2, 6 series 1->3,
  7 series 2->3, 8 parallel 1||2 (summed, then normalized)

RawBoost has no room model: LnL's filters are short band-stop FIRs, not impulse responses.

A port of the reference implementation (github.com/TakHemlata/SSL_Anti-spoofing: RawBoost.py, and
process_Rawboost_feature in data_utils_SSL.py; MIT licence, notice below). Two changes:
- every random draw comes from a passed-in generator instead of the global np.random state, with the same draws in
  the same order, so a legacy np.random.RandomState reproduces the reference exactly (tests/test_rawboost.py);
- every call returns (audio, params) like hearsay/augment.py.
The FIR filtering uses FFT convolution instead of lfilter: the same output to floating-point rounding, about 10x
faster. The reference's quirks are kept on purpose; each is marked "quirk".

The AntiDeepfake checkpoints we fine-tune were post-trained with algo 5 (their model cards: "RawBoost series:
(1)+(2)"; the -nda checkpoints are the ones without it).

    MIT License

    Copyright (c) 2022 Hemlata

    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
    documentation files (the "Software"), to deal in the Software without restriction, including without limitation
    the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
    permit persons to whom the Software is furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all copies or substantial portions of
    the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
    WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
    COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
    OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import dataclasses

import numpy as np
from scipy.signal import fftconvolve, firwin, freqz

from . import config

ALGOS = {0: (), 1: ("lnl",), 2: ("isd",), 3: ("ssi",), 4: ("lnl", "isd", "ssi"), 5: ("lnl", "isd"), 6: ("lnl", "ssi"),
         7: ("isd", "ssi"), 8: ("lnl|isd",)}


@dataclasses.dataclass(frozen=True)
class RawBoostParams:
    """The reference defaults (main_SSL_LA.py); field names and integer types are its command-line flags."""
    nBands: int = 5              # notch filters per FIR (LnL, SSI)
    minF: int = 20               # notch centre frequency, Hz
    maxF: int = 8000
    minBW: int = 100             # notch bandwidth, Hz
    maxBW: int = 1000
    minCoeff: int = 10           # taps per notch filter
    maxCoeff: int = 100
    minG: int = 0                # filter gain, dB
    maxG: int = 0
    minBiasLinNonLin: int = 5    # LnL: gain offset of the non-linear orders, dB
    maxBiasLinNonLin: int = 20
    N_f: int = 5                 # LnL: highest polynomial order (1 = linear only)
    P: int = 10                  # ISD: at most P% of the samples
    g_sd: int = 2                # ISD: gain
    SNRmin: int = 10             # SSI: SNR, dB
    SNRmax: int = 40

    @classmethod
    def parse(cls, spec=""):
        """'maxF=7000,SNRmin=5' -> the defaults with these fields replaced; an unknown field raises."""
        kw = {}
        for kv in filter(None, (s.strip() for s in spec.split(","))):
            k, v = (t.strip() for t in kv.split("="))
            if k not in cls.__dataclass_fields__:
                raise ValueError(f"unknown RawBoost parameter {k!r}")
            kw[k] = int(v)
        return cls(**kw)

    def to_json(self):
        return dataclasses.asdict(self)


def _draw(rng, lo, hi):
    # the reference's randRange, uniform(lo, hi), written out as numpy computes it (lo + (hi - lo) * U): a Generator's
    # uniform() refuses lo > hi, which LnL's quirk needs; a RandomState gives the reference's values either way
    return float(lo + (hi - lo) * rng.random(1)[0])


def norm_wav(x, always):
    """Divide by the peak: always, or only when it exceeds 1 (the reference's normWav)."""
    m = np.amax(np.abs(x)) if len(x) else 0.0
    if (always and m > 0) or m > 1:        # (the reference divides by a zero peak when `always`; it never has one)
        x = x / m
    return x


def notch_filter(rng, prm, min_g, max_g, fs, log=None):
    """nBands cascaded band-stop FIRs with a random gain (the reference's genNotchCoeffs)."""
    b, bands = np.ones(1), []
    for _ in range(prm.nBands):
        fc, bw = _draw(rng, prm.minF, prm.maxF), _draw(rng, prm.minBW, prm.maxBW)
        c = int(_draw(rng, prm.minCoeff, prm.maxCoeff))
        if c % 2 == 0:                     # quirk: band-stop needs an odd tap count, so even counts gain one tap
            c += 1
        f1, f2 = fc - bw / 2, fc + bw / 2
        f1 = 1 / 1000 if f1 <= 0 else f1   # quirk: band edges clamped just inside (0, fs/2)
        f2 = fs / 2 - 1 / 1000 if f2 >= fs / 2 else f2
        # two cutoffs with firwin's default pass_zero=True: a band-stop filter
        b = np.convolve(firwin(c, [f1, f2], window="hamming", fs=fs), b)
        bands.append([fc, bw, c])
    g = _draw(rng, min_g, max_g)
    _, h = freqz(b, 1, fs=fs)
    b = pow(10, g / 20) * b / np.amax(np.abs(h))
    if log is not None:
        log.append(dict(gain_db=g, bands=bands))
    return b


def fir(x, b):
    """The reference's filterFIR: lfilter(b, 1, x padded by len(b) + 1 zeros), then the centre len(x) samples."""
    n = len(b) + 1
    y = fftconvolve(np.pad(x, (0, n)), b)[: len(x) + n]
    return y[int(n / 2): int(len(y) - n / 2)]


def lnl(x, rng, prm, fs, log):
    y = np.zeros(len(x))
    min_g, max_g, orders = prm.minG, prm.maxG, []
    for i in range(prm.N_f):
        if i == 1:   # quirk: the gain bounds shift once, from the second order on: uniform(-5, -20) by default, low > high
            min_g, max_g = min_g - prm.minBiasLinNonLin, max_g - prm.maxBiasLinNonLin
        b = notch_filter(rng, prm, min_g, max_g, fs, orders)
        y = y + fir(np.power(x, i + 1), b)
    log["lnl"] = orders
    return norm_wav(y - np.mean(y), always=False)


def isd(x, rng, prm, log):
    beta = _draw(rng, 0, prm.P)
    y = x.copy()
    n = int(len(x) * (beta / 100))
    p = rng.permutation(len(x))[:n]
    f_r = (2 * rng.random(n) - 1) * (2 * rng.random(n) - 1)
    y[p] = x[p] + prm.g_sd * x[p] * f_r
    log["isd"] = dict(percent=beta, samples=n)
    return norm_wav(y, always=False)


def ssi(x, rng, prm, fs, log):
    noise = rng.normal(0, 1, len(x))
    filt = []
    noise = norm_wav(fir(noise, notch_filter(rng, prm, prm.minG, prm.maxG, fs, filt)), always=True)
    snr = _draw(rng, prm.SNRmin, prm.SNRmax)
    noise = noise / np.linalg.norm(noise, 2) * np.linalg.norm(x, 2) / 10.0 ** (0.05 * snr)
    log["ssi"] = dict(snr_db=snr, **filt[0])
    return x + noise                       # quirk: no renormalization after SSI (the canonical view normalizes)


def apply(x, rng, algo, prm=RawBoostParams(), fs=config.SR):
    """-> (float32 audio of len(x), params). `rng`: a numpy Generator, or a RandomState for parity with the reference.
    Silent input stays silent; empty input is returned as is."""
    if algo not in ALGOS:
        raise ValueError(f"RawBoost algo must be 0-8, got {algo}")
    params = dict(channel=f"rawboost_a{algo}", algo=algo)
    if algo == 0 or len(x) == 0:
        return np.asarray(x, np.float32), params
    y = x
    for stage in ALGOS[algo]:
        if stage == "lnl":
            y = lnl(y, rng, prm, fs, params)
        elif stage == "isd":
            y = isd(y, rng, prm, params)
        elif stage == "ssi":
            y = ssi(y, rng, prm, fs, params)
        else:                              # 8: LnL and ISD of the same input, summed
            y = norm_wav(lnl(x, rng, prm, fs, params) + isd(x, rng, prm, params), always=False)
    return np.asarray(y, np.float32), params
