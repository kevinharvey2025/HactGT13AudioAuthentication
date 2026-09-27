"""Channels no model trains on: evaluation views 100 ("unseen") and 101 ("device") of hearsay/views.py.

View 1 draws from the training augmentation's own distribution (hearsay/augment.py), so it favours whatever was
trained on. View 100 uses only operations that are in neither that chain nor RawBoost (hearsay/rawboost.py), one or
two per clip:

  room         image-method room acoustics (pyroomacoustics shoebox: size, Sabine RT60 0.2-1.0 s, source-microphone
               distance 0.5-4 m); the chain's reverb is exponentially decaying noise
  codec        Vorbis, AMR-NB, MP2, AC-3, G.723.1, WMA v2 or RealAudio 1.0; the chain has MP3, AAC, Opus, G.711,
               G.726 and G.722 (GSM, Speex and AMR-WB have no encoder in Raven's ffmpeg/7.1)
  tandem       two of those codecs in sequence, as on sharing pipelines
  packet_loss  1-5% of 20 ms frames lost: zeroed, or concealed by repeating the previous frame (VoIP)
  agc          automatic gain control: frame level tracked with attack/release, gain -12..+18 dB (phones, conferencing)

View 101 is RawBoost-adjacent, so it never counts towards the unseen-channel headline: a small loudspeaker or
microphone (high-pass, one resonance, low-pass) driven into soft saturation.

Both views apply to either class inside the canonical view's augmentation hook, and draw everything from the clip's
own generator (audio.uid_rng), so every system sees the same audio. The definitions are frozen: a change makes a new
benchmark, not a new version of this one. Codec outputs depend on the ffmpeg build (Raven: ffmpeg/7.1), room
impulse responses on pyroomacoustics 0.10.1.
"""
import numpy as np
from scipy.signal import butter, fftconvolve, sosfilt

from . import config
from .augment import _ffmpeg_roundtrip

SR = config.SR

CODECS = {
    # name: (ffmpeg encode args for a setting, container, encode sample rate, settings)
    "vorbis": (lambda q: ["-c:a", "libvorbis", "-q:a", str(q)], "ogg", SR, [0, 1, 2, 4]),
    "amrnb": (lambda b: ["-c:a", "libopencore_amrnb", "-b:a", str(b)], "amr", 8000, [4750, 5900, 7400, 10200, 12200]),
    "mp2": (lambda b: ["-c:a", "mp2", "-b:a", f"{b}k"], "mp2", SR, [32, 48, 64]),
    "ac3": (lambda b: ["-c:a", "ac3", "-b:a", f"{b}k"], "ac3", 32000, [48, 64, 96]),
    "g723_1": (lambda b: ["-c:a", "g723_1", "-b:a", "6300"], "matroska", 8000, [6.3]),
    "wmav2": (lambda b: ["-c:a", "wmav2", "-b:a", f"{b}k"], "asf", SR, [24, 32, 48]),
    "real_144": (lambda b: ["-c:a", "real_144"], "rm", 8000, [8]),
}


def codec(x, rng, name=None):
    name = name or str(rng.choice(sorted(CODECS)))
    args, fmt, enc_sr, settings = CODECS[name]
    s = settings[int(rng.integers(len(settings)))]
    return _ffmpeg_roundtrip(x, args(s), fmt, enc_sr), dict(channel=f"codec_{name}", codec=name, setting=s)


def tandem(x, rng):
    first, second = rng.choice(sorted(CODECS), size=2, replace=False)
    y, p1 = codec(x, rng, str(first))
    y, p2 = codec(y, rng, str(second))
    return y, dict(channel=f"tandem_{first}>{second}", codecs=[p1, p2])


def room(x, rng):
    import pyroomacoustics as pra
    dims = np.array([rng.uniform(3, 10), rng.uniform(3, 8), rng.uniform(2.4, 4)])
    rt60 = float(rng.uniform(0.2, 1.0))
    absorption, order = pra.inverse_sabine(rt60, dims)
    mic = np.array([rng.uniform(0.5, d - 0.5) for d in dims])
    dist = float(rng.uniform(0.5, 4.0))
    for _ in range(100):                      # a source `dist` away, inside the room (0.3 m from the walls)
        v = rng.standard_normal(3)
        v[2] *= 0.3                           # mostly horizontal
        src = mic + dist * v / np.linalg.norm(v)
        if np.all(src > 0.3) and np.all(src < dims - 0.3):
            break
    else:
        src, dist = np.clip(src, 0.3, dims - 0.3), float(np.linalg.norm(np.clip(src, 0.3, dims - 0.3) - mic))
    r = pra.ShoeBox(dims, fs=SR, materials=pra.Material(absorption), max_order=min(int(order), 20))
    r.add_source(src)
    r.add_microphone(mic)
    r.compute_rir()
    rir = np.asarray(r.rir[0][0], float)
    rir = rir[max(0, int(np.argmax(np.abs(rir))) - 40):]          # start at the direct path (keep the filter's lead)
    y = fftconvolve(x, rir)[: len(x)]
    y = y / (np.abs(y).max() + 1e-9) * np.abs(x).max()
    return y.astype(np.float32), dict(channel="room", rt60=round(rt60, 2), dims=np.round(dims, 2).tolist(),
                                      distance=round(dist, 2), max_order=min(int(order), 20))


def packet_loss(x, rng):
    frame = int(0.02 * SR)
    rate = float(rng.uniform(0.01, 0.05))
    mode = str(rng.choice(["zero", "repeat"]))
    y = x.copy()
    lost = np.flatnonzero(rng.random(len(x) // frame) < rate)
    for k in lost:
        seg = slice(k * frame, (k + 1) * frame)
        y[seg] = y[(k - 1) * frame: k * frame] if mode == "repeat" and k > 0 else 0
    return y.astype(np.float32), dict(channel="packet_loss", mode=mode, rate=round(rate, 3), frames=int(len(lost)))


def agc(x, rng):
    frame = int(0.01 * SR)
    target = float(rng.uniform(-26, -16))                          # dBFS RMS
    attack, release = float(rng.uniform(0.01, 0.05)), float(rng.uniform(0.2, 1.0))   # s
    max_gain = float(rng.uniform(12, 18))
    n = max(1, len(x) // frame)
    lev = 10 * np.log10(np.mean(x[: n * frame].reshape(n, frame) ** 2, axis=1) + 1e-10)
    a_att, a_rel = np.exp(-frame / SR / attack), np.exp(-frame / SR / release)
    env, s = np.empty(n), lev[0]
    for i, v in enumerate(lev):
        a = a_att if v > s else a_rel
        s = a * s + (1 - a) * v
        env[i] = s
    g = np.clip(target - env, -12.0, max_gain)
    gain = 10 ** (np.interp(np.arange(len(x)), (np.arange(n) + 0.5) * frame, g) / 20)
    return (x * gain).astype(np.float32), dict(channel="agc", target_db=round(target, 1), attack_s=round(attack, 3),
                                                release_s=round(release, 2), max_gain_db=round(max_gain, 1))


OPS = {"room": room, "codec": codec, "tandem": tandem, "packet_loss": packet_loss, "agc": agc}
WEIGHTS = {"room": 2, "codec": 2, "tandem": 1, "packet_loss": 1, "agc": 1}


def random_unseen(x, rng):
    """View 100: one or two unseen operations in sequence -> (audio, params) with a combined channel label."""
    names = list(WEIGHTS)
    p = np.array([WEIGHTS[k] for k in names], float)
    ops = rng.choice(names, size=int(rng.choice([1, 1, 2])), replace=False, p=p / p.sum())
    params = []
    for op in ops:
        x, prm = OPS[op](x, rng)
        params.append(prm)
    return x, dict(channel="+".join(q["channel"] for q in params), ops=params)


def device(x, rng):
    """View 101 (RawBoost-adjacent): small loudspeaker/microphone response driven into soft saturation."""
    hp, lp = float(rng.uniform(150, 400)), float(rng.uniform(4500, 7000))
    fr, q, g_db = float(rng.uniform(1000, 4000)), float(rng.uniform(1, 4)), float(rng.uniform(3, 9))
    drive = float(rng.uniform(1.5, 4.0))
    y = sosfilt(butter(2, hp, "highpass", fs=SR, output="sos"), x)
    w0, amp = 2 * np.pi * fr / SR, 10 ** (g_db / 40)              # RBJ peaking biquad
    alpha = np.sin(w0) / (2 * q)
    b = np.array([1 + alpha * amp, -2 * np.cos(w0), 1 - alpha * amp])
    a = np.array([1 + alpha / amp, -2 * np.cos(w0), 1 - alpha / amp])
    y = sosfilt(np.r_[b / a[0], a / a[0]][None], y)
    y = sosfilt(butter(4, lp, fs=SR, output="sos"), y)
    y = y / (np.abs(y).max() + 1e-9)
    y = np.tanh(drive * y) / np.tanh(drive) * np.abs(x).max()
    return y.astype(np.float32), dict(channel="device", highpass_hz=round(hp), lowpass_hz=round(lp),
                                      resonance_hz=round(fr), resonance_db=round(g_db, 1), drive=round(drive, 2))
