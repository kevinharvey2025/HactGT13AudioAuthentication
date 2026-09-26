"""D1: vocoder / codec resynthesis residuals (tests H1 and H3).

H1 ("on-manifold reconstruction", the audio analogue of DIRE): audio that already came out of a
neural vocoder sits on a vocoder's output manifold, so resynthesizing it through a similar
pretrained vocoder changes it less than it changes real audio. H3: which model reconstructs a
clip best hints at the generator family.

Each model maps the canonical 16 kHz clip to its native rate and feature space with the exact
settings from its model card, reconstructs, maps back to 16 kHz, is delay-aligned to the input
and low-passed at 7 kHz like the input. Residual features per model:

  mrstft      multi-resolution log-magnitude L1 (n_fft 256/512/1024)
  mel_l1      80-band log-mel L1
  band_*      log-magnitude L1 in 0-2, 2-4, 4-7 kHz
  voiced/unv  log-magnitude L1 in voiced-like vs noise-like frames
  snr_db      waveform SNR after alignment (phase-sensitive)

plus cross-model features: each model's mrstft relative to the per-clip mean over models (public
vocoders are mostly trained on one speaker, so absolute error mostly measures speaker/model
mismatch) and the argmin model. Real clips passed through these models double as D6
copy-synthesis fakes (real content and speaker, vocoder artifacts only).

Models (all public checkpoints; weights are fetched on first use):
  hifigan_16k   speechbrain/tts-hifigan-libritts-16kHz  GAN vocoder, multi-speaker, 16 kHz
  hifigan_lj    speechbrain/tts-hifigan-ljspeech        GAN vocoder, LJSpeech, 22.05 kHz
  diffwave_lj   speechbrain/tts-diffwave-ljspeech       diffusion vocoder, 6-step fast schedule
  vocos         charactr/vocos-mel-24khz                Fourier-domain GAN vocoder, 24 kHz
  encodec_6k    facebook/encodec_24khz @ 6 kbps         neural codec
  encodec_1k5   facebook/encodec_24khz @ 1.5 kbps       neural codec, low rate
  dac_16k       descript/dac_16khz                      neural codec, 16 kHz
  mp3_64k       ffmpeg libmp3lame round trip            classical codec reference
  bigvgan_22k   nvidia/bigvgan_v2_22khz_80band_256x     optional: needs the NVIDIA BigVGAN repo on PYTHONPATH
"""
import numpy as np
import soxr
import torch

from .. import audio, augment, config

SR = config.SR
PRETRAINED = config.CACHE / "pretrained"
FAST_DIFFWAVE_SCHEDULE = [0.0001, 0.001, 0.01, 0.05, 0.2, 0.5]   # 6 steps, from the model card


def _sb_mel(x, sr):
    """Mel settings from the SpeechBrain HiFi-GAN / DiffWave model cards (80 bins, hop 256, 0-8 kHz)."""
    from speechbrain.lobes.models.HifiGAN import mel_spectrogram
    return mel_spectrogram(sample_rate=sr, hop_length=256, win_length=1024, n_fft=1024, n_mels=80, f_min=0.0,
                           f_max=8000.0, power=1, normalized=False, norm="slaney", mel_scale="slaney",
                           compression=True, audio=x)


class Resynthesizer:
    """One pretrained model: reconstruct(x16k float32) -> float32 at 16 kHz (unaligned)."""

    def __init__(self, key, device=None):
        self.key = key
        self.device = device or config.device()
        self._load()

    def _load(self):
        k, dev = self.key, str(self.device)
        if k in ("hifigan_16k", "hifigan_lj"):
            from speechbrain.inference.vocoders import HIFIGAN
            src = {"hifigan_16k": "speechbrain/tts-hifigan-libritts-16kHz", "hifigan_lj": "speechbrain/tts-hifigan-ljspeech"}[k]
            self.model = HIFIGAN.from_hparams(source=src, savedir=str(PRETRAINED / src), run_opts={"device": dev})
            self.sr = 16000 if k == "hifigan_16k" else 22050
        elif k == "diffwave_lj":
            from speechbrain.inference.vocoders import DiffWaveVocoder
            src = "speechbrain/tts-diffwave-ljspeech"
            self.model = DiffWaveVocoder.from_hparams(source=src, savedir=str(PRETRAINED / src), run_opts={"device": dev})
            self.sr = 22050
        elif k == "vocos":
            from vocos import Vocos
            self.model = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(self.device).eval()
            self.sr = 24000
        elif k.startswith("encodec"):
            from transformers import EncodecModel
            self.model = EncodecModel.from_pretrained("facebook/encodec_24khz").to(self.device).eval()
            self.sr, self.bandwidth = 24000, {"encodec_6k": 6.0, "encodec_1k5": 1.5}[k]
        elif k == "dac_16k":
            from transformers import DacModel
            self.model = DacModel.from_pretrained("descript/dac_16khz").to(self.device).eval()
            self.sr = 16000
        elif k == "bigvgan_22k":
            import bigvgan  # github.com/NVIDIA/BigVGAN (not on PyPI)
            from meldataset import get_mel_spectrogram
            self.model = bigvgan.BigVGAN.from_pretrained("nvidia/bigvgan_v2_22khz_80band_256x", use_cuda_kernel=False)
            self.model.remove_weight_norm()
            self.model = self.model.eval().to(self.device)
            self._bigvgan_mel = get_mel_spectrogram
            self.sr = 22050
        elif k == "mp3_64k":
            self.model, self.sr = None, SR
        else:
            raise KeyError(k)

    @torch.inference_mode()
    def reconstruct(self, x):
        y = soxr.resample(x, SR, self.sr).astype(np.float32) if self.sr != SR else x.astype(np.float32)
        t = torch.from_numpy(y).to(self.device)
        k = self.key
        if k in ("hifigan_16k", "hifigan_lj"):
            out = self.model.decode_batch(_sb_mel(t, self.sr)[None])[0, 0]
        elif k == "diffwave_lj":
            out = self.model.decode_batch(_sb_mel(t, self.sr)[None], hop_len=256, fast_sampling=True,
                                          fast_sampling_noise_schedule=FAST_DIFFWAVE_SCHEDULE)[0, 0]
        elif k == "vocos":
            out = self.model(t[None])[0]
        elif k.startswith("encodec"):
            out = self.model(t[None, None], bandwidth=self.bandwidth).audio_values[0, 0]
        elif k == "dac_16k":
            out = self.model(t[None, None]).audio_values[0]
            out = out[0] if out.dim() > 1 else out
        elif k == "bigvgan_22k":
            mel = self._bigvgan_mel(t[None], self.model.h).to(self.device)
            out = self.model(mel)[0, 0]
        elif k == "mp3_64k":
            return augment.codec(x, np.random.default_rng(0), "mp3", 64)[0]
        out = out.float().cpu().numpy()
        return soxr.resample(out, self.sr, SR).astype(np.float32) if self.sr != SR else out


# ----------------------------------------------------------------------------- residual features
def align(ref, y, max_shift=800):
    """Shift y to best match ref (codecs and vocoders add delay); both trimmed to a common length."""
    n = min(len(ref), len(y))
    ref, y = ref[:n], y[:n]
    c = np.fft.irfft(np.fft.rfft(ref, 2 * n) * np.conj(np.fft.rfft(y, 2 * n)))
    c = np.concatenate([c[-max_shift:], c[: max_shift + 1]])
    lag = int(np.argmax(c)) - max_shift
    y = np.roll(y, lag)
    if lag > 0:
        y[:lag] = 0
    elif lag < 0:
        y[lag:] = 0
    return ref, y, lag


def _logmag(x, n_fft):
    w = torch.hann_window(n_fft)
    S = torch.stft(torch.from_numpy(x), n_fft, n_fft // 4, window=w, return_complex=True).abs().numpy()
    f = np.fft.rfftfreq(n_fft, 1 / SR)
    return np.log(S + 1e-5), f


def _logmel(x):
    import librosa
    m = librosa.feature.melspectrogram(y=x, sr=SR, n_fft=1024, hop_length=256, n_mels=80, fmax=7000, power=1.0)
    return np.log(m + 1e-5)


def residual_features(x, y):
    """x: canonical input, y: aligned + low-passed reconstruction (same length)."""
    out = {}
    errs = []
    for n_fft in (256, 512, 1024):
        (X, f), (Y, _) = _logmag(x, n_fft), _logmag(y, n_fft)
        keep = f <= 7000
        errs.append(np.abs(X[keep] - Y[keep]).mean())
        if n_fft == 512:
            D = np.abs(X - Y)
            for lo, hi in [(0, 2000), (2000, 4000), (4000, 7000)]:
                out[f"band_{lo // 1000}_{hi // 1000}k"] = float(D[(f >= lo) & (f < hi)].mean())
            # voiced-like (energetic, low spectral flatness) vs noise-like (energetic, flat) frames
            P = np.exp(2 * X[keep])
            energy = 10 * np.log10(P.sum(0) + 1e-12)
            flat = np.exp(np.log(P + 1e-12).mean(0)) / (P.mean(0) + 1e-12)
            active = energy > energy.max() - 35
            voiced = active & (flat < np.median(flat[active]) if active.any() else active)
            unvoiced = active & ~voiced
            dk = D[keep]
            out["voiced"] = float(dk[:, voiced].mean()) if voiced.any() else np.nan
            out["unvoiced"] = float(dk[:, unvoiced].mean()) if unvoiced.any() else np.nan
    out["mrstft"] = float(np.mean(errs))
    out["mel_l1"] = float(np.abs(_logmel(x) - _logmel(y)).mean())
    out["snr_db"] = float(10 * np.log10(np.sum(x ** 2) / (np.sum((x - y) ** 2) + 1e-12) + 1e-12))
    return out


class D1Features:
    """Run a set of resynthesizers over clips and assemble per-clip residual features."""

    DEFAULT = ("hifigan_16k", "hifigan_lj", "diffwave_lj", "vocos", "encodec_6k", "dac_16k", "mp3_64k")

    def __init__(self, keys=DEFAULT, device=None, save_dir=None):
        self.models = {k: Resynthesizer(k, device) for k in keys}
        self.save_dir = save_dir   # if set, reconstructions are kept (D6 copy-synthesis material)

    def __call__(self, uid, x):
        row = {}
        for k, m in self.models.items():
            y = audio.lowpass(m.reconstruct(x))
            xr, ya, lag = align(x, y)
            for name, v in residual_features(xr, ya).items():
                row[f"d1_{k}_{name}"] = v
            row[f"d1_{k}_lag"] = lag
            if self.save_dir is not None:
                import soundfile as sf
                p = self.save_dir / k / f"{uid}.wav"
                p.parent.mkdir(parents=True, exist_ok=True)
                sf.write(p, np.clip(ya, -1, 1), SR, subtype="PCM_16")
        mr = {k: row[f"d1_{k}_mrstft"] for k in self.models}
        mean = np.mean(list(mr.values()))
        for k, v in mr.items():
            row[f"d1_{k}_rel_mrstft"] = v - mean
        row["d1_argmin_model"] = min(mr, key=mr.get)
        return row
