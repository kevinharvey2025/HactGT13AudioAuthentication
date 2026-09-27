"""Copy-synthesis (Track D6-R): real clips re-vocoded by public pretrained mel vocoders, used as extra training fakes.

Vocoded real speech keeps the real content and speaker and adds only vocoder artifacts, so a detector trained on it
learns the artifacts instead of who speaks or what is said (scripts/run_d6r.py). Each model maps the canonical 16 kHz
clip to its native rate and mel settings from its model card, reconstructs, and maps back to 16 kHz; `align` removes
the model's delay. Neural codecs were deliberately left out: codec artifacts are a channel, not a sign of synthesis.

  hifigan_16k   speechbrain/tts-hifigan-libritts-16kHz   GAN vocoder, multi-speaker, 16 kHz
  hifigan_lj    speechbrain/tts-hifigan-ljspeech         GAN vocoder, LJSpeech, 22.05 kHz
  diffwave_lj   speechbrain/tts-diffwave-ljspeech        diffusion vocoder, 6-step fast schedule
  vocos         charactr/vocos-mel-24khz                 Fourier-domain GAN vocoder, 24 kHz
"""
import numpy as np
import soxr
import torch

from .. import config

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
    """One pretrained vocoder: reconstruct(x16k float32) -> float32 at 16 kHz (unaligned)."""

    def __init__(self, key, device=None):
        self.key, self.device = key, device or config.device()
        dev = str(self.device)
        if key in ("hifigan_16k", "hifigan_lj"):
            from speechbrain.inference.vocoders import HIFIGAN
            src = {"hifigan_16k": "speechbrain/tts-hifigan-libritts-16kHz", "hifigan_lj": "speechbrain/tts-hifigan-ljspeech"}[key]
            self.model = HIFIGAN.from_hparams(source=src, savedir=str(PRETRAINED / src), run_opts={"device": dev})
            self.sr = 16000 if key == "hifigan_16k" else 22050
        elif key == "diffwave_lj":
            from speechbrain.inference.vocoders import DiffWaveVocoder
            src = "speechbrain/tts-diffwave-ljspeech"
            self.model = DiffWaveVocoder.from_hparams(source=src, savedir=str(PRETRAINED / src), run_opts={"device": dev})
            self.sr = 22050
        elif key == "vocos":
            from vocos import Vocos
            self.model = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(self.device).eval()
            self.sr = 24000
        else:
            raise KeyError(key)

    @torch.inference_mode()
    def reconstruct(self, x):
        y = soxr.resample(x, SR, self.sr).astype(np.float32) if self.sr != SR else x.astype(np.float32)
        t = torch.from_numpy(y).to(self.device)
        if self.key in ("hifigan_16k", "hifigan_lj"):
            out = self.model.decode_batch(_sb_mel(t, self.sr)[None])[0, 0]
        elif self.key == "diffwave_lj":
            out = self.model.decode_batch(_sb_mel(t, self.sr)[None], hop_len=256, fast_sampling=True,
                                          fast_sampling_noise_schedule=FAST_DIFFWAVE_SCHEDULE).reshape(-1)  # [1, T] or [1, 1, T]
        else:
            out = self.model(t[None])[0]
        out = out.float().cpu().numpy()
        return soxr.resample(out, self.sr, SR).astype(np.float32) if self.sr != SR else out


def align(ref, y, max_shift=800):
    """Shift y to best match ref (vocoders add delay); both trimmed to a common length. -> (ref, y, lag)."""
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
