"""D3: latent audio diffusion denoising curve (H2 with a pretrained general-audio prior).

AudioLDM 2 (cvssp/audioldm2, via diffusers) is used as a prior over "what audio looks like":
the clip's log-mel is encoded by the VAE, noised to several timesteps, and the UNet's noise
prediction under the unconditional branch (embeddings of the empty prompt, i.e. the
classifier-free-guidance negative branch) is scored against the true noise. The per-t loss
curve is a likelihood proxy, the same idea as diffusion classifiers. The VAE round-trip error
is a cheap extra feature that needs no diffusion steps. General-audio models cover speech,
noise and environmental sound, so this is secondary to D2 for speech but relevant to scene
manipulation (fabricated backgrounds).

Front-end follows AudioLDM's TacotronSTFT: 16 kHz, n_fft 1024, hop 160, 64 slaney mel bins,
0-8 kHz, magnitude mel, log(clamp(1e-5)), waveform mean-removed and peak-scaled to 0.5.
Scheduler: the pipeline's own (scaled-linear betas, epsilon prediction, T=1000).
"""
import numpy as np
import torch

from .. import config

T_GRID = (25, 50, 100, 200, 300, 400, 600, 800)
FRAME_MULT = 32   # VAE downsamples time by 4 and the UNet by 8 more; pad frames to a multiple of 32


class LatentCurve:
    def __init__(self, repo="cvssp/audioldm2", device=None, dtype=None, t_grid=T_GRID, n_draws=2):
        from diffusers import AudioLDM2Pipeline
        self.device = device or config.device()
        self.dtype = dtype or (torch.float16 if self.device.type == "cuda" else torch.float32)
        pipe = AudioLDM2Pipeline.from_pretrained(repo, torch_dtype=self.dtype).to(self.device)
        self.vae, self.unet, self.scheduler = pipe.vae, pipe.unet, pipe.scheduler
        self.t_grid, self.n_draws = t_grid, n_draws
        with torch.inference_mode():   # unconditional branch = embeddings of the empty prompt
            self.prompt_embeds, self.attention_mask, self.generated_embeds = pipe.encode_prompt(
                [""], self.device, num_waveforms_per_prompt=1, do_classifier_free_guidance=False)
        # the text encoders / GPT-2 / vocoder are released with `pipe` when __init__ returns
        import torchaudio
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, n_fft=1024, win_length=1024, hop_length=160, f_min=0.0, f_max=8000.0,
            n_mels=64, power=1.0, norm="slaney", mel_scale="slaney").to(self.device)

    def logmel(self, x):
        x = x - x.mean()
        x = 0.5 * x / (np.abs(x).max() + 1e-8)
        m = torch.log(torch.clamp(self.melspec(torch.from_numpy(x.astype(np.float32)).to(self.device)), min=1e-5))
        m = m.T                                                    # [frames, 64]
        n = m.shape[0]
        pad = (-n) % FRAME_MULT
        m = torch.nn.functional.pad(m, (0, 0, 0, pad), value=float(np.log(1e-5)))
        return m[None, None].to(self.dtype), n                     # [1, 1, T, 64]

    @torch.inference_mode()
    def __call__(self, x, seed=0):
        mel, n_frames = self.logmel(x)
        post = self.vae.encode(mel).latent_dist
        z0 = post.mean * self.vae.config.scaling_factor           # deterministic encoding
        valid = max(1, n_frames // 4)                              # latent rows covering real audio
        feats = {}
        recon = self.vae.decode(z0 / self.vae.config.scaling_factor).sample
        feats["d3_vae_recon_l1"] = float((recon - mel)[..., :n_frames, :].abs().float().mean())
        g = torch.Generator(device="cpu").manual_seed(seed)
        emb = dict(encoder_hidden_states=self.generated_embeds, encoder_hidden_states_1=self.prompt_embeds,
                   encoder_attention_mask_1=self.attention_mask, return_dict=False)
        for t in self.t_grid:
            losses, lo_band, hi_band = [], [], []
            for _ in range(self.n_draws):
                noise = torch.randn(z0.shape, generator=g).to(self.device, self.dtype)
                tt = torch.tensor([t], device=self.device, dtype=torch.long)
                zt = self.scheduler.add_noise(z0, noise, tt)
                pred = self.unet(zt, tt, **emb)[0]
                err = ((pred - noise) ** 2)[..., :valid, :].float()  # [1, C, rows, 16 mel-latent cols]
                losses.append(err.mean())
                lo_band.append(err[..., :8].mean())                  # lower half of the mel axis
                hi_band.append(err[..., 8:].mean())
            feats[f"d3_loss_t{t}"] = float(torch.stack(losses).mean())
            feats[f"d3_loss_lo_t{t}"] = float(torch.stack(lo_band).mean())
            feats[f"d3_loss_hi_t{t}"] = float(torch.stack(hi_band).mean())
        ts = np.asarray(self.t_grid)
        curve = np.array([feats[f"d3_loss_t{t}"] for t in self.t_grid])
        feats["d3_fine_minus_coarse"] = float(curve[ts <= 100].mean() - curve[ts >= 400].mean())
        return feats
