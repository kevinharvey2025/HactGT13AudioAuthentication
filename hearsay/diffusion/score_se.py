"""D4 (exploratory, lowest priority in Track D): a score-based speech enhancement model as a feature.

SGMSE+ (github.com/sp-uhh/sgmse, not on PyPI; clone it onto PYTHONPATH and download a 16 kHz
checkpoint, e.g. VoiceBank-DEMAND: `gdown 1_H3EXvhcYBhOZ9QNUcD5VZHc6ktrRbwQ`) is trained on
clean real speech in the compressed complex STFT domain, conditioned on a noisy observation.
It is a conditional model (noisy -> clean), not an unconditional prior, so the features are
indirect:

  dsm_t      denoising score-matching error at SDE time t, with the clip as both the clean target
             and the conditioning observation (how well real-speech training explains this clip)
  enh_*      how much reverse-diffusion "enhancement" changes the clip (log-spectral distance, SNR)

Caveat from the plan: enhancement can erase the artifacts a detector needs, so enhanced audio is
only ever an input to the prosody / speaker-drift modules, never to the DL detector.
"""
import numpy as np
import torch

T_GRID = (0.05, 0.1, 0.2, 0.4, 0.6, 0.8)


class ScoreSE:
    def __init__(self, ckpt, device=None, n_reverse=30, t_grid=T_GRID, n_draws=2):
        from sgmse.model import ScoreModel
        from sgmse.util.other import pad_spec
        # complex STFT ops are not fully supported on MPS; CPU unless CUDA is present
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ScoreModel.load_from_checkpoint(ckpt, map_location=str(self.device))
        self.model.eval()
        self.pad_spec, self.n_reverse, self.t_grid, self.n_draws = pad_spec, n_reverse, t_grid, n_draws
        self.pad_mode = "reflection" if self.model.backbone in ("ncsnpp_48k", "ncsnpp_v2") else "zero_pad"
        if self.model.backbone == "ncsnpp_48k":
            raise ValueError("use a 16 kHz checkpoint: the canonical clips are 16 kHz")

    def _spec(self, x):
        y = torch.from_numpy(np.asarray(x, np.float32))[None].to(self.device)
        norm = y.abs().max()
        Y = self.model._forward_transform(self.model._stft(y / norm))[None]
        return self.pad_spec(Y, mode=self.pad_mode), norm, y.shape[1]

    @torch.no_grad()
    def dsm_curve(self, x, seed=0):
        Y, _, _ = self._spec(x)
        g = torch.Generator(device="cpu").manual_seed(seed)
        out = {}
        for t in self.t_grid:
            tt = torch.full((1,), float(t), device=self.device)
            errs = []
            for _ in range(self.n_draws):
                mean, std = self.model.sde.marginal_prob(Y, Y, tt)
                sigma = std[:, None, None, None]
                z = torch.complex(torch.randn(Y.shape, generator=g), torch.randn(Y.shape, generator=g)).to(self.device) / np.sqrt(2)
                F = self.model(mean + sigma * z, Y, tt)
                if self.model.loss_type == "score_matching":
                    err = F * sigma + z                                  # model output is the score
                else:                                                    # denoiser / data prediction
                    err = (F - Y) / sigma
                errs.append((err.abs() ** 2).mean().real)
            out[f"d4_dsm_t{t}"] = float(torch.stack(errs).mean())
        return out

    @torch.no_grad()
    def enhance(self, x):
        Y, norm, n = self._spec(x)
        if self.model.sde.__class__.__name__ == "OUVESDE":
            sampler = self.model.get_pc_sampler("reverse_diffusion", "ald", Y, N=self.n_reverse, corrector_steps=1, snr=0.5)
        else:
            sampler = self.model.get_sb_sampler(sde=self.model.sde, y=Y, sampler_type="ode")
        sample, _ = sampler()
        return (self.model.to_audio(sample.squeeze(), n) * norm).cpu().numpy().astype(np.float32)

    def features(self, x):
        f = self.dsm_curve(x)
        xh = self.enhance(x)
        n = min(len(x), len(xh))
        X = np.abs(np.fft.rfft(x[:n] * np.hanning(n)))
        H = np.abs(np.fft.rfft(xh[:n] * np.hanning(n)))
        f["d4_enh_logspec_dist"] = float(np.mean(np.abs(np.log(X + 1e-6) - np.log(H + 1e-6))))
        f["d4_enh_snr_db"] = float(10 * np.log10(np.sum(x[:n] ** 2) / (np.sum((x[:n] - xh[:n]) ** 2) + 1e-12)))
        return f
