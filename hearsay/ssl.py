"""Self-supervised speech encoders (Track A front-end, also the embedding space for D2/D5).

For each clip-view we keep, per hidden layer, the time-mean and time-std of the frame features:
an array [n_layers, 2, dim] in float16. Middle layers usually carry more artifact information
than the last one, so every layer is kept and the probe picks.
"""
import numpy as np
import torch
from transformers import AutoFeatureExtractor, AutoModel

from . import config

MODELS = {
    "wavlm_base_plus": "microsoft/wavlm-base-plus",
    "xlsr_300m": "facebook/wav2vec2-xls-r-300m",
}
QUANTUM = 4000  # inputs are truncated to a multiple of 0.25 s: MPS re-plans kernels per input shape


def quantize(x, q=QUANTUM):
    n = max(q, len(x) // q * q)
    return x[:n] if len(x) >= n else np.pad(x, (0, n - len(x)))


class SSLEncoder:
    def __init__(self, key="wavlm_base_plus", device=None):
        self.key, self.name = key, MODELS[key]
        self.device = device or config.device()
        fe = AutoFeatureExtractor.from_pretrained(self.name)
        self.normalize = bool(getattr(fe, "do_normalize", False))
        self.model = AutoModel.from_pretrained(self.name).eval().to(self.device)
        cfg = self.model.config
        self.n_layers, self.dim = cfg.num_hidden_layers + 1, cfg.hidden_size

    @torch.inference_mode()
    def frames_batch(self, xs):
        """Equal-length float32 16 kHz clips -> hidden states [n_layers, B, T, dim] (on device)."""
        x = np.stack([quantize(np.asarray(a, dtype=np.float32)) for a in xs])
        if self.normalize:
            x = (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-7)
        out = self.model(torch.from_numpy(x).to(self.device), output_hidden_states=True)
        return torch.stack(out.hidden_states)

    def frames(self, x):
        return self.frames_batch([x])[:, 0]

    def pooled_batch(self, xs):
        h = self.frames_batch(xs)                                     # [L, B, T, D]
        return torch.stack([h.mean(2), h.std(2)], dim=2).permute(1, 0, 2, 3).to(torch.float16).cpu().numpy()

    def pooled(self, x):
        return self.pooled_batch([x])[0]                              # [L, 2, D]
