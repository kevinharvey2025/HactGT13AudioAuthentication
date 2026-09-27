"""NII AntiDeepfake detectors (Ge et al., arXiv 2506.21090) without fairseq.

Each checkpoint (huggingface.co/nii-yamagishilab/<name>-anti-deepfake, CC BY-NC-SA 4.0) is a
fairseq wav2vec 2.0 / HuBERT encoder post-trained on ~74k hours of real and synthetic speech, plus
mean pooling over time of the final encoder output and a linear layer with logits [fake, real].
The official inference code needs fairseq 0.12.2, which does not install on current Python, so
the weights are mapped onto the equivalent `transformers` modules (stable layer norm: layer norm
before each block and after the last; a layer-norm conv feature extractor). Input: 16 kHz mono,
standardized over the whole clip (the model card's layer_norm(wav, wav.shape)).

The mapping is strict: every encoder/feature-extractor tensor must land somewhere and every
target parameter must be filled; only pre-training leftovers (quantizer, projections for the
contrastive/HuBERT losses, mask embedding) are dropped.
"""
import re

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

# name -> (repo, architecture, hidden, layers, ffn); all use 16 heads and a 7-layer conv front-end
VARIANTS = {
    "xlsr_2b": ("nii-yamagishilab/xls-r-2b-anti-deepfake", "wav2vec2", 1920, 48, 7680),
    "xlsr_1b": ("nii-yamagishilab/xls-r-1b-anti-deepfake", "wav2vec2", 1280, 48, 5120),
    "mms_1b": ("nii-yamagishilab/mms-1b-anti-deepfake", "wav2vec2", 1280, 48, 5120),
    "mms_300m": ("nii-yamagishilab/mms-300m-anti-deepfake", "wav2vec2", 1024, 24, 4096),
    "w2v_large": ("nii-yamagishilab/wav2vec-large-anti-deepfake", "wav2vec2", 1024, 24, 4096),
    "hubert_xl": ("nii-yamagishilab/hubert-xlarge-anti-deepfake", "hubert", 1280, 48, 5120),
}
DROP = re.compile(r"^(quantizer\.|project_q\.|final_proj\.|mask_emb$|label_embs_concat$)")


def _config(arch, hidden, layers, ffn):
    from transformers import HubertConfig, Wav2Vec2Config
    kw = dict(hidden_size=hidden, num_hidden_layers=layers, intermediate_size=ffn, num_attention_heads=16,
              feat_extract_norm="layer", do_stable_layer_norm=True, conv_bias=(arch == "wav2vec2"),
              conv_dim=(512,) * 7, conv_kernel=(10, 3, 3, 3, 3, 2, 2), conv_stride=(5, 2, 2, 2, 2, 2, 2),
              num_conv_pos_embeddings=128, num_conv_pos_embedding_groups=16, hidden_act="gelu",
              hidden_dropout=0.0, attention_dropout=0.0, activation_dropout=0.0, feat_proj_dropout=0.0,
              final_dropout=0.0, layerdrop=0.0, apply_spec_augment=False, mask_time_prob=0.0)
    if arch == "hubert":
        return HubertConfig(feat_proj_layer_norm=True, **kw)
    return Wav2Vec2Config(**kw)


def _rename(k):
    """fairseq key (after the m_ssl.model. prefix) -> transformers key, or None to drop."""
    if DROP.match(k):
        return None
    rules = [
        (r"^feature_extractor\.conv_layers\.(\d+)\.0\.", r"feature_extractor.conv_layers.\1.conv."),
        (r"^feature_extractor\.conv_layers\.(\d+)\.2\.1\.", r"feature_extractor.conv_layers.\1.layer_norm."),
        (r"^layer_norm\.", "feature_projection.layer_norm."),
        (r"^post_extract_proj\.", "feature_projection.projection."),
        (r"^encoder\.pos_conv\.0\.bias$", "encoder.pos_conv_embed.conv.bias"),
        (r"^encoder\.pos_conv\.0\.weight_g$", "encoder.pos_conv_embed.conv.WEIGHT_G"),
        (r"^encoder\.pos_conv\.0\.weight_v$", "encoder.pos_conv_embed.conv.WEIGHT_V"),
        (r"^encoder\.layers\.(\d+)\.self_attn\.", r"encoder.layers.\1.attention."),
        (r"^encoder\.layers\.(\d+)\.self_attn_layer_norm\.", r"encoder.layers.\1.layer_norm."),
        (r"^encoder\.layers\.(\d+)\.fc1\.", r"encoder.layers.\1.feed_forward.intermediate_dense."),
        (r"^encoder\.layers\.(\d+)\.fc2\.", r"encoder.layers.\1.feed_forward.output_dense."),
        (r"^encoder\.layers\.(\d+)\.final_layer_norm\.", r"encoder.layers.\1.final_layer_norm."),
        (r"^encoder\.layer_norm\.", "encoder.layer_norm."),
    ]
    for pat, rep in rules:
        if re.match(pat, k):
            return re.sub(pat, rep, k)
    raise KeyError(f"unmapped AntiDeepfake tensor: {k}")


def load(name, device="cpu", pretrained=True):
    """-> (encoder: transformers model, head: torch.nn.Linear [fake, real] logits).
    pretrained=False builds the architecture only (for loading a full fine-tuned state dict offline)."""
    from transformers import HubertModel, Wav2Vec2Model
    repo, arch, hidden, layers, ffn = VARIANTS[name]
    enc = (HubertModel if arch == "hubert" else Wav2Vec2Model)(_config(arch, hidden, layers, ffn))
    if not pretrained:
        return enc.eval().to(device), torch.nn.Linear(hidden, 2).eval().to(device)
    sd = load_file(hf_hub_download(repo, "model.safetensors"))
    target = enc.state_dict()
    # weight norm of the positional conv: plain (weight_g/_v) or parametrized (original0/1) depending on torch
    g_key = next(k for k in target if k.startswith("encoder.pos_conv_embed.conv.") and k.endswith(("weight_g", "original0")))
    v_key = next(k for k in target if k.startswith("encoder.pos_conv_embed.conv.") and k.endswith(("weight_v", "original1")))
    mapped = {}
    for k, v in sd.items():
        if not k.startswith("m_ssl.model."):
            continue
        t = _rename(k[len("m_ssl.model."):])
        if t is None:
            continue
        t = t.replace("encoder.pos_conv_embed.conv.WEIGHT_G", g_key).replace("encoder.pos_conv_embed.conv.WEIGHT_V", v_key)
        mapped[t] = v
    missing = sorted(set(target) - set(mapped))
    unexpected = sorted(set(mapped) - set(target))
    # masked_spec_embed exists in transformers even with spec augment off; it is never used at inference
    missing = [k for k in missing if k != "masked_spec_embed"]
    if missing or unexpected:
        raise RuntimeError(f"{name}: missing {missing[:8]} ({len(missing)}), unexpected {unexpected[:8]} ({len(unexpected)})")
    for k, v in mapped.items():
        if tuple(target[k].shape) != tuple(v.shape):
            raise RuntimeError(f"{name}: shape mismatch for {k}: {tuple(v.shape)} vs {tuple(target[k].shape)}")
    enc.load_state_dict(mapped, strict=False)
    head = torch.nn.Linear(hidden, 2)
    head.load_state_dict({"weight": sd["proj_fc.weight"], "bias": sd["proj_fc.bias"]})
    return enc.eval().to(device), head.eval().to(device)


def standardize(x):
    """The model card's torch.nn.functional.layer_norm(wav, wav.shape), per clip."""
    return torch.nn.functional.layer_norm(x, x.shape[-1:])


def synthetic_logit(logits):
    """[.., 2] logits in [fake, real] order -> log-odds of fake (higher = synthetic)."""
    return logits[..., 0] - logits[..., 1]
