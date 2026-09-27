# The detector

How the submitted score is produced. Data and the canonical view: [DATA.md](DATA.md). Results:
[EVALUATION.md](EVALUATION.md). The explanation layer: [CONCEPTS.md](CONCEPTS.md).

```
audio file ─► triage (ffprobe: container, codec, rate, encoder; trace only, never scored)
           ─► canonical view (16 kHz mono, trim, 7 kHz low-pass, DC removal, peak-normalize, dither)
           ─► 3 fine-tuned anti-spoofing detectors ─► synthetic logits ─► z-normalize ─► mean ─► Platt (30% prior)
           ─► cm-score = P(synthetic); router adds windowed re-scoring / compression cross-checks to the trace
```

## 1. Backbones: AntiDeepfake, run through `transformers`

The NII AntiDeepfake models (Wang et al., arXiv 2506.21090) are wav2vec 2.0 / HuBERT encoders post-trained on
56k hours of real and 18k hours of artefact speech, followed by mean pooling and a linear head with logits
[fake, real]. The released checkpoints are fairseq files, and fairseq 0.12.2 does not install on current Python, so
`hearsay/antideepfake.py` rebuilds each architecture in `transformers` and maps the weights with a strict key map:
every tensor is placed and every shape checked, and a missing or unexpected key is an error. That strictness caught a
positional-convolution weight-norm naming bug before it could silently degrade a model. Input is standardized per
clip, as in the original recipe. The synthetic logit is `logit_fake - logit_real`.

Zero-shot, on our canonical view, XLS-R-2B is the strongest (In-the-Wild minDCF 0.038, EER 1.44%). The table in
[EVALUATION.md](EVALUATION.md) has all six backbones.

## 2. Fine-tuning (`scripts/finetune_ssl.py`, `mpcdf/finetune.sbatch`)

| Setting | Value | Why |
|---|---|---|
| Optimizer | AdamW, LR 2e-6 encoder / 1e-4 head, weight decay 0.01, 10% warm-up + cosine | gentle updates keep the post-training (the literature fine-tunes at 1e-6 to 5e-6) |
| Schedule | 800 steps per epoch, 3-4 epochs; every epoch scored on val / holdout / In-the-Wild, clean and augmented | |
| Checkpoint | epoch with the lowest mean minDCF over (val, In-the-Wild) x (clean, augmented) | picks for generalization and robustness, not in-domain fit |
| Crops | lengths drawn from the test-set duration distribution, start-anchored half the time; clips >= 3.3 s | matches what the model sees at test time |
| Batches | family-balanced: LJ-voice fakes 0.2, cloned fakes 0.2, copy-synthesis 0.1, LJSpeech 0.2, cloned-speaker LibriSpeech 0.15, other LibriSpeech 0.15; class-weighted loss | no source dominates by count |
| Augmentation | a random channel chain on 60% of training clips, both classes | a noisy real clip must not read as fake |
| Memory | bf16 autocast, gradient checkpointing, CNN front end frozen; XLS-R-2B also freezes its bottom 24 of 48 layers (40 GB A100) | |

**Copy-synthesis fakes (Track D6-R, `scripts/run_d6r.py`, `hearsay/diffusion/resynth.py`).** 2,975 train-split real
clips per vocoder are re-vocoded by HiFi-GAN (16 kHz LibriTTS and 22 kHz LJSpeech), DiffWave (a diffusion vocoder)
and Vocos, delay-aligned and labeled synthetic. These fakes carry real content and a real speaker, so the only thing
separating them from their source clip is the vocoder. Plain fine-tuning makes every backbone near-perfect in domain
but costs out-of-domain accuracy with each epoch. D6-R reverses that: out-of-domain accuracy improves while training.
Neural codecs were excluded on purpose, because codec artifacts are a channel, not synthesis.

**WiSE-FT** (`--wise-from`, `--wise-alpha`): weights interpolated between the pretrained and fine-tuned models,
theta = (1 - a) theta_pretrained + a theta_finetuned. It trades robustness against zero-shot generality. It is
used in the interim ensemble, not the final one.

## 3. The final score (`scripts/make_submission.py`, `predict.py`)

- **Ensemble:** XLS-R-2B (epoch 3), XLS-R-1B (epoch 2) and MMS-1B (epoch 3), all fine-tuned with the four-vocoder
  D6-R fakes. That makes three backbones pretrained on different data, so their errors differ. Each model's logit is
  z-normalized with statistics from val + In-the-Wild, and the three are averaged (no fitted weights: fusion weights
  overfit when there is little dev data).
- **Calibration:** Platt scaling fitted on val + In-the-Wild, both views, class-balanced. The log-odds are then
  shifted to the evaluation prior of 30% synthetic, so `cm-score` is P(synthetic) and the Bayes decision for the
  organizers' costs is P > 0.2. minDCF depends only on the ranking; calibration makes the number mean something to a
  reader.
- **Stored parameters:** `submission/fusion.json` (per-model z statistics, Platt coefficients, prior). The stored
  checkpoints are bf16; `predict.py` loads them into fp32 models.

## 4. `predict.py`: inference, routing and traces

`predict.py --input DIR --output DIR [--template FILE]` scores any file ffmpeg can decode.
- **Scoring:** every clip is scored whole and alone, in fp32 on every device. A file's score does not depend on the
  other files in the folder (it moves by less than 3e-5), and CPU and GPU agree.
- **Memory:** the three models are loaded one at a time from memory-mapped checkpoints. Peak memory is about 14.5 GB
  (22 GB when all three were resident).
- **Traces:** for each file it writes a trace (`traces.jsonl`) holding the triage facts, the per-model logits, the
  fused score and every routing decision with its reason:

| Condition | Extra analysis | What it reports |
|---|---|---|
| 0.05 < P < 0.8 (uncertain) | windowed re-scoring: 2 s windows, 1 s hop | logit spread across the clip; a spread > 6 logits flags a possible partial fake (a spliced synthetic segment) |
| clip > 6 s | windowed re-scoring | score drift across the clip |
| lossy codec, or a high declared rate with a narrow measured band | compression / bandwidth cross-check | what the container claims vs what the signal shows |
| confident score, clean container | stop | |

Only the fused detector score reaches the TSV; the routed analyses explain it. Files that fail to decode are listed
in `failures.tsv` and the export is refused rather than filled with invented scores. The output is validated before
it is written: header, one row per template file in template order, finite values in [0, 1], LF line endings.

**Docker** (`Dockerfile`): python:3.11-slim, ffmpeg and CPU torch. The three checkpoints and `fusion.json` go in
`artifacts/diffusion/` at build time, and it runs with `--network none`. Give the container at least 16 GB of memory
(Docker Desktop defaults to less). On CPU, the NSA test set takes about 2 hours at 8 threads, or 21 minutes on one
72-core node with `mpcdf/predict_cpu.sh`, which runs 4 processes over shards of the template
([results/runtime.md](../results/runtime.md)). The same path on all 4,000 labeled In-the-Wild clips gives AUC
0.9994, EER 1.30%, minDCF 0.033.

**The submitted TSV** was written by the earlier version of this path, which scored clips in batches cropped to
their shortest member, in bf16. The organizers scored that file: minDCF 0.0317, EER 1.44%. The current path agrees
with it on 1,668 of 1,671 decisions ([EVALUATION.md](EVALUATION.md) §4).
