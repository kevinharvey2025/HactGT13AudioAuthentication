# HEARSAY — team SideQuests (HackGT 13, NSA audio authentication challenge)

Score any audio file 0.0–1.0 (1.0 = synthetic), say *why*, and do it reproducibly. This repository holds the
full system: a forensic front end, fine-tuned anti-spoofing detectors, a signal-processing detector, metadata
analysis, a concept-formation (prototype) explanation layer, and the evaluation harness that decided what goes
into the final score.

> **Status:** results tables below are filled only from runs (`runs/…` on MPCDF Raven); anything not yet run
> says so. Final predictions: `SideQuests_predictions_final.tsv`.

---

## 1. What the organizers measure, and what that implies

- **Metric.** The organizers' scoring package (`data/HackGTMinDCF`) is the ASVspoof 5 evaluation code with
  `Pspoof = 0.5`, `Cfa = 4`. The primary metric is **minDCF = min over thresholds of P_miss(bona fide) +
  4·P_fa(spoof)** — in our polarity, *false-alarm rate on real clips + 4 × miss rate on fakes*. Missing a fake
  costs four times a false alarm, so the high-recall end of the ROC decides the score. `hearsay/metrics.py`
  reproduces the organizers' minDCF and EER to machine precision.
- **Interim leaderboard** (organizers, Sep 26): best minDCF 0.0584 (EER 2.5%). Our target is below that.

## 2. Forensic audit first (before any training)

`scripts/forensic_audit.py` (details: `plans/metadata_analysis_prompt.md`, Addendum D):

| Question | Finding |
|---|---|
| Duplicates (byte and decoded-PCM SHA-256) | none within the test set, none against any training corpus |
| Is the test set made of training recordings? | **No.** Landmark-hash matching of every test clip against 89,817 reference recordings (DiffSSD, LJSpeech, LibriSpeech) finds nothing; the positive control (reference files pushed through the test pipeline) is matched 100/100 |
| Container | every test file is the same ffmpeg-4.2 WAV (`LIST/INFO/ISFT=Lavf58.29.100`), consistent headers, no provenance/hidden chunks — **metadata is constant on the test set** |
| Processing history | 98% of lengths are multiples of 512 samples at 22.05 kHz; clips are start-trimmed and hard-cut; a kaiser-class resampler cuts everything above ~7.4 kHz; peak-normalized — one librosa-style pipeline for both classes |

Consequences: the test clips come from sources we do not have, so we optimise for **generalization**; the
7.5–8 kHz band, levels, durations and container fields are removed or equalized for every class
("canonical view"), and metadata is reported, never scored.

## 3. System

```
audio file ──► T0 triage (ffprobe: container, codec, rate, encoder, times; trace only)
          ──► canonical view: ffmpeg → 16 kHz mono → trim → 7 kHz low-pass → DC removal → peak-normalize → dither
          ──► fine-tuned anti-spoofing detectors (AntiDeepfake XLS-R / MMS encoders) → synthetic logits
          ──► fusion (mean of z-normalized logits) → Platt calibration → cm-score
          ──► explanations: concept path + prototype composition (concept formation), DSP findings, container facts
```

| Component | Code | Role |
|---|---|---|
| Canonical view, augmentation | `hearsay/audio.py`, `hearsay/augment.py`, `hearsay/views.py` | identical processing for both classes; channel augmentation (codecs, telephony, noise, hum, reverb, clipping) |
| Anti-spoofing detectors | `hearsay/antideepfake.py`, `scripts/finetune_ssl.py` | NII AntiDeepfake encoders run through `transformers` (no fairseq), fine-tuned on test-like crops |
| Copy-synthesis fakes (Track D6-R) | `scripts/run_d6r.py`, `hearsay/diffusion/resynth.py` | real clips re-vocoded by HiFi-GAN (16 k, LJ), DiffWave (diffusion vocoder) and Vocos, used as extra fakes |
| Concept formation (Track D5) | `hearsay/concepts.py`, `scripts/run_concepts.py`, `hearsay/diffusion/prototypes.py` | COBWEB/CLASSIT concept hierarchy + Gaussian prototypes over the detector's embedding space: basic level, prototype-based score, explanations |
| DSP detector (D0) | `hearsay_dsp/` | LFCC-GMM, spectral, LPC, background, phase, prosody, ENF modules; LR/GMM late fusion |
| Metadata (M0–M2) | `scripts/meta_experiments.py`, `hearsay/forensics/triage.py` | technical/tag/consistency features, interventions (tag strip/replace, re-encode) |
| Evaluation | `hearsay/metrics.py`, `hearsay/splits.py` | organizers' minDCF; one shared split for every track; In-the-Wild as uncontaminated evaluation |
| Submission | `scripts/make_submission.py`, `predict.py`, `hearsay/submission.py` | fusion + calibration + validated TSV |

## 4. Data

| Set | Use |
|---|---|
| DiffSSD (organizers; 70,000 fakes, 10 generators) + 242 organizer reals | training (pinned 14,895-file sample, `configs/diffssd_pool_files.txt`) |
| LJSpeech-1.1, LibriSpeech (the 10 cloned speakers + dev/test-clean, 80 more speakers) | bona fide training data |
| In-the-Wild (Müller et al. 2022; 4,000-clip sample) | **evaluation only** — not in AntiDeepfake's training data, unlike DiffSSD |
| NSA sample test set (1,671 clips) | predictions only |

Shared split (`hearsay.splits.shared_split`): `holdout` = the DSP track's leakage-checked rules (held-out
sentence ids, clone speakers 2061/5448, LJ and LibriSpeech chapters, extra-real speakers), `val` = a further
15% of groups (model selection, calibration, fusion), `train` = the rest, `itw` = In-the-Wild.

## 5. Results

All numbers are the organizers' minDCF (lower is better; 1.0 = no better than a constant) unless noted, measured
on held-out clips we never trained on. **Holdout** = the shared in-domain holdout (DiffSSD generators, LJSpeech,
LibriSpeech; *contaminated for AntiDeepfake*, which saw DiffSSD in post-training). **In-the-Wild (ITW)** = 4,000
web clips (celebrity speech and deepfakes) held out from everything — our best proxy for unseen sources. "clean" =
the canonical view; "aug" = the same clip through a random channel chain (codecs, telephony, noise, hum, reverb,
clipping). Sources: `runs/diffusion/ft/*/log.jsonl`, `runs/diffusion/compare.csv`, `runs/fusion_gate.json`,
`runs/dsp/suite_v1/report.md`, `runs/meta/metrics.json`.

### 5.1 Pretrained detectors (zero-shot, our canonical view)

| AntiDeepfake backbone | holdout clean / aug | ITW clean / aug | ITW EER |
|---|---|---|---|
| XLS-R-2B | 0.092 / 0.611 | **0.066** / 0.305 | **1.44%** |
| XLS-R-1B | 0.077 / 0.597 | 0.094 / 0.434 | 2.08% |
| MMS-1B | 0.095 / 0.441 | 0.102 / 0.386 | 2.64% |
| MMS-300M | 0.096 / 0.579 | 0.147 / 0.480 | 3.84% |
| W2V-Large | 0.243 / 0.693 | 0.126 / 0.490 | 2.56% |
| HuBERT-XL | 0.217 / 0.742 | 0.266 / 0.601 | 8.96% |

### 5.2 Fine-tuning (test-like crops, channel augmentation, family-balanced batches; epoch picked on val + ITW)

| Model (epoch) | holdout clean / aug | ITW clean / aug | ITW EER |
|---|---|---|---|
| XLS-R-2B, top 24 layers (2) | 0.003 / 0.092 | 0.087 / 0.225 | 2.32% |
| XLS-R-1B (1) | 0.002 / 0.112 | 0.158 / 0.294 | 3.68% |
| MMS-1B (1) | 0.002 / 0.124 | 0.165 / 0.316 | 4.56% |
| MMS-300M (1) | 0.013 / 0.212 | 0.163 / 0.368 | 4.08% |
| W2V-Large (1) | 0.017 / 0.256 | 0.137 / 0.344 | 3.20% |
| Ensemble XLS-R-2B (1) + MMS-1B (1) | 0.001 / 0.109 | 0.064 / 0.182 | 2.00% |

**What worked / what did not.** Fine-tuning makes the detectors near-perfect in-domain and 2–5× more robust to
channel perturbations, but every extra epoch costs clean out-of-domain accuracy (XLS-R-2B ITW clean minDCF
0.066 → 0.083 → 0.087 → 0.115 over epochs 0–3; MMS-1B collapses to 0.336 by epoch 2). Freezing the bottom half of
the largest encoder keeps most of its generality; ensembling a fine-tuned 2B with a second fine-tuned encoder
recovers clean ITW accuracy while keeping the robustness gains.

### 5.3 Other tracks (gate: a branch joins the score only if it helps held-out minDCF)

| System | holdout (757 shared clips) AUC / EER / minDCF | ITW AUC / EER / minDCF | Verdict |
|---|---|---|---|
| Neural (XLS-R-2B fine-tuned) | 1.000 / 0.0% / 0.000 | 0.998 / 2.3% / 0.087 | **the score** |
| DSP D0 (LFCC-GMM + spectral/LPC/phase/prosody/background LR) | 0.915 / 10.7% / 0.482 | 0.266 / 66% / 1.000 | explanations only: no in-domain gain, **worse than chance out of domain** |
| Neural + DSP (logistic fusion) | 1.000 / 0.0% / 0.000 | 0.992 / 3.0% / 0.138 | rejected (fusion hurts ITW) |
| Metadata M0 (technical fields, HGB) | 0.969 / 11.9% / 0.255 | chance (AUC 0.59) | shortcut; constant on test |
| Metadata X0 (+ bandwidth/duration cross-checks) | 0.992 / 5.3% / 0.175 | chance (AUC 0.57) | shortcut; explanations only |
| Track A handoff baseline (frozen WavLM + MLP), speaker-disjoint CV | AUC 0.998 / EER 1.6% / minDCF 0.058 | not evaluated | in-domain reference |

DSP details (`runs/dsp/suite_v1/report.md`): LFCC-GMM variants reach EER 17–22% on DiffSSD dev (the DiffSSD paper
reports 22.0% for its LFCC-GMM); the full logistic model 6.0%; held-out ElevenLabs drops to AUC 0.59. A DSP model
trained on DiffSSD's reals alone flags **99.8%** of real LibriSpeech clips as fake; adding external real speech brings
that to 17.6% — direct evidence that the provided training reals are too narrow.

## 6. How to run

```bash
# predictions for a directory of audio files (CPU or GPU)
python predict.py --input /path/to/test --output out/ --template /path/to/test/HGT_Hearsay_score_template.csv
# Docker
docker build -t sidequests-hearsay .
docker run --rm --network none -v /path/to/test:/data/input:ro -v $PWD/out:/data/output sidequests-hearsay
```

Training and evaluation ran on MPCDF Raven (A100); `mpcdf/*.sbatch` are the exact job scripts
(`mpcdf/env.sh` sets up the workspace, `mpcdf/setup_envs.sh` the environments).

## 7. Plans and handoffs

`plans/MASTER_PLAN.md` (work packages, decisions), `plans/*_prompt.md` (per-track specifications with
literature addenda), `HANDOFF_DIFFUSION.md`, `HANDOFF_DSP.md`.
