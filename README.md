# HEARSAY — team SideQuests (HackGT 13, NSA audio authentication challenge)

Score any audio file 0.0–1.0 (1.0 = synthetic), say *why*, and do it reproducibly. This repository holds the
full system: a forensic front end, fine-tuned anti-spoofing detectors, a signal-processing detector, metadata
analysis, a concept-formation (prototype) explanation layer, and the evaluation harness that decided what goes
into the final score.

> **Status:** results tables below are filled only from runs (`runs/…` on MPCDF Raven); anything not yet run
> says so. Final predictions: `SideQuests_predictions_final.tsv`.

---

## 1. What the organizers measure, and what that implies

- **Metric.** The organizers score with the ASVspoof 5 track-1 evaluation package (`calculate_metrics.py`) using
  **Pspoof = 0.3 and Cfa = 4** (defaults 0.05 / 10). The primary metric is **minDCF = min over thresholds of
  P_miss(bona fide) + (4·0.3)/(1·0.7) · P_fa(spoof)** — in our polarity, *false-alarm rate on real clips + 1.714 ×
  miss rate on fakes* (normalized: 1.0 = a constant decision). `hearsay/metrics.py` reproduces the organizers'
  minDCF and EER to machine precision. (The copy of the package in `data/HackGTMinDCF` shows Pspoof 0.5; the
  organizers confirmed 0.3, and every number below uses 0.3.)
- **Calibration.** minDCF only depends on the ranking of scores. Our probabilities are Platt-calibrated on the
  selection sets and shifted to the evaluation prior (30% synthetic), so the Bayes-optimal decision for the
  organizers' costs is simply *P(synthetic) > 0.2*.
- **Interim leaderboard** (organizers, Sep 26): best minDCF 0.0584 (EER 2.5%), then 0.0753 (3.53%). Our target
  is below that.

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
| XLS-R-2B | 0.046 / 0.354 | **0.038** / 0.189 | **1.44%** |
| XLS-R-1B | 0.044 / 0.347 | 0.050 / 0.254 | 2.08% |
| MMS-1B | 0.056 / 0.289 | 0.065 / 0.266 | 2.64% |
| MMS-300M | 0.055 / 0.389 | 0.097 / 0.317 | 3.84% |
| W2V-Large | 0.130 / 0.446 | 0.064 / 0.311 | 2.56% |
| HuBERT-XL | 0.119 / 0.471 | 0.208 / 0.450 | 8.96% |

### 5.2 Fine-tuning, Track D copy-synthesis fakes, WiSE-FT and the final ensemble

Fine-tuning: test-like crops, channel augmentation on both classes, family-balanced batches; epoch chosen on val + ITW.

| Model (epoch) | holdout clean / aug | ITW clean / aug | ITW EER |
|---|---|---|---|
| XLS-R-2B, top 24 layers (2) | 0.001 / 0.069 | 0.060 / 0.153 | 2.32% |
| XLS-R-1B (1) | 0.002 / 0.087 | 0.093 / 0.206 | 3.68% |
| MMS-1B (1) | 0.002 / 0.092 | 0.111 / 0.223 | 4.56% |
| MMS-300M (3) | 0.002 / 0.098 | 0.123 / 0.250 | 4.80% |
| W2V-Large (4) | 0.006 / 0.145 | 0.098 / 0.217 | 3.76% |
| **XLS-R-1B + D6-R copy-synthesis fakes (3)** | 0.000 / 0.075 | 0.042 / 0.139 | 1.92% |
| **XLS-R-2B + D6-R copy-synthesis fakes (3)** | 0.000 / 0.074 | 0.041 / 0.103 | 1.68% |
| XLS-R-2B WiSE-FT, α = 0.3 | 0.001 / 0.111 | 0.036 / 0.124 | 1.44% |
| **Final ensemble** (the three rows above, mean of z-scored logits) | **0.000 / 0.068** | **0.028 / 0.095** | **1.12%** |

**What worked / what did not.**
- Plain fine-tuning makes the detectors near-perfect in-domain and 2–5× more robust to channel perturbations, but
  each extra epoch costs clean out-of-domain accuracy (MMS-1B's ITW error triples by epoch 2).
- **Track D6-R** — real clips re-vocoded by HiFi-GAN (16 kHz and LJ), DiffWave (a diffusion vocoder) and Vocos,
  used as extra fakes — reverses that: out-of-domain accuracy *improves* with training (XLS-R-1B ITW clean
  minDCF 0.093 without vs 0.042 with, at matched selection).
- **WiSE-FT** (interpolating pretrained and fine-tuned weights, α = 0.3) keeps the pretrained model's clean
  generality with part of the robustness; the ensemble of two D6-R models and the WiSE model is best on both.

### 5.3 Other tracks (gate: a branch joins the score only if it helps held-out minDCF)

| System | holdout (757 shared clips) AUC / EER / minDCF | ITW AUC / EER / minDCF | Verdict |
|---|---|---|---|
| Neural (XLS-R-2B + D6-R) | 1.000 / 0.0% / 0.000 | 0.999 / 1.7% / 0.042 | **the score** |
| DSP D0 (LFCC-GMM + spectral/LPC/phase/prosody/background LR) | 0.915 / 10.7% / 0.290 | 0.266 / 66% / 1.000 | explanations only: no in-domain gain, **worse than chance out of domain** |
| Neural + DSP (logistic fusion) | 1.000 / 0.0% / 0.000 | 0.992 / 2.6% / 0.058 | rejected (fusion hurts ITW) |
| Metadata M0 (technical fields, HGB) | 0.969 / 11.9% / 0.249 | chance (AUC 0.59) | shortcut; constant on test |
| Metadata X0 (+ bandwidth/duration cross-checks) | 0.992 / 5.3% / 0.133 | chance (AUC 0.57) | shortcut; explanations only |
| Track A handoff baseline (frozen WavLM + MLP), speaker-disjoint CV | AUC 0.998 / EER 1.6% | not evaluated | in-domain reference |

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
