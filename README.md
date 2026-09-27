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

*(filled from runs; see `runs/diffusion/ft/*/log.jsonl`, `runs/dsp/`, `runs/meta/`, `runs/concepts/`)*

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
