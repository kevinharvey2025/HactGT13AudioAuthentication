# HEARSAY — Master plan (restructured around the metadata-analysis prompt)

**Status:** locked-in plan for the final night. Written Sat Sep 26 2026, ~21:00 EDT, after the move to Karthik's
machine + MPCDF Raven. It supersedes the ordering in the individual prompts, not their rules: each prompt in
`plans/` still governs its own track (see its "Addendum" section for the literature notes and updates).

| Prompt | Track | Role in the final system |
|---|---|---|
| `diffusion_cf_prompt.md` | neural detector (Track A), router, fusion, Track D | **detection core** (Track A as fine-tuned anti-spoofing SSL) + Track D (gated) |
| `AASISTand_AntiDeepfake_prompt.md` | unchanged pretrained baselines | **zero-shot benchmark rows** (reference recipe), the starting checkpoints for fine-tuning |
| `signal_processing_prompt.md` | DSP-only detector | **D0** in the metadata matrix; forensic breadth; fusion candidate |
| `metadata_analysis_prompt.md` | metadata alone and with DSP | **evaluation spine**: namespaces, shared split, M0–M2/F0–F2/X0 matrix, interventions |

---

## 1. What changed since the handoffs

1. **Official metric known.** The organizers score with the ASVspoof 5 evaluation package (`calculate_metrics.py`)
   using **`Pspoof = 0.3`, `Cfa = 4`** (defaults 0.05 / 10; confirmed by the organizers on Sep 26 night — the tarball
   copy in `data/HackGTMinDCF` shows 0.5). Primary metric: **minDCF = min over thresholds of FPR(bona fide) +
   1.714·FNR(spoof)** (normalized by min(Cmiss·0.7, Cfa·0.3) = 0.7; 1.0 = constant decision); EER, CLLR, actDCF
   secondary. `hearsay/metrics.py` reproduces their minDCF and EER to machine precision (200 random trials).
   Probabilities are calibrated at the 0.3 prior, so the Bayes decision is P(synthetic) > 0.2.
2. **Targets.** Interim leaderboard (organizers, Sep 26 evening): minDCF/EER 0.0584/2.5%, 0.0753/3.53%, 0.178/6.92%,
   0.258/10.18%, 0.267/10.4%, 0.913/34.6%. **We must beat 0.0584.** The scores confirm the organizers evaluate with
   1.0 = synthetic (their package assumes the opposite polarity; they flip it). They re-analyse in the morning;
   the final TSV must carry a **"final" label** in its name (e.g. `<team>_predictions_final.tsv`).
3. **External data is allowed.** The brief lists ASVspoof 2019/2021/5, WaveFake, In-the-Wild, MLAAD, ADD, VCTK,
   LibriSpeech and VOiCES under "Datasets for training / calibration". This settles the open question of both
   handoffs. Pretrained detectors trained on those sets are fair game.
4. **Compute.** MPCDF Raven: whole 4×A100 nodes start within minutes (single-GPU jobs wait ~10 h); 72-core CPU nodes.
   Workspace `/ptmp/karthiksing/hearsay/{data,cache,runs,artifacts,logs}`; code runs from immutable snapshots
   (`mpcdf/env.sh` links the workspace in). Laptop has <10 GB free: nothing heavy runs locally.
5. **Data on Raven.** Test set (1,671), organizer reals (242), LJSpeech-1.1 (13,100), LibriSpeech clone speakers
   (1,152), LibriSpeech dev+test-clean (5,323; 80 speakers never cloned), DiffSSD: 14,895 fakes (exactly the
   union both manifest builders sample; ~1,480 per generator) with the rest uploading. The organizers' archive has
   no `metadata.csv`; `scripts/rebuild_diffssd_metadata.py` rebuilds it (70,242 rows, identical to the old count).
6. **New detector family.** NII AntiDeepfake checkpoints (wav2vec 2.0 / HuBERT post-trained on ~74k h of real and
   synthetic speech; In-the-Wild EER 1.2–2.9% zero-shot) run through `transformers` via `hearsay/antideepfake.py`
   (no fairseq). They are the starting point for fine-tuning.

---

## 2. Evidence taxonomy (metadata prompt §3, applied to every track)

| Namespace | Examples | Allowed use |
|---|---|---|
| `neural` | fine-tuned/zero-shot SSL detector logits, pooled embeddings | score |
| `dsp` | LFCC-GMM LLR, spectral/LPC/phase/prosody/background features | score (D0), fusion |
| `meta.technical` | codec, rate, channels, bit depth, bitrate, container duration | M0 (training-only experiment) |
| `meta.tags` | encoder family (`Lavf58.29.100`), tag presence | M1 (shortcut audit) |
| `meta.structure` | RIFF chunk layout, padding, trailing bytes | M3 (optional) / diagnostics |
| `meta.consistency` | header byte-rate/block-align, container vs stream duration | M2 |
| `cross` | declared rate vs occupied bandwidth, declared vs decoded duration | X0 only |
| `diagnostic` | paths, file ids, MAC/archive times, parser errors, content-match results | audit/report only |

**Never classifier inputs:** filenames/ids, MAC or archive times, hashes, content matches against training corpora,
row order. These are reported in the forensic audit (WP0), not used to score.

---

## 3. Shared data protocol

- **Canonical view** (both classes, and the test set): ffmpeg → 16 kHz mono PCM16 → trim edge silence → [training:
  crop to a length drawn from the test-duration distribution] → [augmentation] → 7 kHz low-pass → DC removal →
  peak-normalize → 1-LSB dither. Updated by WP0 if the audit identifies the organizers' exact resampling chain.
- **Shared split** (`hearsay.splits.shared_split`, seed 20260926): `holdout` uses the DSP track's rules (20% of
  sentence ids, clone speakers 2061/5448, 20% of LJ chapters, 20% of LibriSpeech (speaker, chapter) groups, 20% of
  extra-real speakers); `val` = a further 15% of the remaining groups; `train` = the rest. The DSP pool's holdout
  rows are holdout here too, so neural, DSP, metadata and fusion scores are compared on the same held-out clips.
- **Selection** only on `val`; `holdout` is reported, never tuned on; the test set is never used for fitting,
  calibration, thresholds or selection (unsupervised diagnostics only: score distribution, cross-model agreement).

---

## 4. Work packages, in order

| WP | What | Runs on | Gate / output |
|---|---|---|---|
| **WP0** | Forensic integrity audit ("cryptographic analysis"): byte/PCM hashes and duplicates, RIFF structure, sample-histogram/LSB/dither/gain history, length quantization, band-edge analysis, identification of the organizers' resampling chain, landmark-hash content matching of test clips against every corpus, watermark/C2PA check, archive-time analysis | CPU node | `reports/forensics/audit.md`; findings may change the canonical view (resampler emulation) and the training pool. **Before any official training.** |
| **WP1** | Detection core: zero-shot benchmark of 6 AntiDeepfake backbones (epoch 0 of `finetune_ssl.py`), then fine-tune the best 2–3 on the pool (test-like crops, channel augmentation, family-balanced batches), one unseen-generator run | 4×A100 node | val/holdout/unseen-generator minDCF per backbone; best checkpoints |
| **WP2** | Handoff Track A (frozen WavLM Base+ + layer-weighted MLP) | A100 | benchmark row |
| **WP3** | DSP D0 (handoff pipeline unchanged): suite → train on dev → holdout once → test predictions | CPU node | D0 row, per-module ablations |
| **WP4** | Metadata M0/M1/M2 (+ consistency) on the DSP pool rows; tag-strip/replace/remux interventions | CPU | expected: near-perfect training AUC from rate/codec/encoder shortcuts, constant on test → documented negative result + forensic-breadth credit |
| **WP5** | Fusion: F0/F1/F2/X0 (metadata × DSP) and N+D (neural + DSP): logistic late fusion on `val` scores, evaluated on `holdout` | CPU | a branch joins the final score only if holdout minDCF improves and robustness views do not degrade |
| **WP6** | Calibration (Platt at prior 0.5 on `val`), final TSV `<team>_predictions_final.tsv`, validator, per-file trace JSON + template explanations | CPU | valid TSV |
| **WP7** | Track D (diffusion H1/H2/H3, D6) | A100 | **only with the user's approval** and if the literature (Addendum of `diffusion_cf_prompt.md`) predicts a gain; otherwise documented as implemented-not-run |
| **WP8** | README, Dockerfile(s), report figures | laptop | deliverables |

WP1–WP4 run in parallel once WP0 is done (different Raven nodes).

---

## 5. Benchmark table (fill from runs only)

| System | Namespaces | val minDCF | holdout minDCF | holdout EER | unseen-gen minDCF | test frac > 0.5 | Notes |
|---|---|---|---|---|---|---|---|
| AntiDeepfake zero-shot (6 backbones) | neural | | | | | | reference recipe on canonical view |
| Fine-tuned AntiDeepfake (best) | neural | | | | | | |
| Track A WavLM+MLP | neural | | | | | | handoff baseline |
| D0 DSP | dsp | | | | | | |
| M0 / M1 / M2 | meta | | | | | | test constant |
| F0 / F1 / F2 / X0 | meta+dsp | | | | | | |
| N+D fusion | neural+dsp | | | | | | |

---

## 6. Open questions for the user

- Team name for `<team>_predictions_final.tsv`.
- Approval to run Track D (diffusion) code (handoff rule: ask first).
- Whether any train–test content overlap found in WP0 should be reported to the organizers (it will not be used for scoring).

---

## 7. WP0 results (forensic integrity audit, done before training)

Full numbers: `plans/metadata_analysis_prompt.md` Addendum D.1 and `runs/forensics/` on Raven.
- **No duplicates**: 0 byte or decoded-PCM duplicate groups within the test set or across test ↔ organizer reals,
  LJSpeech, LibriSpeech, DiffSSD.
- **No content reuse**: landmark-hash matching of all 1,671 test clips against 34,712 reference recordings (LJSpeech
  13,100; LibriSpeech 6,475; organizer reals 242; DiffSSD 14,895) finds nothing (max vote fraction 0.056), while the
  positive control (100 reference files pushed through the test-like pipeline) matches its source 100/100 times
  (median vote fraction 0.93–0.97). The test reals and fakes come from **other sources** than our training corpora
  → optimise for generalization, not DiffSSD fit. (Re-run against all 70,000 DiffSSD files once unpacked.)
- **One container, one pipeline**: identical ffmpeg-4.2 WAV layout, no provenance/hidden chunks, random ids, one
  batch write; 98% of lengths are 512-sample multiples at 22.05 kHz, start-trimmed and hard-cut, kaiser-class
  resampling to 16 kHz (−20 dB at 7.39 kHz), peak-normalized. Metadata is constant on test.

## 8. Decisions locked in after the literature review

1. **Primary detector**: fine-tuned AntiDeepfake (XLS-R-1B, XLS-R-2B top half, MMS-1B), LR 2e-6, 3.2k steps,
   batch 16–24, 3–4.5 s test-like crops, channel augmentation on both classes (p 0.6), family-balanced batches,
   default (not `-nda`) checkpoints. Zero-shot rows for all six backbones come from epoch 0.
2. **Evaluation sets**: shared `val` (selection) and `holdout` (report), plus **In-the-Wild** (4,000 clips,
   uncontaminated, never trained on) — checkpoint selection = mean of val and ITW minDCF (clean+augmented views).
   DiffSSD-based scores for AntiDeepfake backbones are flagged as contaminated in every table.
3. **Fusion**: average of z-normalized logits of the 2–4 best systems (by ITW + val minDCF); logistic fusion with
   effective spoof prior 0.8 only if it beats the average on ITW. No test-set adaptation or normalization.
4. **Final file**: `SideQuests_predictions_final.tsv`, calibrated probabilities (Platt, shifted to the evaluation
   prior 0.3), 10 decimals (no ties).
5. **DSP (D0)** and **metadata (M0–M2)** run as documented tracks; they join the score only through WP5's gate.
6. **Track D**: only D6-R (resynthesized reals as extra fakes) is evidence-backed; not run without approval.
