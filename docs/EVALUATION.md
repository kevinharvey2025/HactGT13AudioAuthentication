# Evaluation: protocol, the final test matrix, results

Every number here comes from `scripts/evaluate.py`, which reads the per-clip scores every system saved and never
re-scores anything. It writes [results/tables.md](../results/tables.md) (all tables, generated) and the CSVs next
to it. Reproduce with `mpcdf/evaluate.sbatch` (see [REPRODUCE.md](REPRODUCE.md)).

## 1. Protocol and pipeline

- **Metric:** the organizers' minDCF (ASVspoof 5 evaluation package, Pspoof 0.3, Cmiss 1, Cfa 4). In our polarity
  it is min over thresholds of FPR(real) + 1.714 · FNR(fake), normalized so that 1.0 = a constant decision. EER is
  reported alongside. `hearsay/metrics.py` matches the organizers' code to 1e-12 (`tests/test_metrics.py` runs their
  module from `data/`). It evaluates only thresholds a detector can realize; with tied scores the package can report
  slightly lower values.
- **Sets** (2,500 label-stratified clips each; [DATA.md](DATA.md)):
  - **val** chooses checkpoints and fits calibration;
  - **holdout** is in domain and never used for any choice, but DiffSSD is contaminated for the AntiDeepfake
    backbones;
  - **In-the-Wild (ITW)** is out of domain and uncontaminated. It is also one of the two sets checkpoint selection
    averages over, so the final ensemble's ITW numbers are mildly optimistic.
- **Views:** every clip is scored clean (canonical view) and "aug": the same clip through one random channel chain,
  identical for every system.
- **Uncertainty:** 1,000-sample stratified bootstrap (reals and fakes resampled separately) → 95% intervals. Each
  system is compared with the final ensemble on the same clips and the same resamples (paired Δ).
- **Never used** for training, selection or calibration: test labels (we have none), test content matches, test
  file times, test-time adaptation.

### The pipeline

`scripts/evaluate.py` runs as one Raven job (`mpcdf/evaluate.sbatch`), after the unit tests. It reads the per-clip
scores the runs saved and re-scores nothing:

```text
INPUTS
  runs/diffusion/ft/<run>/{val,holdout,itw,test}_epoch<e>.parquet   every detector, every epoch, clean + perturbed scores
  runs/diffusion/concepts/<emb>/scores_*.parquet, test_explanations.jsonl   cobweb and TTCG P(fake)
  runs/dsp/..., runs/meta/...                                        DSP and metadata branches (their own out-of-fold scores)
  runs/diffusion/predict_{itw,final}_v*/                             predict.py on all ITW clips and the NSA test set
      |
      v
 1. load_neural + znorm      align on common clips; z-statistics from val + ITW; ensembles = mean of z-scores
 2. load_concepts / dsp_meta add the other branches; concept fusions are fitted on val only
 3. replay                   rebuild every evaluation clip's crop and channel chain with the training RNG
 4. benchmark                AUC, EER, minDCF with stratified bootstrap CIs; paired delta vs the final ensemble
 5. breakdown                per fake generator, real source, channel condition, crop duration; FA / miss at P > 0.2
 6. calibration_table        Platt fitted on {val+ITW, val, ITW}, scored only on the other sets: actDCF, Cllr, ECE
 7. curves                   every run and epoch under the official metric
 8. fusion_gate              logistic fusion (effective prior 0.632) fitted on val, tested on holdout / ITW
 9. test_agreement           NSA test set without labels: Spearman and kappa vs final, flag rate, score vs duration
10. docker_path              predict.py on all 4,000 ITW clips; router statistics
      |
      v
OUTPUTS  results/tables.md, benchmark.csv, breakdown.csv, calibration.csv, curves.csv, fusion.csv,
         test_agreement.csv, docker_path.json
```

The concept tests (E1–E6) come from `scripts/run_concepts.py` (`mpcdf/concepts.sbatch`, plus the CPU passes
`--levels-only` and `--faithfulness-only`); see [APPROACH.md §4.9](APPROACH.md#49-can-the-explanations-be-trusted-tests-e1e6).

## 2. The final test matrix

| # | Test | Question | Where | Result (details below) |
|---|---|---|---|---|
| **0** | **official NSA test score (organizers, hidden labels)** | **how good is the submission?** | organizers | **minDCF 0.0317, EER 1.44%** (best interim leaderboard entry: 0.0584 / 2.5%) |
| A1 | benchmark: 22 detectors, 2 ensembles, concept-based, DSP and metadata systems x 3 sets x 2 views | which system generalizes? | `benchmark.csv` | no system is significantly better than the final ensemble on any set or view; a few tie on holdout aug |
| A2 | paired bootstrap vs the final system | are the differences real? | `benchmark.csv` | ensemble > best single model on ITW clean and aug (CIs exclude 0) |
| A3 | fine-tuning curves, every epoch | does training help out of domain? | `curves.csv` | plain fine-tuning decays on ITW; with D6-R it improves |
| A4 | seed replicate (XLS-R-1B + D6-R) | is a gain bigger than seed noise? | `benchmark.csv` | ITW clean 0.040 vs 0.042, aug 0.140 vs 0.128 |
| B1 | per fake generator | which generators are hard? | `breakdown.csv` | commercial cloners (ElevenLabs, PlayHT) |
| B2 | per real source, false alarms at the shipped threshold | do real clips get flagged? | `breakdown.csv` | 0% clean in domain, 2% ITW; ≤ 5% under perturbation |
| B3 | per channel condition (replayed chains) | which perturbation breaks it? | `breakdown.csv` | **reverberation**; everything else ≤ 0.07 in domain |
| B4 | per crop duration | how short is too short? | `breakdown.csv` | < 2 s clips carry most ITW error; the NSA clips are ≥ 3 s |
| C1 | cross-fitted calibration: actDCF at P > 0.2, Cllr, prior-weighted ECE | can a reader trust the probability? | `calibration.csv` | ~0.01 minDCF lost to calibration across domains |
| C2 | fusion gate (logistic, fitted on val) | do DSP / metadata / concepts add anything? | `fusion.csv` | no: DSP and metadata hurt or do nothing |
| D1 | NSA test set, label-free: rank agreement, decision agreement, flag rate, score vs duration | do the systems agree where we cannot check? | `test_agreement.csv` | neural systems agree (κ ≥ 0.86); metadata tracks duration |
| D2 | shipped code path (`predict.py`): stored-score agreement, order independence, all 4,000 ITW clips | is the TSV what we evaluated, and does a file's score depend on its neighbours? | `test_agreement.csv`, `docker_path.json` | submitted TSV vs stored scores: Spearman 0.999. After the whole-clip fix, a score moves < 3e-5 with the rest of the folder; ITW minDCF 0.033 |
| E1-E6 | concept formation and diffusion prototypes: scores, basic level, leakage, stability, noise-depth, faithfulness | can the explanations be trusted? | [APPROACH.md §4.9](APPROACH.md#49-can-the-explanations-be-trusted-tests-e1e6) | TTCG within ~0.01 of its detector; faithful on 92% of mixed explanations; top source stable (94–95%); held-out basic level at depth 7 |
| F | forensic audit, shortcut checks | is the test set what it seems? | [DATA.md](DATA.md) | no reuse; metadata constant on test |
| G1 | unit tests (89 across tracks) | does the code do what we say? | `tests/` | all pass (§5) |
| G2 | runtime and memory of `predict.py` on CPU | can the judges run it? | [results/runtime.md](../results/runtime.md) | 1,671 clips in 21 min on one 72-core node; ~4.2 s per clip at 8 threads; peak 14.5 GB RAM |

## 3. Headline results (minDCF, [95% CI])

| System | holdout clean | holdout aug | ITW clean | ITW aug | ITW EER |
|---|---|---|---|---|---|
| zero-shot XLS-R-2B (best pretrained) | 0.046 [0.031, 0.059] | 0.354 [0.311, 0.382] | 0.038 [0.024, 0.049] | 0.189 [0.162, 0.212] | 1.44% |
| fine-tuned XLS-R-2B | 0.001 | 0.069 | 0.060 [0.043, 0.071] | 0.153 [0.127, 0.171] | 2.32% |
| XLS-R-2B + D6-R copy-synthesis | 0.001 | 0.089 [0.070, 0.103] | 0.040 [0.026, 0.051] | 0.104 [0.081, 0.119] | 1.52% |
| interim ensemble (v1) | 0.000 | 0.068 [0.051, 0.082] | 0.028 [0.017, 0.039] | 0.095 [0.076, 0.112] | 1.12% |
| **final ensemble (v2)** | **0.000** | **0.073 [0.055, 0.085]** | **0.028 [0.017, 0.040]** | **0.082 [0.064, 0.097]** | **1.20%** |
| DSP detector (757 holdout clips) | 0.290 [0.233, 0.345] | – | 1.000 | – | 66.2% |
| metadata M0 (container fields) | 0.470 | – | – | – | – |

Paired against the final ensemble (Δ = system − final, same clips):
- **Zero-shot XLS-R-2B:** +0.010 [−0.004, 0.021] on ITW clean (not significant) and +0.106 [0.080, 0.133] on ITW
  aug. Fine-tuning buys robustness, not clean accuracy.
- **XLS-R-2B + D6-R (best single model):** +0.012 [0.001, 0.020] clean and +0.022 [0.004, 0.037] aug. The ensemble
  is significantly better than its best member.
- **Interim ensemble:** +0.013 [−0.003, 0.029] on ITW aug and −0.005 [−0.016, 0.007] on holdout aug. The final
  ensemble is better out of domain, but within noise.

On the shipped code path (`predict.py`: whole clips, fp32, CPU), all 4,000 labeled ITW clips give AUC 0.9994, EER
1.30%, minDCF 0.033. At the shipped threshold (P > 0.2) the actual DCF is 0.040.

**Official score.** The organizers scored the submitted TSV on the hidden NSA labels: **minDCF 0.0317, EER 1.44%**.
The interim leaderboard ranged from 0.0584 (EER 2.5%) to 0.913. The official score falls inside our ITW interval for
the final ensemble ([0.017, 0.040]): an uncontaminated out-of-domain set was the right proxy.

## 4. What the tests show

**Fine-tuning, copy-synthesis, ensembling (A1-A4).** ITW minDCF (clean) by epoch, epoch 0 = zero-shot:

| Backbone | plain fine-tuning, epochs 0 → 4 | with D6-R fakes, epochs 0 → 3 |
|---|---|---|
| XLS-R-2B | 0.038 → 0.052 → 0.060 → 0.076 → 0.096 | 0.038 → 0.050 → 0.040 → 0.040 |
| XLS-R-1B | 0.050 → 0.093 → 0.101 → 0.123 → 0.129 | 0.050 → 0.042 → 0.040 → 0.039 |
| MMS-1B | 0.065 → 0.111 → 0.217 → 0.186 → 0.207 | 0.065 → 0.044 → 0.040 → 0.042 |

Every fine-tuned model becomes near-perfect in domain, and 3–5x more robust to channel perturbation there. Without
copy-synthesis fakes, each epoch costs clean out-of-domain accuracy: the model learns what separates DiffSSD from
LJSpeech/LibriSpeech rather than what makes speech synthetic. Adding real clips re-vocoded by four vocoders (the
fake differs from its source only by the vocoder) turns the trend around for all three backbones. WiSE-FT keeps
zero-shot accuracy but gives up part of the robustness (XLS-R-2B, α = 0.3: ITW 0.036 / 0.124). The final ensemble
combines three backbones pretrained on different data (XLS-R-2B, XLS-R-1B, MMS-1B), all trained with the four-vocoder
fakes.

**Where it fails (B1-B4).**
- **Reverberation is the main failure mode.** On reverberant clips minDCF is 0.381 in domain (EER 15%) and 0.248 on
  ITW (EER 11%). Every other channel stays at ≤ 0.07 in domain and ≤ 0.12 on ITW; babble noise is the next worst.
  Reverb is only 1.5 of 13 augmentation weight units in training, so more (and more realistic) room simulation is
  the first thing to add.
- **Short clips.** ITW clips under 2 s: 0.119 (clean). 2–3 s: 0.017. 3 s and longer: 0.000. The NSA test clips are
  3.0–13.6 s.
- **Hardest generators** (holdout, aug): ElevenLabs 0.104, PlayHT 0.075, Grad-TTS 0.061. ElevenLabs is also the only
  generator the zero-shot model misses on clean audio (0.249). This matches the DiffSSD paper's hard tail.
- **False alarms on real speech** at the shipped threshold (P > 0.2): 0% on all three real corpora, 2.0% on ITW
  reals. Under perturbation they rise to 1.5–4.7% (in domain) and 5.0% (ITW).

**Calibration (C1).** Platt scaling fitted on val alone and applied to ITW loses little: actDCF 0.038 vs minDCF 0.028
clean, 0.085 vs 0.082 aug; prior-weighted ECE ≤ 0.016. Fitted on ITW alone and applied to the in-domain holdout:
0.093 vs 0.073 aug. A P(synthetic) from this system means roughly what it says, across domains.

**The gate: what did not make it (C2).** Logistic fusion fitted on val, tested on held-out clips:

| Fusion | holdout minDCF | ITW minDCF |
|---|---|---|
| final ensemble alone | 0.000 | 0.028 |
| + DSP detector | 0.060 (757 clips) | 0.594 |
| + metadata M0 / X0 | 0.000 | not available |

- **The DSP detector** (`hearsay_dsp/`) does not transfer. The components are LFCC-GMM, spectral, LPC, phase,
  prosody, background and ENF features with logistic fusion. It reaches AUC 0.91 in domain but AUC 0.27 on ITW,
  i.e. worse than chance. Its DSP suite (`runs/dsp/suite_v1/report.md`) shows why: trained with DiffSSD's reals
  alone it flags 99.8% of real LibriSpeech as fake.
- **Metadata models** separate DiffSSD's classes through container fields: sample rate, codec, encoder tags. Every
  NSA test file has the same container, and on the test set their scores track clip duration (Spearman 0.65–0.84)
  and nothing the detectors see (Spearman with the final ≈ 0). Both branches stay in the trace as forensic context
  and never enter the score.

**The NSA test set without labels (D1, D2).**
- **Decision agreement:** every D6-R and WiSE-FT model, and the fine-tuned XLS-R and MMS-1B models, flag 27.5–29.0%
  of the test clips at P > 0.2. At the final's flag rate, their decisions agree with the final (κ 0.96–0.99). The
  two weakest fine-tuned backbones flag 33% (W2V-Large) and 43% (MMS-300M).
- **Zero-shot models flag 42–71%.** Their calibration does not transfer to the test pipeline's darker, resampled
  audio; the fine-tuned models, trained on the canonical view with augmentation, do not show this.
- **The submitted TSV** (`predict.py`) matches the evaluated scores: Spearman 0.999, κ 1.00, 28.2% flagged.
- **A file's score no longer depends on the rest of the folder.** The version of `predict.py` that wrote the
  submitted TSV scored clips in batches cropped to their shortest member, in bf16, so a clip's score depended on
  its batch neighbours: up to 0.18 difference on 100 clips. `predict.py` now scores every clip whole, alone, in
  fp32, and a clip's score moves by less than 3e-5 whatever else is in the folder. On the test set the two
  versions agree on 1,668 of 1,671 decisions (Spearman 0.998). One of the three changes is the clip in
  [APPROACH.md §4.8](APPROACH.md#48-a-worked-example), which the concept explanations had flagged.
- **Concept explanations** flag 7 test clips (0.4%) whose score they contradict by more than 0.5. This is a
  review list, not a score ([APPROACH.md §4.8](APPROACH.md#48-a-worked-example)).

## 5. Unit tests

```
pytest tests --ignore=tests/dsp   # core: 30 tests (metric parity, TSV, splits, TTCG, concepts, views)
pytest tests/dsp                  # DSP track: 59 tests
```

| File | Checks |
|---|---|
| `test_metrics.py` | minDCF / EER equal the organizers' `calculate_modules.py` to 1e-12 (three cost settings, three score distributions); tie handling never optimistic; sklearn parity; reference points; P > 0.2 is the Bayes decision (actDCF ≈ minDCF for exact posteriors) |
| `test_submission.py` | template order, 10-decimal output, refusal of missing / NaN / out-of-range scores, the validator catches header / duplicate / order / CRLF errors, Platt prior shift, and the committed final TSV (1,671 rows, template order) |
| `test_splits.py` | no sentence, chapter or speaker group straddles train / val / holdout; held-out clone speakers fully held out; copy-synthesis rows move nothing; seeded and deterministic |
| `test_ttcg.py` | on a Gaussian mixture with known score: mode ascent + Tweedie recover the component means, the Hutchinson covariance equals the analytic posterior variance (1 − ᾱ)s²/(ᾱs² + 1 − ᾱ), and selection + composition explains a query that mixes two concepts dimension by dimension |
| `test_concepts.py` | the cobweb-private wrapper separates two sources (AUC > 0.99), reports the right source and channel for a concept, and walks node → root; the embedding space's inverse map is exact |
| `test_views.py` | canonical views are deterministic per (clip, view), band-limited above 7.6 kHz, and peak-normalized |

On Raven every runnable test passes:
- **DSP:** 58 tests (1 skipped by the suite itself).
- **Core, neural environment:** 29 tests.
- **Concepts environment:** the 6 concept and TTCG tests, including the cobweb-private test that the neural
  environment skips (`mpcdf/concepts.sbatch`).
