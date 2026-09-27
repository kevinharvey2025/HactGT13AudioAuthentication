# HEARSAY: Interpretable Synthetic-Speech Detection with Calibrated Ensembles and Concept Formation

Team SideQuests · NSA HEARSAY Audio Authentication Challenge · HackGT 13

## Abstract

We present a system that assigns each audio file a calibrated probability of being synthetic and explains each
decision. The score is the calibrated mean of three self-supervised speech detectors, fine-tuned on training data
from which we removed format shortcuts and to which we added our own synthetic speech. A concept-formation layer,
based on the COBWEB model of human categorization, reads the detectors' representation. It explains each decision
through learned concepts, diffusion prototypes and real training exemplars.

Our submitted system, System A, achieves a minimum detection cost (minDCF) of 0.0317 and an equal error rate (EER)
of 1.44% on the organizers' held-out test set. On 27,779 In-the-Wild recordings that influenced no design choice, it
achieves minDCF 0.0274 [0.0231, 0.0306] and EER 1.05%. The concept layer reproduces its decisions on the test set with
Cohen's κ = 0.985. A containerized inference path reproduces every prediction to within 3.2 × 10⁻⁶.

Two ablations test whether System A leaves easy gains unclaimed. The first adds raw-waveform augmentation to one
member; the second widens the ensemble to five models. Neither changes more than 3 of 1,671 test decisions, and
neither improves on System A beyond sampling noise.

## 1. Introduction

Zero-shot voice cloning now requires only seconds of reference audio. A fabricated recording can therefore
impersonate a known speaker at negligible cost. An analyst who receives such a recording needs two answers: how likely
the recording is to be synthetic, and on what evidence. Detection alone is insufficient in this setting. Each false
alarm consumes analyst time, and an unexplained score gives the analyst nothing to verify.

Recent detectors built on self-supervised speech models generalize better than earlier hand-crafted systems
[Tak et al. 2022; Ge et al. 2025]. They remain opaque, and they readily learn dataset artifacts instead of synthesis
artifacts [Müller et al. 2021].

### 1.1 Contributions

We build on open-source detectors and published methods. Our contribution lies in how we combine, adapt and test
them. Each claim below rests on a measurement.

1. **A shortcut-free training design.**
   - We audited the provided data and found that file times alone separate the classes perfectly. We then built one
     canonical input view that removes format, bandwidth, level and duration cues for both classes.
   - We generated 11,900 copy-synthesis fakes that differ from their real sources only in synthesis.
   - This design reversed the out-of-domain decay of plain fine-tuning: XLS-R-1B In-the-Wild minDCF moved from
     0.050 → 0.129 without it and 0.050 → 0.039 with it.
2. **A decision-theoretic score.**
   - We traced the metric's false-alarm convention to the organizers' evaluation code and matched that code to
     10⁻¹².
   - We derived the Bayes threshold from the challenge costs and calibrated the ensemble to it.
   - On held-out data, the actual cost at that threshold exceeds the minimum by only 0.004.
3. **Concept formation as an explanation layer for a speech detector.**
   - We pair the COBWEB model of human categorization with diffusion prototypes (TTCG), which we implemented from
     the published method, over the detector's own representation.
   - Each prototype takes the name of a learned concept, so every decision traces to concepts and to real training
     clips (Section 8.1).
4. **Explanations tested as rigorously as the detector.**
   - We measured fidelity (κ = 0.985), faithfulness by deletion (92%), stability across insertion orders, structure
     without labels (98.9% purity) and failure localization.
   - We also measured three cognitive capabilities against conventional baselines (Section 9).
5. **Explanations that found a real defect.** A disagreement between an explanation and its score exposed a
   batch-order bug in our first scorer. Fixing it changed 3 of 1,671 test decisions.
6. **Evidence beyond the leaderboard.**
   - A fully held-out set of 27,779 In-the-Wild recordings confirms the official result.
   - Two predeclared ablations (Section 7) test whether easy gains remain.
   - A new stress view, built from channels no model trains on, exposes simulated rooms as the main remaining
     weakness.
7. **Exact reproducibility.**
   - The Docker image reproduces every prediction to within 3.2 × 10⁻⁶.
   - Re-training from the repository reproduces stored validation results to four decimals.

**Table 0.** What we reused and what we built.

| Reused (open source or published) | Built by us |
|---|---|
| AntiDeepfake checkpoints (XLS-R-2B, XLS-R-1B, MMS-1B) [Ge et al. 2025] | Data audit, canonical view, family-balanced sampling, channel augmentation, and fine-tuning of all three models |
| Neural vocoders (HiFi-GAN, DiffWave, Vocos) | The copy-synthesis data set (11,900 clips) and its split rules |
| cobweb-private, the COBWEB library of the Teachable AI Lab | The concept space over the detector's representation, basic-level selection on held-out data, and the concept atlas |
| The TTCG and DMCF methods (papers) | A DDPM and TTCG implementation for this space, and prototype naming by concepts |
| The ASVspoof 5 metric definition | A parity-tested scorer, calibration, Bayes decisions, the evaluation protocol, and the stress views |
| – | The router, per-file traces, Docker verification, provenance, and all experiments and ablations |

## 2. Task and evaluation metric

The system receives an audio file of at least 2 seconds of English speech, in any container. It outputs P(synthetic)
in [0, 1]. The organizers rank systems with the ASVspoof 5 detection cost function at a spoof prior of 0.3,
C_miss = 1 and C_fa = 4 [Wang et al. 2024].

The ASVspoof 5 package treats bona fide speech as the target class. Its false-alarm cost therefore applies to a
synthetic clip that the system accepts as real. The normalized cost at threshold τ is:

    DCF(τ) = FPR(τ) + (4 · 0.3) / (1 · 0.7) · FNR(τ) = FPR(τ) + 1.714 · FNR(τ)

Here FPR is the share of real clips flagged, and FNR the share of synthetic clips passed. minDCF is the minimum over τ.
Our implementation matches the organizers' module to within 10⁻¹².

After Platt calibration at the evaluation prior [Platt 1999], the Bayes-optimal decision flags a clip when P > 0.2.

## 3. Data

**Provided data.** The organizers supplied DiffSSD [Bhagtani et al. 2024], about 70,000 synthetic clips from ten
systems, and 242 real clips resampled from LJSpeech.

**Added real speech.** We added 19,575 real recordings:
- LJSpeech (13,100 clips);
- the ten LibriSpeech speakers that DiffSSD clones (1,152 clips);
- 80 further LibriSpeech speakers (5,323 clips).

**Copy-synthesis fakes.** We generated 11,900 synthetic clips by re-synthesizing training-split real recordings with
four neural vocoders: two HiFi-GAN models, DiffWave and Vocos. Each such fake shares its speaker and content with a
real recording. The detector can therefore separate the pair only by synthesis artifacts [Wang & Yamagishi 2023].

**Out-of-domain evaluation.** The In-the-Wild dataset [Müller et al. 2022] contains 31,779 web recordings of real and
synthetic speech by public figures. We never trained on it. We used 4,000 of its clips for checkpoint selection and
calibration, and we held out the remaining 27,779 entirely.

**Audit.** Before training, we compared every test clip against 89,817 reference recordings by byte hash, decoded-audio
hash and acoustic landmark matching. No test audio appeared in any training corpus.

The audit also revealed strong shortcuts in DiffSSD:
- File-system timestamps alone separate the classes perfectly (AUC 1.0).
- Container fields reach AUC 0.90.
- Sample rate, codec, level, leading silence and duration all differ by class.

## 4. Method

### 4.1 Input normalization

A canonical view removes the format shortcuts. Every clip, real or synthetic, training or test, goes through the same
steps:
1. decoding to 16 kHz mono;
2. trimming of edge silence;
3. a crop to a test-like duration (training only);
4. a 7 kHz low-pass filter;
5. DC removal and peak normalization;
6. one-LSB dither.

During training, 60% of clips also pass through one or two random channel operations: lossy codecs, telephony,
band-limiting, resampling, spectral tilt, noise (including babble), mains hum, reverberation or clipping. These
operations apply to both classes. A degraded real recording therefore remains a real example.

### 4.2 Detectors

We fine-tune three AntiDeepfake models [Ge et al. 2025]: XLS-R-2B, XLS-R-1B and MMS-1B. These are self-supervised
speech encoders post-trained for spoof detection.

**Training settings.**
- AdamW at learning rate 2 × 10⁻⁶ (10⁻⁴ for the classification head), for three epochs.
- Batches drawn with fixed shares per data family, so no speaker population can stand in for the label.
- Splits by sentence, speaker and chapter, so no speaker or sentence crosses from training to evaluation.

### 4.3 Fusion and calibration

Each detector produces a synthetic logit ℓ_k. We z-normalize each logit with statistics from validation data, average
the three, and apply Platt scaling:

    s(x) = (1/3) Σ_k (ℓ_k(x) − μ_k) / σ_k,    P(synthetic | x) = σ(a · s(x) + b)

The fitted values are a = 6.84 and b = −2.60, including the prior term for π = 0.3.

### 4.4 Adaptive routing

A router selects further analyses for each file:
- Clips with uncertain scores (0.05 < P < 0.8), or longer than 6 seconds, receive windowed re-scoring with 2-second
  windows.
- Lossy or band-limited files receive a compression and bandwidth cross-check.
- All other files stop after the first pass.

The router records each decision and its reason in a per-file trace. On the test set, 96.7% of files stopped after the
first pass.

### 4.5 Concept-based explanation

The explanation layer operates on the XLS-R-2B detector's final representation, whitened to 32 dimensions. The
detector's decision is almost entirely linear in this space (R² = 0.99997).

**The concept hierarchy.** COBWEB [Fisher 1987] builds a hierarchy of concepts incrementally, by maximizing category
utility [Gluck & Corter 1985]. We use the Teachable AI Lab's continuous implementation. Each clip's explanation
concept is its basic level [Rosch et al. 1976]: the node on its path with the highest pointwise mutual information on
held-out data.

**Diffusion prototypes.** A denoising diffusion model over the same space supplies prototypes. Test-time concept
discovery and composition (TTCG) [Wang et al. 2026] finds the modes of the noised data distribution near a clip. It
selects up to three of them by submodular facility location and composes them per dimension. Each prototype is then
named by the COBWEB concept that contains it.

An explanation therefore states:
- which learned concepts the clip belongs to;
- which prototypes account for which share of its representation;
- which real training clips typify those concepts.

## 5. Experimental protocol

We evaluate every system from stored per-clip scores under one protocol.

| Set | Role |
|---|---|
| Validation | Selects checkpoints and fits calibration |
| In-domain holdout | Reports in-domain performance; never informs a choice |
| In-the-Wild | Measures out-of-domain generalization |

Each clip is scored in two views: clean, and through a fixed random channel chain. We report 95% intervals from a
stratified bootstrap, which resamples real and synthetic clips separately. We compare systems with a paired bootstrap
on identical clips. A gate fitted on validation data decides whether any auxiliary branch may enter the score.

## 6. Results for System A

System A is the three-detector ensemble of Section 4, scored by the containerized inference path. It is the system
we submitted, documented and verified.

**Table 1.** minDCF at the organizers' costs (lower is better), with 95% bootstrap intervals.

| System | In-the-Wild, clean | In-the-Wild, perturbed | NSA test (official) |
|---|---|---|---|
| Best pretrained detector, zero-shot | 0.038 | 0.189 | – |
| Fine-tuned XLS-R-2B, no copy-synthesis | 0.060 | 0.153 | – |
| Fine-tuned XLS-R-2B, with copy-synthesis | 0.040 | 0.104 | – |
| **System A (final ensemble)** | **0.028** [0.017, 0.040] | **0.082** [0.064, 0.097] | **0.0317** (EER 1.44%) |

**Copy-synthesis is the decisive intervention.** Plain fine-tuning improved in-domain scores but degraded
out-of-domain performance with every epoch (XLS-R-1B: 0.050 to 0.129). With copy-synthesis fakes, all three models
held or improved (XLS-R-1B: 0.050 to 0.039).

**System A outperforms its best member.** Paired on identical clips, the margin is 0.012 [0.001, 0.020] on clean
audio and 0.022 [0.004, 0.037] under perturbation.

**Held-out confirmation.** On the 27,779 In-the-Wild clips that informed no choice, System A achieves:
- minDCF 0.0274 [0.0231, 0.0306], EER 1.05%, AUC 0.9996;
- actual DCF 0.0317 at the fixed threshold P > 0.2, equal to the official test score.

Each member alone scores 0.033, 0.034 and 0.041 on this set, so the ensemble's advantage holds out of sample. Error
concentrates in clips shorter than 2 seconds (minDCF 0.10). Clips longer than 4 seconds show minDCF of 0.003 or less.

**Calibration.** At the fixed threshold, System A flags 28.2% of the test set. Its actual cost on held-out In-the-Wild
exceeds the minimum by 0.004. The Bayes threshold derived in Section 2 therefore transfers across domains with little
loss.

## 7. Ablations

The three systems differ only in their detector ensemble:
- **System A (submitted):** XLS-R-2B, XLS-R-1B and MMS-1B, each fine-tuned with copy-synthesis fakes and channel
  augmentation. The Docker image reproduces it exactly.
- **Ablation B (robustness):** System A with its XLS-R-1B member replaced by one also trained with RawBoost. It tests
  whether raw-waveform augmentation adds robustness.
- **Ablation C (breadth):** System A plus a second-seed XLS-R-1B and a WiSE-FT XLS-R-2B, five members in all. It tests
  whether more members reduce variance.

Both ablations keep System A's pipeline, training data, fusion recipe and inference path. Each changes only the
ensemble members, which isolates one design question.

- **Ablation B, robustness.** Does raw-waveform augmentation help? We replace the XLS-R-1B member with one trained with
  RawBoost [Tak et al. 2022] on top of our channel augmentation.
  - We fixed the success criteria before training, trained the baseline with two seeds, and chose among three RawBoost
    variants on validation data only.
  - The selected variant (convolutive, impulsive and additive noise in series) cost no training time.
- **Ablation C, breadth.** Do more members help? We add a second-seed XLS-R-1B and a weight-interpolated XLS-R-2B
  (WiSE-FT, α = 0.3 [Wortsman et al. 2022]) to System A's three members.

**Table 2.** minDCF on In-the-Wild (in-sample for calibration, as for System A) and agreement with System A on the
NSA test set. No test labels exist, so the last two columns measure change, not accuracy.

| System | Members | ITW clean | ITW perturbed | Test decisions that differ from A | Spearman with A |
|---|---|---|---|---|---|
| **A (submitted)** | XLS-R-2B, XLS-R-1B, MMS-1B | **0.028** | 0.082 | – | – |
| B, robustness | A with a RawBoost XLS-R-1B | 0.029 | 0.082 | 2 of 1,671 | 0.990 |
| C, breadth | A + second-seed XLS-R-1B + WiSE-FT XLS-R-2B | 0.030 | **0.078** | 3 of 1,671 | 0.987 |

**Ablation B: no measurable gain.**
- **The member alone.** The RawBoost member reduced perturbed In-the-Wild minDCF from 0.140 to 0.121, a 13%
  relative gain, larger than the seed spread (0.012). Its paired 95% interval, [−0.033, 0.004], includes zero. The
  member therefore failed our predeclared criterion.
- **Unseen channels.** On an evaluation view built from channels that no model trains on, the member showed no gain
  (0.296 to 0.290, below the seed spread of 0.036).
- **Inside the ensemble.** The effect shrinks to 2 changed decisions.
- **Why.** The pretrained backbones had already received RawBoost during post-training, which limits any further gain.

**Ablation C: a trade-off, not an improvement.** The wider ensemble improves perturbed In-the-Wild minDCF slightly
(0.078 against 0.082) and worsens clean minDCF slightly (0.030 against 0.028). Both differences lie within sampling
noise. It also costs about 1.7 times System A's inference time.

**Conclusion.** Neither ablation beats System A. Its three members already capture the benefit of ensembling, and its
channel augmentation already covers what RawBoost adds. Both ablations share System A's main weakness: on simulated
room acoustics, every variant scores minDCF 0.51–0.59.

## 8. Interpretability analysis

### 8.1 Example explanations

The explanation layer produces the following for two NSA test clips (`results/concepts/` holds one per test clip):

**HGT7824018.wav, synthetic (P = 0.996).**
- **Concept:** a basic-level concept of 19 training clips, all synthetic: UnitSpeech 42% and XTTS v2 37%, heard
  through a codec.
- **Prototypes:** three, with shares:
  - 39%: YourTTS and XTTS v2 clones under a codec (98% synthetic);
  - 33%: DiffGAN-TTS and WaveGrad 2 under a codec (100% synthetic);
  - 28%: UnitSpeech and XTTS v2 under a codec (100% synthetic).
- **Exemplars:** `unit_speech/speaker_1487/sentence_475` and `xtts_v2/speaker_3654/sentence_59`.
- **The bug it exposed:** our first scorer assigned this clip P = 0.009. The explanation contradicted that score,
  and the contradiction exposed the batch-order defect described in Section 1.1 (contribution 5).

**HGT1013455.wav, real (P = 0.0002).**
- **Concept:** a concept of 598 training clips, 0.2% synthetic. It is made of LibriSpeech speakers; 34% of its clips
  are real recordings of the very speakers that DiffSSD clones.
- **Exemplars:** `real_libri/7995/7995-276908-0031` and `real_libri/100/100-121669-0008`.
- **Reading:** the clip groups with real speech even beside real recordings of the cloned voices. A familiar voice
  alone does not make a clip look synthetic to the detector.

### 8.2 Quantitative evaluation

We evaluate the explanation layer against the detector it explains.

**Fidelity.** At a matched flag rate, the concept layer's decisions agree with the detector's on the NSA test set
with κ = 0.985 [0.975, 0.994]. On perturbed out-of-domain audio, agreement falls to κ = 0.87–0.90. This is where the
detector itself is least certain.

**Structure without labels.** A COBWEB hierarchy built without any labels separates real from synthetic speech at 98.9%
purity with its first five concepts. Deeper levels separate generators and speaker populations. The representation
comes from a supervised detector. This result therefore shows that the representation is organized by authenticity,
not that authenticity emerges from raw audio.

**Faithfulness.** We tested explanations that mix real and synthetic concepts by deletion [Samek et al. 2017]. Removing
the dimensions attributed to synthetic concepts lowers the detector's logit more than removing the real-concept
dimensions, by 0.79 SD [0.67, 0.89], in 92% of the 66 cases. Random dimensions produce effects of 0.18–0.20 SD.

**Stability.** Over three insertion orders, the tree changes (partition ARI 0.03–0.04), but explanations persist: the
leading source agrees for 94–95% of clips.

**Acoustic correlates.** The dimension with the greatest influence on the detector correlates with frame-level
loudness variability (Spearman ρ = 0.42 [0.39, 0.44]). High-band flatness and background-noise statistics follow. In
our data, synthetic speech is cleaner and steadier than real recordings. This cue may partly reflect recording
conditions rather than synthesis, and we treat it as a hypothesis under test.

**Failure localization.** Five concepts contain 27% of all errors on held-out data. They identify the detector's weak
conditions directly: reverberant real speech and two voice-cloning systems (XTTS v2 and YourTTS).

## 9. Cognitive capabilities of the concept layer

COBWEB models incremental human category learning. We measured three resulting capabilities against conventional
baselines.

- **Few-shot attribution without forgetting.** We withheld one generator and inserted k of its clips without any
  gradient update.
  - With a single example, the concept layer attributes 7% of the generator's held-out clips. Refitted kNN attributes
    none, and logistic regression 1%. With three examples, the concept layer and logistic regression tie (15%).
  - Attribution of known generators remains at 0.87 throughout.
  - Logistic regression overtakes the concept layer at k ≥ 10.
- **Novelty detection.** Negative basic-level mutual information detects clips from an unseen generator with
  AUROC 0.87 ± 0.06, against 0.49 for a Mahalanobis baseline.
- **Error triage.** Ranking clips by disagreement between the concept layer and the detector recovers errors at nine
  times the random rate. Combined with the detector's uncertainty, it recovers 95% of errors in the top 10% of clips.

## 10. Negative results

- **Signal-processing detector.** It combines LFCC-GMM, spectral, LPC, phase, prosody, background and mains-frequency
  features. It reached AUC 0.91 in domain but 0.27 on In-the-Wild. Fusion raised In-the-Wild minDCF from 0.028 to
  0.594, so the gate excluded it.
- **Metadata.** Metadata separates the DiffSSD classes perfectly, and it is constant on the test set. It enters only the
  trace.
- **Diffusion features for detection.** We designed and discarded them; we found no precedent for them in speech.
  Diffusion contributes elsewhere instead: DiffWave as a copy-synthesis vocoder, and diffusion prototypes in the
  explanation layer.
- **The concept layer as a scorer.** It remains below the detector, and adding it to the ensemble slightly degraded
  clean out-of-domain performance.
- **RawBoost augmentation and a wider ensemble.** Neither improved on System A (Section 7).

## 11. Reproducibility

- **Containerized inference.** The Docker image scores a directory of audio without network access. Its output matches
  the reference run on all 1,671 test files to within 3.2 × 10⁻⁶, with identical decisions.
- **Order independence.** Each file is scored whole and independently, so no prediction depends on the other files.
- **Provenance.** Each results directory records:
  - the code version;
  - the SHA-256 hash of every input;
  - the package versions;
  - the command.
- **Controlled environments and tests.** Frozen environments pin every package. Continuous integration runs the unit
  tests and a container build. The tests cover metric parity with the organizers' package, the prediction format,
  data splits and input determinism.
- **Re-training reproduces exactly.** Re-training the XLS-R-1B detector from the repository reproduced the stored
  validation results to four decimal places at every epoch.

## 12. Limitations and future work

- **Reverberation and short clips.** Reverberant real speech and clips shorter than 2 seconds remain the main error
  sources.
- **Simulated rooms.** A stress view that no model trains on shows that simulated room acoustics degrade every system
  (minDCF 0.51–0.59). Training with measured room impulse responses is the most direct remedy.
- **Newer generators.** Our evaluation does not yet cover open-weight generators released in 2025–26.
- **Speaker drift.** We did not implement speaker-embedding drift detection.
- **Label-informed tree.** The shipped explanations use a tree fitted with labels. The label-free analysis in
  Section 8 bounds, but does not remove, this dependence.

## References

- Bhagtani et al. DiffSSD: a diffusion-based dataset for speech forensics. 2024.
- Fisher. Knowledge acquisition via incremental conceptual clustering. *Machine Learning*, 1987.
- Ge, Wang, Liu & Yamagishi. Post-training for deepfake speech detection (AntiDeepfake). 2025.
- Gluck & Corter. Information, uncertainty, and the utility of categories. *Proc. Cognitive Science Society*, 1985.
- Müller et al. Speech is silver, silence is golden: what do ASVspoof-trained models really learn? 2021.
- Müller et al. Does audio deepfake detection generalize? *Interspeech*, 2022.
- Platt. Probabilistic outputs for support vector machines. 1999.
- Rosch et al. Basic objects in natural categories. *Cognitive Psychology*, 1976.
- Samek et al. Evaluating the visualization of what a deep neural network has learned. *IEEE TNNLS*, 2017.
- Tak et al. RawBoost: a raw data boosting and augmentation method. *ICASSP*, 2022.
- Tak et al. Spoofing and deepfake detection using wav2vec 2.0 and data augmentation. *Odyssey*, 2022.
- Wang & Yamagishi. Spoofed training data can be efficiently created using neural vocoders. *ICASSP*, 2023.
- Wang et al. ASVspoof 5. 2024.
- Wang, Gupta, Zhu & MacLellan. Test-time compositional generalization in diffusion models via concept discovery. 2026.
- Wortsman et al. Robust fine-tuning of zero-shot models (WiSE-FT). *CVPR*, 2022.

The full bibliography is in `docs/APPROACH.md`.

## Built with

Python, PyTorch, Hugging Face Transformers, AntiDeepfake (XLS-R, MMS), cobweb-private (Teachable AI Lab, Georgia Tech),
SpeechBrain, Vocos, FFmpeg, librosa, scikit-learn, pyroomacoustics, Docker, Apptainer, MPCDF Raven (NVIDIA A100).
