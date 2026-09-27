# HEARSAY: Audio Authentication (team SideQuests)

## Summary

We built a system that assigns every audio file a calibrated probability that it is synthetic. The system shows why it
reached that probability. Three fine-tuned speech detectors produce the score. A concept-formation layer from
cognitive science explains each decision in terms of learned categories and real training examples. On the NSA test
set, the organizers measured a **minDCF of 0.0317 and an EER of 1.44%**, the lowest interim score they posted. A Docker
image reproduces our predictions exactly, and every reported number traces to code, data hashes and a command.

## The problem

Modern voice cloning needs only a few seconds of reference audio. A fabricated call can move money, markets or
people. An analyst therefore needs two answers for every clip: how likely it is to be synthetic, and why. False
alarms carry their own cost: each one consumes analyst time and erodes trust in the tool. The challenge scores
systems with the ASVspoof 5 detection cost function at a 30% spoof prior and a false-alarm cost of 4.

## What the system does

1. **Triage.** The system reads the container, codec, sample rate, encoder tag and file times. It records them in a
   per-file trace. They never enter the score, because they are a shortcut in the training data (below).
2. **Canonical view.** The system decodes the audio to 16 kHz mono, trims edge silence, low-passes at 7 kHz, removes
   DC offset, normalizes the level and adds dither. Training and test audio pass through the same steps, so format,
   bandwidth and level cannot stand in for the label.
3. **Detection.** Three self-supervised speech detectors score the clip: XLS-R-2B, XLS-R-1B and MMS-1B, from the
   AntiDeepfake family. We fine-tuned all three on our data.
4. **Fusion and calibration.** The system z-normalizes each detector's logit and averages them. Platt scaling at the
   30% prior turns the result into a probability, and the Bayes threshold for the challenge costs is P > 0.2.
5. **Routing.** The system chooses further analyses per file. Uncertain or long clips get windowed re-scoring
   (2-second windows). Lossy or band-limited files get a compression and bandwidth cross-check. Every routing
   decision and its reason go into the trace.
6. **Explanation.** The system places the clip in a concept hierarchy built by COBWEB over the detector's
   representation. It reports the clip's concept, the sources that make up that concept, and the most typical training
   clips, which a person can listen to.

## How we built it

**Data.** We started from the organizers' data (DiffSSD fakes and resampled LJSpeech reals) and expanded it.
- **More real speech.** The organizers provided 242 real clips and about 70,000 fakes. We added LJSpeech
  (13,100 clips), the 10 LibriSpeech speakers that DiffSSD clones (1,152 clips), and 80 further LibriSpeech speakers
  (5,323 clips).
- **Our own fakes.** We generated 11,900 copy-synthesis fakes: we re-synthesized training-split real speech with four
  neural vocoders (two HiFi-GANs, DiffWave, Vocos). These pairs differ only in the synthesis step, so the detector must
  learn synthesis artifacts rather than speaker or content.
- **An out-of-domain benchmark.** We held out the In-the-Wild dataset (web recordings of real and faked public
  figures) as an evaluation set that none of our models trained on.
- **An audit before training.** We audited the data for leakage. Exact and content-level matching of every test clip
  against 89,817 reference recordings found no reused audio. The audit also showed that sample rate, codec,
  duration and silence separate the classes in DiffSSD, which motivated the canonical view.

**Training.**
- We split data by sentence, speaker and chapter, so no speaker or sentence crosses from training to evaluation.
- We fine-tuned gently (learning rate 2e-6, three epochs) with fixed family shares, balanced class weights, and random channel augmentation on both classes: codecs, telephony, noise, babble, reverberation, clipping
  and resampling. A noisy real clip must stay real.
- Copy-synthesis fakes reversed the out-of-domain decay that plain fine-tuning showed. A three-backbone ensemble beat
  its best member.

**Metric.** We reimplemented the organizers' scorer and checked it for parity against `calculate_metrics.py` from the
ASVspoof 5 package, with Pspoof 0.3 and Cfa 4. The stock package treats bona fide speech as the target class, so its
false-alarm cost applies to a synthetic clip that passes as real. We optimized the metric exactly as the organizers
compute it.

**Evaluation.**
- One script scores every system from stored per-clip scores.
- It reports 95% bootstrap intervals and paired comparisons on identical clips.
- It breaks results down by generator, real source, channel condition and clip duration.
- Two views of every clip: clean, and through a random channel chain.

## Results

| System | In-the-Wild, clean | In-the-Wild, perturbed | NSA test (official) |
|---|---|---|---|
| Best pretrained detector, zero-shot | 0.038 | 0.189 | – |
| **Final ensemble** | **0.028** [0.017, 0.040] | **0.082** [0.064, 0.097] | **0.0317** (EER 1.44%) |

minDCF at the organizers' costs (lower is better), with 95% bootstrap intervals.

**A fully held-out confirmation.** We used 4,000 In-the-Wild clips to choose checkpoints and fit calibration. The other
27,779 clips never influenced any choice. On them, the Docker inference path scores:
- **minDCF 0.0274 [0.0231, 0.0306], EER 1.05%, AUC 0.9996;**
- **actual DCF 0.0317 at our fixed threshold P > 0.2,** the same value as the official test score.

The ensemble beats each of its members (0.033, 0.034, 0.041). Error concentrates in clips under 2 seconds (minDCF
0.10); clips over 4 seconds are nearly error-free (0.003 or lower).

## Forensic techniques and what each one contributed

| Technique | What we built | Role in the final system |
|---|---|---|
| Deep-learning anti-spoofing | Three fine-tuned self-supervised detectors, fused and calibrated | Produces the score |
| Container and metadata forensics | Triage of container, codec, encoder tag, sample rate, file times; two metadata models | Trace only: metadata separates the classes perfectly in DiffSSD, and every test file shares one container |
| Spectral analysis | LFCC-GMM, spectral statistics, band-energy and bandwidth measurement | Gated out of the score (below) |
| Prosody and phonetics | Pitch contour, pauses, energy dynamics | Gated out; used to name concept dimensions |
| Acoustic environment | Mains-hum (ENF) and background-noise statistics | Gated out; used to name concept dimensions |
| Compression forensics | Codec cross-check in the router; codec augmentation of both classes in training | Routing and trace |
| Splice and discontinuity | Phase-jump features; windowed re-scoring flags local inconsistency | Routing and trace |
| Adaptive orchestration | A router that picks analyses by confidence, duration and codec | Chooses the analyses for each file |
| Concept formation (cognitive science) | COBWEB concept hierarchy and diffusion prototypes over the detector's representation | Explains every decision |

We did not build speaker-embedding drift detection.

## What did not work

- **The signal-processing detector did not generalize.** It reached AUC 0.91 in domain but 0.27 on In-the-Wild, worse
  than chance. Fused with the ensemble, it raised In-the-Wild minDCF from 0.028 to 0.594. A gate on held-out data
  removed it from the score.
- **Metadata is a shortcut, not evidence.** It separates the classes perfectly in DiffSSD, and it is constant on the
  test set.
- **Plain fine-tuning overfit to DiffSSD.** In-domain scores improved while In-the-Wild scores decayed. Copy-synthesis
  fakes fixed this.
- **Diffusion models as detectors.** We designed diffusion-based detection features and dropped them before running
  them: we found no precedent for speech. Diffusion helped elsewhere: DiffWave became a copy-synthesis vocoder, and
  diffusion prototypes became part of the explanation layer.
- **Concept formation as a scorer.** It stays below the detector, so it explains the score instead of producing it.
- **Weak spots.** Reverberant real speech and clips shorter than 2 seconds cause most errors.
- **RawBoost augmentation.** We tested it in a controlled experiment. We fixed the gates before training, selected
  on validation data only, and ran two seeds for the baseline. Adding RawBoost to our channel augmentation reduced
  perturbed In-the-Wild minDCF from 0.140 to 0.121 for the best variant. The paired interval included zero, so the
  gain was not measurable. Our backbones had already been post-trained with RawBoost, which limits the gain.

## Why the system is not a black box

The organizers asked for detectors that are not black boxes. We tested the explanation layer the way we tested the
detector:
- **Fidelity.** The concept layer's decisions agree with the detector's on the NSA test set with Cohen's κ 0.985
  [0.975, 0.994].
- **Structure without labels.** A COBWEB tree built with no labels separates real from synthetic speech at 98.9%
  purity with its first five concepts. Deeper concepts separate generators and speaker sets.
- **Case-based evidence.** Each test clip receives its concept and three typical training clips from public corpora.
- **Acoustic meaning.** The dimension the detector relies on most tracks frame-level loudness variability (ρ 0.42).
  Synthetic speech in our data is cleaner and steadier than real recordings.
- **Faithfulness.** Deleting the evidence an explanation names moves the detector's score as predicted in 92% of the 66
  explanations that mix real and synthetic concepts.
- **Failure localization.** Five concepts hold 27% of all errors. They name the weak spots directly: reverberant real
  speech and two voice-cloning systems.

## Cognitive science as a capability

COBWEB models how people form categories incrementally. In our system this gives capabilities that a classifier
alone lacks, each measured against a baseline:
- **Learning from a few examples.** From 1–3 clips of an unseen generator, the concept layer attributes new clips where
  a refitted kNN or logistic regression cannot. It learns without retraining and forgets nothing: attribution of
  known generators stays at 0.87.
- **Novelty detection.** It flags clips from an unseen generator with AUROC 0.87; a Mahalanobis baseline reaches 0.49.
- **Triage for analysts.** Ranking clips by disagreement between the concept layer and the detector finds the
  detector's errors at nine times the rate of random review.

## Reproducibility

- **Docker.** The image runs inference on a folder of audio with no network. We verified it on the NSA test set: its
  output matches our reference run to within 3.2 × 10⁻⁶ on all 1,671 files, with every decision identical.
- **Determinism.** The system scores each file whole and alone, so no score depends on the other files in the folder.
- **Provenance.** Every results folder records the code version, the SHA-256 of its inputs, package versions and the
  command. Frozen environments pin every package.
- **Tests.** Continuous integration runs the unit tests and a Docker build. The tests cover metric parity with the
  organizers' package, the prediction file format, data splits and view determinism.
- **One command per stage** documents training, evaluation and inference.

## Challenges

- **Shortcuts in the training data.** Sample rate, codec, duration and silence all predicted the label, so we removed
  them before the model could learn them.
- **Few real clips.** The organizers supplied 242 real clips against about 70,000 fakes. We sourced real speech from
  public corpora and matched speakers to the cloned voices.
- **Measuring generalization honestly.** Scores on DiffSSD saturate at zero error, so In-the-Wild became the benchmark
  that decided every design choice.

## What we learned

- Data design mattered more than model size: copy-synthesis fakes and shortcut removal gave the largest gains.
- A detector with near-perfect in-domain scores can fail out of domain, and only a held-out out-of-domain set reveals it.
- Interpretability claims need tests, just as accuracy claims do.

## What's next

- **Real room acoustics.** Measured room impulse responses in training, to address reverberation.
- **Newer generators.** Test sets from 2025–26 open-weight systems.
- **Speaker-embedding drift detection.**
- **Room acoustics.** A new evaluation view uses channels that no model trains on: simulated rooms, unseen codecs,
  packet loss and gain control. It shows that simulated rooms are our largest remaining weakness (minDCF about 0.5).

## Built with

Python, PyTorch, Hugging Face Transformers, AntiDeepfake (XLS-R, MMS), cobweb-private (Teachable AI Lab, Georgia Tech),
SpeechBrain, Vocos, FFmpeg, librosa, scikit-learn, pyroomacoustics, Docker, Apptainer, MPCDF Raven (A100 GPUs).
