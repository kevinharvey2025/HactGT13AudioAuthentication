# HEARSAY: Interpretable Synthetic-Speech Detection with Calibrated Ensembles and Concept Formation

Team SideQuests · NSA HEARSAY Audio Authentication Challenge · HackGT 13

## Abstract

We present a detector that assigns each audio file a calibrated probability of being synthetic and explains each
decision. Three fine-tuned speech models produce the score. A concept-formation layer, based on the COBWEB model of
human categorization, explains it with learned concepts and real training clips.

| | Result |
|---|---|
| **Official NSA test score** | minDCF 0.0317, EER 1.44% |
| **27,779 held-out In-the-Wild clips** | minDCF 0.0274 [0.0231, 0.0306], EER 1.05% |
| **Explanations vs detector** | agreement κ = 0.985 on the test set |
| **Docker image vs reference run** | identical to within 3.2 × 10⁻⁶ |

## 1. Inspiration

Voice cloning now needs only seconds of audio. An analyst who receives a suspicious recording needs two answers: how
likely it is to be synthetic, and why. Modern detectors answer the first question well and the second not at all.
They also learn dataset quirks instead of synthesis artifacts. We set out to build a detector that generalizes beyond
its training data and shows its evidence.

## 2. Primary technique

```
audio file ──> triage (codec, tags; trace only)
     │
     v
canonical view: 16 kHz mono · trim silence · 7 kHz low-pass · normalize · dither
     │
     ├──> XLS-R-2B ──┐
     ├──> XLS-R-1B ──┼──> z-normalize, average ──> Platt at prior 0.3 ──> flag if P > 0.2 ──> TSV + trace
     └──> MMS-1B  ───┘                                                     │
     │                                                                     └──> router: re-score uncertain clips
     v
concept layer: COBWEB tree + diffusion prototypes ──> concept, prototypes, training exemplars per clip
```

1. **Remove shortcuts.** In the provided data, file times, sample rate, codec, duration and silence all predict the
   label. Every clip passes through one canonical view, so none of these cues survive.
2. **Make our own fakes.** We re-synthesized 11,900 real training clips with four neural vocoders. Each fake matches
   its source in speaker and content, so the detector must learn the synthesis itself.
3. **Fine-tune gently, augment both classes.** Three AntiDeepfake speech models (XLS-R-2B, XLS-R-1B, MMS-1B) train at
   a small learning rate. Codecs, noise and reverberation apply to real and fake clips alike.
4. **Calibrate to the challenge's costs.** We average the three z-normalized scores and calibrate the result at the
   30% spoof prior. The cost-optimal decision is then P > 0.2.

We reused the AntiDeepfake checkpoints, the vocoders, the lab's COBWEB library and two published methods. We built
the data design, training, calibration, concept space, prototype implementation, evaluation and Docker path.

**Table 1.** minDCF (lower is better), with 95% intervals.

| System | In-the-Wild, clean | In-the-Wild, perturbed | NSA test |
|---|---|---|---|
| XLS-R-2B, pretrained, no fine-tuning | 0.038 | 0.189 | – |
| XLS-R-2B, fine-tuned without our fakes | 0.060 | 0.153 | – |
| XLS-R-2B, fine-tuned with our fakes | 0.040 | 0.104 | – |
| **System A: final ensemble (submitted)** | **0.028** [0.017, 0.040] | **0.082** [0.064, 0.097] | **0.0317** |

On 27,779 In-the-Wild clips that influenced no choice, System A scores 0.0274 [0.0231, 0.0306]. Its actual cost at
P > 0.2 is 0.0317, the same as the official score.

## 3. Ablations

Two ablations test whether System A leaves easy gains unclaimed. Each keeps everything but the ensemble members.

- **System A (submitted):** XLS-R-2B, XLS-R-1B, MMS-1B.
- **Ablation B, robustness:** the XLS-R-1B member retrained with RawBoost raw-waveform augmentation, selected on
  validation data under criteria fixed in advance.
- **Ablation C, breadth:** System A plus a second-seed XLS-R-1B and a WiSE-FT XLS-R-2B, five members in all.

**Table 2.** In-the-Wild minDCF, and change on the NSA test set (no labels exist there).

| System | ITW clean | ITW perturbed | Test decisions that differ from A |
|---|---|---|---|
| **A (submitted)** | **0.028** | 0.082 | – |
| B, robustness | 0.029 | 0.082 | 2 of 1,671 |
| C, breadth | 0.030 | **0.078** | 3 of 1,671 |

Neither ablation improves on System A beyond sampling noise:
- **B.** The RawBoost member alone improved from 0.140 to 0.121 on perturbed audio, but its 95% interval included zero.
  The pretrained models had already seen RawBoost.
- **C.** The wider ensemble trades a slight clean-audio loss for a slight gain under perturbation.

## 4. Interpretation

The concept layer reads the detector's representation and explains each clip in three steps:
1. the clip's basic-level concept in the COBWEB tree;
2. up to three diffusion prototypes, with their shares;
3. the concepts that name those prototypes, with training clips to listen to.

**HGT7824018.wav, synthetic (P = 0.996).**
- **Concept:** 19 training clips, all synthetic: UnitSpeech and XTTS v2 voice clones heard through a codec.
- **Prototypes:** 39% YourTTS/XTTS v2 clones · 33% DiffGAN-TTS/WaveGrad 2 · 28% UnitSpeech/XTTS v2.
- **Listen to:** `unit_speech/speaker_1487/sentence_475`.
- **The bug it caught:** our first scorer gave this clip 0.009. The contradiction with its explanation exposed a
  scoring bug, and the fix changed 3 of 1,671 test decisions.

**HGT1013455.wav, real (P = 0.0002).**
- **Concept:** 598 real LibriSpeech clips, a third of them from the very speakers the cloners imitate.
- **Reading:** a familiar voice alone does not make a clip look synthetic to the detector.

**Table 3.** Tests of the explanation layer.

| Test | Result |
|---|---|
| Agreement with the detector (NSA test) | κ = 0.985 |
| Faithfulness: deleting named evidence moves the score as predicted | 92% of cases |
| Tree built without labels: real vs synthetic purity at depth 1 | 98.9% |
| Novelty detection for an unseen generator | AUROC 0.87 (baseline 0.49) |
| Error triage: errors caught in the top 10% of clips | 95% |

## 5. Conclusion

- **Data design was the decisive technique.** Removing shortcuts and adding our own fakes mattered more than model
  size.
- **The ensemble and calibration hold out of sample:** 0.0274 on 27,779 unseen clips, matching the official 0.0317
  at the fixed threshold.
- **The ablations found no easy gains left.**
- **The explanations can be tested, and they proved useful:** they agree with the detector, cite real training
  audio, and caught a real bug.

## 6. Next steps

- **Room acoustics.** A stress test with simulated rooms, which no model trains on, is our largest weakness
  (minDCF 0.51–0.59). Next: train with measured room impulse responses.
- **Short clips.** Clips under 2 seconds cause most remaining errors (minDCF 0.10).
- **Newer generators.** Evaluate on open-weight systems released in 2025–26.

## Built with

Python, PyTorch, Hugging Face Transformers, AntiDeepfake (XLS-R, MMS), cobweb-private (Teachable AI Lab, Georgia Tech),
HiFi-GAN, DiffWave, Vocos, FFmpeg, scikit-learn, Docker, MPCDF Raven (NVIDIA A100).

Full method, references and every result: `README.md` and `docs/` in the repository.
