# Not a black box: what the concept layer shows, and how we tested it

The detector is a fine-tuned self-supervised speech model: accurate (official NSA minDCF 0.0317), but opaque on its
own. The concept layer makes it readable. It sorts the detector's own representation into a hierarchy of concepts
with COBWEB, the incremental concept-formation model from cognitive science (Fisher 1987; cobweb-private, Teachable AI
Lab), and names diffusion prototypes (TTCG) by those concepts. [APPROACH.md](APPROACH.md) has the method and its
background. This page collects the evidence, and every claim below comes with a test, a baseline and its limits.

Outputs are in [results/concepts/experiments/](../results/concepts/experiments/). They were made by
`scripts/concept_experiments.py --seeds 3` (the `provenance.json` there records the code version, inputs and
packages) on the XLS-R-2B member's last-layer embedding, whitened to 32 dimensions, with 12,000 training clips.

## At a glance

| Question | Answer | Test |
|---|---|---|
| Does the readable layer reproduce the black box? | Yes: κ 0.985 [0.975, 0.994] with the detector's decisions on the NSA test set (99.4% agreement); 0.90–1.0 on labelled sets | I3 |
| What did the detector learn, globally? | A browsable map: real LJSpeech, LJ-voice TTS, cloned voices, commercial TTS, channel sub-concepts | I1 |
| Is the structure imposed by labels? | No. A tree built with **no labels at all** separates real from synthetic at 98.9% purity with its first 5 concepts | I8 |
| Why this clip? | Its concept, that concept's sources, and training clips a person can listen to | I2 |
| What do the dimensions mean acoustically? | The dimension the detector leans on most tracks background-level variability (ρ 0.42), a possible cleanliness cue | I5 |
| Where does it fail? | Errors cluster: 5 concepts hold 27% of them (XTTS/YourTTS clones; reverberant real speech) | I7, C3 |
| Can it learn a new generator without retraining? | Yes, from a few examples and with no forgetting; a refitted classifier catches up after about 10 | C1 |
| Does it know when something is new? | Unseen generator vs seen: AUROC 0.87 (Mahalanobis 0.49) | C2 |
| Can it point an analyst at mistakes? | Ranking by concept–detector disagreement: 9× random precision; best combined with detector uncertainty | C4 |

## Fidelity: the readable layer agrees with the detector (I3)

For each clip the concept layer makes its own decision: the synthetic share of the clip's basic-level concept
(cobweb), or of its nearest named prototypes (TTCG). It flags the same share of clips as the detector (matched flag
rate), and κ measures chance-corrected agreement, with 95% bootstrap intervals.

| Set | View | TTCG κ | cobweb κ |
|---|---|---|---|
| val / holdout | clean | 1.00 / 1.00 | 1.00 / 1.00 |
| val / holdout | perturbed | 0.97 / 0.95 | 0.97 / 0.95 |
| In-the-Wild | clean | 0.96 [0.93, 0.98] | 0.93 [0.89, 0.96] |
| In-the-Wild | perturbed | 0.90 [0.86, 0.93] | 0.87 [0.82, 0.90] |
| **NSA test** | as delivered | **0.985 [0.975, 0.994]** | **0.985 [0.975, 0.994]** |

The explanation describes the model that actually made the decision, not a different model. Agreement falls where
the detector itself is least sure (out-of-domain, perturbed), which is where an analyst should look anyway (C4).

## The concept atlas: what the detector learned (I1)

The top of the tree, from [atlas.md](../results/concepts/experiments/atlas.md):

- **0.1, real LJSpeech (4,375 clips, 0.6% synthetic).** Sub-concepts include 0.1.2 (356 clips), which is mostly
  reverberant and noisy real speech.
- **0.2, synthetic speech in the LJ voice (3,261 clips, 100% synthetic).** It splits by generator family:
  0.2.1 is DiffGAN-TTS, Grad-TTS and WaveGrad 2 in equal thirds; 0.2.2 is ElevenLabs, PlayHT and UnitSpeech.
- **Further branches:** real LibriSpeech speakers, their cloned voices, and the copy-synthesis fakes.

Each concept lists its size, synthetic share, sources, channels and most typical training clips. These clips come
from public corpora, so a reader can listen to what a concept means.

## No labels needed: the unsupervised tree (I8)

The training tree is label-informed: cobweb-private's category utility has a label term. So we rebuilt the tree
from the embeddings alone (`num_labels=0`) and named its concepts afterwards by their members
([atlas_unsupervised.md](../results/concepts/experiments/atlas_unsupervised.md)).

| Depth | Concepts | Real/synthetic purity | Source purity |
|---|---|---|---|
| 0 | 1 | 0.546 (the base rate) | 0.36 |
| 1 | 5 | **0.989** | 0.58 |
| 2 | 16 | 0.989 | 0.70 |
| 4 | 137 | 0.990 | 0.81 |
| 6 | 1,227 | 0.994 | 0.87 |
| 8 | 4,015 | 0.999 | 0.94 |

The real/synthetic distinction emerges as the first split, unsupervised. Deeper levels refine by source (generator,
speaker set). This is the categorization behaviour COBWEB was designed to model.

**Caveat:** the embedding comes from a detector trained with labels, so this shows that the representation is
organized by authenticity, not that authenticity was discovered from raw audio. Leaf-level AMI is low (0.03–0.06)
only because leaves are near-singletons; purity by depth is the meaningful measure.

## Case-based explanations: "this clip sounds like these" (I2)

Every NSA test clip gets its basic-level concept, the concept's sources, and its most typical training clips. These
are exemplars in the sense of exemplar theory (Medin & Schaffer 1978). The full set is in
[i2_test_exemplars.jsonl](../results/concepts/experiments/i2_test_exemplars.jsonl), and a typical synthetic case
looks like this:

> `HGT1027213.wav` → concept at depth 7 (7 clips, 100% synthetic, all ElevenLabs). Exemplars:
> `elevenlabs/speaker_6167/sentence_260`, `sentence_449` (codec channel), `sentence_284` (resampled).

A real case gets a concept of LibriSpeech speakers that is 0.2% synthetic, with three real LibriSpeech exemplars.

## Acoustic names for the concept dimensions (I5)

We correlated each of the 32 concept dimensions with 93 interpretable DSP measures on 5,283 clips (Spearman, 95%
bootstrap intervals).

- **dim00 dominates the detector.** Its head sensitivity is 143, against 10.2 for the next dimension. Its top
  correlate is frame-level loudness variability (`bg.frame_level_std_db`, ρ 0.42 [0.39, 0.44]).
- **dim04 tracks spectral centroid** (ρ 0.55), and **dim08 tracks SNR** (ρ 0.31).
- **The detector's logit rises with:**
  - frame-level level variability (ρ 0.41),
  - high-band spectral flatness (0.38),
  - noise-floor flatness (0.28),
  - SNR (0.28).
- **The logit falls with** spectral-flux spread (−0.24) and F0 variability (−0.23).

Read together, synthetic speech here tends to be cleaner and steadier than real recordings. That is a plausible
cue, but it could also be a *cleanliness shortcut*: a noisy real clip could be pushed towards "real" for the wrong
reason. The parametric stress tests (ROADMAP T2) and the unseen-channel view of the RawBoost experiment
([results/rawboost/](../results/rawboost/)) test this directly.

## Where it fails, and how the concepts show it (I7, C3)

On held-out and In-the-Wild clips (2,794 clips, 73 errors), errors are not spread evenly: the five worst concepts
hold 27% of them.

| Concept | Size | Synthetic share | What it is | Error rate |
|---|---|---|---|---|
| depth 2 | 1,177 | 99% | XTTS-v2 and YourTTS voice clones | 14% |
| depth 4 | 40 | 13% | real speech, 60% reverberant | 46% |
| depth 3 | 37 | 8% | real speech, 51% reverberant | 50% |
| depth 4 | 43 | 98% | ElevenLabs and PlayHT under noise | 38% |

Typicality (C3) agrees. Under perturbation, error rates are low in quintiles 1–4 and jump in the most specific
quintile: 12.6% in-domain and 15.0% on In-the-Wild, where detector confidence also drops (5.4 and 3.3 against about
7). These are the reverberant concepts. The concept layer therefore names the detector's weak spot, reverberant
real speech, the same weak spot the channel breakdown finds independently.

## Cognitive-science capabilities, each against a baseline

**C1, learning a new family from a few examples, without retraining or forgetting.** We leave one generator out,
insert k of its clips into the tree (no gradient steps), and measure attribution on its held-out clips. Baselines
are kNN (k = 10) and logistic regression, both refitted at every k. Numbers are means over generators and 3 seeds.

| k new clips | cobweb | kNN | logistic regression | old generators (cobweb) |
|---|---|---|---|---|
| 1 | **0.068** | 0.000 | 0.008 | 0.873 |
| 3 | **0.152** | 0.015 | 0.150 | 0.873 |
| 10 | 0.319 | 0.164 | **0.415** | 0.873 |
| 30 | 0.517 | 0.387 | **0.640** | 0.872 |
| 100 | 0.715 | 0.670 | **0.825** | 0.869 |

Cobweb learns first: from 1–3 examples it attributes clips where the baselines cannot. It beats kNN at every k and
forgets nothing (0.873 → 0.869). A classifier refitted from scratch overtakes it at about 10 examples.

On the domain version (C1b: labelled In-the-Wild clips from other speakers), inserting 300 clips cuts cobweb's
minDCF from 0.039 to 0.027. The refitted baselines improve less in absolute terms but start lower (kNN
0.024 → 0.020, LR 0.028 → 0.022). The detector remains best (0.011), so the concept layer is an explanation and
adaptation tool, not a replacement scorer.

**C2, knowing when something is new.** Novelty is the negative of the best held-out pointwise mutual information
along the clip's path.

| Task | Novelty AUROC | Mahalanobis | kNN distance |
|---|---|---|---|
| Unseen generator vs seen | **0.87 ± 0.06** | 0.49 | 0.84 |
| In-the-Wild vs in-domain | 0.82 | 0.81 | **0.92** |

**C4, finding the detector's mistakes.** On In-the-Wild, both views, 1,296 clips and 43 errors, clips are ranked
for review.

| Ranking | Average precision | Precision at top 5% | Recall at top 10% |
|---|---|---|---|
| random | 0.04 | 0.03 | 0.12 |
| novelty | 0.03 | 0.05 | 0.09 |
| concept–detector disagreement | 0.38 | 0.35 | 0.84 |
| detector uncertainty | 0.47 | 0.42 | 0.91 |
| **uncertainty + disagreement** | **0.49** | **0.48** | **0.95** |

Disagreement between the readable layer and the black box is a strong error signal by itself (9× random), and it
adds to the detector's own uncertainty. Novelty does not find errors: new is not the same as wrong.

## Faithfulness and stability (tests E4 and E6, [APPROACH.md §4.9](APPROACH.md#49-can-the-explanations-be-trusted-tests-e1e6))

- **Faithful where they make a claim (E6, deletion).** We took the 66 explanations that mix synthetic and real
  concepts. Deleting the synthetic-concept dimensions lowers the detector's logit more than deleting the
  real-concept ones, by 0.79 SD [0.67, 0.89], on 92% of clips. Random dimensions give only 0.20 and 0.18 SD.
- **Stable across insertion orders (E4).** Over 3 orders the tree itself changes (partition ARI 0.03–0.04), but the
  explanation does not: the top source agrees on 94–95% of clips, and P(fake) has Spearman 0.75.
- **They caught a real scoring bug.** The bug was batch-cropping order dependence, since fixed in `predict.py`.

## Limits, stated plainly

- **The concepts explain the embedding, not raw audio.** The embedding comes from a detector trained with labels.
- **The training tree is label-informed.** I8 shows the same structure appears without labels, but the shipped
  explanations use the label-informed tree.
- **The concept layer is not the scorer.** It is less accurate than the detector (C1b), and the shipped scores are the
  detector's.
- **I5 correlations are associations, not causes.** The cleanliness reading is a hypothesis under test.
- **The C1 comparison has limits.** Attribution only; detection AUC is already about 1.0 because the embedding saw
  every generator, so C1 is about naming a family, not about detecting it.
