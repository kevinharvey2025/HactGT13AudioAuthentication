# Concept formation and diffusion prototypes: saying *why* a clip looks synthetic

A detector outputs a number. An analyst needs a reason: "this clip sits with ElevenLabs clones heard through a phone
line", or "nothing in training explains it". This module arranges what the final detector sees into concepts
learned the way people form categories. It then explains each clip as a composition of prototypes discovered by a
diffusion model and names each prototype through those concepts. The explanations are tested like a detector
(scores, stability, leakage) and against the detector itself (faithfulness). As far as we know, this is the first use
of COBWEB-style concept formation and test-time diffusion prototype composition for audio deepfake analysis.

Code: `hearsay/concepts.py` (cobweb-private wrapper, embedding space), `hearsay/diffusion/ddpm.py`,
`hearsay/diffusion/ttcg.py`, `scripts/concept_subset.py`, `scripts/run_concepts.py`, `mpcdf/concepts.sbatch`.

## 1. Where the ideas come from

| Idea | What we take from it | How it is used |
|---|---|---|
| Prototype theory (Posner & Keele 1968; Rosch 1973, 1975) | a category is summarized by its central tendency; typical members sit near it | every concept is a diagonal Gaussian over the detector's embedding |
| Family resemblance (Rosch & Mervis 1975), exemplar models (Medin & Schaffer 1978; Nosofsky 1986), the varying-abstraction continuum (Vanpaemel & Storms 2008) | prototype and exemplar accounts are two ends of one continuum | a concept hierarchy spans it: shallow nodes are prototypes, leaves are near-exemplars |
| Category utility and the basic level (Gluck & Corter 1985; Rosch et al. 1976) | people prefer one level of a taxonomy ("dog", not "animal" or "beagle") where categories are most informative | each clip is described at its basic-level concept |
| COBWEB / CLASSIT (Fisher 1987; Gennari, Langley & Fisher 1989) | concepts form incrementally: each new instance is added, starts a new concept, merges or splits concepts, whichever maximizes category utility | the lab's implementation, **cobweb-private**, is the ground truth |
| Diffusion models as concept formation (Wang, Singaravadivelan & MacLellan, arXiv 2609.13047) | a diffusion model's marginals and a COBWEB tree describe the same hierarchy of Gaussian prototypes: noise level ↔ depth, the basic level is where D(c) = E[pmi(x; c)] peaks, and a prototype's covariance follows from the denoiser's Jacobian (their Eq. 9) | the basic level via `get_basic`; prototype covariances; the noise-depth test |
| Test-time compositional generalization, TTCG (Wang, Gupta, Zhu & MacLellan, arXiv 2605.07078) | a new input is explained by a few prototypes, each covering different dimensions: greedy facility-location selection (monotone submodular, within 1 − 1/e), product-of-experts composition | the per-clip explanation |

## 2. Pipeline

1. **Space.** The time-mean of the final XLS-R-2B detector's last layer (+ D6-R, the model inside the ensemble),
   exactly the input of its linear head. Standardized, then PCA-whitened to 32 dimensions, fitted on training
   clips. Concepts therefore describe what the detector sees.
2. **Concepts** (cobweb-private `CobwebContinuousTree`, library defaults). 12,000 training clips, stratified by
   source, both views, inserted one at a time. Each carries a multi-hot label:
   - source: one of 10 DiffSSD generators or 3 real corpora;
   - channel: clean or one of 9 perturbation families.

   Three trees in three insertion orders. The tree gives:
   - **P(fake)** from `predict` (best-first over 300 concepts);
   - each clip's **concept**: the node on its path with the largest held-out pmi, log p_c(x) − log p_root(x)
     (`Concepts.informative`). This is the basic level for a clip the tree has never seen; a concept too narrow to
     cover the clip pays for it. The library's closed-form alternative, the path node with the largest D(c)
     (`get_basic`), is reported alongside (§4).
3. **Diffusion prototypes.** An unconditional DDPM over the same space: residual MLP, 4 x 512, T = 1000, 8,000
   steps, both classes and all views. For each query clip, TTCG:
   1. **Discover.** At every noise level t = 50, 75, ..., 400, 32 noised copies of the query climb the model's score
      (150 Adam steps) to modes x*. Each mode gives a prototype with mean x*/√ᾱ_t (Tweedie) and diagonal
      covariance ((1 − ᾱ_t)/√ᾱ_t) · diag(∂x̂₀/∂x), estimated with 4 Rademacher finite-difference probes (Eq. 9).
   2. **Select.** Greedily maximize F(S) = Σ_dims max(root, max_{j∈S} log N(z_d; m_jd, s_jd)), K ≤ 3. The
      whitened root N(0, 1) is the baseline a prototype must beat.
   3. **Compose.** Per-dimension softmax weights (τ = 0.5) give each prototype its share of the dimensions and a
      product-of-experts Gaussian.
4. **Naming prototypes.** Each selected prototype's mean is categorized in the tree and named by its concept
   (held-out pmi, as for clips). The concept's training make-up gives sources, channels and a synthetic share.
   **TTCG P(fake)** is the share-weighted synthetic share of the selected prototypes: every point of the score
   traces back to named concepts.

`tests/test_ttcg.py` checks steps 3.1–3.3 on a Gaussian mixture where the right answers are known in closed form:
the modes, the posterior variance (1 − ᾱ)s²/(ᾱs² + 1 − ᾱ), and a query that mixes two concepts across
dimensions.

## 3. What a test-clip explanation looks like

`results/concepts/test_explanations.jsonl` holds one record per NSA test clip:

```json
{
 "filename": "HGT7824018.wav",
 "cobweb_p_fake": 0.9947, "ttcg_p_fake": 0.9934,
 "concept": {"depth": 6, "size": 19, "pmi": 11.7, "p_fake": 1.0,
             "sources": [{"source": "unit_speech", "share": 0.42}, {"source": "xtts_v2", "share": 0.37}, {"source": "your_tts", "share": 0.11}],
             "channels": [{"channel": "codec", "share": 0.95}, {"channel": "clipping", "share": 0.05}]},
 "diffusion_prototypes": [
  {"noise_level": 50, "share": 0.39, "concept_size": 59, "concept_sources": ["your_tts 0.37", "xtts_v2 0.36"], "concept_channels": ["codec 0.54"], "concept_p_fake": 0.98},
  {"noise_level": 75, "share": 0.33, "concept_size": 9,  "concept_sources": ["diffgan_tts 0.44", "wavegrad2 0.44"], "concept_channels": ["codec"], "concept_p_fake": 1.0},
  {"noise_level": 75, "share": 0.28, "concept_size": 19, "concept_sources": ["unit_speech", "xtts_v2"], "concept_channels": ["codec"], "concept_p_fake": 1.0}],
 "summary": "t=50 prototype (39% of dims) ~ 59-clip concept of your_tts, xtts_v2 under codec; t=75 prototype (32% of dims) ~ 9-clip concept of diffgan_tts, wavegrad2 under codec; t=75 prototype (28% of dims) ~ 19-clip concept of unit_speech, xtts_v2 under codec"
}
```

(trimmed; the file also records the closed-form basic concept and every prototype's depth.)

How to read it: the clip falls in a 19-clip concept of voice-cloning fakes heard through a codec, and all three
prototypes that explain it are synthetic concepts. The submitted TSV gave this clip P = 0.009. That run scored the
test clips in batches cropped to their shortest member, in bf16. Scoring the whole clip in fp32, as `predict.py`
now does, gives 0.997, which agrees with the explanation. Across the test set, 7 of 1,671 clips have an explanation
that contradicts the submitted decision by more than 0.5. This is the review list the concept layer produces; one
of the seven exposed that scoring bug.

The most uncertain test clip, `HGT9417641.wav` (P = 0.50), is explained entirely by real LibriSpeech speakers
**under reverb**. Reverberation is the detector's weakest channel condition ([EVALUATION.md](EVALUATION.md)), so
the explanation names the likely reason for the doubt.

## 4. Tests and results

All numbers: `results/concepts/` (`metrics.json`, `levels.json`, `faithfulness.json`) and, paired against the
detectors, [results/tables.md](../results/tables.md).

**E1. Scores** (minDCF, 1,000 clips per set and view; AUC in brackets):

| Scorer | holdout clean / aug | ITW clean | ITW aug |
|---|---|---|---|
| cobweb P(fake) | 0.000 / 0.086 | 0.078 (0.991) | 0.155 (0.977) |
| **TTCG P(fake)** | 0.000 / 0.095 | **0.043 (0.994)** | **0.097 (0.985)** |
| TTCG, prototypes named by closed-form basic concepts (ablation) | 0.000 / 0.103 | 0.051 (0.985) | 0.100 (0.972) |

- **Against its own detector.** Paired on identical clips with the detector whose embedding it uses (XLS-R-2B +
  D6-R), TTCG trails by +0.010 [0.000, 0.029] on clean ITW and +0.009 [−0.013, 0.041] on perturbed ITW. A score
  that is a readable vote of concepts gives up about 0.01 minDCF out of domain. Cobweb's P(fake) is weaker
  (+0.037 and +0.046, both significant).
- **Not a scorer.** Adding both concept scores to the final ensemble does not help: clean ITW gets worse by +0.016
  [0.002, 0.032]. Concepts explain; the detectors score.
- **NSA test set.** Cobweb and TTCG flag 28.4% and 29.3% of the test clips at P > 0.2 (the final ensemble: 28.0%;
  κ with its decisions 0.985 and 0.979). The concepts the test clips fall in come from real LibriSpeech (71%),
  ElevenLabs (9%), PlayHT (5%), XTTS (4%), UnitSpeech (4%) and YourTTS (3%): the test fakes resemble commercial
  voice cloning.

**E2. The basic level.** The two definitions disagree, and the data decide between them.

| | closed-form D(c) (`get_basic`) | held-out pmi (`informative`) |
|---|---|---|
| depth of a clip's concept | at the leaves (median 10) | interior (peak at 7–9) |
| training clips in the concept (median) | 1 | 8 |
| distinct concepts over 999 held-out clips | 872 | 785 |
| concept agrees with the clip's real/fake label | 100% | 99.97% |

- **Closed-form D(c) never peaks.** It rises monotonically with depth (3.4 at depth 1 to 52.3 at depth 12),
  because a narrower Gaussian always has more expected PMI against the root. Stronger variance smoothing does not
  move it (mean depth 9.85 → 9.84 as `eval_prior_var` goes from the default to 4). It picks exemplars.
- **Held-out pmi has an interior basic level.** It rises and falls with depth: 1.1 (depth 1), 8.9 (3), 13.2 (5),
  **15.8 (7)**, 12.8 (9), 7.7 (11), 1.4 (12). This is the pattern arXiv 2609.13047 reports: informativeness on
  unseen data peaks at an intermediate level.
- **The library's closed form is correct.** Closed-form expected PMI matches its Monte-Carlo estimate within 0.1%
  on 21 multi-clip concepts.

**E3. Leakage.** Chance-adjusted mutual information (AMI) between the held-out basic concepts and each variable:
source 0.069, speaker 0.049, real/fake 0.038, native sample rate 0.034, duration 0.012. The raw NMI values
(0.16–0.38) are inflated by the 785-way partition.
- The concepts are nearly pure in real/fake (99.97%) and structured by source.
- Speaker information is present, as expected from DiffSSD, where every LJ-voice generator speaks with the LJSpeech
  voice and the cloners imitate 10 LibriSpeech speakers. A concept named after a generator may partly be a voice.
- Duration and sample rate carry little: the canonical view removed them.

**E4. Stability across insertion orders.** COBWEB is order-sensitive, and the exact partition changes (ARI
0.03–0.04 across three orders). What an explanation says does not: the top source of a clip's concept agrees in
94–95% of clips, and P(fake) correlates at 0.75 (Spearman).

**E5. Noise level ↔ depth.**
- **Prediction** (arXiv 2609.13047): higher diffusion noise corresponds to more general, shallower concepts.
- **Result:** prototypes from t = 50–300 land in concepts at mean depth 7.4–7.8, and those from t = 325–400 at
  5.2–6.6.
- **Reading:** the direction is right, but only at high noise; the overall rank correlation is −0.01.

**E6. Faithfulness to the detector.**
- **The concept space holds the detector's decision.** The detector's linear head, evaluated through the 32-dim
  concept space, reproduces its logit with R² = 0.99997. The head applied to the stored embeddings matches the
  detector's saved scores (Spearman 0.992).
- **Only mixed explanations make a testable claim.** 66 of 3,000 held-out explanations mix synthetic and real
  concepts. When all prototypes are synthetic, every dimension counts as synthetic evidence, and the deletion test
  cannot tell evidence from random dimensions.
- **On those 66 clips, the detector relies on the explanation's evidence:**

| Test (in SD of the detector logit) | mean [95% CI] | clips in the expected direction |
|---|---|---|
| delete synthetic-concept dims vs delete real-concept dims (lowers the synthetic logit more) | 0.79 [0.67, 0.89] | 92% |
| delete synthetic-concept dims vs as many random dims | 0.20 [0.14, 0.24] | 91% |
| delete real-concept dims vs as many random dims (raises it more) | 0.18 [0.13, 0.22] | 89% |

**Implementation checks.** `tests/test_ttcg.py`:
- mode ascent and Tweedie recover the component means of a Gaussian mixture;
- the Eq. 9 covariance equals the analytic posterior variance;
- selection and composition explain a query that mixes two concepts across dimensions.

`tests/test_concepts.py` exercises the cobweb-private wrapper. All six pass in the concepts environment on Raven.

**Bottom line.**
- **Scoring:** the concept layer turns the detector into explanations that score within about 0.01 minDCF of the
  detector out of domain.
- **Explanations:** they are faithful where they make a claim, stable in what they report, and label-pure. They
  produce an audit list that already caught one real scoring bug.
- **Basic level:** the held-out definition gives the psychologically meaningful level the concept-formation
  literature predicts, where the closed-form definition does not.
- **What it does not do:** improve the score itself. Speaker structure is present in the concepts, and the
  noise-depth correspondence is weak.

## 5. Limitations

- **The concepts inherit the detector's view.** They summarize what the detector's representation separates, not
  acoustics in general. A concept named "ElevenLabs" means "clips the detector places with ElevenLabs training
  clips".
- **Confounds in the sources.** The LJ-voice generators all speak with the LJSpeech voice, and the cloners imitate
  10 LibriSpeech speakers, so source concepts can partly be voice concepts. The leakage test measures this instead
  of assuming it away.
- **The exact partition depends on insertion order** (COBWEB is order-sensitive; Fisher 1996). What an explanation
  reports (top source, P(fake)) is far more stable than the partition; see the stability test.
- **Private dependency.** cobweb-private is the lab's repository and is not redistributed. The Docker image therefore
  ships the detector only, and the explanations for the NSA test set are provided as results.
