# Approach

Team SideQuests, HackGT 13, NSA HEARSAY audio authentication challenge. This document explains the whole system from
first principles: what is measured and why, how the detector is built and trained, how the concept-formation layer
explains its decisions, and how everything is evaluated. Sources are numbered in [References](#references). Data
tables and the forensic audit are in [DATA.md](DATA.md). Every result, with intervals, is in
[EVALUATION.md](EVALUATION.md) and the generated [results/tables.md](../results/tables.md). How to rerun everything:
[REPRODUCE.md](REPRODUCE.md).

**Outcome.** The organizers scored the submitted TSV on the hidden NSA test labels: **minDCF 0.0317, EER 1.44%**. The
best interim leaderboard entry was minDCF 0.0584 / EER 2.5%. Our out-of-domain proxy, In-the-Wild, predicted it:
minDCF 0.028, 95% interval [0.017, 0.040].

---

## Contents

0. [The flow at a glance](#0-the-flow-at-a-glance)
1. [The task and how it is scored](#1-the-task-and-how-it-is-scored)
2. [Data, shortcuts and the canonical view](#2-data-shortcuts-and-the-canonical-view)
3. [The detector](#3-the-detector)
4. [The explanation layer: concepts and prototypes](#4-the-explanation-layer-concepts-and-prototypes)
5. [The evaluation pipeline](#5-the-evaluation-pipeline)
6. [Results](#6-results)
7. [What did not work, and limitations](#7-what-did-not-work-and-limitations)
8. [References](#references)

---

## 0. The flow at a glance

**Inference** (`predict.py`, the Docker entrypoint):

```text
audio file (any container ffmpeg reads)
  |
  +--> T0 triage: container, codec, sample rate, encoder tag, file times ........ trace only, never scored
  |
  v
canonical view: ffmpeg -> 16 kHz mono -> trim edge silence -> 7 kHz low-pass -> DC removal -> peak-normalize -> dither
  |
  +--> XLS-R-2B detector (fine-tuned + copy-synthesis, epoch 3) --> logit_1 --+
  +--> XLS-R-1B detector (fine-tuned + copy-synthesis, epoch 2) --> logit_2 --+--> z-normalize each, average
  +--> MMS-1B  detector (fine-tuned + copy-synthesis, epoch 3) --> logit_3 --+          |
                                                                                        v
                                                        Platt calibration at prior 0.3 -> P(synthetic) = cm-score
                                                                                        |
                                              decision: synthetic if P > 0.2 (Bayes threshold for Cmiss 1, Cfa 4)
  |
  v
router:  0.05 < P < 0.8, or clip > 6 s   -> windowed re-scoring (2 s windows, 1 s hop)
         lossy codec, or narrow band      -> compression / bandwidth cross-check
         otherwise                        -> stop
  |
  v
TSV row (filename, cm-score) + traces.jsonl (triage, per-model logits, fused score, routing decisions and reasons)
```

**Training** (MPCDF Raven, A100):

```text
DiffSSD sample: 14,895 fakes, 10 generators --+
LJSpeech 13,100 + LibriSpeech 6,475 (real) ---+--> shared_split (group-disjoint) --> train | val | holdout
In-the-Wild 4,000 (real + fake) ---------------+--> itw (evaluation only)

train-split reals --> HiFi-GAN 16k, HiFi-GAN LJ, DiffWave, Vocos --> 11,900 copy-synthesis fakes (train only)

train rows --> canonical view (test-like crop) --> random channel chain on 60% --> AntiDeepfake encoder + linear head
           --> class-weighted cross-entropy, source-balanced batches, AdamW 2e-6 (head 1e-4), 800 steps x 3 epochs
           --> after every epoch: score val / holdout / ITW (clean + perturbed) and the test set
           --> keep the epoch with the lowest mean minDCF over {val, ITW} x {clean, perturbed}
```

**Explanation layer** (`scripts/run_concepts.py`):

```text
clip --> final XLS-R-2B detector, time-mean of the last layer (the input of its linear head)
     --> standardize + PCA-whiten to 32 dimensions (fitted on train)  =  concept space z
          |                                              |
          v                                              v
   cobweb-private concept tree                     unconditional DDPM over z
   12,000 train clips, labels = [source | channel]  (both classes, all views)
          |                                              |
          |                                              v
          |                     TTCG: for t = 50, 75, ..., 400: noise z, climb the score to a mode x*
          |                           prototype mean m = x*/sqrt(abar_t), covariance from the denoiser Jacobian
          |                           greedy facility-location selection (K <= 3), product-of-experts composition
          |                                              |
          +-------- name each prototype by its concept (held-out basic level) ----------+
          |                                                                              |
          v                                                                              v
   the clip's own concept: sources, channels, size                 TTCG P(fake) = share-weighted synthetic share
          \------------------------> explanation record (results/concepts/test_explanations.jsonl) <-----------/
```

**Evaluation** (`scripts/evaluate.py`, §5):

```text
per-clip scores of every run, epoch, set and view (runs/diffusion/ft/<run>/<set>_epoch<e>.parquet)
+ concept scores + DSP and metadata scores + predict.py outputs on In-the-Wild and the test set
  |
  v
align on common clips -> z-normalize -> ensembles -> replay each clip's crop and channel chain
  |
  +--> benchmark (AUC, EER, minDCF, 95% bootstrap CI, paired delta vs final) -> breakdowns (generator, real source,
  |    channel, duration) -> cross-fitted calibration -> fine-tuning curves -> fusion gate -> label-free test checks
  |    -> shipped-path check
  v
results/tables.md + CSVs          (unit tests run first, in the same job: mpcdf/evaluate.sbatch)
```

---

## 1. The task and how it is scored

### 1.1 The task

The NSA Research Directorate's tech talk [NSA] defines the task: given audio files, output P(synthetic) in [0, 1] for
each, as a tab-separated `filename  cm-score` file.
- **Clips:** English, longer than 2 s. Some carry counter-measures such as added noise; noise does not change the
  label. No clip is partly synthetic, and intent is not judged.
- **Judging:** 60% minDCF, 20% diversity of techniques (quality, innovation, depth), 20% documentation "in your own
  words".
- **The brief asks teams to:** report what did not work, explain what did, and "minimize black box detectors".

§4 answers the last request.

### 1.2 Detection costs from first principles

A detector maps a clip $x$ to a score $s(x)$. A threshold $\tau$ turns scores into decisions, and every threshold trades
two errors: rejecting real speech and accepting synthetic speech. Bayes decision theory prices the two errors and
weights them by the prior probability of an attack. The ASVspoof 5 evaluation package [3], which the organizers run,
does exactly this. It treats **bona fide speech as the target class** [3, 4]:

$$
\mathrm{DCF}(\tau) = C_{\mathrm{miss}}\,(1-\pi)\,P^{\mathrm{bona}}_{\mathrm{miss}}(\tau) + C_{\mathrm{fa}}\,\pi\,P^{\mathrm{spoof}}_{\mathrm{fa}}(\tau),
\qquad \pi = 0.3,\quad C_{\mathrm{miss}} = 1,\quad C_{\mathrm{fa}} = 4 .
$$

Here $P^{\mathrm{bona}}_{\mathrm{miss}}$ is the share of real clips rejected, i.e. flagged as synthetic.
$P^{\mathrm{spoof}}_{\mathrm{fa}}$ is the share of synthetic clips accepted as real. Normalizing by the cost of the
better constant decision and minimizing over thresholds gives the ranked metric:

$$
\mathrm{minDCF} = \min_\tau \frac{\mathrm{DCF}(\tau)}{\min\{C_{\mathrm{miss}}(1-\pi),\, C_{\mathrm{fa}}\pi\}}
= \min_\tau \Big[\mathrm{FPR}(\tau) + \frac{C_{\mathrm{fa}}\,\pi}{C_{\mathrm{miss}}(1-\pi)}\,\mathrm{FNR}(\tau)\Big]
= \min_\tau \big[\mathrm{FPR}(\tau) + 1.714\,\mathrm{FNR}(\tau)\big].
$$

FPR is the share of real clips flagged; FNR is the share of fakes passed. A value of 0 is perfect; 1 means the
detector is no better than always giving the same answer. minDCF depends only on the ranking of scores. The costs
are equivalent to unit costs at the **effective prior**:

$$
\tilde\pi = \frac{C_{\mathrm{fa}}\,\pi}{C_{\mathrm{fa}}\,\pi + C_{\mathrm{miss}}(1-\pi)} = \frac{1.2}{1.9} \approx 0.632 .
$$

The equal error rate (EER) is the error at the threshold where FPR = FNR.

> **A naming trap.** The tech talk calls "identifying real as synthetic" the false alarm and says it costs 4×. In the
> package, "false alarm" means *accepting a spoof*, because bona fide is the target class. So the 4× cost falls on
> passing a fake, and flagging real speech costs 1. `hearsay/metrics.py` reproduces the package, and
> `tests/test_metrics.py` checks it to $10^{-12}$ against the organizers' own module. The official score was computed
> with the package.

**Ties.** The package walks every sorted score position. With tied scores that includes operating points no threshold
can realize. Our implementation evaluates only realizable thresholds, so it is never optimistic (also unit-tested).

### 1.3 From scores to decisions: calibration and the Bayes threshold

minDCF ignores calibration, but a reader of `cm-score` does not. Let $P = P(\text{spoof}\mid x)$ be a posterior
computed at the evaluation prior $\pi$. Rejecting costs $C_{\mathrm{miss}}(1-P)$ in expectation, and accepting costs
$C_{\mathrm{fa}}P$. Flagging is optimal when

$$
C_{\mathrm{fa}}\,P > C_{\mathrm{miss}}\,(1-P) \iff P > \frac{C_{\mathrm{miss}}}{C_{\mathrm{miss}} + C_{\mathrm{fa}}} = 0.2 .
$$

We produce such posteriors with Platt scaling [6, 7]: a class-balanced logistic regression $\sigma(a\,s + b_{\mathrm{bal}})$
(a prior-0.5 posterior), shifted to the evaluation prior:

$$
P(\text{spoof}\mid s) = \sigma\!\Big(a\,s + b_{\mathrm{bal}} + \log\frac{\pi}{1-\pi}\Big).
$$

Two numbers measure calibration. **actDCF** is the DCF of the decisions $P > 0.2$; its gap to minDCF is the cost of
calibration. $C_{\mathrm{llr}}$ [5] scores the log-likelihood ratios $\lambda = \operatorname{logit}P - \operatorname{logit}\pi$:

$$
C_{\mathrm{llr}} = \frac12\Big[\frac{1}{N_s}\sum_{i\in\text{spoof}}\log_2\big(1+e^{-\lambda_i}\big) + \frac{1}{N_b}\sum_{j\in\text{bona}}\log_2\big(1+e^{\lambda_j}\big)\Big].
$$

### 1.4 Uncertainty

Every metric is reported with a 95% percentile interval from a **stratified bootstrap** [8]: real and synthetic clips
are resampled separately, 1,000 times. Two systems are compared with a **paired** bootstrap on the same clips and the
same resamples, so shared difficulty cancels out.

---

## 2. Data, shortcuts and the canonical view

Details and tables: [DATA.md](DATA.md).

**Sources.**
- **DiffSSD** [29]: the organizers' training data, 10 generators. Four LJ-voice TTS systems speak with the LJSpeech
  narrator's voice; six voice cloners imitate 10 LibriSpeech speakers. We use a pinned 14,895-file sample.
- **Real speech:** LJSpeech-1.1, the 10 cloned LibriSpeech speakers, and 80 more LibriSpeech speakers.
- **In-the-Wild** [20]: 4,000 web recordings of public figures, real and fake. Evaluation only. AntiDeepfake [19] was
  post-trained on most of DiffSSD's fakes, so DiffSSD-based numbers flatter the backbones, while In-the-Wild was held
  out of their training.

**Audit before training** (`scripts/forensic_audit.py`):
- **No reuse.** Byte and decoded-PCM hashes found no duplicates. Landmark hashing [61] of every test clip against 89,817
  reference recordings found no content match; the positive control matched 100 of 100.
- **Constant container.** Every test file has the same FFmpeg-4.2 container, so metadata cannot score on the test set.
- **One test pipeline.** 98% of test lengths are multiples of 512 samples at 22.05 kHz, and a kaiser-class resampler
  cut the band at 7.4 kHz. It is one librosa-style pipeline applied to both classes: augmentation material, never a
  feature.

**Shortcuts.** In DiffSSD the classes differ in ways unrelated to synthesis, and trivial features exploit them
(results/shortcut_checks.md):

| Feature | AUC |
|---|---|
| file-system times | 1.00 |
| container fields | 0.90 |
| level | 0.91–0.96 |
| edge silence | 0.92–0.95 |

This is shortcut learning [22]. ASVspoof 2019 had the same problem: silence alone reached 85% accuracy [21].

**The canonical view.** Every model sees the same processing (`hearsay/audio.py`):
1. ffmpeg decode to 16 kHz mono;
2. trim edge silence;
3. crop to a length drawn from the test durations, start-anchored half the time (training only);
4. [channel chain];
5. 7 kHz low-pass;
6. DC removal, peak normalization to 0.97–1.0, and 1-LSB dither.

Duration drops to AUC 0.53, and the container disappears. The residual level cues are one reason In-the-Wild, not the
in-domain holdout, decides between systems.

**Channel augmentation** (`hearsay/augment.py`). Real test audio may be noisy or compressed, and a noisy real clip must
not read as fake. So 60% of training clips, **of both classes**, pass through one or two random operations:
- codecs: MP3, AAC, Opus, G.711, G.722, G.726;
- filtering: telephony band-pass, band-limiting, resampling through 8–12 kHz, spectral tilt;
- noise: white, pink or brown noise, and babble from train-split reals, at 5–30 dB SNR;
- mains hum, room reverb (RT60 0.15–0.9 s), clipping.

Codec and filtering augmentation is among the most effective generalization tools in the literature [17, 18].

**Splits** (`hearsay/splits.py`).
- **Groups never straddle splits:** a sentence id, an LJSpeech chapter, a LibriSpeech (speaker, chapter) or an extra
  speaker. Two cloned speakers are held out entirely.
- **Copy-synthesis rows** are made only from train-split reals and never move another row.

---

## 3. The detector

### 3.1 Background: from spectral features to self-supervised encoders

1. **Hand-built features.** Early spoofing countermeasures scored hand-built spectral features with Gaussian mixture
   models: LFCC [9] and CQCC [10]. Our DSP detector (`hearsay_dsp/`) belongs to this family.
2. **End-to-end networks** learn from the raw waveform: RawNet2 [11] and AASIST's spectro-temporal graph attention [12].
3. **Self-supervised encoders** changed the field. wav2vec 2.0 [13] learns speech representations without labels by
   masking latent frames and identifying the true quantized latent $q_t$ among distractors $\tilde q \in Q_t$ from
   the context vector $c_t$:

$$
\mathcal{L}_m = -\log \frac{\exp\big(\mathrm{sim}(c_t, q_t)/\kappa\big)}{\sum_{\tilde q \in Q_t} \exp\big(\mathrm{sim}(c_t, \tilde q)/\kappa\big)} .
$$

   - XLS-R [14] and MMS [15] scale this to many languages and 300M–2B parameters.
   - HuBERT [16] predicts cluster targets instead.
   - With a light classifier and data augmentation, these encoders became the most robust spoofing detectors [17].
   - **AntiDeepfake** [19, 59] post-trains them on 56,000 h of real and 18,000 h of artefact speech, adding mean
     pooling and a linear head with logits [fake, real]. We start from its checkpoints.

The field's central problem is **generalization**. Detectors near-perfect on ASVspoof fail on real-world deepfakes [20].
Fine-tuning a pretrained model can also distort its features and lose out-of-distribution accuracy [28]; we observed
exactly this (§6).

### 3.2 Running AntiDeepfake without fairseq

The released checkpoints are fairseq files, and fairseq 0.12 does not install on current Python.
`hearsay/antideepfake.py` rebuilds each architecture in `transformers` and maps the weights with a **strict** key map:
every tensor is placed and every shape checked. That strictness caught a positional-convolution weight-norm naming bug
before it could silently degrade a model. The input is standardized per clip, as in the original recipe. The detector's
score is the synthetic logit $\ell = \text{logit}_{\text{fake}} - \text{logit}_{\text{real}}$.

### 3.3 Fine-tuning gently (`scripts/finetune_ssl.py`)

| Setting | Value | Why |
|---|---|---|
| optimizer | AdamW [60], LR 2e-6 encoder / 1e-4 head, weight decay 0.01, 10% warm-up + cosine | small steps preserve post-training; the literature fine-tunes at 1e-6 to 5e-6 [19, 17] |
| schedule | 800 steps per epoch, 3 epochs; every epoch scored on val / holdout / ITW, clean + perturbed, and on the test set | curves, not just endpoints (§6) |
| checkpoint | lowest mean minDCF over {val, ITW} × {clean, perturbed} | selects for generalization and robustness |
| batches | source-balanced: LJ-voice fakes 0.2, cloned fakes 0.2, copy-synthesis 0.1, LJSpeech 0.2, cloned-speaker LibriSpeech 0.15, other LibriSpeech 0.15; class-weighted cross-entropy | no source dominates by count |
| crops | lengths drawn from the test durations; clips ≥ 3.3 s | train on what the test looks like |
| memory | bf16, gradient checkpointing, CNN front end frozen; XLS-R-2B also freezes its bottom 24 of 48 layers | fits a 40 GB A100 |

**WiSE-FT** [27] interpolates weights between the pretrained and fine-tuned models,
$\theta_\alpha = (1-\alpha)\,\theta_{\mathrm{pre}} + \alpha\,\theta_{\mathrm{ft}}$. It keeps zero-shot generality with
part of the robustness. It is used in the interim ensemble, not the final one.

### 3.4 Copy-synthesis fakes (Track D6-R)

A neural vocoder reconstructs a waveform from a mel spectrogram. Three families are used:
- **HiFi-GAN** [24] trains the generator against multi-period and multi-scale discriminators.
- **DiffWave** [25] is a diffusion model that denoises Gaussian noise into a waveform in a few learned steps.
- **Vocos** [26] predicts Fourier coefficients and inverts them.

All of them must re-invent the phase and fine spectral texture that the mel spectrogram discards; that is where their
artifacts live.

**Copy-synthesis** re-vocodes *real* clips and labels them synthetic [23]. The fake keeps the real words and the real
speaker, so the vocoder is the only thing that separates it from its source, and a detector trained on such pairs has
to learn the vocoder. We re-vocode 2,975 train-split reals with each of HiFi-GAN (16 kHz LibriTTS and 22 kHz LJSpeech),
DiffWave and Vocos, and delay-align them: 11,900 fakes (`scripts/run_d6r.py`, `hearsay/diffusion/resynth.py`). Neural
codecs were left out on purpose; codec artifacts are a channel, not synthesis. Wang & Yamagishi report vocoded training
data cutting In-the-Wild EER roughly in half [23]. For us it reversed the out-of-domain decay of fine-tuning (§6).

### 3.5 Ensemble and calibration (`scripts/make_submission.py`, `submission/fusion.json`)

Three backbones pretrained on different data make different errors: XLS-R-2B (epoch 3), XLS-R-1B (epoch 2) and MMS-1B
(epoch 3), all fine-tuned with copy-synthesis. Each synthetic logit is z-normalized with statistics from val and
In-the-Wild, and the three are averaged. There are no fitted fusion weights, because weights overfit small development
sets, as ASVspoof 5 teams found [3]:

$$
s(x) = \frac{1}{3}\sum_{k=1}^{3} \frac{\ell_k(x) - \mu_k}{\sigma_k},
\qquad
P(\text{synthetic}\mid x) = \sigma\big(a\,s(x) + b\big),\quad a = 6.84,\ b = -2.60 \text{ (prior 0.3 included)} .
$$

### 3.6 `predict.py`: inference, routing, traces

- **Scoring:** every clip is scored **whole and alone, in fp32** on every device. A file's score changes by less than
  $3\times10^{-5}$ whatever else is in the folder, and CPU and GPU agree.
- **Memory:** models are loaded one at a time from memory-mapped checkpoints; peak memory is about 14.5 GB.
- **Router:** the routing decisions and their reasons go into each file's trace.
  - Uncertain scores (0.05 < P < 0.8) and clips longer than 6 s get windowed re-scoring: 2 s windows, 1 s hop, and a
    spread above 6 logits is flagged.
  - Lossy codecs, or a declared rate with a narrow measured band, get a compression cross-check.
  - Everything else stops there. On the test set, 3.3% of clips were routed further.
- **Failures:** files that fail to decode are listed in `failures.tsv`, and the export is refused.
- **Docker** (`Dockerfile`): CPU torch and ffmpeg, runs with `--network none`, needs about 16 GB of memory. On 8 CPU
  threads the test set takes about 2 hours ([results/runtime.md](../results/runtime.md)).

The submitted TSV was written by an earlier version that scored batches cropped to their shortest clip, in bf16. The
current path agrees with it on 1,668 of 1,671 decisions. One of the three changes was found by the explanation layer
(§4.8).

---

## 4. The explanation layer: concepts and prototypes

### 4.1 Why prototypes

People classify by **resemblance to a category's central tendency**, its prototype.
- **Typicality:** after learning distortions of dot patterns, subjects classify the never-seen prototype best
  [31]. Members closer to the prototype are judged more typical and share more attributes with other members
  (*family resemblance*) [32, 33].
- **Exemplars:** exemplar theories instead store instances and classify by summed similarity [35, 36].
- **One continuum:** the varying-abstraction model shows the prototype and exemplar accounts are two ends of one
  continuum [37]. A concept hierarchy spans it: shallow nodes act as prototypes, leaves as exemplars.
- **The basic level:** people prefer one level of a taxonomy, where categories are most informative ("chair", not
  "furniture" or "kitchen chair") [34].

Machine learning uses the same idea for interpretable classifiers:
- nearest class means, and prototypical networks [44];
- "this looks like that" part-prototype networks [45].

Here the explanation points at training material a person can recognize.

For a detector, that is the explanation an analyst can act on: "this clip sits with ElevenLabs clones heard through a
codec", or "nothing in training explains it". An explanation is only useful if it is **faithful**: it must point at
what the model actually uses, not just sound plausible [56, 57]. We test that in §4.9.

### 4.2 COBWEB from first principles

Concept formation builds a hierarchy of probabilistic concepts incrementally, one instance at a time, with no labels
required [39]. Its objective is **category utility** [38]: the expected increase in the number of attribute values one
can guess correctly when the category is known, averaged over the $K$ categories of a partition:

$$
CU(\{C_1,\dots,C_K\}) = \frac{1}{K}\sum_{k=1}^{K} P(C_k) \sum_i \sum_j \Big[ P(A_i = V_{ij} \mid C_k)^2 - P(A_i = V_{ij})^2 \Big].
$$

The term $\sum_j P(A_i = V_{ij}\mid C)^2$ is the probability of guessing attribute $A_i$ correctly by probability
matching. CU rewards partitions whose members are **predictable** (intra-class similarity) and **distinct** from the
whole (inter-class dissimilarity). The level of a taxonomy with the highest CU matches the human basic level [38].

COBWEB [39] sorts each new instance down the tree from the root. At every node it tries four operators and keeps the
one that maximizes CU of the partition below that node:
1. add the instance to the best child;
2. create a new singleton child;
3. merge the two best children;
4. split the best child into its children.

Merge and split let the tree recover from bad early decisions, but the result still depends on insertion order [41].
We test that in §4.9. Anderson's rational model gives the same process a Bayesian reading: each instance goes into
the category that maximizes the posterior, including a new one [42].

**Continuous attributes (CLASSIT)** [40]. Replace the sum of squared probabilities by the integral of a squared Gaussian
density, $\int \mathcal N(v;\mu,\sigma^2)^2\,dv = 1/(2\sqrt\pi\,\sigma)$. Category utility then rewards children that
are tighter than their parent $p$:

$$
CU = \frac{1}{K}\sum_{k} P(C_k) \sum_i \frac{1}{2\sqrt{\pi}}\Big(\frac{1}{\sigma_{ik}} - \frac{1}{\sigma_{ip}}\Big),
$$

with each $\sigma$ floored at an *acuity* so that singletons do not dominate. Each concept is a diagonal Gaussian.
Modern implementations scale this to vision (Cobweb/4V) [43].

### 4.3 cobweb-private: what we use and how

We use the Teachable AI Lab's implementation `cobweb-private`, `CobwebContinuousTree`, with library defaults. It is
private and never committed (`.gitignore`); `hearsay/concepts.py` wraps it.

- **Concepts** are diagonal Gaussians. A node's density uses its *parent's* variance plus a prior variance $v_0$, so a
  concept is never narrower than its parent's spread: $p_c(x) = \mathcal N\big(x;\,\mu_c,\, s^2_{\mathrm{par}(c)} + v_0\big)$.
- **Labels ride along.** Every inserted clip carries a multi-hot label [source one-hot | channel one-hot]. Each concept
  therefore reports which generators or real corpora and which channel conditions it summarizes. `predict` expands
  the 300 best concepts to give P(source), and so P(fake).
- **How informative a concept is** is measured by its expected pointwise mutual information with the root [50]:

$$
\mathrm{pmi}(x;c) = \log p_c(x) - \log p_{\mathrm{root}}(x), \qquad D(c) = \mathbb{E}_{x \sim \mathcal N(\mu_c,\, s_c^2)}\big[\mathrm{pmi}(x;c)\big].
$$

  For diagonal Gaussians with model variances $\sigma^2_c = s^2_{\mathrm{par}(c)} + v_0$, this has a closed form per
  dimension $d$:

$$
D(c) = \sum_d \frac12\left[\log\frac{\sigma^2_{\mathrm{root},d}}{\sigma^2_{c,d}}
+ \frac{s^2_{c,d} + (\mu_{c,d}-\mu_{\mathrm{root},d})^2}{\sigma^2_{\mathrm{root},d}} - \frac{s^2_{c,d}}{\sigma^2_{c,d}}\right].
$$

  `get_basic` returns the node on an instance's path with the largest $D(c)$. We checked the library's closed form
  against its Monte-Carlo estimate: they agree within 0.1%.
- **The held-out basic level.** The closed-form $D(c)$ is an expectation under the concept's *own* distribution, and
  it grows with depth: a narrower Gaussian always looks more informative about its own members (§6.2). For a clip the
  tree has **not** seen, we instead evaluate pmi at the clip itself along its path and take the argmax. That is
  `Concepts.informative`, the held-out definition in [50]. A concept too narrow to cover the clip is penalized.

### 4.4 Diffusion models from first principles

A denoising diffusion probabilistic model (DDPM) [46] defines a forward process that gradually adds Gaussian noise:

$$
q(x_t \mid x_0) = \mathcal N\big(x_t;\ \sqrt{\bar\alpha_t}\,x_0,\ (1-\bar\alpha_t) I\big),\qquad \bar\alpha_t = \prod_{s\le t}(1-\beta_s).
$$

The model learns to predict the added noise with $\mathbb E\,\lVert \varepsilon - \hat\varepsilon_\theta(\sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\,\varepsilon,\ t)\rVert^2$.
The noise prediction is a scaled **score** of the noised marginal $p_t$ [47]:

$$
\nabla_x \log p_t(x) = -\frac{\hat\varepsilon_\theta(x,t)}{\sqrt{1-\bar\alpha_t}} .
$$

**Tweedie's formula** [48] gives the posterior mean of the clean data from the score:

$$
\hat x_0(x_t) = \mathbb E[x_0 \mid x_t] = \frac{x_t + (1-\bar\alpha_t)\,\nabla_x\log p_t(x_t)}{\sqrt{\bar\alpha_t}} .
$$

Its derivative gives the posterior covariance:

$$
\operatorname{Cov}[x_0 \mid x_t] = \frac{1-\bar\alpha_t}{\bar\alpha_t}\Big(I + (1-\bar\alpha_t)\nabla^2_x\log p_t\Big)
= \frac{1-\bar\alpha_t}{\sqrt{\bar\alpha_t}}\,\frac{\partial \hat x_0}{\partial x_t}.
$$

Our model (`hearsay/diffusion/ddpm.py`) is a residual MLP (4 × 512, FiLM time conditioning, T = 1000, linear β, EMA
weights). It is trained unconditionally on the 32-dimensional concept space, both classes and all views, for 8,000
steps.

### 4.5 Diffusion as concept formation (DMCF)

Wang, Singaravadivelan and MacLellan [50] show that a diffusion model's family of noised marginals and a COBWEB tree
describe the same kind of object: a hierarchy of Gaussian prototypes.
- **Noise level ↔ depth.** High noise corresponds to broad, shallow concepts; low noise to specific, deep ones.
- **Prototypes.** Climbing the score of $p_t$ from a noised input reaches a mode $x^*$. At a mode the score is zero,
  so Tweedie maps it back to clean space, and the Jacobian term gives the prototype's covariance (their Eq. 9):

$$
m = \frac{x^*}{\sqrt{\bar\alpha_t}}, \qquad \Sigma = \frac{1-\bar\alpha_t}{\sqrt{\bar\alpha_t}}\ \operatorname{diag}\!\Big(\frac{\partial \hat x_0}{\partial x}\Big)\Big|_{x^*}.
$$

- **Basic level:** it is where held-out informativeness peaks.

We estimate the Jacobian diagonal with Hutchinson's estimator [49] and central finite differences, using Rademacher
probes $v$:

$$
\operatorname{diag}(J) \approx \frac{1}{K}\sum_{k=1}^{K} v_k \odot \frac{\hat x_0(x + \epsilon v_k) - \hat x_0(x - \epsilon v_k)}{2\epsilon}.
$$

**A check with known answers** (`tests/test_ttcg.py`). Take a Gaussian mixture with component variance $s^2$, whose
score is known exactly.
- **Means:** mode ascent plus Tweedie recover the component means.
- **Covariance:** Eq. 9 must equal the posterior variance
  $\operatorname{Var}[x_0\mid x_t] = (1-\bar\alpha)\,s^2 / (\bar\alpha\,s^2 + 1 - \bar\alpha)$. Our implementation matches
  within 10%.

### 4.6 TTCG: explaining one input as a composition of prototypes

Test-time compositional generalization (TTCG) [51] explains a query $z$ with a few prototypes, each responsible for
different dimensions (`hearsay/diffusion/ttcg.py`).

1. **Discover.** At every noise level $t = 50, 75, \dots, 400$, 32 noised copies of $z$ climb $\log p_t$ with Adam for
   150 steps. The modes are de-duplicated. Each gives a prototype $(m_j, \Sigma_j, t_j)$.
2. **Select.** Choose up to $K = 3$ prototypes by greedy maximization of a facility-location objective. Each dimension
   $r$ is served by its best prototype, or by the whitened root $b_r = \log\mathcal N(z_r; 0, 1)$ if no prototype beats
   it:

$$
F(S) = \sum_{r} \max\Big(b_r,\ \max_{j\in S} \ell_{jr}\Big), \qquad \ell_{jr} = \log \mathcal N\big(z_r;\ m_{jr},\ \Sigma_{jr}\big).
$$

   $F(S) - F(\emptyset)$ is monotone submodular, so the greedy choice is within $1 - 1/e$ of the optimum [52]. The root
   baseline makes that guarantee hold even though log-densities can be negative.
3. **Compose.** Per-dimension softmax weights ($\tau = 0.5$) share each dimension among the selected prototypes. The
   composition is a product of Gaussian experts [53]:

$$
w_j(r) = \frac{e^{\ell_{jr}/\tau}}{\sum_{k\in S} e^{\ell_{kr}/\tau}},\qquad
\Sigma^{-1} = \sum_{j\in S} \operatorname{diag}(w_j)\,\Sigma_j^{-1},\qquad
\mu = \Sigma \sum_{j\in S} \operatorname{diag}(w_j)\,\Sigma_j^{-1} m_j .
$$

### 4.7 Our explanation pipeline

- **The space** is the final XLS-R-2B detector's time-mean last layer, exactly the input of its linear head.
  Standardized and PCA-whitened to 32 dimensions on training clips, it holds the detector's decision: the head,
  evaluated through those 32 dimensions, reproduces the detector's logit with $R^2 = 0.99997$.
- **The tree** is fitted on 12,000 source-stratified training clips (both views), in three insertion orders.
- **Naming.** Each selected TTCG prototype is named by its held-out concept, the node on its path with the largest pmi.
  That concept's training make-up gives sources, channels and a synthetic share.
- **The TTCG score** is the share-weighted synthetic share of the selected prototypes, so every point of it traces back
  to named training material:

$$
P_{\mathrm{TTCG}}(\text{fake}\mid z) = \frac{\sum_{j\in S} \bar w_j\ \mathrm{fake}(c_j)}{\sum_{j\in S} \bar w_j}, \qquad \bar w_j = \operatorname{mean}_r\, w_j(r).
$$

Output: one record per NSA test clip in `results/concepts/test_explanations.jsonl`.

### 4.8 A worked example

```json
{"filename": "HGT7824018.wav", "cobweb_p_fake": 0.9947, "ttcg_p_fake": 0.9934,
 "concept": {"depth": 6, "size": 19, "p_fake": 1.0,
             "sources": ["unit_speech 0.42", "xtts_v2 0.37", "your_tts 0.11"], "channels": ["codec 0.95", "clipping 0.05"]},
 "summary": "t=50 prototype (39% of dims) ~ 59-clip concept of your_tts, xtts_v2 under codec; t=75 prototype (32% of dims)
             ~ 9-clip concept of diffgan_tts, wavegrad2 under codec; t=75 prototype (28% of dims) ~ 19-clip concept of
             unit_speech, xtts_v2 under codec"}
```

The clip falls in a 19-clip concept of voice-cloning fakes heard through a codec, and all three prototypes that explain
it are synthetic concepts. The submitted TSV gave it P = 0.009, from the older batch-cropped, bf16 path. Scored whole
in fp32 it gets 0.997, which agrees with the explanation.
- **A review list.** Across the test set, 7 of 1,671 explanations contradict the submitted decision by more than 0.5.
  One of them exposed the scoring bug fixed in §3.6.
- **Naming the doubt.** The most uncertain test clip (P = 0.50) is explained entirely by real LibriSpeech speakers
  **under reverb**. Reverb is the detector's weakest condition (§6.1).

### 4.9 Can the explanations be trusted? (tests E1–E6)

| Test | Result | Reading |
|---|---|---|
| **E1 scores**, In-the-Wild, same clips as its detector | TTCG 0.043 vs 0.033 clean (Δ +0.010 [0.000, 0.029]); 0.112 vs 0.103 perturbed (Δ +0.009 [−0.013, 0.041]) | a score made of named concepts costs about 0.01 minDCF out of domain |
| E1, cobweb P(fake) | +0.037 / +0.046 vs the detector | weaker; explanation only |
| E1, fused into the final score | clean ITW +0.016 [0.002, 0.032] | does not help: concepts explain, detectors score |
| **E2 basic level** | closed-form $D(c)$ rises at every depth (3.4 → 52.3), so it picks single-clip leaves; held-out pmi peaks at depth 7 (15.8), median concept 8 clips | the held-out definition gives an interior basic level, as [50] predicts |
| E2, label purity | 99.97% of clips land in a concept with their own real/fake label | concepts are clean |
| **E3 leakage** (adjusted MI [54]) | source 0.069 · speaker 0.049 · real/fake 0.038 · sample rate 0.034 · duration 0.012 | voice structure is present (DiffSSD's LJ voice and 10 cloned speakers); duration is gone |
| **E4 stability** over 3 insertion orders | partition ARI [55] 0.03–0.04; top source agrees 94–95%; P(fake) Spearman 0.75 | the tree changes, the explanation does not |
| **E5 noise ↔ depth** | mean concept depth 7.4–7.8 for t ≤ 300, 5.2–6.6 for t ≥ 325 | right direction, only at high noise |
| **E6 faithfulness** (deletion [56]), 66 of 3,000 explanations that mix synthetic and real concepts | deleting synthetic-concept dims lowers the detector's logit more than deleting real-concept dims: 0.79 SD [0.67, 0.89], 92% of clips; vs random dims: 0.20 [0.14, 0.24] (91%) and 0.18 [0.13, 0.22] (89%) | where an explanation makes a claim, the detector relies on that evidence |

**More detail on E1–E5.**
- **Scores on 1,000-clip subsets per set and view** (minDCF):

  | Scorer | ITW clean | ITW perturbed | holdout perturbed |
  |---|---|---|---|
  | TTCG | 0.043 | 0.097 | 0.095 |
  | TTCG, prototypes named by the closed-form basic concept (ablation) | 0.051 | 0.100 | 0.103 |
  | cobweb P(fake) | 0.078 | 0.155 | 0.086 |

  Naming prototypes by the held-out concept is slightly better, but within noise on the paired clips.
- **NSA test set:** cobweb and TTCG flag 28.4% and 29.3% at P > 0.2 (the final ensemble: 28.0%). The concepts the test
  clips fall in come from real LibriSpeech (71%), ElevenLabs (9%), PlayHT (5%), XTTS (4%), UnitSpeech (4%) and YourTTS
  (3%).
- **Held-out pmi by depth, 1 → 12:** 1.1, 5.0, 8.9, 11.0, 13.2, 14.9, **15.8**, 14.5, 12.8, 10.9, 7.7, 1.4.
- **Smoothing does not rescue the closed form:** raising the prior variance from the default to 4 moves the closed-form
  basic level only from depth 9.85 to 9.84.

**Faithfulness in detail.** Let $w = w_{\text{fake}} - w_{\text{real}}$ be the detector's linear head. Deleting a set of
concept dimensions $M$ (setting them to the training mean, 0 in whitened units) moves the embedding by
$\Delta h(M)$, obtained by un-whitening, and the logit by $w^\top \Delta h(M)$. For each mixed explanation, compare
deleting the dimensions attributed to synthetic concepts, the real ones, and as many random dimensions.

---

## 5. The evaluation pipeline

`scripts/evaluate.py` evaluates every system under one protocol from the per-clip scores the runs saved. It re-scores
nothing, so a number in the docs can be traced to a file. It runs as one Raven job, `mpcdf/evaluate.sbatch`, after the
unit tests.

The ten steps, with inputs and outputs, are drawn in [EVALUATION.md §1](EVALUATION.md#1-protocol-and-pipeline); the compact version is in §0.

The concept tests (§4.9) come from `scripts/run_concepts.py` (`mpcdf/concepts.sbatch`, plus the CPU passes
`--levels-only` and `--faithfulness-only`).

**Unit tests** (`tests/`, 89 across the tracks):
- **metric parity** with the organizers' package, tie handling, and the Bayes threshold;
- the TSV writer and validator, including the committed final TSV;
- split rules;
- TTCG against closed-form answers;
- the cobweb-private wrapper;
- determinism and band limits of the canonical view;
- the DSP suite.

The full test matrix (A–G) and every table are in [EVALUATION.md](EVALUATION.md).

---

## 6. Results

### 6.1 Detection

| System | In-the-Wild clean | In-the-Wild perturbed | holdout perturbed |
|---|---|---|---|
| zero-shot XLS-R-2B (best pretrained) | 0.038 [0.024, 0.049] | 0.189 [0.162, 0.212] | 0.354 |
| fine-tuned XLS-R-2B | 0.060 [0.043, 0.071] | 0.153 [0.127, 0.171] | 0.069 |
| XLS-R-2B + copy-synthesis | 0.040 [0.026, 0.051] | 0.104 [0.081, 0.119] | 0.089 |
| **final ensemble** | **0.028 [0.017, 0.040]** | **0.082 [0.064, 0.097]** | **0.073** |
| **official, NSA test set** | **0.0317** (EER 1.44%) | | |

- **The ensemble beats its best member**, paired on the same clips: by 0.012 [0.001, 0.020] clean and 0.022
  [0.004, 0.037] perturbed. No system is significantly better than the final ensemble on any set or view.
- **Fine-tuning curves**, In-the-Wild clean, epoch 0 → last:

  | Backbone | plain fine-tuning | with copy-synthesis |
  |---|---|---|
  | XLS-R-1B | 0.050 → 0.129 | 0.050 → 0.039 |
  | MMS-1B | 0.065 → 0.207 | 0.065 → 0.042 |

  This matches the distortion effect in [28], and shows copy-synthesis reversing it.
- **Weak spots:**
  - **reverberation:** 0.25 on In-the-Wild, 0.38 in domain, while every other channel is below 0.08 in domain;
  - **clips under 2 s:** 0.119 on In-the-Wild, against 0.000 at 3–4 s;
  - **commercial cloners:** ElevenLabs 0.104, PlayHT 0.075.
- **Calibration transfers.** Fitted on val only, applied to In-the-Wild: actDCF 0.038 against minDCF 0.028. The shipped
  path flags 50.6% of a 50%-synthetic set.

### 6.2 Explanations

See the table in §4.9.

### 6.3 Engineering

| Measure | Result |
|---|---|
| order independence | a clip's score changes by < 3e-5 with the rest of the folder |
| shipped path, all 4,000 In-the-Wild clips | AUC 0.9994, EER 1.30%, minDCF 0.033 |
| CPU runtime | 21 min for the test set on one 72-core node; ~4.2 s per clip at 8 threads |
| peak memory | 14.5 GB |
| unit tests | 89, all runnable ones pass |

---

## 7. What did not work, and limitations

- **Signal-processing detector** (`hearsay_dsp/`: LFCC-GMM, spectral, LPC, phase, prosody, background, ENF). AUC 0.91 in
  domain but 0.27 on In-the-Wild, worse than chance. Fused with the detector it raises In-the-Wild minDCF from 0.028 to
  0.594. Trained on DiffSSD's reals alone, it flags 99.8% of real LibriSpeech.
- **Metadata** is a perfect shortcut in DiffSSD and constant on the test set. There, the metadata models only track clip
  duration. Kept as trace context.
- **Diffusion as a detector.** DDPM mode-path features, AudioLDM 2 denoising curves and SGMSE+ enhancement features were
  built and dropped: no precedent for speech, never run. They are recoverable from git
  ([archive/README.md](archive/README.md)). Diffusion entered where it helped: DiffWave as a copy-synthesis vocoder
  (not separable on its own: +DiffWave changed XLS-R-2B In-the-Wild from 0.041 to 0.040) and TTCG in the explanation
  layer.
- **Cobweb as a scorer, and the closed-form basic level:** neither worked (§4.9).
- **Limitations.**
  - The concepts inherit the detector's view. "An ElevenLabs concept" means clips the detector places with ElevenLabs
    training clips.
  - Speaker structure is present in the concepts.
  - The exact tree depends on insertion order.
  - `cobweb-private` is private, so the Docker image ships the detector only, and the explanations are delivered as
    results.
  - Reverb and short clips remain the detector's weak spots; more realistic room simulation is the first thing to add.

---

## References

**Challenge and evaluation**
- [NSA] NSA Research Directorate, VISTA Research. *HEARSAY: The Audio Authentication Challenge*, HackGT 13 tech talk, 2026. `docs/NSA_HackGT13_TechTalk.pdf`.
- [1] Wang et al. ASVspoof 2019: a large-scale public database of synthesized, converted and replayed speech. *Computer Speech & Language* 64, 2020. [arXiv:1911.01601](https://arxiv.org/abs/1911.01601)
- [2] Yamagishi et al. ASVspoof 2021: accelerating progress in spoofed and deepfake speech detection. 2021. [arXiv:2109.00537](https://arxiv.org/abs/2109.00537)
- [3] Wang et al. ASVspoof 5: crowdsourced speech data, deepfakes, and adversarial attacks at scale. 2024. [arXiv:2408.08739](https://arxiv.org/abs/2408.08739); evaluation package [github.com/asvspoof-challenge/asvspoof5](https://github.com/asvspoof-challenge/asvspoof5)
- [4] Kinnunen et al. t-DCF: a detection cost function for the tandem assessment of spoofing countermeasures and automatic speaker verification. *Odyssey* 2018. [arXiv:1804.09618](https://arxiv.org/abs/1804.09618)
- [5] Brümmer & du Preez. Application-independent evaluation of speaker detection. *Computer Speech & Language* 20, 2006. [doi:10.1016/j.csl.2005.08.001](https://doi.org/10.1016/j.csl.2005.08.001)
- [6] Platt. Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods. *Advances in Large Margin Classifiers*, 1999.
- [7] Niculescu-Mizil & Caruana. Predicting good probabilities with supervised learning. *ICML* 2005. [doi:10.1145/1102351.1102430](https://doi.org/10.1145/1102351.1102430)
- [8] Efron & Tibshirani. *An Introduction to the Bootstrap*. Chapman & Hall, 1993.

**Detectors and speech representations**
- [9] Sahidullah, Kinnunen & Hanilçi. A comparison of features for synthetic speech detection. *Interspeech* 2015. [ISCA](https://www.isca-archive.org/interspeech_2015/sahidullah15_interspeech.html)
- [10] Todisco, Delgado & Evans. Constant Q cepstral coefficients: a spoofing countermeasure for automatic speaker verification. *Computer Speech & Language* 45, 2017. [doi:10.1016/j.csl.2017.01.001](https://doi.org/10.1016/j.csl.2017.01.001)
- [11] Tak et al. End-to-end anti-spoofing with RawNet2. *ICASSP* 2021. [arXiv:2011.01108](https://arxiv.org/abs/2011.01108)
- [12] Jung et al. AASIST: audio anti-spoofing using integrated spectro-temporal graph attention networks. *ICASSP* 2022. [arXiv:2110.01200](https://arxiv.org/abs/2110.01200)
- [13] Baevski et al. wav2vec 2.0: a framework for self-supervised learning of speech representations. *NeurIPS* 2020. [arXiv:2006.11477](https://arxiv.org/abs/2006.11477)
- [14] Babu et al. XLS-R: self-supervised cross-lingual speech representation learning at scale. 2021. [arXiv:2111.09296](https://arxiv.org/abs/2111.09296)
- [15] Pratap et al. Scaling speech technology to 1,000+ languages (MMS). 2023. [arXiv:2305.13516](https://arxiv.org/abs/2305.13516)
- [16] Hsu et al. HuBERT: self-supervised speech representation learning by masked prediction of hidden units. 2021. [arXiv:2106.07447](https://arxiv.org/abs/2106.07447)
- [17] Tak et al. Automatic speaker verification spoofing and deepfake detection using wav2vec 2.0 and data augmentation. *Odyssey* 2022. [arXiv:2202.12233](https://arxiv.org/abs/2202.12233)
- [18] Tak et al. RawBoost: a raw data boosting and augmentation method applied to automatic speaker verification anti-spoofing. *ICASSP* 2022. [arXiv:2111.04433](https://arxiv.org/abs/2111.04433)
- [19] Ge, Wang, Liu & Yamagishi. Post-training for deepfake speech detection (the AntiDeepfake models, NII Yamagishi Lab). 2025. [arXiv:2506.21090](https://arxiv.org/abs/2506.21090)
- [20] Müller et al. Does audio deepfake detection generalize? *Interspeech* 2022. [arXiv:2203.16263](https://arxiv.org/abs/2203.16263)
- [21] Müller et al. Speech is silver, silence is golden: what do ASVspoof-trained models really learn? 2021. [arXiv:2106.12914](https://arxiv.org/abs/2106.12914)
- [22] Geirhos et al. Shortcut learning in deep neural networks. *Nature Machine Intelligence* 2, 2020. [doi:10.1038/s42256-020-00257-z](https://doi.org/10.1038/s42256-020-00257-z)
- [23] Wang & Yamagishi. Spoofed training data for speech spoofing countermeasure can be efficiently created using neural vocoders. *ICASSP* 2023. [arXiv:2210.10570](https://arxiv.org/abs/2210.10570)
- [24] Kong, Kim & Bae. HiFi-GAN: generative adversarial networks for efficient and high fidelity speech synthesis. *NeurIPS* 2020. [arXiv:2010.05646](https://arxiv.org/abs/2010.05646)
- [25] Kong et al. DiffWave: a versatile diffusion model for audio synthesis. *ICLR* 2021. [arXiv:2009.09761](https://arxiv.org/abs/2009.09761)
- [26] Siuzdak. Vocos: closing the gap between time-domain and Fourier-based neural vocoders for high-quality audio synthesis. 2023. [arXiv:2306.00814](https://arxiv.org/abs/2306.00814)
- [27] Wortsman et al. Robust fine-tuning of zero-shot models (WiSE-FT). *CVPR* 2022. [arXiv:2109.01903](https://arxiv.org/abs/2109.01903)
- [28] Kumar et al. Fine-tuning can distort pretrained features and underperform out-of-distribution. *ICLR* 2022. [arXiv:2202.10054](https://arxiv.org/abs/2202.10054)
- [29] Bhagtani et al. DiffSSD: a diffusion-based dataset for speech forensics. 2024. [arXiv:2409.13049](https://arxiv.org/abs/2409.13049)

**Concepts and categorization**
- [31] Posner & Keele. On the genesis of abstract ideas. *Journal of Experimental Psychology* 77, 1968. [doi:10.1037/h0025953](https://doi.org/10.1037/h0025953)
- [32] Rosch. Cognitive representations of semantic categories. *JEP: General* 104, 1975. [doi:10.1037/0096-3445.104.3.192](https://doi.org/10.1037/0096-3445.104.3.192)
- [33] Rosch & Mervis. Family resemblances: studies in the internal structure of categories. *Cognitive Psychology* 7, 1975. [doi:10.1016/0010-0285(75)90024-9](https://doi.org/10.1016/0010-0285(75)90024-9)
- [34] Rosch et al. Basic objects in natural categories. *Cognitive Psychology* 8, 1976. [doi:10.1016/0010-0285(76)90013-X](https://doi.org/10.1016/0010-0285(76)90013-X)
- [35] Medin & Schaffer. Context theory of classification learning. *Psychological Review* 85, 1978. [doi:10.1037/0033-295X.85.3.207](https://doi.org/10.1037/0033-295X.85.3.207)
- [36] Nosofsky. Attention, similarity, and the identification–categorization relationship. *JEP: General* 115, 1986. [doi:10.1037/0096-3445.115.1.39](https://doi.org/10.1037/0096-3445.115.1.39)
- [37] Vanpaemel & Storms. In search of abstraction: the varying abstraction model of categorization. *Psychonomic Bulletin & Review* 15, 2008. [doi:10.3758/PBR.15.4.732](https://doi.org/10.3758/PBR.15.4.732)
- [38] Gluck & Corter. Information, uncertainty, and the utility of categories. *Proc. Cognitive Science Society* 7, 1985; Corter & Gluck. Explaining basic categories: feature predictability and information. *Psychological Bulletin* 111, 1992. [doi:10.1037/0033-2909.111.2.291](https://doi.org/10.1037/0033-2909.111.2.291)
- [39] Fisher. Knowledge acquisition via incremental conceptual clustering. *Machine Learning* 2, 1987. [doi:10.1007/BF00114265](https://doi.org/10.1007/BF00114265)
- [40] Gennari, Langley & Fisher. Models of incremental concept formation. *Artificial Intelligence* 40, 1989. [doi:10.1016/0004-3702(89)90046-5](https://doi.org/10.1016/0004-3702(89)90046-5)
- [41] Fisher. Iterative optimization and simplification of hierarchical clusterings. *JAIR* 4, 1996. [doi:10.1613/jair.276](https://doi.org/10.1613/jair.276)
- [42] Anderson. The adaptive nature of human categorization. *Psychological Review* 98, 1991. [doi:10.1037/0033-295X.98.3.409](https://doi.org/10.1037/0033-295X.98.3.409)
- [43] Barari, Lian & MacLellan. Incremental concept formation over visual images without catastrophic forgetting (Cobweb/4V). 2024. [arXiv:2402.16933](https://arxiv.org/abs/2402.16933)
- [44] Snell, Swersky & Zemel. Prototypical networks for few-shot learning. *NeurIPS* 2017. [arXiv:1703.05175](https://arxiv.org/abs/1703.05175)
- [45] Chen et al. This looks like that: deep learning for interpretable image recognition. *NeurIPS* 2019. [arXiv:1806.10574](https://arxiv.org/abs/1806.10574)

**Diffusion, prototypes and composition**
- [46] Ho, Jain & Abbeel. Denoising diffusion probabilistic models. *NeurIPS* 2020. [arXiv:2006.11239](https://arxiv.org/abs/2006.11239)
- [47] Song et al. Score-based generative modeling through stochastic differential equations. *ICLR* 2021. [arXiv:2011.13456](https://arxiv.org/abs/2011.13456)
- [48] Efron. Tweedie's formula and selection bias. *JASA* 106, 2011. [doi:10.1198/jasa.2011.tm11181](https://doi.org/10.1198/jasa.2011.tm11181)
- [49] Hutchinson. A stochastic estimator of the trace of the influence matrix for Laplacian smoothing splines. *Communications in Statistics – Simulation and Computation* 18, 1989. [doi:10.1080/03610918908812806](https://doi.org/10.1080/03610918908812806)
- [50] Wang, Singaravadivelan & MacLellan. DMCF: diffusion models and concept formation (the basic level as peak held-out informativeness; prototype covariances, Eq. 9). 2026. [arXiv:2609.13047](https://arxiv.org/abs/2609.13047)
- [51] Wang, Gupta, Zhu & MacLellan. Test-time compositional generalization in diffusion models via concept discovery (TTCG). 2026. [arXiv:2605.07078](https://arxiv.org/abs/2605.07078)
- [52] Nemhauser, Wolsey & Fisher. An analysis of approximations for maximizing submodular set functions. *Mathematical Programming* 14, 1978. [doi:10.1007/BF01588971](https://doi.org/10.1007/BF01588971)
- [53] Hinton. Training products of experts by minimizing contrastive divergence. *Neural Computation* 14, 2002. [doi:10.1162/089976602760128018](https://doi.org/10.1162/089976602760128018)

**Measuring explanations and clusterings**
- [54] Vinh, Epps & Bailey. Information theoretic measures for clusterings comparison. *JMLR* 11, 2010. [jmlr.org](https://jmlr.org/papers/v11/vinh10a.html)
- [55] Hubert & Arabie. Comparing partitions. *Journal of Classification* 2, 1985. [doi:10.1007/BF01908075](https://doi.org/10.1007/BF01908075)
- [56] Samek et al. Evaluating the visualization of what a deep neural network has learned. *IEEE TNNLS* 28, 2017. [doi:10.1109/TNNLS.2016.2599820](https://doi.org/10.1109/TNNLS.2016.2599820)
- [57] Jacovi & Goldberg. Towards faithfully interpretable NLP systems: how should we define and evaluate faithfulness? *ACL* 2020. [arXiv:2004.03685](https://arxiv.org/abs/2004.03685)

**Tools**
- [59] AntiDeepfake model cards, e.g. [huggingface.co/nii-yamagishilab/xls-r-2b-anti-deepfake](https://huggingface.co/nii-yamagishilab/xls-r-2b-anti-deepfake)
- [60] Loshchilov & Hutter. Decoupled weight decay regularization (AdamW). *ICLR* 2019. [arXiv:1711.05101](https://arxiv.org/abs/1711.05101)
- [61] Wang, A. An industrial-strength audio search algorithm (landmark hashing). *ISMIR* 2003. [PDF](https://www.ee.columbia.edu/~dpwe/papers/Wang03-shazam.pdf)
