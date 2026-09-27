# HEARSAY — NSA audio authentication challenge (HackGT 13)

Working doc for our entry. Covers the official requirements, scoring, the approach options (with a dedicated **diffusion track**), the open-source models we'd use, the build plan, evaluation, submission checklist, risks and open questions.

Status: planning. Last updated Sat Sep 26, 2026, 6 PM. Sources: official HEARSAY challenge PDF, team notes.

**How to read this doc:** the core detector (Track A) and forensic modules (Track B) carry the score and the rubric. The **diffusion track (Track D, section 7)** is one option on top of them: it builds on open-source pretrained diffusion models plus two recent papers from our lab, and it has go/no-go gates so it can be dropped without hurting the submission.

---

## 1. Official requirements (from the PDF)

### 1.1 Task
- Take **any audio file** (WAV, MP3, M4A, MP4 audio, OGG, etc.) and decide whether it's **real or synthesized**.
- Output a **synthetic probability score from 0.0 to 1.0**, where **1.0 = synthetic**.
- **Orchestrate multiple forensic techniques automatically**: audio, spectrograms, signal processing, file metadata.
- **Explain results**: which techniques flagged a clip, and which worked or had no effect overall.
- Real clips **may be augmented with noise and other perturbations**. A noisy real clip must not read as fake.
- Clips are **at least 2 seconds** and **in English**.
- Time limit: **36 hours**.
- Allowed: cloud services, open-source LLMs, HexLabs LLM resources, university compute, speech models, signal-processing toolkits. We can also **record our own audio** (cartoonish voices, rapid speech, etc.) to stress-test.

### 1.2 Data
- **Training set:** labeled **bonafide vs. spoof** only. No manipulation-type labels.
- **Sample test set:** labels held out, used for the detection score.
- Test data may include: bona fide speech (studio, smartphone, telephony, field; various codecs), fully synthetic speech (zero-shot TTS, neural vocoders), voice conversion, replay attacks, scene manipulation (real voice, fabricated background), laundered fakes (transcoded, band-limited, noised), and metadata-spoofed containers.
- The PDF also lists: word insertion/replacement in real recordings (partial fakes), fabricated room acoustics, spoofed container metadata and MAC times.

### 1.3 Deliverables
1. **Source code** in a GitHub/GitLab repo with a **README** (approach, architecture, how to run).
2. **Docker image** that runs inference on the test set without major configuration changes.
3. **Prediction file** (`.tsv`) generated from the sample test set.

### 1.4 Prediction file format

```
filename	cm-score
file1.wav	0.80
file2.mp3	0.10
file3.m4a	0.40
file4.m4a	1.00
```

- Tab-delimited, header row required.
- `filename`: file name including extension. `cm-score`: probability the file is synthetic, 0.0–1.0.
- File name: `<teamName>_predictions.tsv`.
- **Team note:** the final TSV has **~77k entries**, one per audio file. A **prefilled TSV is on the Google Drive**; its filename list is the source of truth. Location: `[TK: link to shared folder]`.

### 1.5 NSA draft review
- One-time review of a draft TSV to show current performance. Optional but recommended.
- Plan: send it once the fused model is ready (section 11, Phase 4), with time left to act on it.

### 1.6 Scoring rubric

| Weight | Category | What earns points |
|---|---|---|
| **60%** | Detection performance | the 0–1 synthetic score on the test set |
| **20%** | Forensic diversity | each distinct technique actually used, plus an **agentic orchestration bonus** |
| **20%** | Documentation and presentation | clear documentation; insight into *why* a clip is real or synthetic |

Techniques the rubric names:
1. Container, file forensics and metadata (encoder tags, timestamps, codec chain, MAC times)
2. Spectral / frequency-domain (spectrograms, band-limiting, vocoder harmonics)
3. Prosody and phonetics (breath, pause, coarticulation, pitch contour, repeatability, monotonicity, emotion)
4. Acoustic-environment consistency (ENF mains hum)
5. Compression forensics (double encoding, transcoding artifacts)
6. Speaker-embedding consistency (voice drift across the clip)
7. Deep-learning anti-spoofing detectors
8. Splice / discontinuity detection (phase breaks, DC offset, background seams)
9. **Agentic bonus:** the system chooses which analyses to run based on file type, codec, initial findings, or confidence.

**Implication:** 60% rides on one number, so the core detector and fusion come first. The other 40% rewards breadth, a real routing policy, and clear explanations.

---

## 2. Things to check first (Phase 0)

- [ ] Download the prefilled TSV. Count rows, check filename format and extensions.
- [ ] Confirm submission and draft-review deadlines. `[TK]`
- [ ] Confirm our team name string. `[TK]`
- [ ] Training set: size, class balance, formats, sample rates, clip lengths.
- [ ] Compare train vs. test distributions (formats, codecs, durations, sample rates).
- [ ] Ask NSA (booth or Discord):
  - which metric scores detection (AUC, EER, log loss)?
  - are external datasets and self-generated fakes allowed for training?
  - are scene-manipulated clips (real voice, fake background) labeled spoof?
- [ ] Check that every open-source checkpoint we plan to use downloads, runs, and has a license that allows this use (section 8). Bake them into the Docker image.
- [ ] Measure throughput per module on 500 clips; extrapolate to 77k (section 9).

---

## 3. Score polarity warning

The column name `cm-score` comes from ASVspoof, where **a higher score usually means bona fide**. Here **1.0 = synthetic**. Many pretrained anti-spoofing models output a bona fide score. **Flip pretrained outputs before fusion**, and add a dev-set check that spoof clips score higher than bonafide clips.

---

## 4. Approach options at a glance

| Track | What it is | Role | Priority |
|---|---|---|---|
| **A. Core DL detector** | open-source SSL encoders (WavLM, XLS-R) + pretrained anti-spoof heads (AASIST family), fine-tuned with heavy augmentation | carries most of the 60% | must have |
| **B. Forensic modules** | metadata, spectral, prosody, ENF/environment, compression, speaker drift, splice | covers rubric breadth; fusion features; explanations | must have |
| **C. Agentic router + trace** | policy that picks which analyses run per file; per-file trace JSON | agentic bonus; explanation source | must have |
| **D. Diffusion track** | pretrained open-source diffusion models (vocoders, latent audio models, speech score models) + our own small diffusion model on embeddings; also diffusion/flow TTS to generate training fakes | extra detector signals, better training data, strong explanation story | option, gated |
| **E. LLM explainer** | open-source LLM reads the trace and writes the explanation | presentation; demo | should have |

Tracks A–C and E are enough for a complete submission. Track D adds signal and a research angle, and each piece of it has a go/no-go check (section 7.4).

---

## 5. Framing

### 5.1 The problem is compositional
Every clip is a combination of factors:

| Factor | Example values |
|---|---|
| Authenticity | bona fide, TTS, voice conversion, partial fake |
| Vocoder / generator | GAN vocoder, diffusion vocoder, flow-matching TTS, neural codec |
| Channel | MP3/AAC/Opus, telephony band-limit, resampling |
| Environment | reverb, background noise, mains hum, replay loudspeaker |
| Container | encoder tags, timestamps, codec chain |

Only authenticity is labeled. We can **label channel and environment ourselves** because we apply them as augmentation, and external datasets add generator labels (ASVspoof attack IDs, WaveFake vocoder names, MLAAD model names). The goal is to separate what the channel did from what the generator did, so noise or compression isn't mistaken for synthesis.

### 5.2 Architecture

```
audio file
  │
  ▼
[T0] Triage (every file, ms)
     ffprobe + ExifTool + os.stat → container, codec, sample rate, bitrate, duration,
     encoder tags, timestamps, effective bandwidth; decode → 16 kHz mono + native-rate copy
  │
  ▼
[T1] Core detectors (every file)
     SSL anti-spoof detector · LFCC model · metadata/bandwidth features ·
     D4 mode-path on cached embeddings (cheap)
  │
  ▼
[R]  Router (agentic policy)
  │
  ▼
[T2] Targeted analyses (only when triggered)
     compression · ENF/environment · speaker drift · splice · prosody · replay ·
     D1 vocoder resynthesis · D3 latent-diffusion denoising curve
  │
  ▼
[F]  Fusion (gradient boosting, missing = not run) → isotonic calibration → cm-score
  │
  ▼
[O]  TSV row · per-file trace JSON · optional LLM explanation
```

The LLM never produces the score. It reads the trace and writes the explanation.

---

## 6. Tracks A and B: core detector and forensic modules

Each module outputs a score or features and a short finding string for the trace.

### 6.1 Deep-learning anti-spoofing (rubric 7) — highest priority
- **Open-source front-ends:** WavLM Base+/Large, wav2vec 2.0 XLS-R 300M, optionally the Whisper encoder. Use several layers; middle layers usually keep more artifact information than the last.
- **Heads:** logistic regression on pooled layers → attention-pooling MLP → AASIST-style head. Start from a public SSL+AASIST checkpoint pretrained on ASVspoof if one is available, then fine-tune. **Flip its output** (section 3).
- **Second model:** pretrained AASIST or RawNet2 on raw waveform.
- **Training data:** provided train set + external sets if allowed (ASVspoof 2019 LA / 2021 LA+DF / ASVspoof 5, WaveFake, In-the-Wild, MLAAD) + bona fide references (VCTK, LibriSpeech, VOiCES) + self-generated fakes (D6).
- **Augmentation, applied to both classes** so the model can't learn "noisy = real":
  - codecs: MP3, AAC, Opus, AMR/GSM-like, several bitrates
  - telephony band-limit (300–3400 Hz), down/up resampling
  - noise (babble, street, hum, birdsong, sirens), SNR 5–30 dB
  - room impulse responses
  - gain changes, clipping
- Log applied augmentations; they're the channel labels for D5.

### 6.2 Container, file forensics and metadata (rubric 1)
- Tools: `ffprobe`, `mediainfo`, `exiftool`, `os.stat`.
- Features: container, codec, sample rate, bit depth, bitrate, channels, duration, encoder tag and family, container timestamps vs. filesystem MAC times.
- Consistency flags:
  - declared 44.1/48 kHz but energy stops at 4 or 8 kHz → upsampled from a low-quality source
  - header duration ≠ decoded duration
  - encoder tag claims a device whose defaults don't match the stream
  - impossible timestamps (future dates, created after modified)
- Caveats: dataset MAC times were likely reset during copy/download (check whether they vary at all). The test set includes spoofed containers, and encoder tags may correlate with labels as a dataset artifact. Use as fusion features, measure reliance, report it.

### 6.3 Spectral / frequency-domain (rubric 2)
- LFCC (optionally CQCC) + small LCNN, the ASVspoof baseline family.
- Handcrafted: effective bandwidth, high-band energy ratio, spectral flatness and flux stats, periodic peaks in spectral autocorrelation (vocoder upsampling patterns), harmonic regularity across bands.

### 6.4 Prosody and phonetics (rubric 3)
- Tools: openSMILE (eGeMAPS), Praat via `parselmouth`, Silero VAD.
- Features: F0 range/variance/monotonicity, jitter, shimmer, HNR, pause count and duration distribution, breath events in pauses, speaking rate and its variance, repeated near-identical F0 contours.

### 6.5 Acoustic-environment consistency (rubric 4)
- **ENF:** narrowband STFT around 50/60 Hz and harmonics on native-rate audio. Hum present, stable, continuous across speech and non-speech?
- **Background:** noise floor level and spectrum in non-speech vs. under speech.
- **Reverb:** decay estimates across segments; big differences suggest splicing or scene manipulation.
- **Replay:** low-frequency roll-off, doubled reverb, loudspeaker resonances.
- Routed: needs enough non-speech audio and bandwidth.

### 6.6 Compression forensics (rubric 5)
- Spectral cutoffs typical of a lower bitrate inside a higher-quality container; MDCT histogram irregularities from double quantization; band-limiting that doesn't match the declared codec.
- Output a guessed codec chain for the trace. A transcode chain alone doesn't mean fake; it's context and a router trigger.

### 6.7 Speaker-embedding consistency (rubric 6)
- ECAPA-TDNN (SpeechBrain, pretrained on VoxCeleb) on sliding windows (1.5 s, 0.5 s hop).
- Features: mean and min adjacent cosine similarity, variance, largest jump and its time.
- Catches partial voice conversion or inserted words; weak on fully synthetic clips.

### 6.8 Splice / discontinuity (rubric 8)
- Window-level detector scores → jumps between windows.
- Phase breaks (group delay discontinuities), DC offset shifts, background seams.
- Output suspicious time ranges and a clip-level score (max or top-k mean).

---

## 7. Track D: diffusion models (option)

### 7.1 Why diffusion here

Three hypotheses, each testable on dev in an hour or two:

- **H1 — On-manifold reconstruction.** Audio produced by a neural vocoder already sits on a vocoder's output manifold, so passing it through a similar pretrained vocoder changes it less than real audio. This is the audio version of DIRE (diffusion reconstruction error for detecting generated images).
- **H2 — Coarse right, fine wrong.** Compared with a model of real speech, fakes look normal at high noise levels (phonetic content, speaker) and diverge at low noise levels (phase, micro-prosody, breath, fine texture). This comes from the two lab papers (section 7.6): a datum's modes across noise levels form a coarse-to-fine concept hierarchy.
- **H3 — Generator fingerprints.** Diffusion vocoders, GAN vocoders and flow-matching TTS leave different residual patterns (for example in high bands and unvoiced regions). If so, "which pretrained model reconstructs this clip best" hints at the generator family, which helps explanations.

If a hypothesis fails on dev, that option is dropped and reported as a finding.

### 7.2 Options

Ordered by expected value per hour. Each lists the open-source models it uses.

#### D1 — Vocoder resynthesis residual (tests H1, H3)
- **What:** compute a mel spectrogram, resynthesize the waveform with several pretrained vocoders, and measure how much the audio changes. Include codec round-trips as a second family.
- **Open-source models:**
  - diffusion vocoders: **DiffWave** (supports a fast few-step schedule), **WaveGrad**, **PriorGrad** if checkpoints are available
  - GAN/other vocoders for contrast: **BigVGAN v2**, **HiFi-GAN**, **Vocos**
  - neural codecs: **EnCodec**, **Descript Audio Codec (DAC)**
- **Features per model:** multi-resolution STFT distance, mel L1, per-band error (0–4, 4–8, 8+ kHz), error in voiced vs. unvoiced frames. Plus the argmin model ("reconstructs best with DiffWave") as a categorical feature.
- **Notes:**
  - resample to each model's native rate; match its mel settings exactly or the residual is dominated by mismatch
  - many public vocoder checkpoints are trained on one speaker (e.g., LJSpeech). Use relative errors across models, not absolute error, and check on dev
  - codec and noise augmentation change the residual too, so evaluate on the robustness dev set
- **Cost:** medium (GAN vocoders and codecs are fast; diffusion vocoders depend on step count). Routed to uncertain clips unless throughput allows all.
- **Rubric:** deep-learning detector + spectral analysis.

#### D2 — Mode-path features in embedding space (tests H2)
- **What:** train our own small diffusion model on SSL embeddings of **bona fide clips, including augmented bona fide**, then read each clip's mode path across noise levels.
- **Open-source models:** WavLM or XLS-R for embeddings (from Track A). The diffusion model is a small MLP we train ourselves in minutes.
- **Method:**
  - embeddings (pooled or per-window) → standardize → PCA to 64–256 dims
  - DDPM ε-prediction MLP, linear β schedule, T = 1000
  - for each t on a grid, dense at low t (10, 20, 30, 50, 75, 100, 150, 200, 300, 400): noise to t, run score ascent (Adam, ~100 steps) to a mode x*_t
  - prototype mean m = x*_t / sqrt(ᾱ_t); diagonal covariance from Hutchinson probes, or a fixed per-t variance as the cheap first version
  - features per t: log p(z | c) under the diagonal Gaussian, ‖z − m‖, ascent path length
  - stack across t → curve → small classifier
- **Why train on augmented bona fide:** otherwise noise itself looks anomalous, and the PDF says real clips can be noisy.
- **Cost:** cheap on cached embeddings; can run on every file.
- **Rubric:** deep-learning detector; strongest explanation figure (the curve).

```python
def mode_path_features(z, score_net, alpha_bar, t_grid, n_steps=100, lr=0.05):
    feats = []
    for t in t_grid:
        a = alpha_bar[t]
        x0 = a**0.5 * z + (1 - a)**0.5 * torch.randn_like(z)
        x = x0.clone().requires_grad_(True)
        opt = torch.optim.Adam([x], lr=lr)
        for _ in range(n_steps):
            opt.zero_grad()
            x.grad = -score_net(x, t)      # ascend log p_t
            opt.step()
        x_star = x.detach()
        m = x_star / a**0.5
        var = diag_cov(score_net, x_star, t)   # Hutchinson, or fixed per-t
        feats += [diag_gauss_loglik(z, m, var), (z - m).norm(), (x_star - x0).norm()]
    return torch.stack(feats, dim=-1)
```

#### D3 — Latent audio diffusion denoising curve (tests H2 with a pretrained model)
- **What:** use a pretrained latent audio diffusion model as a general "what audio looks like" prior. Encode the clip, add noise at several t, predict the noise with the unconditional branch (empty prompt), and record the denoising loss per t. That curve works like a likelihood proxy, the same idea as diffusion classifiers.
- **Open-source models:** **AudioLDM 2** (Hugging Face `diffusers`), **Stable Audio Open** (gated; check license). Speech is in their training data, but they're general audio models, so treat this as secondary to D2.
- **Cheap sub-feature:** the VAE reconstruction error alone, without diffusion steps.
- **Cost:** medium to heavy (UNet forward pass per t). Routed.
- **Rubric:** deep-learning detector; also useful for scene manipulation, since the prior covers environmental sound.

#### D4 — Pretrained speech score model (exploratory)
- **What:** score-based speech enhancement models (e.g., **SGMSE+**) are trained on clean real speech in the complex STFT domain. Two uses:
  - as a feature: denoising score-matching loss at several noise levels, and how much the "enhanced" output differs from the input
  - as preprocessing: enhance noisy clips before some forensic modules (prosody, speaker drift) so background noise doesn't swamp them
- **Caveats:** these are conditional models (noisy → clean), not unconditional priors, so the features are indirect. Enhancement can also erase the artifacts the detector needs, so **never** enhance before the DL detector. Lowest priority in Track D.

#### D5 — Hierarchy and composition for explanations (from the lab papers)
- **Channel/generator prototypes:** Gaussian prototypes per channel condition (from our augmentation labels) and per generator family (from external dataset labels and D6), found by mode ascent with an unconditional model trained on all embeddings, or fit directly per subset (faster; start there).
- **Composition:** for a test clip, run the per-dimension submodular selection from the 2605 paper: F(S) = Σ_r max_{j∈S} log N(z_r; m_{j,r}, σ²_{j,r}), greedy with K = 2–4, per-dimension softmax weights with temperature τ ≈ 0.5.
- **Output:** "channel = telephony + noise; closest generator = diffusion vocoder family". Channel dims get explained by channel prototypes, so the authenticity judgment rests on the remaining dims.
- **Basic level:** compute distinctiveness of the bonafide/spoof partition at each t (2609 paper's D(c), mean pointwise mutual information). The peak shows which abstraction level carries the signal. Report it either way.
- **Optional Cobweb/4V** over embeddings as an interpretable, incremental comparison.
- **Limit:** composition interpolates inside the training hull. An unseen generator comes back as "no good generator match", which D2 (real-only) still flags.

#### D6 — Diffusion and flow models as data generators
- **What:** generate modern fakes the provided training set may not cover, so Track A generalizes better.
- **Open-source models:**
  - zero-shot TTS / voice cloning: **F5-TTS** (flow matching), **StyleTTS 2** (style diffusion), **CosyVoice** (flow matching), **OpenVoice**, **XTTS** (check license)
  - vocoders from D1 to re-vocode real speech (copy-synthesis): real content and speaker, vocoder artifacts only. Good for teaching the detector vocoder fingerprints.
  - environmental sound: **AudioLDM 2** / **Stable Audio Open** to generate fake backgrounds, mixed under real speech, for scene-manipulation examples `[TK: confirm these count as spoof]`
- **Recipes:**
  - clone bona fide training speakers from 3–10 s references, speaking held-out text
  - partial fakes: splice 1–3 generated words into real clips at word boundaries (use forced alignment or VAD)
  - launder generated clips with the same codec/noise chain as 6.1
- **Rules:** only clone consenting team members or speakers from datasets whose licenses allow it; keep generated audio internal; tag every generated file with its generator for D5 labels.
- **Cost:** GPU time for generation; run in the background from Phase 1 on.

### 7.3 Suggested order and timeboxes

1. **D2 mode-path** (~2 h). Cheap, uses the cached embeddings, tests H2 fast.
2. **D6 data generation** (~2–3 h setup, then runs in the background). Feeds Track A.
3. **D1 vocoder resynthesis** (~2 h). Start with BigVGAN + EnCodec + DiffWave.
4. **D5 prototypes + selection** (~1.5 h). Mainly for explanations and the demo.
5. **D3 latent diffusion curve** (~1.5 h) if GPU time is available.
6. **D4** only if everything else is done.

### 7.4 Go/no-go gates
- A Track D feature set joins fusion only if it improves dev AUC or EER, **or** clearly improves the robustness subsets (laundered spoof, noisy bona fide, self-recorded real).
- Otherwise it stays out of the final model and goes into the ablation table as "no measurable effect". The PDF asks for that explicitly.
- D6 is judged by whether Track A trained with the generated data does better on dev and on the leave-one-generator-out split.

### 7.5 Compute
- D2: seconds per thousand clips on cached embeddings.
- D1, D3: depends on model and step count. Measure on 500 clips. If the extrapolated time for 77k is too long, run them only on router-selected clips (uncertain T1 score or specific triggers, section 10).
- D6: offline GPU jobs; not part of inference.

### 7.6 Papers behind D2 and D5
- **Wang, Gupta, Zhu, MacLellan.** *Test-Time Compositional Generalization in Diffusion Models via Concept Discovery.* arXiv 2605.07078. Mode ascent on the unconditional score at several noise levels recovers clean-space Gaussian prototypes (via Tweedie). A monotone submodular per-dimension coverage objective selects prototypes that explain a query, composed as a product of experts. Interpolates new primitives inside the training hull, not outside it.
- **Wang, Singaravadivelan, MacLellan.** *Diffusion Models and Concept Formation.* arXiv 2609.13047. Diffusion marginals and Cobweb trees are the same kind of hierarchical Gaussian-prototype model: noise level ↔ tree depth, mode Gaussian ↔ node Gaussian, coarse-to-fine denoising ↔ root-to-leaf categorization. A basic level appears where distinctiveness peaks (t ≈ 150 on MNIST/Fashion-MNIST).

### 7.7 Pitch angle for Track D
- "Deepfakes get the big picture right and the fine detail wrong. We measure that directly with diffusion models, level by level."
- Show the mode-path curve for a real and a fake clip, the "reconstructs best with" result from D1, and the composed explanation from D5.
- Be honest about which hypotheses held.

---

## 8. Open-source model inventory

Verify availability, license and exact checkpoint names in Phase 0. Bake weights into the Docker image.

| Role | Model | Used in | Notes |
|---|---|---|---|
| SSL speech encoder | WavLM Base+ / Large (Microsoft) | A, D2, D5 | main front-end |
| SSL speech encoder | wav2vec 2.0 XLS-R 300M (Meta) | A | alternative front-end |
| SSL encoder | Whisper encoder (OpenAI) | A | optional third front-end |
| Anti-spoof head | AASIST, AASIST-L, SSL+AASIST checkpoints | A | pretrained on ASVspoof; **flip polarity** |
| Anti-spoof baseline | RawNet2, LFCC-LCNN / LFCC-GMM (ASVspoof baselines) | A, B | raw waveform and spectral baselines |
| Speaker embeddings | ECAPA-TDNN (SpeechBrain, VoxCeleb) | B (6.7) | window-level drift |
| VAD | Silero VAD | B | pauses, non-speech segments |
| Prosody | openSMILE (eGeMAPS), Praat/parselmouth | B (6.4) | handcrafted features |
| Diffusion vocoder | DiffWave, WaveGrad, PriorGrad | D1 | resynthesis residual; few-step schedules |
| GAN vocoder | BigVGAN v2 (NVIDIA), HiFi-GAN, Vocos | D1, D6 | resynthesis contrast; copy-synthesis fakes |
| Neural codec | EnCodec (Meta), DAC (Descript) | D1 | round-trip residual |
| Latent audio diffusion | AudioLDM 2, Stable Audio Open | D3, D6 | denoising curve; fake backgrounds; Stable Audio Open is gated |
| Speech score model | SGMSE+ | D4 | exploratory; never before the DL detector |
| Zero-shot TTS / VC | F5-TTS, StyleTTS 2, CosyVoice, OpenVoice, XTTS | D6 | training fakes; check each license |
| Forced alignment | Montreal Forced Aligner or WhisperX | D6 | word boundaries for partial fakes |
| ASR (optional) | Whisper | B, E | transcript for the explainer; repeated-phrase checks |
| LLM explainer | open-weight instruct model (e.g., Qwen or Llama family) via HexLabs or local | E | reads trace JSON, writes explanation |
| Fusion + attribution | LightGBM / XGBoost, SHAP | F | per-file top contributors |
| Signal / file tools | librosa, torchaudio, scipy.signal, PyWavelets, FFmpeg/ffprobe, MediaInfo, SoX, ExifTool | all | — |

Community deepfake detectors on Hugging Face also exist. Evaluate any of them on our dev set before trusting them, and check their training data for overlap with the test conditions.

---

## 9. Compute budget for ~77k files

- Rough volume: 77k × ~5 s ≈ 100+ hours of audio. Check the real average duration.
- Measure each module on 500 clips, extrapolate, then decide what runs on every file and what is routed.
- Expected cost ordering (verify): cheap = metadata, bandwidth, LFCC, D2 on cached embeddings; medium = SSL embeddings, ECAPA windows, eGeMAPS, GAN vocoder/codec round-trips; expensive = diffusion-vocoder resynthesis, latent diffusion curves, fine phase analysis, ENF on long native-rate audio.
- Cache everything keyed by filename. Every experiment reads from the cache.
- GPU for SSL, ECAPA, vocoders, diffusion; CPU pool for ffprobe, openSMILE, Praat, ENF.
- No per-file LLM calls in the batch run. The LLM explains a sample and runs live in the demo.

---

## 10. Agentic router (rubric bonus)

A documented policy picks which T2 analyses run per file from T0 findings and T1 confidence. Every decision goes into the trace.

### 10.1 Example policy

| Trigger | Runs | Why |
|---|---|---|
| Always | metadata, bandwidth, SSL detector, LFCC, D2 | cheap and strongest |
| T1 score 0.2–0.8 (uncertain) | D1 resynthesis, prosody, speaker drift, splice | spend effort where the call is close |
| Still uncertain after that | D3 latent diffusion curve | most expensive, last |
| Declared codec ≠ effective bandwidth, or lossy container | compression forensics | possible laundering |
| Clip > 6 s, or window scores vary a lot | speaker drift, splice | room for edits |
| Non-speech > 1 s and native rate ≥ 16 kHz | ENF, background and reverb consistency, D3 | enough material; scene checks |
| Low-frequency roll-off or doubled reverb | replay checks | replay attack |
| Metadata inconsistency flags | full metadata report + compression | possible container spoof |
| T1 very confident (< 0.05 or > 0.95) | stop, unless a metadata flag fires | save compute |

- Fusion treats "not run" as missing; train it on dev data routed the same way.
- Tune thresholds for accuracy vs. compute; report both.
- Stretch: a learned policy that runs the analysis with the largest expected change in score.

### 10.2 Trace format (per file)

```json
{
  "filename": "file2.mp3",
  "cm_score": 0.91,
  "container": {"format": "mp3", "sr": 44100, "bitrate": 128000, "encoder": "Lavf60"},
  "analyses_run": ["metadata", "bandwidth", "ssl", "lfcc", "mode_path", "compression", "vocoder_resynth", "prosody"],
  "router_reasons": {
    "compression": "declared 44.1 kHz, energy stops at 3.9 kHz",
    "vocoder_resynth": "ssl score 0.64 in uncertain band"
  },
  "findings": [
    {"module": "compression", "finding": "telephony band → MP3 transcode chain"},
    {"module": "mode_path", "finding": "fits real speech at coarse levels, diverges below t=50"},
    {"module": "vocoder_resynth", "finding": "lowest resynthesis error with a diffusion vocoder"},
    {"module": "prosody", "finding": "no breath sounds in 4 pauses; low F0 variance"}
  ],
  "top_contributors": [["ssl", 0.41], ["mode_path", 0.22], ["vocoder_resynth", 0.11]]
}
```

`top_contributors` comes from SHAP on the fusion model.

---

## 11. Build plan

Deadline `[TK]`. Assumes the 36-hour window ends Sunday; adjust hours.

**Phase 0 — data, plumbing, submission skeleton (~1.5 h)**
- Section 2 checklist. Loader (any format → 16 kHz mono + native-rate copy, decode-failure handling).
- TSV writer + validator; dummy TSV (all 0.5) from the prefilled list, validated.
- Download and smoke-test the open-source checkpoints.
- Shortcut sanity test (section 13).

**Phase 1 — core detector + first real TSV (~4 h)**
- Cache SSL embeddings for train, dev, test.
- Head with augmentation; LFCC model; metadata + bandwidth features.
- Simple fusion + calibration → **first valid TSV** (fallback).
- Start D6 generation jobs in the background.

**Phase 2 — forensic breadth (~5 h, parallel)**
- Compression, prosody, ENF/environment, speaker drift, splice. Each writes features + findings to cache and trace.

**Phase 3 — diffusion track + router (~5 h, parallel)**
- D2 → go/no-go. D1 on a dev subset → go/no-go. D5 prototypes and selection.
- Retrain Track A with D6 data if it's ready; compare on dev.
- Router policy + trace JSON.

**Phase 4 — fusion, ablations, draft review (~3 h)**
- Full fusion on routed dev data, calibration, SHAP, ablation table.
- **Send the draft TSV to NSA.** Leave at least ~6 hours after it.

**Phase 5 — fixes, Docker, docs, final TSV (~4 h)**
- Act on feedback; build and test Docker from a clean clone; README; final TSV.

**Phase 6 — demo and pitch (~2 h)**
- LLM explainer live on a few files; figures (mode-path curves, traces, splice timeline, ablation table).

### Team split (4 people)

| Person | Owns |
|---|---|
| A | Data loading, augmentation, embedding cache, TSV writer/validator, Docker |
| B | Track A detector, LFCC, fusion, calibration, SHAP, submissions |
| C | Track D: D2, D1, D5 (and D3 if time); router policy |
| D | Track B modules (metadata, compression, prosody, ENF, drift, splice); D6 generation jobs; LLM explainer |

---

## 12. Submission checklist

### 12.1 TSV
- [ ] Header exactly `filename<TAB>cm-score`.
- [ ] One row per file in the prefilled TSV; same filenames with extensions; row count matches (~77k).
- [ ] No duplicates; filename set equals the prefilled set.
- [ ] Every score a float in [0, 1]; no NaN or blanks.
- [ ] Polarity: 1.0 = synthetic.
- [ ] Decode failures get a fallback score from whatever ran (metadata-only model); logged.
- [ ] Tab-delimited, UTF-8, `\n` line endings; named `<teamName>_predictions.tsv`.

```python
def validate_tsv(pred_path, prefilled_path):
    pred = pd.read_csv(pred_path, sep="\t")
    ref = pd.read_csv(prefilled_path, sep="\t")
    assert list(pred.columns) == ["filename", "cm-score"]
    assert len(pred) == len(ref) and pred.filename.is_unique
    assert set(pred.filename) == set(ref.filename)
    s = pred["cm-score"]
    assert s.notna().all() and s.between(0, 1).all()
```

### 12.2 Repo
- [ ] README: approach, architecture diagram, modules mapped to rubric categories, router policy, Track D hypotheses and results, how to run, ablations, what worked and what didn't.
- [ ] Model inventory with licenses and checkpoint sources.
- [ ] `predict.py --input DIR --output FILE [--gpu] [--explain N] [--no-diffusion]`.

### 12.3 Docker
- [ ] `docker run -v /path/to/test:/data -v /path/to/out:/out <image>` writes `/out/<team>_predictions.tsv` with no other config.
- [ ] All weights inside the image; no internet at run time.
- [ ] CPU works; GPU used if available. Document runtimes. A `--no-diffusion` flag skips Track D for fast CPU runs.
- [ ] Tested from a clean clone on a second machine.

---

## 13. Shortcut and leakage checklist

- [ ] Trivial model on duration, RMS, silence ratio, sample rate, codec. If it scores high, normalize those away for the DL detector and decide deliberately whether to keep them in fusion.
- [ ] Leading/trailing silence by class (a known earlier ASVspoof issue).
- [ ] Sample rate, bit depth, codec, container, loudness by class.
- [ ] Encoder tags and MAC times by class.
- [ ] Speaker overlap between train and dev.
- [ ] Noise present in only one class → augment both.
- [ ] Self-generated fakes (D6) all from one pipeline can add their own shortcut (e.g., identical loudness or sample rate). Normalize and launder them like everything else.

---

## 14. Evaluation and "what worked"

### 14.1 Splits
- Dev split from training data, stratified.
- Robustness dev set: dev clips through unseen codecs, band-limiting, noise, reverb.
- Leave-one-dataset-out or leave-one-generator-out if external or D6 data is used.
- Self-recorded real clips (cartoon voices, fast speech, whispering, phone speaker). Should score low.
- Separate calibration split.

### 14.2 Metrics
- ROC-AUC, EER; log loss, Brier, reliability plot.
- Per-condition breakdown: clean, codec, telephony, noisy, reverb, self-recorded, partial fakes.
- Official metric once confirmed. `[TK]`

### 14.3 Ablation table (fill in)

| Configuration | AUC | EER | Log loss | Notes |
|---|---|---|---|---|
| SSL detector only | | | | |
| + D6 generated training data | | | | does generated data help generalization? |
| + LFCC | | | | |
| + metadata / container | | | | watch for shortcut reliance |
| + compression forensics | | | | |
| + prosody | | | | |
| + ENF / environment | | | | |
| + speaker drift | | | | |
| + splice | | | | |
| + D2 mode-path | | | | tests H2 |
| + D1 vocoder resynthesis | | | | tests H1 / H3 |
| + D3 latent diffusion curve | | | | |
| Full, run-everything | | | | |
| Full, routed | | | | compute saved vs. accuracy lost |

Also report drop-one-module deltas and mean |SHAP| per module. Modules with no effect stay in the table as "no measurable effect".

---

## 15. Demo and pitch

- **One line:** "A forensic analyst's workflow, automated: cheap checks on every file, deeper analyses where they're needed, and a trace that says why."
- Show:
  1. A file goes in; router decisions appear step by step; score and explanation come out.
  2. A laundered fake: compression forensics finds the transcode chain; the detector still flags it.
  3. A noisy real clip: channel prototypes explain the noise; score stays low.
  4. A partial fake: speaker drift + splice timeline.
  5. Track D: mode-path curves (real vs. fake) and "reconstructs best with a diffusion vocoder".
  6. The ablation table.
- Name the rubric categories when showing modules.

---

## 16. Risks and fallbacks

| Risk | Sign | Fallback |
|---|---|---|
| Score polarity flipped | spoof scores lower on dev | section 3 check; fix before fusion |
| Noisy real clips flagged | high scores on augmented bonafide dev | augment both classes; D2 trained on augmented bonafide; D5 channel prototypes |
| Laundered fakes missed | low scores on transcoded spoof dev | codec augmentation on spoof; compression forensics |
| Metadata shortcut | fusion leans on encoder tags | cap or drop metadata features; report it |
| H1/H2/H3 fail | no dev separation | drop that Track D option; report the negative result |
| Vocoder mismatch dominates D1 | residual tracks sample rate or mel settings, not authenticity | match settings exactly; use relative errors |
| Checkpoint won't download or license blocks use | — | swap for another model in the same row of section 8 |
| D6 data adds its own shortcut | model detects "our generator" only | normalize and launder; leave-one-generator-out check |
| 77k files too slow | extrapolated runtime too long | route expensive modules; `--no-diffusion`; batch; cache |
| Docker fails for judges | — | clean-clone test; bundled weights; CPU path |
| TSV errors | validator fails | build validator in Phase 0 |
| Draft review too late | — | send by end of Phase 4 |
| Out of time | — | Phase 1 TSV is always ready |

---

## 17. Open questions

- Detection metric? `[TK: ask NSA]`
- External datasets and self-generated fakes allowed? `[TK: ask NSA]`
- Are scene-manipulated clips labeled spoof? `[TK: ask NSA]`
- Prefilled TSV location; does the test set match its filenames 1:1? `[TK]`
- Deadlines and team name. `[TK]`
- D2: pooled or per-window embeddings? Which SSL layer?
- D2: fixed per-t variance or Hutchinson covariance? Start cheap.
- D1: which diffusion vocoder checkpoints are available at 16 or 22 kHz, and how many steps are affordable?
- D3: AudioLDM 2 or Stable Audio Open (license, speed)?
- Router thresholds: tune for accuracy, or for a fixed compute budget?
- Which LLM for explanations?

---

## 18. Additional comments

- 60% of the grade is one number. Get Track A with good augmentation submitted before any Track D work.
- Track D is worth doing because it adds signals the SSL detector may not use (reconstruction behavior, level-by-level fit) and gives the clearest explanation figures. But each option has to earn its place on dev.
- Use pretrained open-source models wherever possible. The only model we train from scratch in Track D is the small MLP in D2, which takes minutes.
- "Noisy real clips" is stated outright in the PDF. Augment both classes, and train any real-only model (D2) on augmented bona fide.
- Metadata is double-edged: a scoring category and a likely trap. Use it, measure it, report it.
- Keep the LLM out of the scoring path. It orchestrates the demo and writes explanations.
- Build the trace format early so every module, including Track D, writes to it from the start.
- Negative results are still slides: "we tested whether fakes reconstruct more easily through diffusion vocoders; here's what we found."
- Record our own clips early (phone playback for replay, cartoon voices, fast speech, whispering). Cheap, and good demo material.

---

## 19. References

- HEARSAY: The Audio Authentication Challenge (NSA challenge brief, HackGT 13).
- Wang, Gupta, Zhu, MacLellan. *Test-Time Compositional Generalization in Diffusion Models via Concept Discovery.* arXiv 2605.07078 (2026). https://arxiv.org/pdf/2605.07078
- Wang, Singaravadivelan, MacLellan. *Diffusion Models and Concept Formation.* arXiv 2609.13047 (2026). https://arxiv.org/pdf/2609.13047
- Sclocchi, Favero, Wyart. *A phase transition in diffusion models reveals the hierarchical nature of data.* PNAS (2025).
- Barari, Lian, MacLellan. *Incremental concept formation over visual images without catastrophic forgetting* (Cobweb/4V). ACS (2024).
- DIRE: diffusion reconstruction error for detecting diffusion-generated images (idea behind D1).
- DiffWave, WaveGrad, PriorGrad (diffusion vocoders); BigVGAN, HiFi-GAN, Vocos (GAN/other vocoders); EnCodec, DAC (neural codecs).
- AudioLDM 2, Stable Audio Open (latent audio diffusion); SGMSE+ (score-based speech enhancement).
- F5-TTS, StyleTTS 2, CosyVoice, OpenVoice, XTTS (open TTS / voice cloning).
- ASVspoof and ADD challenge overview papers; AASIST, RawNet2/3; WavLM, wav2vec 2.0 XLS-R, Whisper; ECAPA-TDNN.
- Datasets: ASVspoof 2019 LA / 2021 LA+DF / ASVspoof 5, WaveFake, In-the-Wild, MLAAD, ADD; VCTK, LibriSpeech, VOiCES.
- Tools: librosa, torchaudio, scipy.signal, PyWavelets, FFmpeg/ffprobe, MediaInfo, SoX, Praat/parselmouth, openSMILE, ExifTool, SpeechBrain, Silero VAD, LightGBM, SHAP.
- DARPA SemaFor program (semantic forensics).
---

## Addendum — literature review, verified facts and plan changes (Sat Sep 26 2026, night)

Added after the move to MPCDF Raven. `plans/MASTER_PLAN.md` sequences all tracks; this addendum records what changes
for Track A (detector), fusion and Track D. Numbers are from the cited sources or from our own runs where marked.

### A.1 Verified facts that change this plan
- **Metric.** The organizers' package (`data/HackGTMinDCF`) is ASVspoof 5's evaluation code with `Pspoof = 0.5`,
  `Cfa = 4` (was 0.05 / 10). minDCF = min over thresholds of P_miss(bona fide) + 4·P_fa(spoof), i.e. in our polarity
  FPR on reals + 4·FNR on fakes; `hearsay/metrics.py` matches their code exactly. EER at its own threshold costs 5×EER,
  so the leader's **0.0584** needs roughly ≤1.5% missed fakes at ≤5.8% false alarms: the hardest ~1% of fakes decide
  the ranking. Calibration does not change minDCF; it matters for fusion and actDCF only [S21].
- **External data allowed** (the brief lists ASVspoof, WaveFake, In-the-Wild, MLAAD, ADD, VCTK, LibriSpeech, VOiCES).
- **Test pipeline (our forensic audit, `reports/forensics/audit.md`).** Every test file is the same ffmpeg-4.2 WAV
  (`Lavf58.29.100`). 98.0% of test lengths are exact multiples of 512 samples at 22.05 kHz (real corpora: 0–2.5%),
  clips start trimmed and end hard-cut, and the 7.5–8 kHz band is ~44 dB down with a −20 dB edge at 7.39 kHz: a
  librosa-style 22.05 kHz pipeline (trim/crop in 512-sample frames) resampled to 16 kHz with a resampy/kaiser-class
  filter. It applies to both classes, so it is augmentation material, never a feature. Our canonical view already
  removes > 7 kHz; training crops are now start-anchored with p = 0.5 (`audio.START_ANCHOR_P`).

### A.2 Literature: SSL detectors (for Track A)
- **Start from AntiDeepfake** (NII; wav2vec 2.0/HuBERT post-trained on 56k h real + 18k h artefact speech, mean-pool +
  linear head, logits [fake, real], input standardized per clip) [S1, S2]. Zero-shot In-the-Wild (ITW) EER:
  XLS-R-2B 1.23%, XLS-R-1B 1.35%, MMS-1B 1.82%, W2V-Large 1.91%, MMS-300M 2.90% [S2]. ITW clips average 4.3 s, like
  ours [S18]. Run through `transformers` via `hearsay/antideepfake.py` (fairseq 0.12.2 does not install on current
  Python); the key map is strict (every tensor placed, shapes checked).
- **Contamination.** AntiDeepfake's post-training set includes ~140 h of DiffSSD's ~146 h of fakes, plus MLAAD,
  ASVspoof 5, DFADD, SpoofCeleb, CodecFake and vocoded LibriTTS/VoxCeleb2 [S1]. **DiffSSD/LibriSpeech validation of
  these backbones is optimistic.** Use an uncontaminated dev set as well: ITW (held out in [S1]) through the test
  pipeline, plus our held-out speakers.
- **Fine-tune gently** [S1, S8, S22, S24]: AdamW, LR 1e-6 (≤ 5e-6), 3–6k steps, early stopping on dev, 3–4 s crops
  matched to the test (6 s crops beat 4 s in one ASVspoof 5 system [S24]; score whole test clips), weighted CE and
  per-generator caps [S4, S8]. Post-training helped most at 4 s: XLS-R-1B 26.76 → 11.86% EER on Deepfake-Eval-2024
  after fine-tuning, 19.96% without post-training [S1, S2]. Start from default, not `-nda`, checkpoints [S2].
- **Augment both classes**, highest value first: codecs/band-limiting (40.73 → 5.18% EER on LA21 [S27]; codec
  re-synthesis cut ITW 23.7 → 9.6% [S28]), reverb and noise (reverb alone took AASIST from 0.83 to 50.19% [S26]),
  RawBoost only when matched (LA21 4.48 → 0.82% with algos 1+2; mismatched 6.64%) [S8, S39].
- **Diversity beats volume**: 53 generation methods gave 13.03% out-of-domain EER vs 17.5–19.7% for larger
  single-source sets [S40]; vocoded real speech as extra fakes: ITW 6.78% vs 13.52% [S10].
- **Fusion** of 3–6 diverse systems: most top ASVspoof 5 systems were ensembles; average z-normalized logits, or
  logistic fusion with effective spoof prior 0.8 (Cfa = 4); keep it low-parameter, dev→eval gaps were large
  (SZU 0.027 → 0.115 minDCF) [S20, S22–S25]. A second model family for diversity: DF_Arena_1B_V_1 (self-reported ITW
  0.91%) [S6].
- **Do not** adapt on the unlabeled test set: AS-norm on a mixed cohort took ITW EER from 11.18 to 60.16%, and
  pseudo-labels reinforce confident misses on the hard fakes that dominate our metric [S34].
- **DiffSSD itself** [S30]: DiffGAN-TTS, PlayHT and UnitSpeech are its test-only generators (use them for our
  unseen-generator split); trained on the 7 seen generators, Wav2Vec2 reached 3.00% and PaSST 3.53% EER; the hard
  tail is ElevenLabs (PaSST 73% accuracy) and PlayHT (86%); AntiDeepfake-2B caught 81.9% of ElevenLabs fakes [S4].

### A.3 Literature: Track D (diffusion) — evidence-based ranking
1. **D6-R: resynthesize real clips with the H1 reconstructor bank, label them spoof, pair with the original** — the
   best-supported option: vocoded training data + RawBoost took ITW EER 26.65 → 7.55% [D-W23]; vocoder/codec
   pseudo-spoofs ITW 6.9 → 2.1% [D-MS25]; SemantiCodec hard negatives cut DiffSSD EER 21.59 → 14.48% [D-C26]. Some
   reconstructors hurt (HiFi-GAN hard negatives raised DiffSSD EER to 38.99%) [D-C26], so gate each one.
2. **D6-T: new TTS families** (flow matching first) not already in DiffSSD or AntiDeepfake's data [D-DF, D-Y26].
3. **H1 residuals as a branch fused with SSL**: no minDCF gain in-domain (ASVspoof 5 eval 0.1846 vs 0.1753 baseline)
   but better ITW EER out of domain [D-Mo26]. Pilot only; continue if residual-only AUC ≥ 0.80 on held-out generators.
4. **H3 fingerprints**: explanation/attribution only; detectors generalize within a decoder family, not across
   [D-AF24, D-CF+]. Grad-TTS/ProDiff/DiffGAN-TTS render through HiFi-GAN [D-DoC].
5. **H2 (DDPM on embeddings, AudioLDM 2 curves)**: no speech precedent found; demo only unless a 1-h pilot reaches
   held-out AUC ≥ 0.75 not explained by an SNR baseline.
- Go bar for any Track D block: ≥ 10% relative minDCF drop on held-out generators with a CI excluding 0, and ≤ 5%
  relative in-domain loss, over 3 seeds (ITW EER seed std reached ~5 points in [D-Mo26]). AntiDeepfake already saw
  ~7k h of vocoded speech, so expect smaller gains than published.
- Credible explanations (D5): validate against paired real/resynthesized ground truth [D-GR25], compare deletion to
  random masks [D-APEX], and show prototypes are not silence/speaker/corpus clusters [MU21].

### A.4 Resulting changes
- Track A = fine-tuned AntiDeepfake (`scripts/finetune_ssl.py`, LR default lowered to 2e-6, ≤ 3.2k steps), zero-shot
  rows as the benchmark; the frozen-WavLM head stays as the handoff baseline row.
- Dev evidence = shared split (`splits.shared_split`) + unseen generators (diffgan_tts, playht, unit_speech) + ITW
  through the test pipeline (uncontaminated).
- Final score = low-parameter fusion of 2–4 diverse neural systems (+ DSP only if it helps held-out minDCF).
- Track D: D6-R is the only option worth GPU time tonight, and only with the user's approval.

### References
[S1] https://arxiv.org/abs/2506.21090 · [S2] https://huggingface.co/nii-yamagishilab/xls-r-2b-anti-deepfake ·
[S4] https://aclanthology.org/2026.acl-long.796.pdf · [S6] https://huggingface.co/Speech-Arena-2025/DF_Arena_1B_V_1 ·
[S8] https://arxiv.org/abs/2202.12233 · [S10] https://arxiv.org/abs/2309.06014 · [S18] https://arxiv.org/abs/2203.16263 ·
[S20] https://arxiv.org/abs/2408.08739 · [S21] https://github.com/asvspoof-challenge/asvspoof5/blob/main/evaluation-package/calculate_modules.py ·
[S22] https://arxiv.org/abs/2409.01695 · [S23] https://www.isca-archive.org/asvspoof_2024/rohdin24_asvspoof.pdf ·
[S24] https://arxiv.org/abs/2408.09933 · [S25] https://arxiv.org/abs/2408.10361 · [S26] https://arxiv.org/abs/2408.14712 ·
[S27] https://arxiv.org/abs/2110.10491 · [S28] https://arxiv.org/abs/2405.04880 · [S30] https://arxiv.org/abs/2409.13049 ·
[S34] https://arxiv.org/abs/2606.21584 · [S39] https://github.com/TakHemlata/SSL_Anti-spoofing · [S40] https://arxiv.org/abs/2606.08038 ·
[D-W23] https://arxiv.org/abs/2210.10570 · [D-MS25] https://arxiv.org/abs/2509.26471 · [D-C26] https://arxiv.org/abs/2604.26465 ·
[D-Mo26] https://arxiv.org/abs/2607.26472 · [D-DF] https://arxiv.org/abs/2409.08731 · [D-Y26] https://arxiv.org/abs/2606.08038 ·
[D-DoC] https://arxiv.org/abs/2410.06796 · [D-AF24] https://arxiv.org/abs/2405.04181 · [D-CF+] https://arxiv.org/abs/2501.08238 ·
[D-GR25] https://arxiv.org/abs/2506.03425 · [D-APEX] https://arxiv.org/abs/2605.10153 · [MU21] https://arxiv.org/abs/2106.12914

---

## Addendum E — D5 grounded in the cognitive science of concepts (Sat Sep 26 2026, night; user-approved Track D)

Implemented in `hearsay/concepts.py`, `scripts/run_concepts.py`, `scripts/run_basic_level.py`,
`hearsay/diffusion/prototypes.py`. The embedding space is the fine-tuned detector's own last-layer time-mean,
PCA-whitened (32 dims) on train rows, so concepts describe what the detector actually sees.

### E.1 Theory we operationalize
| Idea (source) | Definition we use | Where |
|---|---|---|
| **Prototype theory** (Posner & Keele 1968; Rosch 1973, 1975) | a category is summarized by its central tendency and graded typicality; members closer to the prototype are more typical | one diagonal Gaussian per source (generator / vocoder / real corpus) and per channel condition; typicality = 1 − χ²_d CDF of the Mahalanobis distance to the concept |
| **Family resemblance** (Rosch & Mervis 1975) | typicality correlates with shared attributes (r = .84–.94) | typicality percentile reported per explanation |
| **Exemplar theory / GCM** (Medin & Schaffer 1978; Nosofsky 1986) and the **varying-abstraction continuum** (Vanpaemel & Storms 2008) | categories as stored exemplars; prototype and exemplar models are the two ends of one continuum | the concept tree spans the continuum: shallow nodes ≈ prototypes, leaves ≈ exemplars |
| **Category utility** (Gluck & Corter 1985; Corter & Gluck 1992) | CU(c) = P(c) Σ_k [P(f_k∣c)² − P(f_k)²]; the level maximizing CU matches the human **basic level** (Rosch et al. 1976) | continuous form below |
| **COBWEB** (Fisher 1987) / **CLASSIT** (Gennari, Langley & Fisher 1989) | incremental hierarchical clustering; each instance is sorted top-down and at every node the operator (add to best child, new child, merge two best, split best) with the highest partition CU wins; for Gaussian attributes CU ∝ (1/K) Σ_k P(C_k) Σ_i (1/σ_ik − 1/σ_ip), σ floored at an **acuity** | `ConceptTree.ifit` (acuity 0.25 in whitened units) |
| **Diffusion ↔ concept formation** (Wang, Singaravadivelan & MacLellan, arXiv 2609.13047) | diffusion marginals and Cobweb trees are the same hierarchical Gaussian-prototype model: noise level ↔ depth, mode Gaussian ↔ node; the basic level is where held-out **D(c) = E_{x∼c}[pmi(x; c)] = KL(p(x∣c) ‖ p(x))** peaks (depth 3 for Cobweb/4V, t ≈ 150 for diffusion on MNIST) | held-out I(X;C) per depth (tree) and I(label; level-t concept) per noise level (DDPM) |
| **Prototype composition** (Wang, Gupta, Zhu & MacLellan, arXiv 2605.07078) | F(S) = Σ_r max_{j∈S} ℓ_{j,r}(x) is a facility-location function (monotone submodular); greedy K ≤ 3; per-dimension softmax weights (τ = 0.5) form a product of experts | `prototypes.greedy_select` / `explain`; the whitened root N(0, I) is the per-dimension baseline, which restores the (1 − 1/e) greedy guarantee for possibly negative log-likelihoods (Nemhauser, Wolsey & Fisher 1978) |

### E.2 What we compute and report
1. **Basic level, both definitions** (they can disagree): (a) the depth whose held-out I(X;C) = mean pmi is highest
   (DMCF), (b) the path node maximizing P(c)·KL(c ∥ root) (the lab code's `get_basic`), plus category utility,
   I(fake; C_d) and I(source; C_d) per depth; and the diffusion counterpart: an unconditional DDPM over the same space
   and I(real/fake; level-t mode concept) over t ∈ {10 … 800}.
2. **Prototype scores** gated like any detector (val / holdout / In-the-Wild, clean + augmented): the concept-tree
   posterior and the prototype-mixture LLR log Σ_fake π_j N(x; m_j, S_j) − log Σ_real π_j N(x; m_j, S_j).
3. **Explanations** per test clip: basic-level concept (training make-up by source, typicality), deepest concept,
   the composed prototypes (e.g. "channel = telephony; closest generator = xtts_v2"), and a **novelty flag** when no
   prototype explains the clip better than 99% of training clips (Cobweb's *create*, Anderson's new-cluster prior).
4. **Checks** (so explanations are not voice/corpus clusters in disguise): NMI of basic-level concepts with speaker,
   source, native sample rate and duration, versus with the real/fake label; ARI of the basic-level partition across
   three insertion orders (COBWEB is order-sensitive; Fisher 1996).

### E.3 Known confounds (DiffSSD) and how they show up
The LJ-voice systems (Grad-TTS, ProDiff, WaveGrad 2, DiffGAN-TTS) all speak with the LJSpeech voice; the cloners use
10 LibriSpeech speakers; native rates differ (16/22.05/24/44.1 kHz); fake and real clips read different texts. A
"Grad-TTS prototype" can therefore be "LJ voice + text domain". Our mitigations: the canonical view (rate, band, level,
duration equalized), speaker-matched real LJSpeech and LibriSpeech, and the leakage check above. First measurement
(frozen WavLM space, smoke test): basic-level concepts carry more speaker (NMI 0.28) than real/fake (0.10)
information, and the prototype-mixture LLR (val AUC 0.93) beats the tree posterior (0.76) — the fine-tuned space is
where the module is meant to operate.

### References (E)
Posner & Keele 1968 https://doi.org/10.1037/h0025953 · Rosch 1973 https://doi.org/10.1016/0010-0285(73)90017-0 ·
Rosch 1975 https://doi.org/10.1037/0096-3445.104.3.192 · Rosch & Mervis 1975 https://doi.org/10.1016/0010-0285(75)90024-9 ·
Rosch et al. 1976 https://doi.org/10.1016/0010-0285(76)90013-X · Medin & Schaffer 1978 https://doi.org/10.1037/0033-295X.85.3.207 ·
Nosofsky 1986 https://doi.org/10.1037/0096-3445.115.1.39 · Vanpaemel & Storms 2008 https://doi.org/10.3758/PBR.15.4.732 ·
Gluck & Corter 1985 (Proc. CogSci 7, 283–287) · Corter & Gluck 1992 https://doi.org/10.1037/0033-2909.111.2.291 ·
Fisher 1987 https://doi.org/10.1007/BF00114265 · Gennari, Langley & Fisher 1989 https://doi.org/10.1016/0004-3702(89)90046-5 ·
Fisher 1996 https://doi.org/10.1613/jair.276 · Anderson 1991 https://doi.org/10.1037/0033-295X.98.3.409 ·
Love, Medin & Gureckis 2004 https://doi.org/10.1037/0033-295X.111.2.309 · Kruschke 1992 https://doi.org/10.1037/0033-295X.99.1.22 ·
Cobweb/4V https://arxiv.org/abs/2402.16933 · DMCF https://arxiv.org/abs/2609.13047 · TTCG https://arxiv.org/abs/2605.07078 ·
Sclocchi et al. https://arxiv.org/abs/2402.16991 · Nemhauser, Wolsey & Fisher 1978 https://doi.org/10.1007/BF01588971 ·
Snell et al. 2017 https://arxiv.org/abs/1703.05175 · ProtoPNet https://arxiv.org/abs/1806.10574 ·
lab code https://github.com/Teachable-AI-Lab/cobweb · https://github.com/cmaclell/concept_formation
