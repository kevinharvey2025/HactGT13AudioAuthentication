# Roadmap: branch `experimental-1`

`main` holds the submitted system: official NSA minDCF 0.0317, EER 1.44%. This branch expands the evidence, builds
on what already works, and makes cognitive science a first-class part of what we offer. **Reproducibility is the
primary goal:** nothing here counts until it can be rerun from the repository by someone else.

Four rules decide what gets merged back into `main`:
1. **Reproducible first.** Every result is traceable to a commit, input hashes, a locked environment and a command.
   The Docker image must reproduce the submitted scores.
2. **More evidence before more model.** A change is adopted only if it improves the fully held-out In-the-Wild set
   (T1) or the stress curves (T2), with a paired bootstrap interval that excludes 0. It must also cause no
   significant loss elsewhere and keep the Docker regression passing.
3. **Scaffold on what works.** The four things that worked were copy-synthesis fakes, gentle fine-tuning, a
   z-normalized ensemble, and concepts as explanations rather than scores. Each new idea extends one of them.
4. **Interpretable, and tested as such.** The concept layer is the headline. Every interpretability or
   cognitive-science claim (workstream I) has a measurable test and a baseline.

---

## R. Reproducibility (primary)

| # | Item | Done when |
|---|---|---|
| R1 | **Docker, verified (on Raven via Apptainer).** `Dockerfile` builds a CPU image (python:3.11-slim, ffmpeg, CPU torch, `predict.py` entrypoint, `--network none`). Pin the base image by digest. Make weights mountable at run time (`-v weights:/app/artifacts/diffusion:ro`) as well as bakeable. Record the image digest. | The image, run on the NSA test set, reproduces `runs/diffusion/predict_final_v3` (max abs difference ≤ 1e-4) on linux/amd64 and linux/arm64 |
| R2 | **Memory.** Docker Desktop's default VM has 8 GB; fp32 inference peaks at 14.5 GB. `--precision bf16` was added. **Measured:** it agrees (Spearman 0.998, all decisions equal) but peaks at 13.2 GB on IceLake (no native bf16), so it is *not* the fix; next: score one model at a time with the others' weights unmapped, or dynamic int8 on the linear layers | bf16 agrees with fp32 (Spearman ≥ 0.998, decisions ≥ 99.5%) and fits in 8 GB |
| R3 | **Weights distribution.** SHA-256 manifest for `artifacts/diffusion/`, model cards, hosting on the Hugging Face Hub (consumed by the Dockerfile's `WEIGHTS_REPO`) | `predict.py` refuses weights whose hashes do not match |
| R4 | **Locked environments.** `uv pip compile --generate-hashes` lock files for the neural, concepts, DSP and Docker environments; record torch, CUDA and driver versions | a fresh venv from the lock reproduces the unit tests and a 100-clip golden run |
| R5 | **Data manifests.** SHA-256 of every training and evaluation file (pool, copy-synthesis, In-the-Wild), plus `scripts/verify_data.py` | verification passes on Raven |
| R6 | **Provenance.** `evaluate.py` and `run_concepts.py` write `provenance.json` next to their outputs: git commit, input file hashes, package versions | every file in `results/` has one |
| R7 | **Golden regression.** Deterministic synthetic WAVs (seeded tones, noise, chirps: redistributable, unlike challenge audio) with stored expected scores | `tests/test_golden.py` passes wherever the weights are present |
| R8 | **Public per-clip scores.** Per-clip scores of every benchmarked system in `results/scores/`, with `evaluate.py --scores results/scores` | every table in `results/tables.md` regenerates on a laptop, without audio or GPUs |
| R9 | **CI.** A GitHub Actions workflow runs the CPU unit tests on every push, and a Docker build smoke test without weights | green on `experimental-1` |
| R10 | **One command per level.** `make test`; `make eval` (from scores); `mpcdf/pipeline.sh` (the Raven job chain, in order, with dependencies) | documented in REPRODUCE.md |

**R1 status (2026-09-27): passed on linux/amd64.** Image `sha256:ab09aa0b…` (built from commit `1260aef`'s
`predict.py`), run by Apptainer on Raven, scored all 1,671 NSA test clips: max |ΔP| 3.2e-6 against the stored CPU
run, Spearman 0.99999999, every decision identical; 4 shards of ~19 min, 14.6 GB peak each. linux/arm64 is not
verified yet.

Our laptop has 19 GB of free disk, so the image is built without weights (672 MB, linux/amd64) and verified on
Raven. The saved image is converted with Apptainer (`apptainer build hearsay.sif docker-archive://...`) and run with
the weights bind-mounted at `/app/artifacts/diffusion`, on the NSA test set; its output is compared with the stored
CPU run.

## T. Expanding the testing suite

| # | Test set | Size | Why |
|---|---|---|---|
| T1 | **In-the-Wild, fully held out.** The 27,779 clips never used for selection or calibration (all 31,779 exist on Raven) | 27,779 | unbiased out-of-domain estimate with tight intervals; per-speaker breakdown from `meta.csv` |
| T2 | **Parametric stress (dose-response).** Sweeps instead of random chains: RT60 with *real* room impulse responses (OpenSLR 28), SNR with real noise, music and babble (MUSAN), codec bitrate (MP3/AAC/Opus/AMR/GSM), 8 kHz telephony, clip length 1–10 s, and laundering (repeated transcoding, loudness normalization, resampling) | ~8 conditions × 6 levels × 2,000 clips | shows where performance breaks and by how much; reverb is our known weak spot |
| T3 | **Fresh generators.** New fakes from 2025–26 open-weight TTS and voice-cloning models absent from DiffSSD and from AntiDeepfake's training data (e.g. F5-TTS, CosyVoice 2, Kokoro, StyleTTS 2, Parler-TTS, MaskGCT; Qwen3-TTS, which the NSA talk names). Prompted by held-out LibriSpeech speakers and paired with those speakers' real clips | ~6 × 500 | the talk's advice ("test yourself, generate your own synthetic speech"); truly unseen generators |
| T4 | **External benchmarks** (evaluation only, licences permitting): ASVspoof 2019 LA eval, ASVspoof 2021 DF subset, WaveFake, MLAAD (English), FakeOrReal (including its re-recorded split) | subsets of 5–10k | comparability with the literature; replay and re-recording |
| T5 | **Code tests:** Docker smoke test, golden regression (R7), checksum test (R3), determinism test, generator tests for the stress suite | – | the suite grows with every feature |

Every new set goes through `scripts/evaluate.py`, so it gets bootstrap intervals, paired comparisons, breakdowns and
calibration automatically.

## D. Detection, built on what works

| # | Idea | Builds on | Adopt if |
|---|---|---|---|
| D1 | **Real-RIR reverb augmentation** at higher weight, continuing the three copy-synthesis models | augmentation + gentle fine-tuning | reverb minDCF (T2) drops ≥ 30%, no loss on T1 |
| D2 | **More copy-synthesis vocoders:** BigVGAN v2, UnivNet, Parallel WaveGAN / MB-MelGAN, WaveGlow | copy-synthesis (the single biggest win) | T1 and T3 improve |
| D3 | **Short clips:** mixed 1–3 s crops in training; multi-crop averaging at test time for long clips | canonical view | clips under 2 s improve on T1 and T2 |
| D4 | **A fourth ensemble member** (W2V-Large or HuBERT-XL with copy-synthesis) | the ensemble | paired improvement on T1 |
| D5 | **bf16 inference** (see R2) | `predict.py` | agreement criteria in R2 |
| D6 | **RawBoost** (Tak et al. 2022) on top of the channel chain, measured on a new unseen-channel view (100) nobody trains on; protocol and gates fixed in advance ([results/rawboost/decisions.md](../results/rawboost/decisions.md)) | augmentation + gentle fine-tuning | the predeclared gates pass for both seeds |

## I. Interpretability through concept formation (the headline)

The claim to prove: **this is not a black box.** The detector's representation can be read as a hierarchy of concepts
formed the way people form categories (COBWEB). Every decision can be traced to concepts, and to real training clips
a person can listen to. The concept layer reproduces the detector's decisions, and it points at the evidence the
detector actually uses.

| # | What it shows | How | Done when |
|---|---|---|---|
| I1 | **A concept atlas: what the detector learned, globally.** | Walk the COBWEB tree: for each concept, size, synthetic share, sources, channels, category utility, and its most typical training clips (ids in public corpora). Render the top levels as an annotated map | `results/concepts/atlas.md` (+ a page in the artifact) |
| I2 | **Case-based explanations: "this clip sounds like these".** | For each explained clip, the most typical training clips of its concept and of each prototype's concept (nearest to the concept mean). This is exemplar theory made operational [Medin & Schaffer 1978] | every test explanation lists exemplars |
| I3 | **Surrogate fidelity: how much of the black box the readable layer reproduces.** | Agreement (accuracy, κ, rank correlation) between the detector's decisions and the concept layer's, on labelled sets and the NSA test set, with intervals | reported per set; currently κ = 0.98 on the test set at matched flag rate |
| I4 | **Contrastive and counterfactual explanations** ("why synthetic rather than real", Miller 2019). | For each clip, the nearest real concept on its path vs its synthetic prototypes: which dimensions separate them, and how far the clip must move to change concept | faithfulness by deletion, like E6 |
| I5 | **Acoustic names for concept dimensions.** | Correlate each of the 32 concept dimensions with interpretable acoustic measures from the DSP track (spectral tilt and roll-off, band energies, harmonics-to-noise ratio, jitter, pauses, LFCC statistics). Explanations then read "brighter high band, steadier pitch than real speech" | every dimension has its top acoustic correlates with confidence intervals |
| I6 | **Faithfulness, extended.** | E6 on more clips (all views, all sets), plus a sanity check that the explanations of a model with randomized weights change [Adebayo et al. 2018] | passes |

**Cognitive-science capabilities** that come with the concepts, each tested against a baseline:

| # | Claim | Test | Baseline |
|---|---|---|---|
| C1 | **Learns a new family from a handful of examples, without retraining or forgetting.** COBWEB sorts new clips into the tree with no gradient steps [Fisher 1987; Cobweb/4V] | Leave a generator out; insert k = 0…100 of its clips; measure attribution and detection on its held-out clips, and forgetting on the others. Domain version: k labelled In-the-Wild clips, tested on held-out speakers | kNN and logistic regression refitted at every k |
| C2 | **Knows when something is new** (COBWEB's new-concept operator; Anderson's rational model) | Novelty = −max held-out pmi. AUROC for unseen generators and for In-the-Wild vs in-domain | Mahalanobis, kNN distance |
| C3 | **Typicality predicts reliability** (prototype theory: Rosch & Mervis 1975) | Error rate by typicality quintile | – |
| C4 | **Finds the detector's mistakes** for an analyst | Precision@k of errors when In-the-Wild clips are ranked by concept–detector disagreement and novelty | random; detector uncertainty |
| C6 | **Explanations people find useful** (optional user study) | Team members rate basic-level vs leaf vs root explanations, blind to condition | – |

## Sequencing

| Phase | Items | Compute |
|---|---|---|
| **1 (now)** | R1 Docker verified on Raven (Apptainer), R2/D5 bf16, I1 atlas, I3 fidelity, C1–C4, T1 scoring | CPU: ~3 node-hours for T1; the concept experiments take minutes |
| 2 | I2, I4, I5, I6; T2 stress suite; R3–R9 | CPU for the interpretability work; GPU ~2–4 node-hours for T2 |
| 3 | D1–D4 retraining, T3 generation, T4 downloads, C6 | GPU: ~15–25 node-hours |

## Decisions needed

1. **Weight hosting** (R3): a Hugging Face repo, private or public, and a GHCR image. This needs your account.
2. **Pushing `experimental-1`** to GitHub, and when.
3. **External datasets** (T4): evaluation-only use under their licences (MLAAD is CC BY-NC-ND).
4. **GPU budget** for phase 3.
