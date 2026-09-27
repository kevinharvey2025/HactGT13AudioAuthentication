# Handoff: diffusion / neural track (plans/diffusion_cf_prompt.md)

For the next Claude agent picking up this track. Read this whole file before touching code or starting jobs.
State as of **Sat Sep 26 2026, ~20:15 local**. The previous session was paused on the user's request.

---

## 1. What you are continuing

The repo is our HackGT 13 entry for the NSA **HEARSAY** challenge: score any audio file 0.0–1.0 (1.0 = synthetic), explain why, and ship a GitHub repo with README, a Docker image and a prediction TSV. The brief is `HackGT13_Hearsay_Audio Authentication.pdf`.

Two plans live in `plans/`:

- `plans/diffusion_cf_prompt.md` is **this track**. It covers Track A (SSL detector), B (forensic modules), C (agentic router + trace), D (diffusion models D1–D6), E (LLM explainer) and fusion/TSV/Docker.
- `plans/signal_processing_prompt.md` is a **separate DSP-only track** being built by *another* Claude session in the same working directory (section 8). Don't implement DSP-track work here.
- `plans/AASISTand_AntiDeepfake_prompt.md` appeared during the session. It's untracked and not ours, so leave it alone.

### User instructions so far (in order; all still in force)
1. "Run diffusion prompt plan. The data is located at kharvey33/DiffSSD on Hugging Face."
2. "The test data is also in my downloads." → `~/Downloads/HackGTHearsayTesting/`
3. **"Make sure you don't run the diffusion model. Just implement it."** The previous session read this as: no Track D code (D1–D6) gets executed. That means no DDPM training or mode ascent, and no running or downloading DiffWave, AudioLDM 2, SGMSE+, vocoders or codecs for D1, D5 fits or D6 generation. Static checks (`py_compile`, imports, signature inspection, reading model cards/source) were fine. **Ask the user before running any Track D code.** Track A (WavLM embeddings + head) is not a diffusion model, and the user allowed it to keep running.
4. **"Make sure that you are committing to the repo."** Commit work as you go (section 9 has the git protocol).
5. "Pause this now and make a comprehensive handoff .md document" (this file).
6. "Also finish writing your commits and push."

---

## 2. Environment

| Thing | Value |
|---|---|
| Machine | Apple M4, 10 cores, **24 GB RAM (often under memory pressure: other apps + the DSP session's joblib workers)**, no CUDA; PyTorch uses **MPS** |
| Python env | conda env **`hearsay`** at `/opt/homebrew/Caskroom/miniconda/base/envs/hearsay` (Python 3.11) |
| Always do first | `export PATH=/opt/homebrew/Caskroom/miniconda/base/envs/hearsay/bin:$PATH` (puts the env's `python`, `ffmpeg`, `ffprobe` first) |
| Key packages | torch 2.14 (MPS), torchaudio 2.11, transformers 5.17, pandas **3.0.6**, scikit-learn 1.9, lightgbm 4.7 (+ conda `llvm-openmp`, needed for libomp), shap 0.51, librosa 0.11, soxr, soundfile, speechbrain 1.1.1, vocos, diffusers 0.40, accelerate, tabulate, pyarrow, tqdm |
| ffmpeg | 6.1 from conda-forge (encoders: libmp3lame, aac, libopus, pcm_mulaw/alaw, g722, g726, mp2, wmav2, ac3; **no** AMR/GSM encoders) |
| HF auth | logged in as `kharvey33` (token in `~/.cache/huggingface`) |
| Not yet written | `requirements.txt` / `environment.yml`: create them from the list above |

The DSP session uses its own `.venv/` (pip). Don't use or modify it.

---

## 3. Data (all under gitignored `data/`, mostly symlinks)

| Path | What | Notes |
|---|---|---|
| `data/DiffSSD/metadata.csv`, `README.md` | copied from HF `kharvey33/DiffSSD` (private) | 70,242 rows: 70,000 fakes from 10 generators + 242 reals |
| `data/DiffSSD/generated_speech` → `~/Downloads/DiffSSD/generated_speech` | **complete** local DiffSSD (all 10 generators) | **The HF copy is incomplete**: only diffgan_tts, grad_tts, elevenlabs, openvoicev2 and playht (3,500/5,000). pro_diff, wavegrad2, unit_speech, xtts_v2 and your_tts are missing. Local files are sha256-identical to HF where both exist (spot-checked) |
| `data/DiffSSD/real_speech` → `~/Downloads/resampled` | the organizers' 242 real LJSpeech clips | 16 kHz PCM16, RIFF `ISFT=Lavf58.29.100` |
| `data/hearsay_test` → `~/Downloads/HackGTHearsayTesting` | **NSA sample test set: 1,671 WAVs** + `HGT_Hearsay_score_template.csv` | template is **tab-separated** despite `.csv`: header `filename<TAB>cm-score`, prefilled 0.006. Its filename list is the source of truth for the TSV. (The plan's "~77k entries" does not match this sample set) |
| `data/external/librispeech_10spk/<spk>/<utt>.flac` + `index.csv` | 1,152 real LibriSpeech train-clean-360 utterances (~4 h) from **the 10 speakers DiffSSD's cloning systems imitate** (100, 1487, 2061, 3654, 4490, 5448, 6167, 6575, 7995, 8848) | fetched by `scripts/fetch_librispeech_speakers.py` (only the needed parquet row groups of `openslr/librispeech_asr`). Gives **speaker-matched** reals for the clone fakes |
| `data/external/LJSpeech-1.1/` | full LJSpeech (13,100 clips, 22.05 kHz) + pipe-separated `metadata.csv` (no header) | speaker-matched reals for the 4 LJ-voice TTS generators. The `.tar.bz2` is also there (2.6 GB, deletable) |

DiffSSD generators: **LJ-voice TTS** (trained on LJSpeech): `diffgan_tts, grad_tts, pro_diff, wavegrad2` (22.05 kHz wav). **Voice cloning** of the 10 LibriSpeech speakers: `elevenlabs` (44.1 kHz mp3), `openvoicev2` (22.05 kHz, 5 accent styles), `playht` (24 kHz, stored as mp3 despite the `.wav` name), `unit_speech` (22.05 kHz), `xtts_v2` (24 kHz), `your_tts` (16 kHz).

The DSP session reads `data/external/` read-only. Don't move or rewrite it.

---

## 4. Key findings (put these in the README; several are presentation material)

### 4.1 The provided data has a perfect, useless format shortcut
- The 242 real clips are ffmpeg's **default resample** of the LJSpeech originals: `ffmpeg -i LJ.wav -ac 1 -ar 16000 pcm_s16le` reproduces them to corr 1.000000, ≤3 LSB. **Every** test file carries the same `Lavf58.29.100` tag. Every DiffSSD fake is in its native rate/codec with no tag.
- So sample rate, codec, encoder tag and MAC times separate real from fake perfectly in training and carry nothing on test. **Container/metadata/MAC features are diagnostics and routing inputs only, never classifier inputs.**

### 4.2 The test set differs from all training data (measured on the 1,671 test WAVs)
- All are 16 kHz mono PCM16, one encoder (`Lavf58.29.100`), so container forensics can't separate classes on this set.
- **Peak-normalized**: 92% peak ≥ 0.977 full scale, 40% have full-scale samples, median −18 dBFS.
- **Low-passed near 7.5 kHz**: the 7.5–8 kHz band sits ~44 dB below 6.5–7 kHz, vs ~4 dB for the 242 real training clips. The DSP session measured the same.
- **Darker spectrum**: energy fraction above 4 kHz is median 0.007, vs 0.012–0.049 for every training source, real or fake. It's a genuine distribution shift, handled by the tilt/low-pass/codec augmentations.
- **Durations** 3.02–13.58 s (median 3.41 s, 80% in 3–4 s, hard floor at 3.0 s), vs 5–9 s median in DiffSSD. Clips look like whole utterances filtered to ≥3 s, often hard-cut at the end.
- **Frame-quantized lengths**: only 181 distinct lengths; 98% are whole multiples of **256 samples at 22.05 kHz** (the common lengths are 512 apart). A librosa `trim` simulation did **not** reproduce this or the 7.5 kHz low-pass, so the mechanism is unknown. Since 98% share it, it's not usable as a class signal. Report it as a compression/container-forensics observation only.
- Several test spectrograms show a **steady low-frequency line** (possible added mains hum or "electrical background"), which is relevant to ENF.
- File mtimes span Sep 22 15:43–16:11. Not used.

### 4.3 Canonical view (removes the shortcuts; `hearsay/audio.py`)
Applied identically to both classes and the test set: ffmpeg → 16 kHz mono PCM16 (the organizers' resampler) → trim edge silence (−40 dB, 50 ms pad) → **train/dev only:** crop to a duration drawn from the empirical test-duration distribution → [augmentation for views ≥ 1] → **7 kHz low-pass** (513-tap Kaiser FIR) → **DC removal** (generator-specific DC offsets, e.g. openvoicev2 −0.009) → peak-normalize to U(0.97, 1.0) → **1-LSB TPDF dither** (xtts_v2/your_tts end in exact digital zeros; test never does).

### 4.4 Shortcut checks (`reports/diffusion/shortcuts.md`; grouped 5-fold CV AUC of a small GBM)
| Feature group | raw | canonical |
|---|---|---|
| filesystem MAC times | **1.000** | n/a |
| container (sr, codec, bitrate, tags) | 0.90 | n/a |
| duration | 0.70 | **0.53** |
| edge silence + digital zeros | 0.95 | 0.66 |
| bandwidth | 0.81 | 0.67 |
| level (peak, RMS, DC) | 0.96 | 0.83 → DC removal added afterwards; rerun to update |
| all signal-trivial, canonical, speaker-CV | n/a | 0.85 |

Single features on the canonical view are weak (0.50–0.59); the 0.85 comes from multivariate generator fingerprints (crest factor, spectral tilt), which are partly legitimate signal. **Rerun `scripts/shortcut_checks.py`**: the DC-removal step was added after this report was generated.

---

## 5. Code map (what exists, and what has actually been run)

Legend: **RUN** = executed and verified; **PARTIAL** = executed on a subset/smoke test; **UNRUN** = written and compile-checked only.

| File | Purpose | Status |
|---|---|---|
| `hearsay/config.py` | paths (`HEARSAY_DATA/CACHE/RUNS` env overrides), SR=16k, generator lists, `ffmpeg_bin()`, `device()` | RUN |
| `hearsay/audio.py` | `decode` (ffmpeg), `trim_silence`, `crop`, `lowpass`, `dither`, `peak_normalize`, `TestLikeDurations`, `canonical(x, rng, durations, trim, aug)`, `uid_rng(uid, view, seed)` | RUN |
| `hearsay/manifest.py` | builds the manifest: 10 generators × 1,000 fakes (LJ-voice: the same 1,000 sentence ids for all four; clones: 100 per speaker), 242 organizer reals + 2,000 LJSpeech + 1,152 LibriSpeech reals, 1,671 test rows. label 1 = fake, 0 = real, −1 = test | RUN |
| `hearsay/forensics/triage.py` | T0: ffprobe container/codec/encoder, MAC times, signal stats (cutoff, HF drop, zero runs, clipping, DC) | RUN |
| `hearsay/metrics.py` | AUC, EER, log loss, Brier, stratified bootstrap CI, `check_polarity` | RUN |
| `hearsay/splits.py` | `text_folds` (grouped by sentence, stratified by generator), `speaker_folds` (2 LibriSpeech speakers held out per fold; clone fakes follow their speaker), `logo_splits`, subsets `lj_matched`, `clone_matched`, `nsa_real_only` | RUN |
| `hearsay/augment.py` | codecs (mp3/aac/opus/g711/g726/g722 via ffmpeg pipes), telephony, band-limit, resample chain, spectral tilt, white/pink/brown/babble noise, 50/60 Hz hum, synthetic RIR reverb, clipping; `random_chain` returns params with a `channel` label (the D5 channel labels) | RUN (smoke) |
| `hearsay/views.py` | `ViewMaker(uid, view, is_test)`: the single source of truth for the audio a model sees. View 0 = clean canonical; views ≥ 1 = augmented. Babble pool = real clips only | RUN |
| `hearsay/ssl.py` | `SSLEncoder` (WavLM Base+ / XLS-R 300M): pooled mean+std per layer `[13, 2, 768]` fp16. **Inputs are truncated to multiples of 0.25 s** (`quantize`) because MPS re-plans kernels per shape (189 ms vs 40 ms/clip) | RUN |
| `hearsay/embeddings.py` | `EmbeddingStore` over `cache/emb/<model>/` | UNRUN |
| `hearsay/heads.py` | `LayerMLP` (softmax layer weights → MLP), `Scaled` (standardization baked in), `train_layer_mlp`, `predict`, `layer_weights` | UNRUN |
| `scripts/fetch_librispeech_speakers.py` | targeted LibriSpeech download | RUN |
| `scripts/prepare_data.py` | manifest → `cache/wav16k/<uid>.wav` → `cache/triage.parquet` → `cache/test_durations.npy` | RUN (15,065 clips, 0 decode failures) |
| `scripts/shortcut_checks.py` | section 13 report | RUN (before DC removal) |
| `scripts/extract_ssl.py` | cached embeddings for all clip-views; resumable; length-bucketed batches | **PARTIAL: paused at 15,776 / 41,853 rows** |
| `scripts/train_track_a.py` | Track A CV (text + speaker), subsets, augmented views, optional LOGO, final head → test scores | UNRUN (expect small bugs) |
| `hearsay/diffusion/ddpm.py` | **D2**: `EmbeddingSpace` (standardize, optional speaker-LDA nuisance projection, PCA-whiten), `EpsMLP`, `Schedule` (linear β 1e-4→0.02, T=1000), `train_ddpm` (EMA), `mode_ascent` (Adam, lr ∝ σ_t, cosine decay), `hutchinson_diag_hessian`, `prototype_variance_from_hessian`, `D2Model` (fixed-variance calibration, curve `[n, 10 t, 5 feats]` = loglik, dist, disp, path, DSM loss; `flat()` adds fine-minus-coarse contrast; logistic curve classifier; save/load), `OneClassControls` (Mahalanobis, GMM, kNN) | UNRUN (do not run without approval) |
| `hearsay/diffusion/resynth.py` | **D1**: `Resynthesizer` for hifigan_16k, hifigan_lj, diffwave_lj (6-step), vocos, encodec_6k/1k5, dac_16k, mp3_64k, optional bigvgan_22k; `align`, `residual_features`, `D1Features` (per-model mrstft/mel/band/voiced/unvoiced/SNR, relative-to-mean, argmin; optional `save_dir` for D6 copy-synthesis) | UNRUN |
| `hearsay/diffusion/latent.py` | **D3**: `LatentCurve` on AudioLDM 2 (empty-prompt branch via `pipe.encode_prompt`; UNet called exactly as the pipeline does; scheduler `add_noise`; per-t loss, lo/hi mel-band loss, VAE recon L1) | UNRUN |
| `hearsay/diffusion/score_se.py` | **D4**: `ScoreSE` wrapper for SGMSE+ (clone github.com/sp-uhh/sgmse, 16 kHz checkpoint): DSM curve + enhancement residual; CPU default (complex STFT on MPS) | UNRUN |
| `hearsay/diffusion/prototypes.py` | **D5**: `Prototype`, `fit_direct`, `fit_by_mode_ascent`, `greedy_select` (facility-location F(S) with background N(0,1)), `compose` (per-dim softmax τ, PoE), `explain` (channel share, closest generator, bona fide margin on non-channel dims), `distinctiveness_curve` (I(C; K_t) via k-means + classifier estimate) | UNRUN |

API facts already verified against installed packages and model cards (don't re-derive them):
- SpeechBrain 1.1.1 spells it `mel_spectrogram` (not `mel_spectogram`). `HifiGAN.mel_spectrogram` and `FastSpeech2.mel_spectrogram(min_max_energy_norm=...)` give the **same mel**: that flag only rescales the returned energy. Settings for all SpeechBrain vocoders: hop 256, win 1024, n_fft 1024, 80 mels, 0–8 kHz, power 1, slaney/slaney, compression=True.
- DiffWave: `decode_batch(mel, hop_len=256, fast_sampling=True, fast_sampling_noise_schedule=[0.0001, 0.001, 0.01, 0.05, 0.2, 0.5])`.
- transformers: `DacModel.forward(input_values[B,1,T], n_quantizers=None)`; `EncodecModel.forward(input_values, padding_mask=None, bandwidth=...)`; Vocos: `vocos(y_24k[B,T])`.
- AudioLDM 2 (`cvssp/audioldm2`): DDIM scheduler, scaled-linear β 0.0015→0.0195, **epsilon** prediction, VAE ×4 downsampling (scaling 0.4111, 8 latent channels), mel hop 160 at 16 kHz, 64 bins. `encode_prompt(...)` returns `(prompt_embeds, attention_mask, generated_prompt_embeds)`. UNet call: `unet(z_t, t, encoder_hidden_states=generated_prompt_embeds, encoder_hidden_states_1=prompt_embeds, encoder_attention_mask_1=attention_mask, return_dict=False)[0]`.
- SGMSE+: `ScoreModel.load_from_checkpoint`, `_forward_transform(_stft(y))`, `pad_spec`, `sde.marginal_prob(x0, y, t)`, `forward(x_t, y, t)`, `get_pc_sampler('reverse_diffusion', 'ald', Y, N=30, corrector_steps=1, snr=0.5)`, `to_audio`.

---

## 6. Caches and outputs (gitignored)

| Path | Contents |
|---|---|
| `cache/manifest.parquet` | 15,065 rows: uid, path, label, generator, family, speaker, style, sentence_id, text_group, source, filename |
| `cache/wav16k/<uid>.wav` | ffmpeg-decoded 16 kHz PCM16, full length, no other processing |
| `cache/triage.parquet` | T0 triage per uid |
| `cache/test_durations.npy` | test clip durations (drives the training crops) |
| `cache/signal_raw.parquet`, `cache/signal_canonical.parquet` | trivial signal stats per uid |
| `cache/emb/wavlm_base_plus/pooled.npy` (41,853 × 13 × 2 × 768 fp16, memmap) + `index.parquet` (uid, view, is_test, channel, params JSON, **done**) | **partial**; only rows with `done=True` are valid |
| `cache/pretrained/` | intended home for SpeechBrain checkpoints (nothing downloaded yet) |
| `runs/diffusion/shortcuts.json` | shortcut-check numbers |
| `logs/*.log` | downloads, extraction |

Index order: labeled clips × views 0,1,2 first, then 1,671 test rows (view 0 only). The paused run finished only labeled rows, so **no test embeddings exist yet**.

---

## 7. Next steps, in priority order

The plan's rule: 60% of the grade is one number, so get Track A and a valid TSV done before any Track D work.

1. **Resume the WavLM extraction** (~30 min; allowed, not diffusion):
   `python scripts/extract_ssl.py --model wavlm_base_plus --views 3 --workers 5 --batch 16 > logs/extract_wavlm.log 2>&1 &`
   It resumes from `done` flags and checkpoints every 4,000 items. Check that `index.done.all()` holds when it finishes.
2. **Run Track A** (allowed): `python scripts/train_track_a.py --model wavlm_base_plus` then `--logo`. Untested code, so fix what breaks. Memory: the script holds all views in RAM as fp16 (~1.6 GB). Report clean vs augmented AUC/EER, the `lj_matched` / `clone_matched` / `nsa_real_only` subsets, per-generator results, LOGO, and learned layer weights. Use `metrics.check_polarity` (plan section 3).
3. **Rerun `scripts/shortcut_checks.py`** so the report reflects DC removal.
4. **TSV writer + validator + `predict.py`** (not written yet). Use the plan's `validate_tsv`: header exactly `filename\tcm-score`, one row per template filename, floats in [0,1], no NaN, named `<teamName>_predictions.tsv` (team name unknown; `config.TEAM` reads `HEARSAY_TEAM`). Calibrate Track A scores to probabilities (isotonic or Platt on OOF scores) and write the **first valid TSV** for the 1,671 test files. Also look at the test score distribution: what fraction is flagged, and whether scores correlate with the §4.2 fingerprints (they shouldn't).
5. **Track D run scripts: write them, but don't execute without the user's go-ahead.**
   - `scripts/run_d2.py`, nested protocol per speaker fold. Split training-fold bona fide into A (70%, by text group) for EmbeddingSpace + DDPM (all views) and B (30%) for fixed-variance calibration + curve classifier (B reals vs training-fold fakes). Then score the test fold on all views. Fit `OneClassControls` on A. Embedding: pooled mean(+std) of the layer(s) with the largest Track A layer weights, PCA 64. Try `remove_speaker_dims` 0 vs 8 and compare on the speaker-disjoint folds. Outputs: OOF parquet, curves for the figure (real vs fake mean curves per generator).
   - `scripts/run_d1.py`: subset (~300 per generator + all reals, views 0–1) + all test clips; parquet of `D1Features`; H1/H3 tables (per-model AUC on `lj_matched`, `clone_matched`; argmin-model confusion vs generator). Key H1 check: `hifigan_lj` on the LJ subset (several LJ-voice systems use HiFi-GAN-style vocoders; verify per paper).
   - `scripts/run_d3.py` (AudioLDM 2, ~2k clips + test, fp32 on MPS), `scripts/run_d5.py` (prototypes from `index.parquet` channel labels + generator labels, `explain()` strings, `distinctiveness_curve` needing an unconditional DDPM trained on all embeddings).
   - `hearsay/diffusion/generate.py` (D6, not written): copy-synthesis from `D1Features(save_dir=...)` on real clips; partial fakes by splicing speaker-matched clone segments into LibriSpeech reals at low-energy boundaries (crossfade, keep splice times); laundering via `augment.random_chain(n_ops=2–3)`. The TTS hook (F5-TTS etc.) is optional; check licenses and consent rules (plan 7.2 D6).
   - **Gate script (plan 7.4):** a D feature block joins fusion only if it improves OOF AUC/EER or clearly helps the robustness subsets (augmented views, speaker-matched subsets, LOGO) under the *same folds*. Otherwise it goes in the ablation table as "no measurable effect".
6. **Router (C) + trace JSON (plan §10)**, **fusion** (LightGBM, missing = not run; isotonic calibration; SHAP top contributors), and **template explanations** (E; an LLM is optional and never in the scoring path). Ingest DSP outputs from `runs/dsp/<run>/scores.tsv` (keyed by test filename; columns filename, cm-score, raw sub-scores) when they exist.
7. **Docker** (`Dockerfile` for this track; the DSP track has `Dockerfile.dsp`), `requirements.txt` / `environment.yml`, **`README_DIFFUSION.md`** (approach, architecture, rubric mapping, findings §4, results, ablations, how to run), tests under `tests/` (not `tests/dsp/`).

Open questions for the user or organizers (plan §17, updated): official metric; whether external data (LibriSpeech/LJSpeech) is allowed (the PDF lists them as recommended resources); team name; deadline; whether the final test set is larger than this 1,671-file sample.

---

## 8. The parallel DSP session (same repo, same working tree)

- A second Claude session (`hactgt13audioauthentication-7e`, reachable via `SendMessage` while it's alive) implements `plans/signal_processing_prompt.md`.
- **Its paths; never modify or commit these:** `hearsay_dsp/`, `configs/dsp*.yaml`, `tests/dsp/`, `scripts/dsp_*`, `reports/dsp/`, `Dockerfile.dsp`, `Dockerfile.dsp.dockerignore`, `requirements-dsp.txt`, `README_DSP.md`, `.venv/`, `cache/dsp/`, `runs/dsp/`, `artifacts/dsp/`.
- **Our paths:** `hearsay/`, `scripts/` (non-`dsp_`), `configs/` and `tests/` (non-dsp names), `reports/diffusion/`, `runs/diffusion/`, `artifacts/diffusion/`, `cache/` (non-`dsp` subdirs), `data/external/`, `logs/`, `predict.py`, `Dockerfile`, `requirements.txt`, `environment.yml`, `README_DIFFUSION.md`, this file. README.md is shared; leave it alone unless the user asks.
- It commits to branch **`dsp-track`** using a private `GIT_INDEX_FILE` + `commit-tree`/`update-ref`, so it never moves HEAD or touches the shared index. It treats `data/` as read-only and writes per-file scores to `runs/dsp/<run>/scores.tsv`.
- It shares the CPU (joblib workers at ~70% each). Expect slower CPU-bound jobs and use ~5 data-loader workers, not 8+.

---

## 9. Git protocol (the user asked for commits and a push)

- Commit to **`main`** with explicit pathspecs only, so the other session's untracked files and staged index are never touched:
  `git add <our paths> && git commit -m "..." -- <our paths>`
  **Never** `git add -A`, `git add .` or `git commit -a`. Never switch branches in this working tree; that would move HEAD under the DSP session.
- End commit messages with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Remote: `origin https://github.com/kevinharvey2025/HactGT13AudioAuthentication.git`. Push `main` only (`git push origin main`). The DSP session pushes `dsp-track` itself.
- Commits from the previous session: `fed29a9` Phase 0, `416709e` Track A, `7f7fca4` D2, `7c534cd` D1/D3/D4/D5, plus the commit adding this file.

---

## 10. Gotchas already hit

- **pandas 3**: `groupby(...).apply(...)` drops the grouping columns. Use `groupby(...).sample(n=...)` or keep keys explicitly.
- **MPS shape churn**: variable input lengths make WavLM ~5× slower. Always go through `ssl.quantize` / `pooled_batch`, and batch equal lengths.
- **Memory pressure** (24 GB machine, other apps, DSP jobs): keep embeddings fp16 and avoid upcasting all views to fp32 at once.
- ffmpeg must be on PATH (the env's `bin`), otherwise `subprocess` can't find it.
- Codec round-trips add delay (MP3/AAC priming), so waveform correlation with the input is ~0. Use `resynth.align` before any sample-level comparison.
- HF rate-limits (429) when many parquet shards are scanned in parallel. Keep ≤8 threads and don't run HF downloads concurrently.
- `pgrep -f "a|b"` on macOS matched every process (just a listing, nothing killed). Use `pgrep -P <pid>` for children.
- Canonical crops draw lengths from `cache/test_durations.npy` (181 discrete values), so identical crop lengths across clips are expected, not a bug.
- `views.ViewMaker` builds its babble pool from real clips only, so background talkers never carry spoof audio.
