# Handoff: HEARSAY DSP track (signal-processing-only detector)

**Written:** 2026-09-26, ~19:40 EDT, by the Claude session that built the track.
**For:** the next Claude agent continuing this work.
**Spec you are implementing:** `plans/signal_processing_prompt.md`. The challenge brief `HackGT13_Hearsay_Audio Authentication.pdf` overrides it on any challenge rule.
**Git:** branch `dsp-track`, pushed to `origin` (GitHub `kevinharvey2025/HactGT13AudioAuthentication`). Tip at handoff: this document's commit, on top of `d9e1a3a`.

---

## 0. Status in one screen

| Area | State |
|---|---|
| Package `hearsay_dsp/` (decode, cache, 10 analysis modules, routing, GMM/LR/HGB, calibration, late fusion, bundle, CLI, traces, metrics, augmentation, ablation suite, report renderer) | **Implemented.** 59 tests pass (`tests/dsp`, 56 unit + 3 CLI integration on synthetic fixtures). |
| Manifests with leakage-checked dev/holdout split | **Done**: `runs/dsp/manifests/pool.tsv` (8,394 rows), `hearsay_test.tsv` (1,671 rows). Regenerate with `scripts/dsp_build_manifests.py`; they are gitignored. |
| Feature extraction (cache `cache/dsp/`) | **Test set: 1,671/1,671 done, 0 errors.** Pool: **6,542/8,394 done.** All 6,000 fakes and DiffSSD's 242 reals are done. Remaining: LJSpeech-1.1 700 of 1,000 and LibriSpeech all 1,152 (1,852 files). The extraction was stopped deliberately for this handoff and is resumable; see §8. |
| Ablation suite / model training / holdout evaluation / robustness / test predictions | **Not run yet.** These code paths are unit-tested only on fixtures, and `suite.py`, `bundle.train` on real data, `evaluate`, `dsp_robustness.py`, `dsp_augment_train.py` and `report.py` have never executed on real data. Expect some debugging. |
| Docker image (`Dockerfile.dsp`) | Written, **never built**. The build needs a trained bundle at `artifacts/dsp/model/`, which does not exist yet. |
| README_DSP.md, final report with numbers, figures | **Not written.** |
| Submission TSV | **Does not exist.** Blocker: the model has not been trained yet. |
| Performance numbers | **None established.** Do not claim any until they come from `runs/dsp/*` JSON. |

**Very next command** (resume extraction; cached files are skipped):
```bash
cd /Users/kevinharvey/HactGT13AudioAuthentication
.venv/bin/python -m hearsay_dsp.cli extract --manifest runs/dsp/manifests/pool.tsv \
    --config configs/dsp.yaml --out runs/dsp/extract_pool
```

---

## 1. Shared workspace: read before touching git or files

Another Claude session works in the **same working tree**: session name `hactgt13audioauthentication-54`, the diffusion/neural track. Its handoff is `HANDOFF_DIFFUSION.md` on `main`. Rules we agreed on:

- **Our paths (DSP):** `hearsay_dsp/`, `configs/dsp.yaml`, `tests/dsp/`, `scripts/dsp_*`, `reports/dsp/`, `Dockerfile.dsp`, `Dockerfile.dsp.dockerignore`, `requirements-dsp.txt`, `README_DSP.md`, `HANDOFF_DSP.md`, `.venv/`, `cache/dsp/`, `runs/dsp/`, `artifacts/dsp/`.
- **Their paths (never modify):** `hearsay/`, `scripts/*` without the `dsp_` prefix, other files in `configs/` and `tests/`, `cache/` other than `cache/dsp`, `runs/diffusion/`, `artifacts/diffusion/`, `reports/diffusion/`, `logs/`, `predict.py`, `Dockerfile`, `requirements.txt`, `environment.yml`, `README_DIFFUSION.md`, `HANDOFF_DIFFUSION.md`.
- **`data/` is theirs and read-only for us:** they created the symlinks and downloaded `data/external/`.
- **Git protocol:**
  - They commit their paths to `main` with explicit pathspecs.
  - We commit only to `dsp-track`, using `scripts/dsp_git_commit.sh <message-file>`. It builds the commit through a private `GIT_INDEX_FILE`, so it never switches HEAD, never touches the shared index, and never stages their files.
  - **Never** run `git checkout`/`git switch` in this tree, and never `git add -A`, `git add .` or `git commit -a`.
  - Push with `git push origin dsp-track`.
  - `dsp-track` branched from `main` at `7f7fca4`, so it contains their first three commits as ancestors. Merging `dsp-track` into `main` is the user's call, and the file sets are disjoint.
- You can message that session with the `SendMessage` tool (find it via `ListAgents`). They offered to ingest our per-file test scores for their fusion if we write them keyed by test filename under `runs/dsp/<run>/`.
- Plain files in `plans/`, such as `plans/AASISTand_AntiDeepfake_prompt.md`, belong to the user. Leave them alone.

## 2. Environment

- Machine: Apple M4, 10 cores, 24 GB RAM, macOS; ~84 GB disk free at handoff.
- **Memory pressure is severe:** swap was ~17.6 of 18.4 GB, and load average peaked around 119. The causes are desktop apps (a Teams renderer used ~450% CPU at one point, plus Chrome, VS Code and others) and the other session's WavLM extraction. Pool extraction fell from ~15 to ~2–4 files/s. **None of the throughput measured so far is representative.** Re-benchmark (§10, step 10) and report the load conditions.
- Python: `.venv/` (Python 3.13.13, created from miniconda's `python3.13`). Always run `.venv/bin/python`. Pinned runtime deps are in `requirements-dsp.txt`; the venv also has matplotlib, pytest and huggingface_hub.
- There is no system `ffmpeg`, and none is needed: PyAV (`av`) bundles the FFmpeg libraries, with encoders libmp3lame, aac, libopus and pcm_mulaw verified. The other session's conda env `hearsay` has its own ffmpeg; we do not use it.
- Docker 29.3.1 is installed; the daemon runs linux/aarch64. It has not been used yet.
- Hugging Face CLI is logged in as `kharvey33` (token in `~/.cache/huggingface`).
- Tests: `.venv/bin/python -m pytest tests/dsp -q -p no:cacheprovider` takes ~11 s when the machine is quiet.

## 3. Data (all verified)

| Path | What | Notes |
|---|---|---|
| `data/DiffSSD/` | `metadata.csv` + symlinks: `generated_speech → ~/Downloads/DiffSSD/generated_speech`, `real_speech → ~/Downloads/resampled` | Local staging copy of HF `kharvey33/DiffSSD` (private). 70,242 metadata rows: 70,000 fake from 10 generators + 242 real LJSpeech. All present locally. |
| `data/hearsay_test/` → `~/Downloads/HackGTHearsayTesting` | **HEARSAY test set**: 1,671 WAV (16 kHz, mono, PCM_16) + `HGT_Hearsay_score_template.csv` | The template is **tab-delimited despite `.csv`**, with header `filename<TAB>cm-score` and every score a placeholder `0.006`. It is the submission reference: its filenames and order are authoritative. Team notes said "~77k test entries"; the actual count is **1,671**. |
| `data/external/librispeech_10spk/` | 1,152 FLAC files (16 kHz) from LibriSpeech train-clean-360 for the **10 DiffSSD voice-clone reference speakers** (100, 1487, 2061, 3654, 4490, 5448, 6167, 6575, 7995, 8848); `index.csv` (file, speaker, chapter, utt_id, text, sr, dur) | Downloaded by the other session, raw and untouched. 107–131 utterances per speaker, 23 chapters, median 14 s. CC BY 4.0. |
| `data/external/LJSpeech-1.1/` | All 13,100 original LJSpeech WAVs (22.05 kHz PCM16); `metadata.csv` (pipe-separated, no header, `quoting=3`) | Public domain. The 242 DiffSSD reals are a subset: the other session verified they are plain ffmpeg-default resamples of these files. |
| `cache/dsp/external/pre_trained_LA_LFCC-GMM.mat` | Official ASVspoof 2021 LA pretrained LFCC-GMM | Fetched by `scripts/dsp_fetch_pretrained.py`; the zip's SHA-256 is verified. |

**HF upload status (as of 22:33 UTC):** incomplete, still uploading from `~/Downloads/hf_upload/upload_to_hf.py`.
- Present: 43,745 files, 12.94 GB — diffgan_tts, elevenlabs, grad_tts, openvoicev2, 7 of 10 playht speakers, and real_speech.
- Missing: pro_diff, unit_speech, wavegrad2, xtts_v2, your_tts, and playht speakers 6575, 7995, 8848.
- The local staging copy is complete and identical where comparable: 642 sampled files (all 242 reals + 400 random fakes) match the HF LFS SHA-256s, and `metadata.csv` is byte-identical.
- No duplicate content among the 43,742 HF LFS files.

**External data is an open question** (§12). The challenge PDF lists LibriSpeech and LJSpeech-family sets as recommended bona fide references; the plan says external-data permission is unresolved. The design keeps a DiffSSD-only view (`--where dataset=diffssd`), and the ablation suite includes an audit comparing the two.

## 4. Measured findings (facts; use them in the report)

**4.1 Technical properties separate the classes in training.** Sample: 150 files per generator.

| Source | Native format | Median dur (s) |
|---|---|---|
| real (DiffSSD `real_speech`) | 16 kHz PCM16 WAV; RIFF INFO `ISFT=Lavf58.29.100` | 7.11 |
| diffgan_tts / grad_tts / pro_diff / wavegrad2 (LJ-voice TTS) | 22.05 kHz PCM16 WAV | 6.0–6.5 |
| openvoicev2 / unit_speech | 22.05 kHz PCM16 WAV | 7.3–7.5 |
| xtts_v2 | 24 kHz PCM16 WAV | 7.42 |
| your_tts | 16 kHz PCM16 WAV | 8.12 |
| elevenlabs | 44.1 kHz MP3 (`.mp3`) | 6.65 |
| playht | 24 kHz **MP3 inside a `.wav` extension** | 6.77 |
| HEARSAY test | 16 kHz mono PCM16 WAV; the other session reports the same `Lavf58.29.100` tag | 3.41 (min 3.02, p95 5.11, max 13.58) |

Sample rate, codec, extension mismatch and encoder tag therefore separate real from fake in training almost perfectly, yet carry no information on the test set. All of them are diagnostics only; the shortcut audit in the suite quantifies them.

**4.2 The test set differs from training.** From `scripts/dsp_profile_quick.py`: 120 clips per generator and 600 test clips, measured at 16 kHz after `resample_poly`. Data is in `reports/dsp/profile_quick.csv`; cells are the 10th/50th/90th percentiles.

| Tag | dur s | RMS dBFS | first 100 ms rel. p95 (dB) | last 100 ms rel. p95 (dB) | 7.5–8 kHz vs 6.5–7 kHz (dB) | 4–8 vs 0–4 kHz (dB) |
|---|---|---|---|---|---|---|
| TEST | 3.1/3.4/4.7 | −20.8/−18.0/−15.4 | −45/−30/−23 | −38/−27/−13 | **−48.6/−43.6/−33.9** | −26/−21/−16 |
| real (LJ, 16 kHz) | 4.7/6.9/9.3 | −25.5/−24.0/−22.4 | −27/−7/−1 | −44/−40/−35 | −9.3/−4.4/0.0 | −15/−11/−7 |
| xtts_v2 | 5.7/7.9/11.2 | −19.0/−17.2/−15.0 | −12/−5/−1 | **−191/−189/−188** | −5.9/−4.1/−2.1 | −23/−19/−10 |
| your_tts | 6.2/8.6/13.0 | −20.4/−18.9/−15.3 | −25/−18/−11 | **−190/−188/−186** | −7.6/−6.2/−4.5 | −23/−16/−11 |
| openvoicev2 | 5.7/7.2/11.7 | −28.3/−25.9/−20.4 | −54/−46/−36 | −68/−51/−43 | −41/−34/−27 | −22/−17/−12 |

The full table (all generators, plus digital-silence fraction and occupied bandwidth) can be regenerated from the CSV. Consequences:

- The test set is **low-passed near 7.5 kHz** (occupied bandwidth 7.45–7.60 kHz, 10th–90th percentile), whereas training reals and most fakes reach 8 kHz. So **every classifier feature is capped at 7 kHz** (`features.max_hz`).
- xtts_v2 and your_tts **end in exact digital zeros**; the test set never contains digital silence. Frames below one 16-bit LSB are therefore excluded, never treated as evidence.
- Test clips are **peak-normalized** (median peak 0.999 per the other session), around −18 dBFS, and short (80% are 3–4 s). Speech often runs to the clip edge (hard cuts). Hence gain-invariant features (LFCC c0 dropped), and planned test-like crop/peak-norm conditions.

**4.3 Speaker confound.** DiffSSD's only real speaker is LJSpeech, while the six clone generators imitate 10 LibriSpeech speakers. Trained on DiffSSD alone, a model can learn "LJSpeech voice or recording chain = real". This is why the external matched reals were added, and it is tested by `speaker_audit` in the suite.

**4.4 The official pretrained LFCC-GMM carries a digital-silence component.** `spoofGMM` has a component with c0 mean −130.967 (= log10(eps)·√70, the value of an all-zero frame, verified in a unit test) and variance at the 1e-4 floor. This is the known ASVspoof 2019 LA silence shortcut; on DiffSSD it would reward xtts_v2/your_tts trailing zeros.

**4.5 Official baseline details** (read from `github.com/asvspoof-challenge/2021`, `LA/Baseline-LFCC-GMM`):
- MATLAB `lfcc_bp.m`: 30 ms Hamming window, 15 ms hop (`buffer` with 50% overlap, `nodelay`), NFFT 1024, 70 `trimf` filters over 0–4 kHz, log10(E+eps), orthonormal DCT, 19 coefficients including c0, and deltas `(x[t+1]-x[t-1])/2` applied twice: 57 dims. GMMs are `vl_gmm` with 512 components and 10 iterations; score = mean llk(genuine) − mean llk(spoof).
- The Python variant (spafe) uses 20 coefficients and unnormalized deltas (60 dims), 512 components, `max_iter=10` on every 10th file.
- **Our polarity is the opposite:** `raw_spoof_score = mean ll(synthetic GMM) − mean ll(real GMM)`, and `OfficialPretrainedGMM` negates the official score. A unit test checks this against a direct port of MATLAB `lgmmprob`/`compute_llk`.
- Exact numerical parity with MATLAB itself was **not** tested (no MATLAB or Octave available). Only the building blocks (buffer, trimf, deltas, the DCT convention and the silence value) are verified.

## 5. Design decisions already made (with reasons; do not re-litigate without new evidence)

1. **Analysis path.** Decode to float64 in [-1, 1) (int PCM divided by 2^(bits−1)), mono by mean, `scipy.signal.resample_poly` with a Kaiser(5.0) window to 16 kHz, then round to the 16-bit grid (`decode.quantize_bits: 16`), a no-op for 16 kHz PCM16 inputs like the test set. Native-rate audio is kept for bandwidth and ENF diagnostics.
2. **Decoder policy.** The container is sniffed from magic bytes, not the extension. WAV/FLAC/AIFF/OGG go to soundfile first; MP3/MP4/AAC/other go to PyAV first; the other decoder is the fallback, and both attempts are logged. Streams are chosen by the `first` policy (lowest-index audio stream), or `most_channels` if configured.
3. **Namespaces.** `features` are candidate classifier inputs, selected only through the explicit prefix allowlist `cep. spec. lpc. bg. phase. pros.`. `q.*` holds quality/applicability. `diag.*` holds diagnostics (stream properties, container/RIFF/ID3 tags, MAC times, levels, digital-silence fraction, native bandwidth, absolute median F0, ENF) and **never** enters the classifier.
4. **7 kHz cap** for all classifier features (see 4.2). Band availability: `eligible_max = min(7 kHz, 0.95·native Nyquist, occupied bandwidth)`. Occupied bandwidth is the lower of a level rule (within 50 dB of the 0.3–3 kHz median) and a cliff rule (everything ≥300 Hz above the edge is at least 25 dB below the 700 Hz just under it, edge ≥2.5 kHz). Full-band (0.05–7 kHz) features become NaN (unavailable) when the band is limited; low-band features use the 3.4 kHz telephone band.
5. **Two LFCC front-ends** in one module:
   - `official`: the exact `lfcc_bp` configuration, 0–4 kHz with c0 (57 dims).
   - `custom`: same framing, 0–7 kHz, c0 dropped for gain invariance (57 dims).
   - Frame masks `nonsilent` (RMS ≥ 2^-15) and `active` are stored with them. The GMM frame policy is configurable as `all`, `nonsilent` or `active`; the default is custom + nonsilent + 64 components.
6. **Modules and eligibility** (router in `routing.py`):
   - Core: `container` (diagnostic), `lfcc`, `spectral`.
   - Extended: `lpc` (≥50 active frames), `background` (≥30 low-energy frames), `phase` (≥50 active frames), `prosody` (≥0.5 s active; the module also needs ≥0.5 s voiced and ≥40 pulses for jitter/shimmer/HNR), `enf` (presence check always; continuity only if present at ≥10 dB and ≥10 s long; diagnostic only).
   - `compression` is always `not_applicable`, because no validated bitstream/MDCT analysis exists.
   - Module status is one of `ok / not_applicable / insufficient_signal / error`, always with a reason. Results with status `error` are not cached, so a rerun retries them.
7. **Background discontinuities** use per-window local floors (1 s windows, 0.5 s hop) and a robust local-difference rule: a jump is a candidate if it is ≥ max(absolute minimum, median + 4·1.4826·MAD); floor jumps need ≥6 dB, shape jumps ≥4 dB. Candidates are labelled **"possible discontinuity"**, never "splice".
8. **Phase:** modified group delay without unwrapping (α 0.4, γ 0.9, lifter 30, 12 cepstra, unit-RMS frames), plus IF-jump continuity (phase-vocoder IF, `princarg`). Spikes are reported as "possible phase discontinuity" (diagnostic).
9. **Prosody** uses Praat (parselmouth): F0 in semitones relative to the clip median, so the speaker's pitch level is excluded. Jitter/shimmer/HNR use standard voice-report arguments; pauses are interior low-activity runs of at least 150 ms.
10. **Models.**
    - Components: `gmm` (GMM LLR), `lr_core` (LR on `cep+spec`), and `lr_full` (LR on all enabled groups). LR uses median imputation with missing indicators, standardization and balanced class weights; C is chosen by inner grouped CV (highest AUC, the most regularized within 1e-4).
    - **Late fusion:** logistic regression on out-of-fold component scores, balanced to prior 0.5, so the fused logit is a calibrated log-odds. `core = gmm + lr_core`, `full = gmm + lr_full`. Early fusion (GMM score as an LR feature) is supported in `Detector` but needs nested OOF; late fusion was chosen for cost.
    - **Calibration prior is 0.5** because test prevalence is unknown; `calibration.py` documents how to shift it.
11. **Routing modes.** `all_eligible` (default) runs everything. `confidence` computes `p_core` first and runs extended modules only if `p_core` falls inside a band chosen on training OOF scores: the narrowest band whose OOF AUC is within 0.002 and balanced log loss within 0.01 of the full model.
12. **Splits** (`scripts/dsp_build_manifests.py`, seed 20260926; deterministic hashes, independent of row order):
    - 20% of DiffSSD sentence ids are held out (995 ids).
    - 2 clone speakers are held out entirely (**2061, 5448**), including their LibriSpeech reals and their clones, which are drawn only from held-out sentences.
    - LJ chapters are held out at 20% (dev 41 chapters, holdout 9).
    - LibriSpeech (speaker, chapter) groups are held out at 20% (dev 12, holdout 11).
    - Checks: 0 groups and 0 sentence ids in both splits, and no held-out speaker in dev.
    - Fakes are content-paired: the 4 LJ-voice generators share 600 sentence ids, and the 6 clone generators share 60 ids per speaker. OpenVoiceV2 styles cycle by hash.
    - **Counts:** dev 5,797 (1,573 real = 193 DiffSSD real + 795 LJSpeech-1.1 + 585 LibriSpeech; 4,224 fake = 474 per LJ-voice generator + 388 per clone generator). Holdout 2,597 (821 real = 49 + 205 + 567; 1,776 fake = 126 per LJ-voice generator + 212 per clone generator).
13. **Grouping for folds and bootstrap:** `group_id` = `S:<sentence_id>` for fakes, `LJ:<chapter>` for LJSpeech, `LS:<speaker>-<chapter>` for LibriSpeech. Groups are class-pure. CV uses `StratifiedGroupKFold` with 5 folds. Bootstrap CIs resample groups within class, implemented as multiplicity weights.
14. **Augmentation** (`evaluation/augment.py`): matched transforms applied to both classes, written as 16 kHz PCM16 WAV with a JSON provenance sidecar. `EVAL_CONDITIONS` (robustness) and `TRAIN_CONDITIONS` use deliberately different settings. AAC priming (1,024 samples) is trimmed, and all codec round trips are length-aligned with the source.
15. **Export:** header `filename<TAB>cm-score`, exact reference coverage and order, finite scores in [0,1]. **Export is blocked** (exit code 2, `<out>.failures.tsv`) if any file lacks a valid score; there is no fill-in value.

## 6. Code map

| File | Role | Tested? |
|---|---|---|
| `hearsay_dsp/config.py` | defaults, YAML merge, `MODULE_VERSIONS` (bump to invalidate a module's cache), feature-group prefixes, cache keys | yes |
| `hearsay_dsp/io/manifest.py` | label map (ambiguous labels rejected), manifest read/validate, `select_training_rows` (blocks eval/test/holdout splits) | yes |
| `hearsay_dsp/io/decode.py` | sniffing, soundfile/PyAV decoding, resampling, quantization, `probe_file` | yes |
| `hearsay_dsp/io/cache.py` | content-hash record + npz arrays, atomic writes | yes |
| `hearsay_dsp/io/export.py` | reference TSV, blocking checks, writer, validator | yes |
| `hearsay_dsp/features/*.py` | common (framing, STFT, activity), lfcc, spectral, lpc, background, phase, prosody, enf, container | yes (synthetic signals) |
| `hearsay_dsp/routing.py`, `pipeline.py` | eligibility, per-file `analyze_file`, parallel `extract_paths`, `records_to_table` | yes; ran on real data (test set + 6.5k pool files) |
| `hearsay_dsp/models/gmm.py` | `GMMPair`, `OfficialPretrainedGMM`, frame selection | yes |
| `hearsay_dsp/models/detector.py` | `DetectorSpec`, `Detector` (GMM / LR / HGB / early fusion), contributions, save/load (refuses on scikit-learn version mismatch) | yes (fixtures) |
| `hearsay_dsp/models/calibration.py` | `SigmoidCalibrator` | yes |
| `hearsay_dsp/models/bundle.py` | `ModelBundle.train` (OOF components → fusion → routing band → final fits), scoring, save/load | fixtures only |
| `hearsay_dsp/evaluation/experiments.py` | `FrameStore`, folds, `gmm_oof`, `detector_oof` (parallel), cross-fitted calibration/fusion, `LateFusion`, `summarize` | partly (fixtures) |
| `hearsay_dsp/evaluation/metrics.py` | AUC, AP, interpolated EER, log loss/Brier (plain and class-balanced), confusion, reliability, group bootstrap | yes |
| `hearsay_dsp/evaluation/augment.py` | transforms, `make_variant` | transforms yes; `make_variant` no |
| `hearsay_dsp/evaluation/suite.py` | ablation suite (gmm, pretrained, features, fusion, logo, audit) | **never run** |
| `hearsay_dsp/evaluation/report.py` | markdown from run JSON | **never run** |
| `hearsay_dsp/trace.py` | per-file findings and template explanations | yes (via predict fixture) |
| `hearsay_dsp/cli.py` | validate, extract, train, evaluate, predict, report, experiment | train/predict yes (fixtures); others not |
| `scripts/dsp_build_manifests.py` | manifests + leakage asserts | ran |
| `scripts/dsp_fetch_pretrained.py` | official .mat download with checksum | ran |
| `scripts/dsp_robustness.py` | frozen-bundle robustness on the holdout sample | **never run** |
| `scripts/dsp_augment_train.py` | augmented training manifest | **never run** |
| `scripts/dsp_profile_quick.py` | technical profile (§4.2) | ran (from the scratchpad version) |
| `scripts/dsp_git_commit.sh` | private-index commit to `dsp-track` | used for commits 3 and 4 |

## 7. Known risks and gaps in not-yet-run code (check these first)

- `suite.py`:
  - The `speaker_audit` compares against `oof["fusion:<gmm>+lr:all"]`, so the `fusion` part must run in the same invocation.
  - LOGO reuses the C selected by the `features` part; if run alone, it falls back to the middle of the grid.
  - The shortcut audit feeds bool/object `diag.*` columns through `pd.to_numeric`, so the dtypes need verifying.
  - The 512-component official-reproduction GMM is the slowest variant.
  - Memory: `FrameStore` holds ~0.5 GB per LFCC variant for the dev set.
- `bundle.choose_band`: with near-perfect AUCs the band may collapse to the narrowest candidate. Sanity-check `meta.routing.candidates`.
- `cmd_predict`: per-feature contributions are computed only for rows routed to the full model (`lr_full`). Core-routed rows have none: add `lr_core` contributions. It also writes no per-file component-score TSV for the other session's fusion: add `--scores-out` or similar.
- `Dockerfile.dsp` copies `artifacts/dsp/model/`, but `artifacts/` is gitignored. Either add a `.gitignore` exception for the small bundle, or document "train before build". Verify that the Python 3.13 wheels of every pin exist for linux/aarch64 and linux/amd64.
- LPC is the slowest module (~0.1–0.25 s per file, a Python loop over frames). Vectorize it if runtime matters.
- `records_to_table` merges on the raw `path` string (with `..` segments from relative manifests). Always build tables from the same manifest object that was extracted.
- Not implemented: figures (spectrograms with candidate boundaries, reliability plot), README_DSP.md, and the final report.

## 8. Cache and resumption

- Layout: `cache/dsp/features/<sha[:2]>/<sha>.json` holds per-module `{key, result}`; `<sha>.lfcc.<key>.npz` holds the frame arrays (`official`, `custom`, `nonsilent`, `active`).
- A module entry is reused only if its key (module version + params + decode identity) matches and its arrays exist. A changed parameter recomputes only that module.
- Size at handoff: 1.8 GB.
- Any CLI command that reads a manifest calls `extract_paths` first (cache hits cost ~1 ms/file), so the suite and training fill any gaps automatically. Running `extract` first is only clearer and parallel.
- Augmented audio goes to `cache/dsp/augmented/<condition>.v1/<source_sha>.wav` with a `.json` provenance sidecar.

## 9. How to run everything (commands)

```bash
cd /Users/kevinharvey/HactGT13AudioAuthentication
PY=.venv/bin/python
$PY -m pytest tests/dsp -q -p no:cacheprovider                                  # 59 tests
$PY scripts/dsp_build_manifests.py                                              # regenerate manifests (deterministic)
$PY -m hearsay_dsp.cli validate --manifest runs/dsp/manifests/pool.tsv --out runs/dsp/validate/pool.json
$PY -m hearsay_dsp.cli validate --manifest runs/dsp/manifests/hearsay_test.tsv \
    --reference data/hearsay_test/HGT_Hearsay_score_template.csv --out runs/dsp/validate/test.json
$PY -m hearsay_dsp.cli extract    --manifest runs/dsp/manifests/pool.tsv --config configs/dsp.yaml --out runs/dsp/extract_pool
$PY -m hearsay_dsp.cli experiment --manifest runs/dsp/manifests/pool.tsv --config configs/dsp.yaml --out runs/dsp/suite_v1
$PY -m hearsay_dsp.cli report     --run runs/dsp/suite_v1
$PY -m hearsay_dsp.cli train      --manifest runs/dsp/manifests/pool.tsv --config configs/dsp.yaml --out artifacts/dsp/model
$PY -m hearsay_dsp.cli evaluate   --manifest runs/dsp/manifests/pool.tsv --model artifacts/dsp/model --splits holdout --out runs/dsp/eval_holdout
$PY -m hearsay_dsp.cli report     --run runs/dsp/eval_holdout
$PY scripts/dsp_robustness.py --manifest runs/dsp/manifests/pool.tsv --model artifacts/dsp/model --out runs/dsp/robustness
$PY -m hearsay_dsp.cli predict --manifest runs/dsp/manifests/hearsay_test.tsv --model artifacts/dsp/model \
    --reference data/hearsay_test/HGT_Hearsay_score_template.csv --output runs/dsp/submission/TEAMNAME_predictions.tsv
docker build -f Dockerfile.dsp -t hearsay-dsp .
docker run --rm --network none -v ~/Downloads/HackGTHearsayTesting:/data/input:ro -v $PWD/runs/dsp/docker_out:/data/output \
    hearsay-dsp predict --input /data/input --model /app/artifacts/dsp/model --output /data/output/predictions.tsv \
    --reference /data/input/HGT_Hearsay_score_template.csv --cache-dir /tmp/c
```
The `--where column=value` filter (repeatable) works on `validate`, `extract`, `train`, `evaluate` and `experiment`. Example: `--where dataset=diffssd` gives the DiffSSD-as-provided view.

## 10. Next steps (ordered, with decision rules)

1. **Finish extraction** (§0 command). Afterwards, check `runs/dsp/extract_pool/extract_summary.json`: `decode_errors` should be 0, and review the module status counts.
2. **Run the suite** (`experiment ... --out runs/dsp/suite_v1`) and debug as needed. Parts can be run separately with `--parts gmm,pretrained,features,fusion,logo,audit`; run `features` and `fusion` before `logo` and `audit`.
3. **Freeze the configuration from dev evidence only** and record the rules in the report:
   - GMM: the custom variant with the highest dev OOF AUC (`gmm_selected` in `suite_results.json`). Write it to `configs/dsp.yaml` under `model.gmm`. Report the official reproduction and the pretrained model as separate rows.
   - Feature groups: keep group *g* in `model.feature_groups` if dropping it lowers dev OOF AUC with the paired 95% CI entirely below 0 (see `paired_vs_lr_all`). If every drop-one CI includes 0 (redundant groups), keep the smallest set within 0.002 AUC of `lr:all`, preferring cheap groups. Excluded modules stay as diagnostics, labelled "not used in score".
   - Use HGB only if it beats LR on paired dev deltas.
   - Treat the shortcut audit as a warning: if `stream_properties`/`levels_and_silence` alone give a high AUC, say so. They are excluded by design.
   - Speaker audit: report the genuine false-positive rate on LibriSpeech/LJSpeech-1.1 for the DiffSSD-only model vs the extended model. This is the key evidence for (or against) using external reals.
4. **Train the final bundle** on dev rows. `train` excludes holdout automatically, and `bundle.json/meta.splits_used` should be `["dev"]`.
5. **Evaluate the holdout once** (`--splits holdout`) and do not tune afterwards. Report AUC/EER with CIs, per generator (the held-out-speaker clones 2061/5448 matter most), the genuine false-positive rate per real source, and reliability.
6. **Robustness:** run `scripts/dsp_robustness.py`. It reports AUC, EER, genuine FPR and sensitivity per condition, including `crop_3to4s` (test-like) and `lowpass_7500`.
7. *(Optional)* **Augmentation experiment:** run `scripts/dsp_augment_train.py`, then `train` on `pool_aug.tsv` with a different `--out`, and compare clean holdout vs robustness. Adopt it only if robustness improves without a clean-holdout loss.
8. **Predict the test set** and inspect `<out>.summary.json`: the score quantiles and `frac_ge_0.5`. An extreme fraction suggests domain shift, not prevalence. Keep the traces.
   - The TSV file name should be `<teamName>_predictions.tsv`, but the team name is unknown. Use a placeholder and flag it.
   - Also write per-file component scores for the other session.
9. **Build and run Docker offline** (`--network none`), then diff the TSV against the local run.
10. **Throughput:** take a ≤500-clip stratified sample (include long LibriSpeech and short test clips), run `extract` with a fresh `--cache-dir`, and record wall time, files/s, per-module mean runtime (from records) and worker max RSS. Record the machine load at the time. Extrapolate to 1,671 test files.
11. **Write `README_DSP.md` and `reports/dsp/*.md`** from saved JSON only. Include the ablation table (plan §11 layout), what helped and what did not, limitations, and figures.
    - Then commit with `scripts/dsp_git_commit.sh msg.txt` and `git push origin dsp-track`.
    - Ask the user whether to merge into `main` or open a PR (GitHub suggested `https://github.com/kevinharvey2025/HactGT13AudioAuthentication/pull/new/dsp-track`).
12. Do **not** submit anything to the organizers, contact them, or use the NSA one-time draft review. Tell the user that they can.

## 11. Mapping to the plan's completion report (plan §18)

| Item | Status |
|---|---|
| 1. What was implemented (paths, commands) | Code: §6. Commands: §9. |
| 2. Modules affecting score vs diagnostics | Designed: `lfcc` (GMM + cep), `spectral`, `lpc`, `background`, `phase`, `prosody` are candidates; `container`, `enf`, `compression` and all `diag.*` are diagnostic only. The final set is decided in step 3 of §10. |
| 3. Split provenance | §5.12–13; the bundle stores `training_rows.tsv`, `training_oof_scores.tsv` and the training content-hash digest. |
| 4. Tests, metrics, coverage, throughput | Tests: 59 pass. Metrics, coverage, throughput: **not yet measured**. |
| 5. What helped / didn't / untested | Pending the suite. |
| 6. Valid submission? | **No.** Blocker: no trained bundle yet (§10, steps 2–8). |
| 7. Limitations and next command | §7, §12; next command in §0. |

## 12. Open questions (for the user or organizers; none block the next steps)

- Is external bona fide data (LibriSpeech, LJSpeech-1.1) permitted by the organizers? The challenge PDF recommends such datasets; the plan leaves it open.
- What is the team name (needed for the TSV file name)? What is the official detection metric, and what is the deadline?
- Should `dsp-track` be merged into `main`, and should the trained bundle be committed so the Docker build works from a clean clone?
- Label policy for edited or replay audio: not needed for DiffSSD, but relevant to the HEARSAY test set.

## 13. Things not to do

- Do not fit anything (scalers, imputers, calibrators, thresholds, routing band) on holdout or test rows. Do not tune after looking at holdout results.
- Do not add stream properties, duration, level, digital-silence fraction, encoder tags, file names or MAC times to the classifier.
- Do not call a background or phase candidate a "splice", and do not name a generator family from spectral quirks.
- Do not claim accuracy from synthetic fixtures (`tests/dsp` fixtures test mechanics only).
- Do not write into the other session's paths, and do not switch branches in the shared tree (§1).
