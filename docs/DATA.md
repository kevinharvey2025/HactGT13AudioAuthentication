# Data, forensic audit and leakage control

What we trained and evaluated on, what we established about the NSA test set before training anything, and how
the splits and the input view keep shortcuts out of the numbers.

## 1. Sources

| Set | Clips | Label | Role |
|---|---|---|---|
| DiffSSD (organizers), pinned sample `configs/diffssd_pool_files.txt` | 14,895 (~1,480 per generator) | synthetic | training / val / holdout |
| &nbsp;&nbsp;LJ-voice TTS: DiffGAN-TTS, Grad-TTS, ProDiff, WaveGrad 2 | 5,908 | synthetic | |
| &nbsp;&nbsp;voice cloning: ElevenLabs, OpenVoice v2, PlayHT, UnitSpeech, XTTS v2, YourTTS | 8,987 | synthetic | |
| LJSpeech-1.1 | 13,100 | real | the voice of the LJ-voice generators |
| LibriSpeech, the 10 speakers DiffSSD clones (`scripts/fetch_librispeech_speakers.py`) | 1,152 | real | speaker-matched reals for the cloners |
| LibriSpeech dev-clean + test-clean, 80 other speakers (`scripts/fetch_extra_reals.sh`) | 5,323 | real | real-speech diversity |
| Copy-synthesis fakes, D6-R (`scripts/run_d6r.py`): train-split reals re-vocoded by HiFi-GAN 16 kHz, HiFi-GAN LJ, DiffWave, Vocos | 11,900 (2,975 per vocoder) | synthetic | training only |
| In-the-Wild (Müller et al., Interspeech 2022), random sample | 4,000 (2,000 per class) | both | **evaluation only** |
| NSA sample test set | 1,671 (3.0-13.6 s, median 3.4 s) | unknown | predictions only |

In-the-Wild matters more than any other set: the AntiDeepfake backbones were post-trained on ~140 h of DiffSSD's
~146 h of fakes (Wang et al., arXiv 2506.21090), so DiffSSD-based numbers flatter them, while In-the-Wild was held
out of their training. It is our closest stand-in for sources nobody has seen.

## 2. Forensic audit of the test set (before any training)

`scripts/forensic_audit.py` (outputs under `runs/forensics/`); every positive finding has a control.

| Question | Finding | Consequence |
|---|---|---|
| Exact reuse? Byte and decoded-PCM SHA-256 over test, organizer reals, LJSpeech, LibriSpeech, DiffSSD | 0 duplicate groups, within the test set or across sets | |
| Content reuse? Landmark-hash matching of every test clip against 89,817 reference recordings (all 70,000 DiffSSD files, LJSpeech, LibriSpeech) | no matches; the positive control (reference files pushed through the test pipeline) matches 100/100 | the test clips come from sources we do not have: optimize for generalization |
| Container | every file: `fmt LIST data`, PCM mono 16 kHz 16-bit, `ISFT=Lavf58.29.100` (FFmpeg 4.2), consistent header arithmetic, nothing after `data`, no C2PA/ID3/bext chunks | metadata is constant on test: it can never score |
| Processing history | 98.0% of lengths are exact multiples of 512 samples at 22.05 kHz (real corpora 0-2.5%); start-trimmed, hard-cut ends; 7.5-8 kHz band 43.5 dB down with a -20 dB edge at 7.39 kHz (a kaiser-class resampler, 22.05 -> 16 kHz); peak-normalized | one librosa-style pipeline for both classes: augmentation material, never a feature |
| Hidden data | LSB parity 0.500, no histogram combs, no provenance chunks | nothing to decode |

Test labels, content matches and file times were never used to score, select or calibrate anything.

## 3. Shortcuts in the training data, and the canonical view

In DiffSSD the classes differ in ways that have nothing to do with synthesis: native sample rate (16 / 22.05 / 24 /
44.1 kHz), codec (some generators ship MP3), encoder tags, level, leading/trailing silence and duration. Small
gradient-boosting models on such trivial features alone (`scripts/shortcut_checks.py`,
[results/shortcut_checks.md](../results/shortcut_checks.md), grouped 5-fold CV):

| Feature group | AUC on the raw files | AUC on the canonical view |
|---|---|---|
| container (rate, codec, bitrate, tags) | 0.90 | removed |
| file-system times | 1.00 | removed |
| duration | 0.70 | 0.53 |
| level (peak, RMS, clipping, DC) | 0.91-0.96 | 0.79-0.83 |
| edge silence, digital zeros | 0.92-0.95 | 0.63-0.66 |
| bandwidth (cutoff, HF drop) | 0.77-0.81 | 0.64-0.67 |

So every model sees one **canonical view** (`hearsay/audio.py`, `hearsay/views.py`): ffmpeg decode to 16 kHz mono
(the organizers' resampler) -> edge-silence trim -> a crop whose length is drawn from the 1,671 test durations,
start-anchored half the time like the test clips -> [channel augmentation] -> 7 kHz low-pass -> DC removal ->
peak-normalize to 0.97-1.0 -> 1-LSB dither. Rate, band edge, duration and level cues are equalized for both classes;
test clips are only trimmed, never cropped. What remains (residual level and edge cues; all trivial features together
still reach AUC 0.85 in domain) is why In-the-Wild, not the in-domain holdout, decides between systems.

## 4. Splits

`hearsay.splits.shared_split` gives every track the same split (seed 20260926), with groups that never straddle it:

| Family | Group (never split) | train | val | holdout |
|---|---|---|---|---|
| DiffSSD fakes | sentence id (20% of ids held out); all clones of 2 held-out speakers | 8,989 | 1,554 | 4,352 |
| LJSpeech | chapter | 7,045 | 3,366 | 2,689 |
| LibriSpeech, cloned speakers | (speaker, chapter); the 2 held-out speakers entirely | 585 | 0 | 567 |
| LibriSpeech, other speakers | speaker | 3,153 | 900 | 1,270 |
| copy-synthesis (D6-R) | made only from train-split reals | 11,900 | 0 | 0 |
| In-the-Wild | own split `itw` | 0 | 0 | 0 (4,000 in `itw`) |

`val` selects checkpoints, fits the z-normalization and the calibration (together with In-the-Wild) and trains every
fusion; `holdout` is touched only for reporting. Evaluation uses a label-stratified sample of 2,500 clips per set.
`tests/test_splits.py` checks the group rules and that adding copy-synthesis rows moves no other row.

## 5. Channel augmentation

Real clips in the test set may be noisy or compressed, and a noisy real clip must not read as fake. Training applies
the same random channel chains to both classes (probability 0.6; `hearsay/augment.py`): one or two of codecs (MP3
24-128 kbps, AAC, Opus 8-32 kbps, G.711, G.722, G.726), telephony band-pass, band-limiting, resampling through
8-12 kHz, spectral tilt (-6 to +2 dB/octave; the test set is darker than every training source), white/pink/brown or
babble noise at 5-30 dB SNR (babble from train-split reals only), mains hum, room reverb (RT60 0.15-0.9 s) and
clipping. Every evaluation clip also gets one fixed chain (view 1), identical for all systems, so "aug" numbers are
paired.
