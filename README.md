# HEARSAY: is this voice real? (team SideQuests, HackGT 13, NSA audio authentication challenge)

`predict.py` gives any audio file a score from 0 to 1 (1 = synthetic) and a trace of why. It runs the same way in
Docker. Deliverable: [`submission/SideQuests_predictions_final.tsv`](submission/SideQuests_predictions_final.tsv).

## Results

**Official score on the NSA test set** (organizers, hidden labels), for the submitted TSV: **minDCF 0.0317, EER
1.44%**. The best interim leaderboard entry was minDCF 0.0584 / EER 2.5%. Our out-of-domain proxy predicted it
(In-the-Wild minDCF 0.028 [0.017, 0.040], EER 1.2%).

minDCF is the organizers' metric (Pspoof 0.3, Cfa 4; lower is better, 1.0 = a constant decision), [95% CI].
In-the-Wild (ITW) is web speech and deepfakes from sources none of our models saw. "Perturbed" = the same clips
through random codec / telephony / noise / reverb chains.

| System | ITW clean | ITW perturbed | in-domain holdout, perturbed |
|---|---|---|---|
| best pretrained detector (AntiDeepfake XLS-R-2B, zero-shot) | 0.038 [0.024, 0.049] | 0.189 [0.162, 0.212] | 0.354 |
| **final: 3 fine-tuned detectors + copy-synthesis fakes, ensembled** | **0.028 [0.017, 0.040]** | **0.082 [0.064, 0.097]** | **0.073** |
| concept-based scorer: TTCG prototypes named by cobweb concepts, so every score traces to named concepts (1,000-clip subsets) | 0.043 | 0.097 | 0.095 |

Final system on ITW: EER 1.20%. The shipped code path (`predict.py`: whole clips, CPU) on all 4,000 labeled ITW
clips: AUC 0.9994, EER 1.30%, minDCF 0.033. The concept-based scorer is within about 0.01 minDCF of the detector it
is built on, measured on identical clips.
Unseen sources and channel perturbations are exactly what the NSA test set brings. All systems, confidence
intervals, breakdowns and the tests behind every claim: [docs/EVALUATION.md](docs/EVALUATION.md).

## How it works

```
audio ─► forensic triage (container, codec, rate; trace only)
      ─► canonical view: 16 kHz, trim, 7 kHz low-pass, level-normalize (removes dataset shortcuts)
      ─► XLS-R-2B, XLS-R-1B, MMS-1B anti-spoofing detectors, fine-tuned ─► z-normalized mean ─► calibrated P(synthetic)
      ─► router: windowed re-scoring (partial fakes), compression cross-checks ─► trace
      ─► concept formation + diffusion prototypes: "which learned concepts explain this clip?"
```

1. **Audit before training** ([docs/DATA.md](docs/DATA.md)). The test set shares no audio with any training corpus
   (landmark hashing against 89,817 recordings, positive-controlled). Every test file has the same container. So
   metadata cannot score, and the job is generalization.
2. **Remove shortcuts.** In the training data, container fields, file times, level and leading silence alone
   separate real from fake (AUC 0.90–1.00). Every model sees one canonical view, and every clip, real or fake, gets
   the same channel perturbations.
3. **Detect** ([docs/METHOD.md](docs/METHOD.md)). NII AntiDeepfake encoders, ported to `transformers` with a strict
   weight map, then fine-tuned gently. Plain fine-tuning costs out-of-domain accuracy with every epoch. Adding
   *copy-synthesis* fakes (real clips re-vocoded by HiFi-GAN, DiffWave and Vocos) reverses that. The three backbones
   are ensembled and calibrated so that P > 0.2 is the organizers' Bayes decision.
4. **Explain** ([docs/CONCEPTS.md](docs/CONCEPTS.md)). The detector's representation is organized into concepts
   the way COBWEB models human category learning (the lab's cobweb-private). Each clip is then explained by
   diffusion prototypes (Wang et al., TTCG) named by those concepts, e.g. "39% of the evidence ~ a 59-clip concept
   of YourTTS / XTTS clones under a codec". The explanations are tested like a detector, and against it:
   - **Faithfulness:** where they say which evidence is synthetic, deleting that evidence moves the detector in
     92% of clips.
   - **Stability:** their top source agrees 94–95% of the time across insertion orders.
   - **The basic level:** the held-out one sits at an intermediate depth, as the concept-formation literature
     predicts.
   - **Auditing:** the explanations flag test clips whose score they contradict; one of these exposed a scoring
     bug, since fixed.
5. **Report what failed.** A signal-processing detector (AUC 0.91 in domain, worse than chance on ITW) and metadata
   models (a shortcut) were gated out. Reverberation and clips under 2 s are the remaining weak spots.

## Run it

```bash
python predict.py --input /path/to/audio --output out/ --template /path/to/HGT_Hearsay_score_template.csv
docker build -t sidequests-hearsay . && \
  docker run --rm --network none -v /path/to/audio:/data/input:ro -v $PWD/out:/data/output sidequests-hearsay
pytest tests --ignore=tests/dsp && pytest tests/dsp
```

Training, the concept analysis and the evaluation ran on MPCDF Raven (A100); every step is a job script in
`mpcdf/`. See [docs/REPRODUCE.md](docs/REPRODUCE.md).

## Repository

| Path | Contents |
|---|---|
| `predict.py`, `Dockerfile` | inference: TSV + per-file traces |
| `hearsay/` | audio and canonical view, augmentation, data manifests and splits, metric, TSV writer, AntiDeepfake port, concept formation (`concepts.py`), diffusion (`diffusion/`: DDPM, TTCG, copy-synthesis), forensic triage |
| `hearsay_dsp/` | the signal-processing detector (LFCC-GMM, spectral, LPC, phase, prosody, background, ENF) |
| `scripts/` | pipeline stages: data, forensic audit, fine-tuning, copy-synthesis, embeddings, concepts, evaluation, submission |
| `mpcdf/` | Raven job scripts and environment setup |
| `tests/` | unit tests (metric parity with the organizers' code, TSV, splits, TTCG, concepts, views; DSP) |
| `results/` | generated evaluation tables and CSVs, concept-analysis outputs, test-set explanations |
| `submission/` | the final TSV, its summary, `fusion.json` |
| `docs/` | DATA, METHOD, CONCEPTS, EVALUATION, REPRODUCE; `archive/`: plans, handoffs, what was removed |

## Credits

- Detectors: AntiDeepfake (Wang et al., NII Yamagishi Lab, arXiv 2506.21090).
- Concept formation: cobweb-private (Teachable AI Lab, Georgia Tech; not redistributed here).
- Diffusion prototypes: Wang, Gupta, Zhu & MacLellan (arXiv 2605.07078).
- Diffusion as concept formation: Wang, Singaravadivelan & MacLellan (arXiv 2609.13047).
- Data: DiffSSD, LJSpeech, LibriSpeech, In-the-Wild (Müller et al. 2022).
