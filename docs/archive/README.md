# Archive: how the project got here

Kept for provenance; the current system is described in the docs one level up. Paths inside these files refer to
the repository as it was when they were written.

| File | What it is |
|---|---|
| `HANDOFF_DIFFUSION.md`, `HANDOFF_DSP.md` | handoffs from the first machine: the neural/diffusion track and the DSP track (built by a teammate), before the move to MPCDF Raven |
| `plans/MASTER_PLAN.md` | work packages and decisions after the move to Raven |
| `plans/AASISTand_AntiDeepfake_prompt.md` | detector plan; Addendum: SSL-detector literature (fine-tuning, augmentation, fusion, contamination) |
| `plans/diffusion_cf_prompt.md` | challenge requirements and rubric (§1), the diffusion track (§7), Addenda A (literature), E (the cognitive science of concepts), E.4 (switch to cobweb-private + TTCG) |
| `plans/signal_processing_prompt.md` | DSP-track specification |
| `plans/metadata_analysis_prompt.md` | metadata / forensics specification; Addendum D: the pre-training forensic audit and its literature |

## Removed in the consolidation

Nothing below was used by the final system or its evaluation; `git show 4c0b05d:<path>` recovers any of it.

| Path | Why removed |
|---|---|
| `hearsay/heads.py`, `hearsay/embeddings.py`, `scripts/train_track_a.py` | Track A handoff baseline (frozen WavLM + MLP; speaker-disjoint CV AUC 0.998 / EER 1.6% in domain). Evaluated under a different protocol, never on In-the-Wild; superseded by the fine-tuned AntiDeepfake detectors |
| `hearsay/diffusion/latent.py`, `hearsay/diffusion/score_se.py` | options D3 (AudioLDM 2 denoising curve) and D4 (SGMSE+ features); implemented, never run, no speech precedent |
| `hearsay/diffusion/prototypes.py`, `scripts/run_basic_level.py`, the D2 mode-path model in `hearsay/diffusion/ddpm.py` | our own prototype code and mode-path features; replaced by cobweb-private + TTCG (`hearsay/concepts.py`, `hearsay/diffusion/ttcg.py`) |
| old `hearsay/concepts.py` (CLASSIT re-implementation) | replaced by the wrapper around the lab's cobweb-private |
| residual features in `hearsay/diffusion/resynth.py` (option D1) and the codec models | only the four copy-synthesis vocoders are used |
| `scripts/compare_systems.py`, `scripts/fusion_gate.py` | merged into `scripts/evaluate.py` (the fusion gate now uses the official effective prior, 0.632, instead of 0.8) |
| `scripts/score_files.py` | zero-shot scoring of arbitrary files; covered by `predict.py` and the epoch-0 evaluations |
| `Dockerfile.dsp`, `Dockerfile.dsp.dockerignore` | a DSP-only image (never built); the DSP detector is not part of the submission |
| `scripts/dsp_git_commit.sh`, `scripts/dsp_profile_quick.py`, `reports/dsp/profile_quick.csv` | a commit helper for two sessions sharing one working tree, and a profiling one-off |
| `lightgbm`, `shap`, `matplotlib`, `mutagen`, `diffusers`, `accelerate` in `requirements.txt` | no remaining imports |
