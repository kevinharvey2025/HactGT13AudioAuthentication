# Results (generated; do not edit by hand)

Copied from the Raven workspace after the final runs; the docs quote these files.

| File | Made by | Contents |
|---|---|---|
| `tables.md` | `scripts/evaluate.py` | every table in one place: all systems, paired comparisons with the final ensemble, breakdowns, calibration, fusion gate, test-set agreement, fine-tuning curves, the shipped code path |
| `benchmark.csv` | `scripts/evaluate.py` | system x set x view: n, AUC, EER, minDCF with 95% bootstrap intervals, the final ensemble on the same clips, paired Δ |
| `breakdown.csv` | `scripts/evaluate.py` | per fake generator, real source, channel condition and crop duration, with false-alarm / miss rates at P > 0.2 |
| `calibration.csv` | `scripts/evaluate.py` | cross-fitted Platt calibration: actDCF, minDCF, Cllr, prior-weighted ECE, flag rate |
| `curves.csv` | `scripts/evaluate.py` | every fine-tuning run and epoch under the official metric |
| `fusion.csv` | `scripts/evaluate.py` | logistic fusion of the final score with DSP / metadata (fitted on val) |
| `test_agreement.csv` | `scripts/evaluate.py` | NSA test set, label-free: Spearman and κ with the final, flag rate, score vs duration |
| `docker_path.json` | `scripts/evaluate.py` | `predict.py` on all 4,000 labeled In-the-Wild clips; router decisions |
| `runtime.md` | `mpcdf/run.sbatch` + `predict.py` | CPU inference time and memory |
| `concepts/` | `scripts/run_concepts.py` | `metrics.json` (scores), `levels.json` (basic level, leakage, stability, noise-depth), `faithfulness.json`, `scores_*.parquet`, `test_explanations.jsonl` (one explanation per NSA test clip) |
| `shortcut_checks.md` | `scripts/shortcut_checks.py` | how far trivial features go, raw vs canonical view |
| `forensics/` | `scripts/forensic_audit.py` | container structure and duplicate hashes; content matching (test clips: at most 33 aligned hash votes, none matched; positive control: median 484 votes, 100/100 matched) |
| `dsp/`, `metadata/` | `hearsay_dsp`, `scripts/meta_experiments.py` | the gated-out branches' own reports |
