# Candidate prediction files

The organizers score two files per team: the initial (interim) submission and the final one. They count the better
of the two (Thao, 9/26, 3:45 PM). After a team receives its interim score, only the final remains (PuzzleMaster,
9/26, 5:14 PM). Our interim scored **minDCF 0.0317, EER 1.443%** (9/27, 1:21 AM).

| File | SHA-256 (first 16) | System | How it was scored |
|---|---|---|---|
| `v1_interim.tsv` | `b6210912c84a2e71` | interim ensemble: D6-R XLS-R-2B and XLS-R-1B (3 vocoders) + WiSE-FT XLS-R-2B a=0.3 | `predict.py` before the order-independence fix (Raven `runs/diffusion/submission`, 10:36 PM EDT) |
| `v2_final.tsv` | `658418f42bc84ec7` | final ensemble: D6-R XLS-R-2B, XLS-R-1B, MMS-1B (4 vocoders), `submission/fusion.json` | `predict.py` before the fix; identical to `submission/SideQuests_predictions_final.tsv` |
| `v3_final_wholeclip.tsv` | `8640b624cfbf2c96` | the final ensemble, same weights and calibration | `predict.py` after the fix: each file scored whole and alone, fp32. **The Docker image reproduces this file** (max \|ΔP\| 3.2e-6) |

The v1 file most likely matches the interim DM by its time stamp (10:36 PM EDT); the DM itself is the record.

**What distinguishes them (no test labels were used).**
- v1 and v2 rank the test clips differently (Spearman 0.93) but agree on 99.6% of decisions at P > 0.2.
- v2 and v3 agree on 99.8% of decisions (Spearman 0.998).
- On In-the-Wild, the two ensembles tie on clean audio (0.028 each). The final ensemble is better under channel
  perturbations (0.082 vs 0.095), but the paired interval of that difference includes 0.

**Recommendation:** v3 as the final file. The Docker image that judges run produces v3, and v3 follows the same
recipe as v2 without the batch-order dependence. Since the better score counts, the final cannot lower our result
below the interim's 0.0317.

## The three options (9/27, 5:10 AM EDT)

Named in the organizers' format, `TeamName_predictions.tsv` with the final label: A is the verified option, B robustness, C
breadth. Their fusion files are `fusion_B.json` and `fusion_C.json`; A's is `submission/fusion.json`.

All three come from the same pipeline and the shipped fusion recipe (`scripts/candidate.sh`: z statistics and Platt at
the 30% prior on val + In-the-Wild). All three are scored through `predict.py`, the Docker path. They differ only in
the ensemble members.

| File | SHA-256 (first 16) | Tactic | Members |
|---|---|---|---|
| `SideQuests_predictions_final_A.tsv` | `8640b624cfbf2c96` | exact reproducibility (= `v3_final_wholeclip.tsv`) | XLS-R-2B@3, XLS-R-1B@2, MMS-1B@3 |
| `SideQuests_predictions_final_B.tsv` | `cddd231828def976` | RawBoost augmentation in the XLS-R-1B member | XLS-R-2B@3, R3 (RawBoost LnL→ISD→SSI)@3, MMS-1B@3 |
| `SideQuests_predictions_final_C.tsv` | `77e7cea28138076e` | more members across seeds and recipes | A's three + XLS-R-1B seed 1@3 + WiSE-FT XLS-R-2B (α 0.3) |

`comparison.csv` (from `scripts/compare_candidates.py`) holds the comparison below. In-the-Wild minDCF (2,500 clips)
is in-sample for calibration, as in the shipped recipe.

| Option | ITW clean | ITW perturbed | Test flagged (P > 0.2) | Test decisions equal to A | Spearman with A |
|---|---|---|---|---|---|
| A | 0.0281 | 0.0824 | 28.19% | – | – |
| B | 0.0287 | 0.0822 | 28.19% | 99.88% | 0.990 |
| C | 0.0297 | 0.0782 | 28.25% | 99.82% | 0.987 |

**Reading.** The options differ by 2–3 test decisions out of 1,671, and none differs from A outside the noise.
- B's RawBoost member did not pass the predeclared gates (`results/rawboost/decisions.md`).
- C trades a little clean accuracy for a little robustness under perturbation.
- A stays the recommendation: it is the file the Docker image reproduces and the system every document describes.
