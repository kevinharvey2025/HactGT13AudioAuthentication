# RawBoost experiment: decisions log

Append-only. Each entry is written before the results it governs are opened.

## 0. Protocol, fixed before stage 1 (2026-09-27)

**Code.** Commit `4698c9a` (branch `experimental-1`). The unseen-channel views are defined by `hearsay/unseen.py` at
that commit: view 100 ("unseen") and view 101 ("device", RawBoost-adjacent, never part of a headline number). They run
with ffmpeg 7.1.1 (Raven module `ffmpeg/7.1`) and pyroomacoustics 0.10.1. The definitions are frozen: any change
gets a new view id.

**Phase 0 findings.**
- The training path is byte-identical to the snapshot that trained the stored runs (`61e593e-dirty-4af30fc6`). The
  RawBoost and view additions are inert when off; the tests show bit-identical batches and frozen views 0 and 1.
  So R0 at seed 0 should reproduce `xlsr1b_d6rall` up to GPU nondeterminism, and the report's paired R0-vs-stored
  difference is the reproduction check.
- A stored XLS-R-1B run takes 0.40 s/step, about 25 minutes end to end, far less than the 3-5 h budgeted.
- The reference implementation is MIT-licensed, so the port carries its notice. Parity: max |diff| 1.8e-7 across
  algos 0-8.
- **The AntiDeepfake checkpoints we fine-tune** (the default ones, not `-nda`) were post-trained and fine-tuned with
  RawBoost algo 5 (LnL+ISD). This comes from their model cards and paper, §IV-A and §IV-D1. Expect little from R2.
  SSI (R1) is the one RawBoost model these backbones have not seen.
- RawBoost has no room model: LnL is band-stop FIR filtering of polynomial powers. It is not expected to fix reverb.
- Literature corrections:
  - the RawBoost paper's "27%" is min t-DCF; its EER gain is 44%;
  - the Odyssey 2022 figures (0.82% / 2.85%) are the best of three seeds; the means are 1.00% / 3.69%;
  - Wang & Yamagishi's 7.55% In-the-Wild EER also needs a contrastive loss and paired batches; vocoded data plus
    RawBoost alone gives 12.08%.

**Runs.**

| Stage | Runs | Settings |
|---|---|---|
| 1 | R0 (seeds 0 and 1); R1 (algo 3, SSI); R2 (algo 5, LnL→ISD); R3 (algo 4, LnL→ISD→SSI) | RawBoost arms at `--p-rawboost 0.5`, seed 0 |
| 1, alongside | `views_<member>` for the three shipped members | evaluation only, all four views |

Every run copies `xlsr1b_d6rall/config.json` and adds `--select val --eval-views 0,1,100,101`.

The `views_<member>` runs give the shipped ensemble an unseen-channel number. They involve no training and no choice.

**Operating points.**
- (a) the val-selected epoch: the lowest val minDCF, as the mean of the clean and aug views; ties go to the earliest
  epoch;
- (b) the last epoch.

Gates are judged at (a); (b) is reported as a check.

**Stage decisions.** Made with `python scripts/rawboost_report.py --val-only`, on val minDCF (mean of clean and aug)
at each arm's val-selected epoch. The lowest wins. Ties within 0.001 go to the arm with fewer RawBoost stages
(SSI, then LnL→ISD, then LnL→ISD→SSI). Each decision is written here before the next stage runs. Val clean is
saturated at 0 for fine-tuned XLS-R-1B, so decisions rest on val aug (view 1). View 1 is drawn from the chain, which
favours arms that keep it; this is accepted and stated.

**Gates**, relative to R0 of the same seed:
1. **Target.** ITW aug or ITW unseen minDCF falls by ≥ 10% relative. The paired 95% interval of the difference must
   exclude 0, and the drop must exceed the seed spread. The seed spread is max(|R0 s0 − R0 s1|, |stored s0 − stored s1|)
   for that set, view and operating point. The stored pair has no view 100, so R0's pair alone sets the unseen spread.
2. **Clean non-inferiority.** ITW clean and holdout clean are no worse than R0 by more than max(5% relative, 0.005).
3. **Seeds.** The view(s) passing gate 1 improve for both seeds of the selected arm, each against its own R0.
4. **Cost.** s/step ≤ 1.25 × R0's.

If no arm passes, the conclusion is "no measurable benefit on top of the current augmentation", and stage 3 does not
run. Stage 3 also needs the user's approval.

**Confirmation.** The arm that passes and R0 are re-scored on all 4,000 labelled ITW clips through `predict.py`'s
whole-clip fp32 path. The 27,779 held-out ITW clips (T1) are a further check and are never used for a choice.

## 1. Stage 1 decision (2026-09-27, 04:45 EDT), from `rawboost_report.py --val-only` only

Holdout, In-the-Wild and the unseen views of these runs were not opened before this entry.

| Run | Val-selected epoch | Val minDCF, mean of clean and aug |
|---|---|---|
| R0 seed 0 | 2 | 0.0168 |
| R0 seed 1 | 3 | 0.0158 |
| R1 (SSI) | 3 | 0.0167 |
| R2 (LnL→ISD) | 2 | 0.0149 |
| **R3 (LnL→ISD→SSI)** | 3 | **0.0121** |

- **Reproduction check:** R0 seed 0 and seed 1 reproduce the stored `xlsr1b_d6rall` and `xlsr1b_d6rall_s1` val numbers
  to four decimals at every epoch.
- **Decision:** R3 has the lowest val minDCF, so it is the selected arm. Val clean is 0 for every run, so the decision
  rests on val aug (view 1, drawn from the chain).
- **Seed spread on val:** 0.0010 (R0 seed 0 vs seed 1). R3's lead over R0 is 0.0037 to 0.0047.
- **Stage 2** (R4, R5, R3 seed 1) is not run tonight: it cannot finish before the team's 6 AM cutoff. The gates stay
  unjudged until R3 seed 1 exists.
- **Submission candidate B** uses R3 in place of the shipped XLS-R-1B member. The user asked for it as an option,
  and it has not passed the gates.
