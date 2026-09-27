#!/bin/bash
# A candidate submission end to end: fuse RUNs with the shipped recipe (scripts/make_submission.py: z statistics and
# class-balanced Platt at the 30% prior on val + ITW), gather the members' checkpoints, and score the NSA test set
# through predict.py (whole clips, fp32: the Docker path). Run inside a CPU job with the neural venv:
#   bash scripts/candidate.sh TAG RUN [RUN ...]        (each RUN at its saved best.pt epoch)
# Writes runs/diffusion/candidates/TAG/{fusion.json, artifacts/, predict/SideQuests_predictions_final.tsv}.
set -eo pipefail
TAG=$1; shift
OUT=runs/diffusion/candidates/$TAG
python scripts/make_submission.py --runs "$@" --calib val,itw --label "$TAG" --out "$OUT"
mkdir -p "$OUT/artifacts"
cp "$OUT/fusion.json" "$OUT/artifacts/fusion.json"
for r in "$@"; do
  mkdir -p "$OUT/artifacts/$r"
  ln -sfn "$(readlink -f "runs/diffusion/ft/$r/best.pt")" "$OUT/artifacts/$r/best.pt"
done
T=$(readlink -f data/hearsay_test)
ARTIFACTS=$OUT/artifacts bash mpcdf/predict_cpu.sh "$T" "$T/HGT_Hearsay_score_template.csv" "$OUT/predict" final 4
