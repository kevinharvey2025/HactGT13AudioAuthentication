#!/bin/bash
# Final submission build (run inside a GPU job from the code snapshot, neural venv active):
#   bash mpcdf/final.sh RUN[@EPOCH] [RUN[@EPOCH] ...]
# 1. scripts/make_submission.py: fused, calibrated TSV from the runs' stored test scores + fusion.json
# 2. artifacts/diffusion/: fusion.json + each run's best.pt (what predict.py and the Docker image load)
# 3. predict.py on the NSA test set (the Docker code path) -> SideQuests_predictions_final.tsv + traces
# 4. check that both TSVs agree (rank correlation, max abs difference)
set -eo pipefail
RUNS=("$@")
OUT=runs/diffusion/submission
python scripts/make_submission.py --runs "${RUNS[@]}" --label final --team SideQuests
mkdir -p artifacts/diffusion
cp $OUT/fusion.json artifacts/diffusion/fusion.json
for r in "${RUNS[@]}"; do
  n=${r%%@*}
  mkdir -p artifacts/diffusion/$n
  cp runs/diffusion/ft/$n/best.pt artifacts/diffusion/$n/best.pt
  cp runs/diffusion/ft/$n/config.json artifacts/diffusion/$n/config.json
done
python predict.py --input data/hearsay_test --output runs/diffusion/predict_final \
    --template data/hearsay_test/HGT_Hearsay_score_template.csv --artifacts artifacts/diffusion --label final
python - <<'PY'
import pandas as pd
a = pd.read_csv("runs/diffusion/submission/SideQuests_predictions_final.tsv", sep="\t")
b = pd.read_csv("runs/diffusion/predict_final/SideQuests_predictions_final.tsv", sep="\t")
m = a.merge(b, on="filename", suffixes=("_submission", "_predict"))
print("rows", len(a), len(b), "| spearman", round(m.iloc[:, 1].corr(m.iloc[:, 2], method="spearman"), 5),
      "| max abs diff", round((m.iloc[:, 1] - m.iloc[:, 2]).abs().max(), 5))
PY
