#!/bin/bash
# The shipped inference path (predict.py, CPU, fp32: what the Docker image runs) over a large set, in parallel
# shards of the template, merged in template order and validated. Run inside a CPU job, neural venv active:
#   bash mpcdf/predict_cpu.sh INPUT_DIR TEMPLATE OUT_DIR LABEL [N_SHARDS]
# Writes OUT_DIR/SideQuests_predictions_<LABEL>.tsv and OUT_DIR/traces.jsonl. ARTIFACTS=DIR picks another fusion.json + checkpoints.
set -eo pipefail
IN=$1; TPL=$2; OUT=$3; LABEL=$4; N=${5:-4}
mkdir -p "$OUT/shards"
python - "$TPL" "$OUT" "$N" <<'PY'
import sys, pandas as pd
tpl, out, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
t = pd.read_csv(tpl, sep="\t")
for i in range(n):
    t.iloc[i::n].to_csv(f"{out}/shards/template_{i}.tsv", sep="\t", index=False)
PY
THREADS=$(( ${SLURM_CPUS_PER_TASK:-72} / N ))
pids=()
for i in $(seq 0 $((N - 1))); do
  OMP_NUM_THREADS=$THREADS /usr/bin/time -v python predict.py --input "$IN" --output "$OUT/shards/$i" \
      --template "$OUT/shards/template_$i.tsv" --device cpu --label "$LABEL" ${ARTIFACTS:+--artifacts "$ARTIFACTS"} \
      > "$OUT/shards/log_$i.txt" 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
grep -h "Elapsed\|Maximum resident" "$OUT"/shards/log_*.txt
python - "$TPL" "$OUT" "$N" "$LABEL" <<'PY'
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, ".")
from hearsay import submission
tpl, out, n, label = sys.argv[1], Path(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
name = f"SideQuests_predictions_{label}.tsv"
parts = pd.concat([pd.read_csv(out / "shards" / str(i) / name, sep="\t") for i in range(n)])
submission.write(parts.set_index("filename")["cm-score"], out / name, template=tpl)
with open(out / "traces.jsonl", "w") as fh:
    for i in range(n):
        fh.write((out / "shards" / str(i) / "traces.jsonl").read_text())
print("merged", len(parts), "rows ->", out / name)
PY
