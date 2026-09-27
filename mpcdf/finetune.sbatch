#!/bin/bash -l
# Up to four scripts/finetune_ssl.py runs in parallel on one 4xA100 node, one per GPU (18 cores each).
# Each positional argument is one GPU's work: one run's argument string, or several joined by " ;; " (run in
# sequence on that GPU), e.g.
#   mpcdf.py submit raven mpcdf/finetune.sbatch -- \
#     "--backbone adf_mms_300m --name mms300m --epochs 4" \
#     "--backbone adf_xlsr_1b --name xlsr1b --epochs 4 --grad-ckpt --batch 24"
# --epochs 0 gives the zero-shot benchmark only (val/holdout/test scored once).
#SBATCH --job-name=hearsay-ft
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
set -eo pipefail
source "$CODE_DIR/mpcdf/env.sh"
use_venv neural
cd "$CODE_DIR"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
i=0
pids=()
for gpu_work in "$@"; do
  (
    IFS=$'\n'
    for spec in $(echo "$gpu_work" | sed 's/ ;; /\n/g'); do
      name=$(echo "$spec" | sed -n 's/.*--name \([^ ]*\).*/\1/p')
      echo "=== GPU $i: $spec"
      unset IFS
      CUDA_VISIBLE_DEVICES=$i OMP_NUM_THREADS=2 python scripts/finetune_ssl.py $spec --workers 16 \
          > "$RUN_DIR/ft_${name:-run$i}.log" 2>&1 || echo "run $name failed"
      IFS=$'\n'
    done
  ) &
  pids+=($!)
  i=$((i + 1))
done
status=0
for p in "${pids[@]}"; do wait "$p" || status=1; done
for f in "$RUN_DIR"/ft_*.log; do echo "=== $f"; grep -E "^\{|best:|Error|error" "$f" | tail -8; done
exit $status
