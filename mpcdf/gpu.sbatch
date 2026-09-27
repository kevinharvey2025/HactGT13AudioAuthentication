#!/bin/bash -l
# Up to four shell commands on one 4xA100 node, one per GPU (CUDA_VISIBLE_DEVICES=i), in the neural venv:
#   mpcdf.py submit raven mpcdf/gpu.sbatch -- "CMD0" "CMD1" ...
# A command may chain several steps with && or ;. Logs: $RUN_DIR/gpu<i>.log
#SBATCH --job-name=hearsay-gpu
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
set -eo pipefail
source "$CODE_DIR/mpcdf/env.sh"
use_venv neural
cd "$CODE_DIR"
i=0
pids=()
for cmd in "$@"; do
  echo "=== GPU $i: $cmd"
  CUDA_VISIBLE_DEVICES=$i OMP_NUM_THREADS=4 bash -c "$cmd" > "$RUN_DIR/gpu$i.log" 2>&1 &
  pids+=($!)
  i=$((i + 1))
done
status=0
for p in "${pids[@]}"; do wait "$p" || status=1; done
for f in "$RUN_DIR"/gpu*.log; do echo "=== $f"; tail -5 "$f"; done
exit $status
