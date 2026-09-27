#!/bin/bash -l
# Run shell commands on one CPU node inside a venv:  mpcdf.py submit raven mpcdf/run.sbatch -- VENV "CMD" ["CMD" ...]
# (VENV = neural | dsp; commands run in sequence from the code snapshot; stop at the first failure)
#SBATCH --job-name=hearsay-run
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --exclusive
set -eo pipefail
source "$CODE_DIR/mpcdf/env.sh"
use_venv "$1"; shift
cd "$CODE_DIR"
for cmd in "$@"; do
  echo "=== $(date +%T) $cmd"
  bash -c "$cmd"
done
echo "=== done $(date +%T)"
