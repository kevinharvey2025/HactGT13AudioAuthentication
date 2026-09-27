#!/bin/bash -l
# Data preparation on one CPU node (no training):
#   neural: fine-tuning pool + the shortcut-check sample (cache/manifest.parquet) -> cache/wav16k, triage tables,
#           test durations
#   DSP:    leakage-checked manifests + feature extraction for the pool and the test set (cache/dsp)
#   mpcdf.py submit raven mpcdf/prep.sbatch [-- pool|tracka|neural|dsp ...]   (default: neural dsp; neural = pool tracka)
#SBATCH --job-name=hearsay-prep
#SBATCH --time=03:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --exclusive
set -eo pipefail
source "$CODE_DIR/mpcdf/env.sh"
cd "$CODE_DIR"
parts=("$@")
[ ${#parts[@]} -eq 0 ] && parts=(neural dsp)
N=${SLURM_CPUS_PER_TASK:-72}
for part in "${parts[@]}"; do
  if [ "$part" = neural ] || [ "$part" = pool ] || [ "$part" = tracka ]; then
    use_venv neural
    if [ "$part" != tracka ]; then
      echo "=== neural pool $(date +%T)"
      python scripts/prepare_data.py --pool --workers "$N"     # incremental: only clips not triaged yet
    fi
    if [ "$part" != pool ]; then
      echo "=== shortcut-check manifest $(date +%T)"
      python scripts/prepare_data.py --workers "$N"
    fi
    deactivate
  elif [ "$part" = dsp ]; then
    use_venv dsp
    echo "=== dsp manifests $(date +%T)"
    python scripts/dsp_build_manifests.py
    for m in hearsay_test pool; do
      echo "=== dsp extract $m $(date +%T)"
      python -m hearsay_dsp.cli extract --manifest runs/dsp/manifests/$m.tsv --config configs/dsp.yaml \
          --out runs/dsp/extract_$m --n-jobs $((N - 2))
    done
    deactivate
  fi
done
echo "=== done $(date +%T)"
