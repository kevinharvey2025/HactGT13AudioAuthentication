#!/bin/bash -l
# Pre-training forensic integrity audit (scripts/forensic_audit.py): structure/hashes, signal history,
# resampling-pipeline identification, landmark-hash content matching. CPU only.
#   mpcdf.py submit raven mpcdf/forensics.sbatch [-- STAGE ...]     (default: all stages)
#SBATCH --job-name=hearsay-forensics
#SBATCH --time=01:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --mem=100000
set -eo pipefail
source "$CODE_DIR/mpcdf/env.sh"
use_venv neural
cd "$CODE_DIR"
stages=("$@")
[ ${#stages[@]} -eq 0 ] && stages=(structure signal pipeline fingerprint)
for s in "${stages[@]}"; do
  echo "=== $s $(date +%T)"
  python scripts/forensic_audit.py "$s" --workers "$SLURM_CPUS_PER_TASK"
done
cp -r runs/forensics "$RUN_DIR/" 2>/dev/null || true
echo "=== done $(date +%T)"
