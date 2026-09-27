#!/bin/bash -l
# Final checks on one CPU node: unit tests (DSP venv, neural venv) and the full evaluation (scripts/evaluate.py).
#   mpcdf.py submit raven mpcdf/evaluate.sbatch [-- EXTRA evaluate.py ARGS]
# Results: runs/eval/ (benchmark.csv, breakdown.csv, calibration.csv, curves.csv, fusion.csv, test_agreement.csv,
# tables.md), copied to $RUN_DIR. The cobweb-private tests run in mpcdf/concepts.sbatch (their venv).
#SBATCH --job-name=hearsay-eval
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --exclusive
set -eo pipefail
source "$CODE_DIR/mpcdf/env.sh"
cd "$CODE_DIR"
export PYTHONDONTWRITEBYTECODE=1
use_venv dsp
python -m pytest tests/dsp -q -p no:cacheprovider 2>&1 | tail -3
deactivate
use_venv neural
python -m pytest tests -q -rs --ignore=tests/dsp -p no:cacheprovider 2>&1 | tail -6
python scripts/evaluate.py --out runs/eval --workers 64 "$@"
cp -r runs/eval "$RUN_DIR/"
