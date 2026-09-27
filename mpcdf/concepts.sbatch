#!/bin/bash -l
# Concept formation (cobweb-private) + TTCG diffusion prototypes over the final detector's embedding space.
#   mpcdf.py submit raven mpcdf/concepts.sbatch [-- EXTRA run_concepts.py ARGS]
# venv-concepts = requirements.txt on a uv-managed Python 3.11 + cobweb-private (pip install -e, built with gcc/13;
# see /ptmp/$USER/hearsay/ext/build_cobweb.sh and ext/cobweb-private/{GIT_REVISION,LOCAL_PATCHES}).
#SBATCH --job-name=hearsay-concepts
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
set -eo pipefail
source "$CODE_DIR/mpcdf/env.sh"
module load gcc/13                      # libstdc++ of the compiler that built the cobweb extension
use_venv concepts
cd "$CODE_DIR"
export CUDA_VISIBLE_DEVICES=0
EMB=xlsr2b_d6rall_last
python scripts/concept_subset.py
python scripts/extract_ssl.py --model adf_xlsr_2b --checkpoint runs/diffusion/ft/xlsr2b_d6rall/best.pt \
    --manifest manifest_concepts --layers last --views 2 --workers 60 --batch 16 --name $EMB
python scripts/run_concepts.py --emb $EMB "$@"
cp -r runs/diffusion/concepts/$EMB "$RUN_DIR/" 2>/dev/null || true
