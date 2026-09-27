# Sourced by every Raven job script and by the login-node setup scripts (bash).
#
# Code runs from immutable snapshots (/ptmp/$USER/jobs/HackGT-13-Hearsay/code/<version>/, made by
# the mpcdf helper). Everything stateful lives in one workspace outside them and is linked into
# each snapshot under the names the code already uses (data/, cache/, runs/, artifacts/):
#   WS      /ptmp/$USER/hearsay         data, caches, run outputs, trained models
#   SHARED  .../HackGT-13-Hearsay/shared  uv, managed Pythons, the two virtualenvs
export WS=${WS:-/ptmp/$USER/hearsay}
export SHARED=${SHARED_DIR:-/ptmp/$USER/jobs/HackGT-13-Hearsay/shared}

module purge >/dev/null 2>&1 || true
module load ffmpeg/7.1                          # ffmpeg + ffprobe 7.1.1 (mp3/aac/opus/g711/g722/g726 encoders)

export PATH="$SHARED/bin:$PATH"                 # uv
export UV_CACHE_DIR=/ptmp/$USER/uv-cache
export UV_PYTHON_INSTALL_DIR=$SHARED/uv-python
export HF_HOME=/ptmp/$USER/hf_cache
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}    # process pools do the parallelism; raise per job if needed
export MPLBACKEND=Agg

if [ -n "${CODE_DIR:-}" ]; then
  for d in data cache runs artifacts; do
    mkdir -p "$WS/$d"
    # never replace an existing link: concurrent jobs share the snapshot
    [ -L "$CODE_DIR/$d" ] || [ -e "$CODE_DIR/$d" ] || ln -s "$WS/$d" "$CODE_DIR/$d" 2>/dev/null || true
  done
fi

# venv-neural: requirements.txt;  venv-dsp: requirements-dsp.txt;  venv-concepts: requirements.txt + cobweb-private
use_venv() { source "$SHARED/venv-$1/bin/activate"; }
