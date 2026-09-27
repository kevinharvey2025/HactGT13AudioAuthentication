#!/bin/bash -l
# Build the virtualenvs on a Raven login node (internet, no GPU needed):
#   bash mpcdf/setup_envs.sh [neural|dsp|concepts|all]      (run from a code snapshot or a checkout)
# venv-neural:   Python 3.11 + requirements.txt (torch wheels from PyPI bundle the CUDA runtime).
# venv-dsp:      Python 3.13 + requirements-dsp.txt (+ pytest, matplotlib for tests/figures).
# venv-concepts: uv-managed Python 3.11 (the C++ extension needs Python headers) + requirements.txt + cobweb-private,
#                built with GCC 13 against Eigen. Needs $WS/ext/cobweb-private (a clone of the lab's private repo;
#                revision in its GIT_REVISION file; GCC 13 needs '#include <stack>' in src/cobweb_discrete_tree.cpp)
#                and $WS/ext/eigen (Eigen 3 headers).
# Each venv is rebuilt only when its requirements file changes (stamp = sha256 of the file).
set -eo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
source "$HERE/mpcdf/env.sh"
mkdir -p "$SHARED/bin"
if ! command -v uv >/dev/null; then
  # uv from the earlier HackGT-13 project; otherwise bootstrap it from python-waterboa
  if [ -x /ptmp/$USER/jobs/HackGT-13/shared/uv-boot/bin/uv ]; then
    cp /ptmp/$USER/jobs/HackGT-13/shared/uv-boot/bin/uv "$SHARED/bin/uv"
  else
    module load python-waterboa/2025.06
    python -m venv "$SHARED/uv-boot" && "$SHARED/uv-boot/bin/pip" install -q uv
    ln -sf "$SHARED/uv-boot/bin/uv" "$SHARED/bin/uv"
  fi
fi
uv --version

build() {  # build NAME PYTHON REQFILE [extra packages...]
  local name=$1 py=$2 req=$3; shift 3
  local venv="$SHARED/venv-$name"
  local stamp="$venv/.req-$(cat "$req" <(echo "$@") | sha256sum | cut -c1-16)"
  (
    flock 9
    if [ -f "$stamp" ]; then echo "venv-$name up to date"; exit 0; fi
    rm -rf "$venv"
    uv venv --python "$py" "$venv"
    VIRTUAL_ENV="$venv" uv pip install -r "$req" "$@"
    touch "$stamp"
    echo "venv-$name built"
  ) 9>"$SHARED/.venv-$name.lock"
}

what=${1:-all}
if [ "$what" = neural ] || [ "$what" = all ]; then
  build neural 3.11 "$HERE/requirements.txt" pytest
  "$SHARED/venv-neural/bin/python" -c "import torch, transformers, sklearn; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'transformers', transformers.__version__)"
fi
if [ "$what" = dsp ] || [ "$what" = all ]; then
  build dsp 3.13 "$HERE/requirements-dsp.txt" pytest matplotlib
  "$SHARED/venv-dsp/bin/python" -c "import numpy, scipy, sklearn, av, parselmouth; print('dsp ok', numpy.__version__, sklearn.__version__, av.__version__)"
fi
if [ "$what" = concepts ] || [ "$what" = all ]; then
  module load gcc/13 cmake/3.26
  UV_PYTHON_PREFERENCE=only-managed build concepts 3.11 "$HERE/requirements.txt" pytest
  (cd "$WS/ext/cobweb-private" && VIRTUAL_ENV="$SHARED/venv-concepts" CMAKE_PREFIX_PATH="$WS/ext/eigen" CC=gcc CXX=g++ uv pip install -e .)
  "$SHARED/venv-concepts/bin/python" -c "from cobweb.cobweb_continuous import CobwebContinuousNode as N; assert hasattr(N, 'get_basic'); print('cobweb-private ok')"
fi
