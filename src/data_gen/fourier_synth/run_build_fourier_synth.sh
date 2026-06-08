#!/bin/bash
set -euo pipefail

DEVICE=${DEVICE:-cuda:0}
DATA_ROOT=${DATA_ROOT:-/workspace/data}
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)

export DATA_ROOT
export VESSL_DATA_ROOT=${VESSL_DATA_ROOT:-$DATA_ROOT}
export HF_HOME=${HF_HOME:-$DATA_ROOT/.cache/huggingface}
export HF_HUB_CACHE=${HF_HUB_CACHE:-$HF_HOME/hub}

python "$PROJECT_ROOT/src/data_gen/fourier_synth/build_fourier_synth.py" \
  --device "$DEVICE" \
  --hf_cache_dir "$HF_HUB_CACHE" \
  --skip_existing \
  "$@"
