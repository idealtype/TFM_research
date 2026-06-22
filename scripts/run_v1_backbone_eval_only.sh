#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-/workspace/data}"
WORK_ROOT="${WORK_ROOT:-/tmp/tfm_v1_eval}"
DEVICE="${DEVICE:-cuda:0}"
HF_HOME="${HF_HOME:-$DATA_ROOT/.cache/huggingface}"
HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
HF_CACHE_DIR="${HF_CACHE_DIR:-$HF_HOME}"
if [[ -d "$REPO_ROOT/timesfm_origin/v1/src" ]]; then
  V1_REPO_DIR="${V1_REPO_DIR:-$REPO_ROOT/timesfm_origin}"
else
  V1_REPO_DIR="${V1_REPO_DIR:-$DATA_ROOT/.cache/timesfm_origin}"
fi
V1_CKPT_DIR="${V1_CKPT_DIR:-$DATA_ROOT/.cache/timesfm-1.0-200m-pytorch}"
V1_CKPT_PATH="$V1_CKPT_DIR/torch_model.ckpt"
COMPACT_DST="${COMPACT_DST:-$DATA_ROOT/data_v1_backbone}"
REAL_EVAL_DST="${REAL_EVAL_DST:-$DATA_ROOT/real_eval_lot_ett_v1_backbone}"
BASE_CKPT_ROOT="${BASE_CKPT_ROOT:?set BASE_CKPT_ROOT to the trained v1_backbone checkpoint root}"
DECODER_CKPT_ROOT="${DECODER_CKPT_ROOT:?set DECODER_CKPT_ROOT to the trained decoder_adjusted checkpoint root}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M)_v1_eval}"
RESULTS_ROOT="${RESULTS_ROOT:-$DATA_ROOT/results/backbone_adjustment/$RUN_TAG}"
LOCALIZE_CACHE="${LOCALIZE_CACHE:-1}"
EVAL_PARALLEL_PROCESSES="${EVAL_PARALLEL_PROCESSES:-1}"

mkdir -p "$WORK_ROOT" "$HF_HOME" "$HF_HUB_CACHE" "$V1_CKPT_DIR" "$RESULTS_ROOT"
export DATA_ROOT HF_HOME HF_HUB_CACHE
export TIMESFM_V1_SRC="$V1_REPO_DIR/v1/src"
export TIMESFM_V1_CHECKPOINT_PATH="$V1_CKPT_PATH"

if [[ ! -d "$V1_REPO_DIR/.git" ]]; then
  git clone --depth 1 https://github.com/idealtype/timesfm_origin.git "$V1_REPO_DIR"
fi

python -c "import utilsforecast" 2>/dev/null || pip install --quiet utilsforecast

if [[ ! -f "$V1_CKPT_PATH" ]]; then
  python - "$V1_CKPT_DIR" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download(
    "google/timesfm-1.0-200m-pytorch",
    local_dir=sys.argv[1],
    allow_patterns=["torch_model.ckpt"],
)
PY
fi

python "$REPO_ROOT/src/data_prep/prepare_v1_backbone_data.py" \
  --src_data_root "$DATA_ROOT" \
  --dst_data_root "$COMPACT_DST" \
  --lotsa_cache_root "$DATA_ROOT/data_lotsa/lotsa_cache" \
  --real_eval_root "$DATA_ROOT/real_eval_lot_ett" \
  --real_eval_dst_root "$REAL_EVAL_DST" \
  --exp_dir "$REPO_ROOT/src/experiments/backbone_adjustment/soft_warm_s10_oldloss_best_v1_backbone" \
  --v1_src "$V1_REPO_DIR/v1/src" \
  --checkpoint_path "$V1_CKPT_PATH" \
  --hf_cache_dir "$HF_CACHE_DIR" \
  --device "$DEVICE" \
  --skip_train_cache

RUN_REAL_EVAL_ROOT="$REAL_EVAL_DST"
if [[ "$LOCALIZE_CACHE" == "1" ]]; then
  RUN_REAL_EVAL_ROOT="$WORK_ROOT/real_eval_lot_ett_v1_backbone"
  if [[ "$(readlink -f "$REAL_EVAL_DST")" == "$(readlink -f "$RUN_REAL_EVAL_ROOT")" ]]; then
    echo "[localize] REAL_EVAL_DST is already local: $RUN_REAL_EVAL_ROOT"
  else
    echo "[localize] copying real eval cache to local workspace"
    rm -rf "$RUN_REAL_EVAL_ROOT"
    mkdir -p "$RUN_REAL_EVAL_ROOT"
    cp -a "$REAL_EVAL_DST"/. "$RUN_REAL_EVAL_ROOT"/
  fi
fi

DEVICE="$DEVICE" python \
  "$REPO_ROOT/src/experiments/backbone_adjustment/soft_warm_s10_oldloss_best_v1_backbone/eval_real_parallel.py" \
  --checkpoint_root "$BASE_CKPT_ROOT" \
  --results_root "$RESULTS_ROOT/v1_backbone_real_lot_ett" \
  --real_root "$RUN_REAL_EVAL_ROOT" \
  --hf_cache_dir "$HF_CACHE_DIR" \
  --parallel_processes "$EVAL_PARALLEL_PROCESSES" \
  --timesfm_metrics_csv none \
  --run_tfm_zeroshot

DEVICE="$DEVICE" python \
  "$REPO_ROOT/src/experiments/backbone_adjustment/soft_warm_s10_oldloss_best_v1_backbone_decoder_adjusted/eval_real_parallel.py" \
  --checkpoint_root "$DECODER_CKPT_ROOT" \
  --results_root "$RESULTS_ROOT/v1_backbone_decoder_adjusted_real_lot_ett" \
  --real_root "$RUN_REAL_EVAL_ROOT" \
  --hf_cache_dir "$HF_CACHE_DIR" \
  --parallel_processes "$EVAL_PARALLEL_PROCESSES" \
  --timesfm_metrics_csv none \
  --skip_tfm
