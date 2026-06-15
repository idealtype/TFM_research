#!/usr/bin/env bash
# Build TimesFM v1 caches, then train/evaluate the recorded-best soft warm-start setup.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-/workspace/data}"
WORK_ROOT="${WORK_ROOT:-/tmp/tfm_v1_backbone}"
DEVICE="${DEVICE:-cuda:0}"
HF_HOME="${HF_HOME:-$WORK_ROOT/hf}"
HF_CACHE_DIR="${HF_CACHE_DIR:-$HF_HOME}"
V1_REPO_DIR="${V1_REPO_DIR:-$WORK_ROOT/timesfm_origin}"
V1_CKPT_DIR="${V1_CKPT_DIR:-$WORK_ROOT/timesfm-1.0-200m-pytorch}"
V1_CKPT_PATH="$V1_CKPT_DIR/torch_model.ckpt"
COMPACT_DST="${COMPACT_DST:-$WORK_ROOT/data_v1_backbone}"
REAL_EVAL_DST="${REAL_EVAL_DST:-$WORK_ROOT/real_eval_lot_ett_v1_backbone}"
EXP_DIR="${EXP_DIR:-src/experiments/backbone_adjustment/soft_warm_s10_oldloss_best_v1_backbone}"
EXP_NAME="$(basename "$EXP_DIR")"
RESULTS_ROOT="${RESULTS_ROOT:-$WORK_ROOT/results/$EXP_NAME}"

mkdir -p "$WORK_ROOT" "$HF_HOME" "$V1_CKPT_DIR" "$(dirname "$RESULTS_ROOT")"
export DATA_ROOT HF_HOME
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

if [[ "${SKIP_RECACHE:-0}" == "1" ]]; then
  echo "[skip] SKIP_RECACHE=1: reusing existing cache at $COMPACT_DST"
else
  python "$REPO_ROOT/src/data_prep/prepare_v1_backbone_data.py" \
    --src_data_root "$DATA_ROOT" \
    --dst_data_root "$COMPACT_DST" \
    --lotsa_cache_root "$DATA_ROOT/data_lotsa/lotsa_cache" \
    --real_eval_root "$DATA_ROOT/real_eval_lot_ett" \
    --real_eval_dst_root "$REAL_EVAL_DST" \
    --v1_src "$V1_REPO_DIR/v1/src" \
    --checkpoint_path "$V1_CKPT_PATH" \
    --hf_cache_dir "$HF_CACHE_DIR" \
    --device "$DEVICE" \
    --batch_size 1024 \
    --fourier_batch_size 1024 \
    --encode_batch_size 1024 \
    --num_workers 4 \
    --fourier_warmup_steps 125 \
    --mixed_steps 2500 \
    --residual_steps 500 \
    --synth_interval 10 \
    --real_group_chunk_steps 63 \
    --skip_lotsa_subsets HZMETRO SHMETRO
fi

DATA_ROOT="$COMPACT_DST" \
REAL_ROOT="$REAL_EVAL_DST" \
RESULTS_ROOT="$RESULTS_ROOT" \
DEVICE="$DEVICE" \
SYNTH_INTERVAL=10 \
"$REPO_ROOT/$EXP_DIR/run_warm_real_mix.sh"
