#!/usr/bin/env bash
# Build TimesFM v1 caches, then train/evaluate the recorded-best soft warm-start setup.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-/workspace/data}"
WORK_ROOT="${WORK_ROOT:-/tmp/tfm_v1_backbone}"
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
EXP_DIR="${EXP_DIR:-src/experiments/backbone_adjustment/soft_warm_s10_oldloss_best_v1_backbone}"
EXP_NAME="$(basename "$EXP_DIR")"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M)_$EXP_NAME}"
RESULTS_ROOT="${RESULTS_ROOT:-$DATA_ROOT/results/backbone_adjustment/$RUN_TAG}"
LOCALIZE_CACHE="${LOCALIZE_CACHE:-1}"

mkdir -p "$WORK_ROOT" "$HF_HOME" "$HF_HUB_CACHE" "$V1_CKPT_DIR" "$(dirname "$RESULTS_ROOT")"
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

cache_ready() {
  [[ -d "$COMPACT_DST" ]] || return 1
  [[ -d "$REAL_EVAL_DST" ]] || return 1
  find "$COMPACT_DST" -name 'backbone_emb*.pt' -print -quit | grep -q . || return 1
  find "$COMPACT_DST" -name 'futures*.pt' -print -quit | grep -q . || return 1
  find "$REAL_EVAL_DST" -name 'backbone_emb*.pt' -print -quit | grep -q . || return 1
  find "$REAL_EVAL_DST" -name 'futures*.pt' -print -quit | grep -q . || return 1
  find "$REAL_EVAL_DST" -name 'raw_contexts_c*.pt' -print -quit | grep -q . || return 1
}

compact_ready() {
  [[ -d "$COMPACT_DST" ]] || return 1
  find "$COMPACT_DST" -name 'backbone_emb*.pt' -print -quit | grep -q . || return 1
  find "$COMPACT_DST" -name 'futures*.pt' -print -quit | grep -q . || return 1
}

RUN_RECACHE=0
SKIP_TRAIN_CACHE_FLAG=""
if [[ "${FORCE_RECACHE:-0}" == "1" ]]; then
  echo "[recache] FORCE_RECACHE=1: rebuilding v1 compact caches"
  RUN_RECACHE=1
elif [[ "${SKIP_RECACHE:-0}" == "1" ]]; then
  echo "[skip] SKIP_RECACHE=1: reusing existing cache at $COMPACT_DST"
  if ! cache_ready; then
    echo "[error] requested SKIP_RECACHE=1, but v1 compact caches are incomplete" >&2
    echo "[error] COMPACT_DST=$COMPACT_DST REAL_EVAL_DST=$REAL_EVAL_DST" >&2
    exit 1
  fi
elif cache_ready; then
  echo "[skip] existing v1 compact caches found; reusing COMPACT_DST=$COMPACT_DST REAL_EVAL_DST=$REAL_EVAL_DST"
elif compact_ready; then
  echo "[partial] compact cache exists; building only real_eval cache (skipping train cache plan)"
  RUN_RECACHE=1
  SKIP_TRAIN_CACHE_FLAG="--skip_train_cache"
else
  echo "[recache] v1 compact caches not found; building from DATA_ROOT=$DATA_ROOT"
  RUN_RECACHE=1
fi

if [[ "$RUN_RECACHE" == "1" ]]; then
  python "$REPO_ROOT/src/data_prep/prepare_v1_backbone_data.py" \
    --src_data_root "$DATA_ROOT" \
    --dst_data_root "$COMPACT_DST" \
    --lotsa_cache_root "$DATA_ROOT/data_lotsa/lotsa_cache" \
    --real_eval_root "$DATA_ROOT/real_eval_lot_ett" \
    --real_eval_dst_root "$REAL_EVAL_DST" \
    --exp_dir "$REPO_ROOT/$EXP_DIR" \
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
    --skip_lotsa_subsets HZMETRO SHMETRO \
    ${SKIP_TRAIN_CACHE_FLAG}
fi

RUN_COMPACT_ROOT="$COMPACT_DST"
RUN_REAL_EVAL_ROOT="$REAL_EVAL_DST"
if [[ "$LOCALIZE_CACHE" == "1" ]]; then
  RUN_COMPACT_ROOT="$WORK_ROOT/data_v1_backbone"
  RUN_REAL_EVAL_ROOT="$WORK_ROOT/real_eval_lot_ett_v1_backbone"
  if [[ "$(readlink -f "$COMPACT_DST")" == "$(readlink -f "$RUN_COMPACT_ROOT")" ]]; then
    echo "[localize] COMPACT_DST is already local: $RUN_COMPACT_ROOT"
  else
    echo "[localize] copying train compact cache to local workspace"
    rm -rf "$RUN_COMPACT_ROOT"
    mkdir -p "$RUN_COMPACT_ROOT"
    cp -a "$COMPACT_DST"/. "$RUN_COMPACT_ROOT"/
  fi
  if [[ "$(readlink -f "$REAL_EVAL_DST")" == "$(readlink -f "$RUN_REAL_EVAL_ROOT")" ]]; then
    echo "[localize] REAL_EVAL_DST is already local: $RUN_REAL_EVAL_ROOT"
  else
    echo "[localize] copying real eval cache to local workspace"
    rm -rf "$RUN_REAL_EVAL_ROOT"
    mkdir -p "$RUN_REAL_EVAL_ROOT"
    cp -a "$REAL_EVAL_DST"/. "$RUN_REAL_EVAL_ROOT"/
  fi
fi

DATA_ROOT="$RUN_COMPACT_ROOT" \
REAL_ROOT="$RUN_REAL_EVAL_ROOT" \
RESULTS_ROOT="$RESULTS_ROOT" \
DEVICE="$DEVICE" \
SYNTH_INTERVAL=10 \
"$REPO_ROOT/$EXP_DIR/run_warm_real_mix.sh"
