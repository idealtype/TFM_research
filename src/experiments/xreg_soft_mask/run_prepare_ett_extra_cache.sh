#!/usr/bin/env bash
# Build ETTh2/ETTm1/ETTm2 real-eval caches for XReg evaluation.
set -euo pipefail

EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_ROOT="${SOURCE_ROOT:-/workspace/data/external/ett}"
REAL_ROOT="${REAL_ROOT:-/workspace/data/real_eval_lot_ett}"
DATASETS=(${DATASETS:-ETTh2 ETTm1 ETTm2})
HORIZONS=(${HORIZONS:-96 192 336 720})
CONTEXT_LEN="${CONTEXT_LEN:-512}"
STRIDE="${STRIDE:-512}"
BATCH_SIZE="${BATCH_SIZE:-128}"
DEVICE="${DEVICE:-cuda:0}"

python "$EXP_DIR/prepare_ett_xreg_cache.py" \
  --source_root "$SOURCE_ROOT" \
  --real_root "$REAL_ROOT" \
  --datasets "${DATASETS[@]}" \
  --horizons "${HORIZONS[@]}" \
  --context_len "$CONTEXT_LEN" \
  --stride "$STRIDE" \
  --batch_size "$BATCH_SIZE" \
  --device "$DEVICE" \
  "$@"
