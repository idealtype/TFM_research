#!/usr/bin/env bash
# Evaluate the decoder-adjusted soft-mask checkpoint with TimesFM-style XReg overlay.
set -euo pipefail

EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEVICE="${DEVICE:-cuda:0}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$EXP_DIR/results}"
RESULTS_ROOT="${RESULTS_ROOT:-$EXP_DIR/results/real_lot_ett_xreg}"
REAL_ROOT="${REAL_ROOT:-/workspace/data/real_eval_lot_ett}"
HORIZONS=(${HORIZONS:-96 192 336 720})
BATCH_SIZE="${BATCH_SIZE:-64}"
SAMPLES_PER_DATASET="${SAMPLES_PER_DATASET:-0}"
XREG_RIDGE="${XREG_RIDGE:-0.0}"

python "$EXP_DIR/eval_real_xreg.py" \
  --checkpoint_root "$CHECKPOINT_ROOT" \
  --results_root "$RESULTS_ROOT" \
  --real_root "$REAL_ROOT" \
  --horizons "${HORIZONS[@]}" \
  --batch_size "$BATCH_SIZE" \
  --samples_per_dataset "$SAMPLES_PER_DATASET" \
  --xreg_ridge "$XREG_RIDGE" \
  --run_tfm_zeroshot
