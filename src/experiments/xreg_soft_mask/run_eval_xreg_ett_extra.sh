#!/usr/bin/env bash
# Evaluate the existing checkpoint with XReg on ETTh2/ETTm1/ETTm2 caches.
set -euo pipefail

EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEVICE="${DEVICE:-cuda:0}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$EXP_DIR/results}"
RESULTS_ROOT="${RESULTS_ROOT:-$EXP_DIR/results/real_lot_ett_xreg_ett_extra}"
REAL_ROOT="${REAL_ROOT:-/workspace/data/real_eval_lot_ett}"
DATASETS=(${DATASETS:-ETTh2 ETTm1 ETTm2})
HORIZONS=(${HORIZONS:-96})
BATCH_SIZE="${BATCH_SIZE:-8}"
SAMPLES_PER_DATASET="${SAMPLES_PER_DATASET:-12}"
PLOT_SAMPLES_PER_DATASET="${PLOT_SAMPLES_PER_DATASET:-6}"
XREG_RIDGE="${XREG_RIDGE:-0.0}"

python "$EXP_DIR/eval_real_xreg.py" \
  --checkpoint_root "$CHECKPOINT_ROOT" \
  --results_root "$RESULTS_ROOT" \
  --real_root "$REAL_ROOT" \
  --datasets "${DATASETS[@]}" \
  --horizons "${HORIZONS[@]}" \
  --batch_size "$BATCH_SIZE" \
  --samples_per_dataset "$SAMPLES_PER_DATASET" \
  --plot_samples_per_dataset "$PLOT_SAMPLES_PER_DATASET" \
  --xreg_ridge "$XREG_RIDGE" \
  --timesfm_metrics_csv none \
  --run_tfm_zeroshot
