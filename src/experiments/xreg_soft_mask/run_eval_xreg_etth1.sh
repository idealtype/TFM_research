#!/usr/bin/env bash
# Quick ETTh1-only XReg evaluation for checking covariate contribution plots.
set -euo pipefail

EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEVICE="${DEVICE:-cuda:0}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$EXP_DIR/results}"
RESULTS_ROOT="${RESULTS_ROOT:-$EXP_DIR/results/real_lot_ett_xreg_etth1_smoke}"
REAL_ROOT="${REAL_ROOT:-/workspace/data/real_eval_lot_ett}"
HORIZONS=(${HORIZONS:-96})
BATCH_SIZE="${BATCH_SIZE:-8}"
SAMPLES_PER_DATASET="${SAMPLES_PER_DATASET:-12}"
PLOT_SAMPLES_PER_DATASET="${PLOT_SAMPLES_PER_DATASET:-6}"
XREG_RIDGE="${XREG_RIDGE:-0.0}"

python "$EXP_DIR/eval_real_xreg.py" \
  --checkpoint_root "$CHECKPOINT_ROOT" \
  --results_root "$RESULTS_ROOT" \
  --real_root "$REAL_ROOT" \
  --datasets ETTh1 \
  --horizons "${HORIZONS[@]}" \
  --batch_size "$BATCH_SIZE" \
  --samples_per_dataset "$SAMPLES_PER_DATASET" \
  --plot_samples_per_dataset "$PLOT_SAMPLES_PER_DATASET" \
  --xreg_ridge "$XREG_RIDGE" \
  --timesfm_metrics_csv none \
  --run_tfm_zeroshot
