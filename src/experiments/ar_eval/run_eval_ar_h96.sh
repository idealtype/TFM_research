#!/usr/bin/env bash
# Evaluate the best soft-mask warm-mix h96 decoder autoregressively.
set -euo pipefail

DEVICE=${DEVICE:-cuda:0}
PROJECT=/home/sia2/project/6.1AR_temp
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-/home/sia2/project/5.30soft_mask/results/fourier_warm_real_mix_scratch}
RESULTS=${RESULTS_ROOT:-$PROJECT/results/ar_h96_from_soft_warm_mix_scratch_real_lot_ett}
HORIZONS=(${HORIZONS:-96 192 336 720})
BATCH_SIZE=${BATCH_SIZE:-32}
SAMPLES_PER_DATASET=${SAMPLES_PER_DATASET:-0}
LOG_DIR="$RESULTS/logs"
LOG_FILE="$LOG_DIR/run_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=== soft-mask AR h96 real-eval start $(date '+%Y-%m-%d %H:%M:%S') ==="
echo "DEVICE=$DEVICE"
echo "CHECKPOINT_ROOT=$CHECKPOINT_ROOT"
echo "RESULTS=$RESULTS"
echo "LOG_FILE=$LOG_FILE"
echo "HORIZONS=${HORIZONS[*]}"
echo "BATCH_SIZE=$BATCH_SIZE"
echo "SAMPLES_PER_DATASET=$SAMPLES_PER_DATASET"

DEVICE=$DEVICE python "$PROJECT/eval_real_ar_h96.py" \
  --checkpoint_root "$CHECKPOINT_ROOT" \
  --results_root "$RESULTS" \
  --horizons "${HORIZONS[@]}" \
  --batch_size "$BATCH_SIZE" \
  --samples_per_dataset "$SAMPLES_PER_DATASET" \
  --skip_tfm

echo "=== soft-mask AR h96 real-eval complete $(date '+%Y-%m-%d %H:%M:%S') ==="
