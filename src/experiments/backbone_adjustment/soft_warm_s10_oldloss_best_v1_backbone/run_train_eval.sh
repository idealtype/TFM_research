#!/usr/bin/env bash
set -euo pipefail

SOFT_MASK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_ROOT="${RESULTS_ROOT:-$SOFT_MASK_DIR/results}"
DEVICE="${DEVICE:-cuda:0}"
SKIP_TFM="${SKIP_TFM:-1}"

cd "$SOFT_MASK_DIR"

python "$SOFT_MASK_DIR/train.py" \
  --device "$DEVICE" \
  --results_root "$RESULTS_ROOT" \
  --skip_nonf

TFM_ARGS=()
if [[ "$SKIP_TFM" == "1" ]]; then
  TFM_ARGS+=(--skip_tfm)
fi

# Synthetic evaluation is intentionally disabled for the current soft-mask runs.
# python "$SOFT_MASK_DIR/eval_synth_fourier.py" \
#   --device "$DEVICE" \
#   --checkpoint_root "$RESULTS_ROOT" \
#   --results_root "$RESULTS_ROOT/fourier_synth" \
#   "${TFM_ARGS[@]}"
#
# python "$SOFT_MASK_DIR/eval_synth_nonf.py" \
#   --device "$DEVICE" \
#   --checkpoint_root "$RESULTS_ROOT" \
#   --results_root "$RESULTS_ROOT/nonfourier_synth" \
#   "${TFM_ARGS[@]}"

python "$SOFT_MASK_DIR/eval_real.py" \
  --device "$DEVICE" \
  --checkpoint_root "$RESULTS_ROOT" \
  --results_root "$RESULTS_ROOT/real_lot_ett" \
  "${TFM_ARGS[@]}"
