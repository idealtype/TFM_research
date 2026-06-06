#!/usr/bin/env bash
set -euo pipefail

EXP_DIR="/home/sia2/project/5.22syn_cent/train_nonF_rawtarget"
OLD_EXP_DIR="/home/sia2/project/4.28basis/basis_dec/experiment/func_dec_syn_cent"
EVAL_NONF_ROOT="${EVAL_NONF_ROOT:-/home/sia2/project/data/synthetic_nonF/synth_eval_nonfourier}"
REAL_ROOT="${REAL_ROOT:-/home/sia2/project/data/real_eval_lot_ett}"
RESULT_ROOT="${RESULT_ROOT:-${EXP_DIR}/results}"
RUN_NAME="${RUN_NAME:-nonfourier_finetune_from_simple_complex}"
CHECKPOINT_RUN_DIR="${CHECKPOINT_RUN_DIR:-${RESULT_ROOT}/train/${RUN_NAME}}"
DEVICE="${DEVICE:-cuda:0}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-64}"
HORIZONS=(${HORIZONS:-96 192 336 720})
RESIDUAL_DISTRIBUTIONS=(${RESIDUAL_DISTRIBUTIONS:-normal student_t exponential gamma weibull pareto})
LOG_DIR="${RESULT_ROOT}/logs"

mkdir -p "${LOG_DIR}"
cd "${EXP_DIR}"

echo "[eval_existing_nonF] checkpoint_run_dir=${CHECKPOINT_RUN_DIR}"
echo "[eval_existing_nonF] result_root=${RESULT_ROOT}"
echo "[eval_existing_nonF] started_at=$(date '+%Y-%m-%d %H:%M:%S')"

for horizon in "${HORIZONS[@]}"; do
  ckpt="${CHECKPOINT_RUN_DIR}/checkpoints/funcdec_h${horizon}.pt"
  if [[ ! -e "${ckpt}" ]]; then
    echo "missing checkpoint: ${ckpt}" >&2
    exit 1
  fi
done

echo "[eval_existing_nonF] evaluate non-Fourier synthetic"
python "${OLD_EXP_DIR}/eval_nonfourier_single_model.py" \
  --nonfourier_root "${EVAL_NONF_ROOT}" \
  --checkpoint_run_dir "${CHECKPOINT_RUN_DIR}" \
  --results_root "${RESULT_ROOT}" \
  --output_name "nonfourier_single_model" \
  --horizons "${HORIZONS[@]}" \
  --stages stage1_S stage2_T_S stage3_T_S_R \
  --residual_distributions "${RESIDUAL_DISTRIBUTIONS[@]}" \
  --batch_size "${EVAL_BATCH_SIZE}" \
  --device "${DEVICE}" \
  2>&1 | tee "${LOG_DIR}/eval_nonfourier_${RUN_NAME}.log"

echo "[eval_existing_nonF] evaluate real LOTSA/ETTh1"
python "${OLD_EXP_DIR}/eval_real_lot_ett_single_model.py" \
  --real_root "${REAL_ROOT}" \
  --checkpoint_run_dir "${CHECKPOINT_RUN_DIR}" \
  --results_root "${RESULT_ROOT}" \
  --output_name "real_lot_ett_single_model" \
  --horizons "${HORIZONS[@]}" \
  --batch_size "${EVAL_BATCH_SIZE}" \
  --device "${DEVICE}" \
  2>&1 | tee "${LOG_DIR}/eval_real_${RUN_NAME}.log"

echo "[eval_existing_nonF] build extra summary plots"
python "/home/sia2/project/5.22syn_cent/plot_extra_result_summaries.py" "${RESULT_ROOT}" \
  2>&1 | tee "${LOG_DIR}/plot_extra_${RUN_NAME}.log"

echo "[eval_existing_nonF] completed_at=$(date '+%Y-%m-%d %H:%M:%S')"
