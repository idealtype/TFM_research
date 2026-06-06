#!/usr/bin/env bash
set -euo pipefail

EXP_DIR="/home/sia2/project/5.22syn_cent/train_nonF_projtarget"
OLD_EXP_DIR="/home/sia2/project/4.28basis/basis_dec/experiment/func_dec_syn_cent"
TRAIN_ROOT="${TRAIN_ROOT:-/home/sia2/project/data/synthetic_nonF/synth_train_nonfourier}"
PROJECTION_ROOT="${PROJECTION_ROOT:-${EXP_DIR}/projection_targets}"
EVAL_NONF_ROOT="${EVAL_NONF_ROOT:-/home/sia2/project/data/synthetic_nonF/synth_eval_nonfourier}"
REAL_ROOT="${REAL_ROOT:-/home/sia2/project/data/real_eval_lot_ett}"
RESULT_ROOT="${RESULT_ROOT:-${EXP_DIR}/results}"
RUN_NAME="${RUN_NAME:-nonfourier_projection_from_simple_complex}"
INIT_CHECKPOINT_DIR="${INIT_CHECKPOINT_DIR:-${OLD_EXP_DIR}/results/simple_complex_synth_fixed_phase_scale/train/simple_complex_coeff_residual_tail}"
DEVICE="${DEVICE:-cuda:0}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-256}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-64}"
MAX_STEPS="${MAX_STEPS:-3000}"
EVAL_BATCHES="${EVAL_BATCHES:-64}"
VAL_SPLIT="${VAL_SPLIT:-0.1}"
LEARNING_RATE="${LEARNING_RATE:-3e-4}"
HORIZONS=(${HORIZONS:-96 192 336 720})
STAGES=(${STAGES:-stage1_S stage2_T_S stage3_T_S_R})
RESIDUAL_DISTRIBUTIONS=(${RESIDUAL_DISTRIBUTIONS:-normal student_t exponential gamma weibull pareto})
LOG_DIR="${RESULT_ROOT}/logs"
SKIP_EXISTING_PROJECTION="${SKIP_EXISTING_PROJECTION:-1}"

mkdir -p "${LOG_DIR}"
cd "${EXP_DIR}"

echo "[projtarget] exp_dir=${EXP_DIR}"
echo "[projtarget] result_root=${RESULT_ROOT}"
echo "[projtarget] projection_root=${PROJECTION_ROOT}"
echo "[projtarget] train_root=${TRAIN_ROOT}"
echo "[projtarget] init_checkpoint_dir=${INIT_CHECKPOINT_DIR}"
echo "[projtarget] policy: seasonal decoder learns projected Fourier coefficients"
echo "[projtarget] policy: residual decoder learns seasonal projection remainder"
echo "[projtarget] policy: distribution noise from stage3 is excluded from training targets"
echo "[projtarget] started_at=$(date '+%Y-%m-%d %H:%M:%S')"

PROJECTION_ARGS=()
if [[ "${SKIP_EXISTING_PROJECTION}" == "1" ]]; then
  PROJECTION_ARGS+=(--skip_existing)
fi

echo "[projtarget] build projection targets"
python build_projection_targets.py \
  --train_root "${TRAIN_ROOT}" \
  --output_root "${PROJECTION_ROOT}" \
  "${PROJECTION_ARGS[@]}" \
  2>&1 | tee "${LOG_DIR}/build_projection_targets.log"

echo "[projtarget] train projection target model"
python train_projection_finetune.py \
  --train_root "${TRAIN_ROOT}" \
  --projection_root "${PROJECTION_ROOT}" \
  --init_checkpoint_dir "${INIT_CHECKPOINT_DIR}" \
  --results_dir "${RESULT_ROOT}/train" \
  --run_name "${RUN_NAME}" \
  --horizons "${HORIZONS[@]}" \
  --stages "${STAGES[@]}" \
  --residual_distributions "${RESIDUAL_DISTRIBUTIONS[@]}" \
  --max_steps "${MAX_STEPS}" \
  --batch_size "${TRAIN_BATCH_SIZE}" \
  --eval_batches "${EVAL_BATCHES}" \
  --val_split "${VAL_SPLIT}" \
  --learning_rate "${LEARNING_RATE}" \
  --device "${DEVICE}" \
  --save_checkpoint \
  2>&1 | tee "${LOG_DIR}/train_${RUN_NAME}.log"

CHECKPOINT_RUN_DIR="${RESULT_ROOT}/train/${RUN_NAME}"

echo "[projtarget] evaluate real LOTSA/ETTh1"
python "${OLD_EXP_DIR}/eval_real_lot_ett_single_model.py" \
  --real_root "${REAL_ROOT}" \
  --checkpoint_run_dir "${CHECKPOINT_RUN_DIR}" \
  --results_root "${RESULT_ROOT}" \
  --output_name "real_lot_ett_single_model" \
  --horizons "${HORIZONS[@]}" \
  --batch_size "${EVAL_BATCH_SIZE}" \
  --device "${DEVICE}" \
  2>&1 | tee "${LOG_DIR}/eval_real_${RUN_NAME}.log"

echo "[projtarget] evaluate non-Fourier synthetic"
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

echo "[projtarget] build extra summary plots"
python "/home/sia2/project/5.22syn_cent/plot_extra_result_summaries.py" "${RESULT_ROOT}" \
  2>&1 | tee "${LOG_DIR}/plot_extra_${RUN_NAME}.log"

echo "[projtarget] completed_at=$(date '+%Y-%m-%d %H:%M:%S')"
