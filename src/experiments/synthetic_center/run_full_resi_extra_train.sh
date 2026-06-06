#!/usr/bin/env bash
set -euo pipefail

cd /home/sia2/project/5.22syn_cent

DEVICE="${DEVICE:-cuda:0}"
MAX_STEPS="${MAX_STEPS:-1000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
GROUP_CHUNK_STEPS="${GROUP_CHUNK_STEPS:-200}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-64}"
MODEL_ONLY_EVAL="${MODEL_ONLY_EVAL:-1}"
RUN_NAME="${RUN_NAME:-residual_extra_from_full_phasefix_lr1e4_s1000}"
EVAL_OUTPUT_NAME="${EVAL_OUTPUT_NAME:-real_lot_ett_single_model_lr1e4_s1000}"

EXP_DIR="/home/sia2/project/5.22syn_cent/train_syn_real_raw"
RESULT_ROOT="${EXP_DIR}/results/full_resi_extra_train"
TRAIN_ROOT="${RESULT_ROOT}/train/${RUN_NAME}"
LOG_DIR="${RESULT_ROOT}/logs"

mkdir -p "${LOG_DIR}"

echo "[1/2] residual-only extra training from full phasefix checkpoints"
python "${EXP_DIR}/train/train_residual_extra_finetune.py" \
  --device "${DEVICE}" \
  --max_steps "${MAX_STEPS}" \
  --batch_size "${BATCH_SIZE}" \
  --group_chunk_steps "${GROUP_CHUNK_STEPS}" \
  --learning_rate "${LEARNING_RATE}" \
  --init_checkpoint_dir "${EXP_DIR}/results/train/domain_real_finetune_phasefix" \
  --results_dir "${TRAIN_ROOT}" \
  --save_checkpoint 2>&1 | tee "${LOG_DIR}/train_residual_extra_finetune_${RUN_NAME}.log"

echo "[2/2] real-only evaluation"
MODEL_ONLY_ARGS=()
if [[ "${MODEL_ONLY_EVAL}" == "1" || "${MODEL_ONLY_EVAL}" == "true" ]]; then
  MODEL_ONLY_ARGS+=(--model_only --skip_tfm)
fi
python "${EXP_DIR}/train/eval_residual_extra_domain_models.py" \
  --device "${DEVICE}" \
  --batch_size "${EVAL_BATCH_SIZE}" \
  --checkpoint_root "${TRAIN_ROOT}" \
  --results_root "${RESULT_ROOT}" \
  --output_name "${EVAL_OUTPUT_NAME}" \
  "${MODEL_ONLY_ARGS[@]}" 2>&1 | tee "${LOG_DIR}/eval_residual_extra_domain_models_${RUN_NAME}.log"

python /home/sia2/project/5.22syn_cent/plot_extra_result_summaries.py \
  "${RESULT_ROOT}" 2>&1 | tee "${LOG_DIR}/plot_extra_result_summaries_${RUN_NAME}.log"

echo "done"
echo "train checkpoints: ${TRAIN_ROOT}"
echo "eval output: ${RESULT_ROOT}/${EVAL_OUTPUT_NAME}"
