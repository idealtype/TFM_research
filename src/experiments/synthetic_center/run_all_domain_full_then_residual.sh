#!/usr/bin/env bash
set -euo pipefail

cd /home/sia2/project/5.22syn_cent

DEVICE="${DEVICE:-cuda:0}"
HORIZONS="${HORIZONS:-96 192 336 720}"
BATCH_SIZE="${BATCH_SIZE:-256}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-64}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
FULL_STEPS="${FULL_STEPS:-10000}"
FULL_GROUP_CHUNK_STEPS="${FULL_GROUP_CHUNK_STEPS:-250}"
RESIDUAL_STEPS="${RESIDUAL_STEPS:-4000}"
RESIDUAL_GROUP_CHUNK_STEPS="${RESIDUAL_GROUP_CHUNK_STEPS:-100}"
SYNTH_SAMPLES_PER_GROUP="${SYNTH_SAMPLES_PER_GROUP:-8}"
SYNTH_GROUP_STRIDE="${SYNTH_GROUP_STRIDE:-2}"
SYNTH_GROUP_OFFSET="${SYNTH_GROUP_OFFSET:-0}"

EXP_DIR="/home/sia2/project/5.22syn_cent/train_syn_real_raw"
OLD_EXP_DIR="/home/sia2/project/4.28basis/basis_dec/experiment/func_dec_syn_cent"
RESULT_ROOT="${EXP_DIR}/results/all_domain_full_then_residual"
TRAIN_ROOT="${RESULT_ROOT}/train"
FULL_RUN_NAME="all_domain_full"
RESIDUAL_RUN_NAME="all_domain_full_residual_extra"
INIT_CHECKPOINT_DIR="${INIT_CHECKPOINT_DIR:-/home/sia2/project/5.22syn_cent/train_nonF_rawtarget/results/train/nonfourier_finetune_from_simple_complex}"
REAL_ROOT="${REAL_ROOT:-/home/sia2/project/data/real_eval_lot_ett}"
F_SYNTH_CACHE_ROOT="${F_SYNTH_CACHE_ROOT:-/home/sia2/project/data/synthetic/func_dec_syn_cent_complex_eval_cache_10_4_8_fixed_phase_scale}"
F_SYNTH_ROOT="${F_SYNTH_ROOT:-/home/sia2/project/data/synthetic/func_dec_syn_cent_complex_eval_fixed_phase_scale}"
NONF_SYNTH_ROOT="${NONF_SYNTH_ROOT:-/home/sia2/project/data/synthetic_nonF/synth_eval_nonfourier}"
LOG_DIR="${RESULT_ROOT}/logs"

mkdir -p "${LOG_DIR}"

echo "[1/4] train all-domain full decoders then residual-only"
python "${EXP_DIR}/train_all_domain_real_then_residual.py" \
  --device "${DEVICE}" \
  --horizons ${HORIZONS} \
  --batch_size "${BATCH_SIZE}" \
  --learning_rate "${LEARNING_RATE}" \
  --full_steps "${FULL_STEPS}" \
  --full_group_chunk_steps "${FULL_GROUP_CHUNK_STEPS}" \
  --residual_steps "${RESIDUAL_STEPS}" \
  --residual_group_chunk_steps "${RESIDUAL_GROUP_CHUNK_STEPS}" \
  --init_checkpoint_dir "${INIT_CHECKPOINT_DIR}" \
  --results_root "${RESULT_ROOT}" \
  2>&1 | tee "${LOG_DIR}/train_all_domain_full_then_residual.log"

echo "[2/6] evaluate real targets: full"
python "${EXP_DIR}/eval_real_lot_ett_global_model.py" \
  --device "${DEVICE}" \
  --horizons ${HORIZONS} \
  --real_root "${REAL_ROOT}" \
  --checkpoint_root "${TRAIN_ROOT}/${FULL_RUN_NAME}" \
  --results_root "${RESULT_ROOT}/eval" \
  --output_name "real_lot_ett_full" \
  --batch_size "${EVAL_BATCH_SIZE}" \
  --skip_tfm \
  2>&1 | tee "${LOG_DIR}/eval_real_global_full.log"

echo "[3/6] evaluate real targets: residual-extra"
python "${EXP_DIR}/eval_real_lot_ett_global_model.py" \
  --device "${DEVICE}" \
  --horizons ${HORIZONS} \
  --real_root "${REAL_ROOT}" \
  --checkpoint_root "${TRAIN_ROOT}/${RESIDUAL_RUN_NAME}" \
  --results_root "${RESULT_ROOT}/eval" \
  --output_name "real_lot_ett_residual_extra" \
  --batch_size "${EVAL_BATCH_SIZE}" \
  --skip_tfm \
  2>&1 | tee "${LOG_DIR}/eval_real_global_residual_extra.log"

echo "[4/6] evaluate F synthetic half groups: full and residual-extra"
python "${OLD_EXP_DIR}/eval_simple_on_complex.py" \
  --run_root "${TRAIN_ROOT}" \
  --run_names "${FULL_RUN_NAME}" "${RESIDUAL_RUN_NAME}" \
  --cache_root "${F_SYNTH_CACHE_ROOT}" \
  --synth_root "${F_SYNTH_ROOT}" \
  --results_dir "${RESULT_ROOT}/eval/synth_F" \
  --horizons ${HORIZONS} \
  --samples_per_group "${SYNTH_SAMPLES_PER_GROUP}" \
  --group_stride "${SYNTH_GROUP_STRIDE}" \
  --group_offset "${SYNTH_GROUP_OFFSET}" \
  --batch_size "${EVAL_BATCH_SIZE}" \
  --device "${DEVICE}" \
  2>&1 | tee "${LOG_DIR}/eval_synth_F_half.log"

echo "[5/6] evaluate non-F synthetic half groups: full"
python "${OLD_EXP_DIR}/eval_nonfourier_single_model.py" \
  --nonfourier_root "${NONF_SYNTH_ROOT}" \
  --checkpoint_run_dir "${TRAIN_ROOT}/${FULL_RUN_NAME}" \
  --results_root "${RESULT_ROOT}/eval" \
  --output_name "synth_nonF_full" \
  --horizons ${HORIZONS} \
  --samples_per_group "${SYNTH_SAMPLES_PER_GROUP}" \
  --group_stride "${SYNTH_GROUP_STRIDE}" \
  --group_offset "${SYNTH_GROUP_OFFSET}" \
  --batch_size "${EVAL_BATCH_SIZE}" \
  --device "${DEVICE}" \
  2>&1 | tee "${LOG_DIR}/eval_synth_nonF_full_half.log"

echo "[6/6] evaluate non-F synthetic half groups: residual-extra"
python "${OLD_EXP_DIR}/eval_nonfourier_single_model.py" \
  --nonfourier_root "${NONF_SYNTH_ROOT}" \
  --checkpoint_run_dir "${TRAIN_ROOT}/${RESIDUAL_RUN_NAME}" \
  --results_root "${RESULT_ROOT}/eval" \
  --output_name "synth_nonF_residual_extra" \
  --horizons ${HORIZONS} \
  --samples_per_group "${SYNTH_SAMPLES_PER_GROUP}" \
  --group_stride "${SYNTH_GROUP_STRIDE}" \
  --group_offset "${SYNTH_GROUP_OFFSET}" \
  --batch_size "${EVAL_BATCH_SIZE}" \
  --device "${DEVICE}" \
  2>&1 | tee "${LOG_DIR}/eval_synth_nonF_half.log"

echo "done"
echo "results: ${RESULT_ROOT}"
echo "real eval full: ${RESULT_ROOT}/eval/real_lot_ett_full"
echo "real eval residual-extra: ${RESULT_ROOT}/eval/real_lot_ett_residual_extra"
echo "F synth eval: ${RESULT_ROOT}/eval/synth_F"
echo "nonF synth eval full: ${RESULT_ROOT}/eval/synth_nonF_full"
echo "nonF synth eval residual-extra: ${RESULT_ROOT}/eval/synth_nonF_residual_extra"
