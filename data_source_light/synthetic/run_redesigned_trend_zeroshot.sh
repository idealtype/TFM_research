#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/workspace/data}"
SYNTH_ROOT="${SYNTH_ROOT:-${DATA_ROOT}/synthetic}"
PROJECT_ROOT="${PROJECT_ROOT:-/workspace}"
EXP_DIR="${EXP_DIR:-${PROJECT_ROOT}/4.28basis/basis_dec/experiment/func_dec_np_trend}"
RUN_DIR="${RUN_DIR:-${PROJECT_ROOT}/4.28basis/basis_dec/experiment/func_dec_np_trend/runs/elecdemand_holdout_03_030807}"
RESULTS_DIR="${RESULTS_DIR:-${RUN_DIR}/zeroshot_synth_trend_redesigned}"
DEVICE="${DEVICE:-cuda:0}"
BATCH_SIZE="${BATCH_SIZE:-64}"
SEED="${SEED:-42}"
CONTEXT_LEN="${CONTEXT_LEN:-512}"
N_SAMPLES="${N_SAMPLES:-4}"
LEVELS=(${LEVELS:-T1 T2 T3 T4 T5 T6})
HORIZONS=(${HORIZONS:-96 192 336 720})

HF_CACHE_ARGS=()
if [[ -n "${HF_CACHE_DIR:-}" ]]; then
  HF_CACHE_ARGS=(--hf_cache_dir "${HF_CACHE_DIR}")
fi

echo "============================================================"
echo "[synthetic trend zero-shot] started at $(date '+%Y-%m-%d %H:%M:%S')"
echo "SYNTH_ROOT=${SYNTH_ROOT}"
echo "EXP_DIR=${EXP_DIR}"
echo "RUN_DIR=${RUN_DIR}"
echo "RESULTS_DIR=${RESULTS_DIR}"
echo "DEVICE=${DEVICE}"
echo "LEVELS=${LEVELS[*]}"
echo "HORIZONS=${HORIZONS[*]}"
echo "============================================================"

echo ""
echo "[1/3] Generate redesigned trend synthetic data"
python "${SYNTH_ROOT}/synth_generator.py" \
  --config "${SYNTH_ROOT}/synth_config.yaml" \
  --output_dir "${SYNTH_ROOT}/trend" \
  --levels "${LEVELS[@]}" \
  --horizons "${HORIZONS[@]}" \
  --context_len "${CONTEXT_LEN}" \
  --n_samples "${N_SAMPLES}" \
  --seed "${SEED}"

echo ""
echo "[2/3] Build Monash-compatible synthetic cache"
python "${SYNTH_ROOT}/prepare_synth_cache.py" \
  --category trend \
  --levels "${LEVELS[@]}" \
  --horizons "${HORIZONS[@]}" \
  --seed "${SEED}" \
  --batch_size "${BATCH_SIZE}" \
  --device "${DEVICE}" \
  "${HF_CACHE_ARGS[@]}"

echo ""
echo "[3/3] Run FuncDec synthetic zero-shot evaluation"
cd "${EXP_DIR}"
python zeroshot_eval_synth.py \
  --run_dir "${RUN_DIR}" \
  --cache_root "${SYNTH_ROOT}/cache" \
  --synth_root "${SYNTH_ROOT}" \
  --category trend \
  --levels "${LEVELS[@]}" \
  --horizons "${HORIZONS[@]}" \
  --batch_size "${BATCH_SIZE}" \
  --device "${DEVICE}" \
  --results_dir "${RESULTS_DIR}" \
  --plot_samples "${N_SAMPLES}" \
  --plot_seed "${SEED}"

echo ""
echo "[synthetic trend zero-shot] done at $(date '+%Y-%m-%d %H:%M:%S')"
echo "results=${RESULTS_DIR}"
