#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/workspace/data}"
SYNTH_ROOT="${SYNTH_ROOT:-${DATA_ROOT}/synthetic}"
PROJECT_ROOT="${PROJECT_ROOT:-/workspace}"
EXP_DIR="${EXP_DIR:-${PROJECT_ROOT}/4.28basis/basis_dec/experiment/func_dec_np_trend}"
RUN_DIR="${RUN_DIR:-${PROJECT_ROOT}/4.28basis/basis_dec/experiment/func_dec_np_trend/runs/elecdemand_holdout_03_030807}"
RESULTS_DIR="${RESULTS_DIR:-${RUN_DIR}/zeroshot_synth_all_redesigned}"
DEVICE="${DEVICE:-cuda:0}"
BATCH_SIZE="${BATCH_SIZE:-64}"
SEED="${SEED:-42}"
CONTEXT_LEN="${CONTEXT_LEN:-512}"
N_SAMPLES="${N_SAMPLES:-4}"
TREND_LEVELS=(${TREND_LEVELS:-T1 T2 T3 T4 T5 T6})
SEASONAL_LEVELS=(${SEASONAL_LEVELS:-S1 S2 S3 S4 S5 S6 S7 S8 S9 S10})
HORIZONS=(${HORIZONS:-96 192 336 720})

HF_CACHE_ARGS=()
if [[ -n "${HF_CACHE_DIR:-}" ]]; then
  HF_CACHE_ARGS=(--hf_cache_dir "${HF_CACHE_DIR}")
fi

echo "============================================================"
echo "[full synthetic zero-shot] started at $(date '+%Y-%m-%d %H:%M:%S')"
echo "SYNTH_ROOT=${SYNTH_ROOT}"
echo "EXP_DIR=${EXP_DIR}"
echo "RUN_DIR=${RUN_DIR}"
echo "RESULTS_DIR=${RESULTS_DIR}"
echo "DEVICE=${DEVICE}"
echo "TREND_LEVELS=${TREND_LEVELS[*]}"
echo "SEASONAL_LEVELS=${SEASONAL_LEVELS[*]}"
echo "HORIZONS=${HORIZONS[*]}"
echo "============================================================"

echo ""
echo "[1/4] Generate redesigned trend and seasonal synthetic data"
python "${SYNTH_ROOT}/generate_all_synth.py" \
  --config "${SYNTH_ROOT}/synth_config.yaml" \
  --trend_output_dir "${SYNTH_ROOT}/trend" \
  --seasonal_output_dir "${SYNTH_ROOT}/seasonal" \
  --trend_levels "${TREND_LEVELS[@]}" \
  --seasonal_levels "${SEASONAL_LEVELS[@]}" \
  --horizons "${HORIZONS[@]}" \
  --context_len "${CONTEXT_LEN}" \
  --n_samples "${N_SAMPLES}" \
  --seed "${SEED}"

echo ""
echo "[2/4] Build trend synthetic cache"
python "${SYNTH_ROOT}/prepare_synth_cache.py" \
  --category trend \
  --levels "${TREND_LEVELS[@]}" \
  --horizons "${HORIZONS[@]}" \
  --seed "${SEED}" \
  --batch_size "${BATCH_SIZE}" \
  --device "${DEVICE}" \
  "${HF_CACHE_ARGS[@]}"

echo ""
echo "[3/4] Build seasonal synthetic cache"
python "${SYNTH_ROOT}/prepare_synth_cache.py" \
  --category seasonal \
  --levels "${SEASONAL_LEVELS[@]}" \
  --horizons "${HORIZONS[@]}" \
  --seed "${SEED}" \
  --batch_size "${BATCH_SIZE}" \
  --device "${DEVICE}" \
  "${HF_CACHE_ARGS[@]}"

echo ""
echo "[4/4] Run FuncDec synthetic zero-shot evaluation for all synthetic data"
cd "${EXP_DIR}"
python zeroshot_eval_synth.py \
  --run_dir "${RUN_DIR}" \
  --cache_root "${SYNTH_ROOT}/cache" \
  --synth_root "${SYNTH_ROOT}" \
  --category all \
  --levels "${TREND_LEVELS[@]}" "${SEASONAL_LEVELS[@]}" \
  --horizons "${HORIZONS[@]}" \
  --batch_size "${BATCH_SIZE}" \
  --device "${DEVICE}" \
  --results_dir "${RESULTS_DIR}" \
  --plot_samples "${N_SAMPLES}" \
  --plot_seed "${SEED}"

echo ""
echo "[full synthetic zero-shot] done at $(date '+%Y-%m-%d %H:%M:%S')"
echo "results=${RESULTS_DIR}"
