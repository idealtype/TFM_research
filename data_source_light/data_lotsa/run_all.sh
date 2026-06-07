#!/usr/bin/env bash
set -euo pipefail

MAX_HORIZON="720"
HORIZONS=("96" "192" "336" "720")
DEVICE="cuda"
DATA_ROOT="${DATA_ROOT:-/workspace/data}"
WORK_DIR="${DATA_ROOT}/data_lotsa"
HF_CACHE_ARGS=()

usage() {
  cat <<'EOF'
Usage:
  bash run_all.sh [--max_horizon 720] [--horizons 96 192 336 720] [--device cuda] [--hf_cache_dir PATH]

Examples:
  nohup bash run_all.sh > run_all.log 2>&1 &
  nohup bash run_all.sh --max_horizon 720 --horizons 96 192 336 720 > run_all.log 2>&1 &
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --max_horizon)
      MAX_HORIZON="${2:-}"
      shift 2
      ;;
    --horizons)
      shift
      HORIZONS=()
      while [[ $# -gt 0 && "$1" != --* ]]; do
        HORIZONS+=("$1")
        shift
      done
      ;;
    --device)
      DEVICE="${2:-}"
      shift 2
      ;;
    --hf_cache_dir)
      HF_CACHE_ARGS=(--hf_cache_dir "${2:-}")
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown argument: $1"
      usage
      exit 1
      ;;
  esac
done

if [[ ${#HORIZONS[@]} -eq 0 ]]; then
  echo "[ERROR] At least one horizon is required."
  usage
  exit 1
fi

cd "${WORK_DIR}"

echo "============================================================"
echo "[RUN_ALL] Started at $(date '+%Y-%m-%d %H:%M:%S')"
echo "[RUN_ALL] work_dir=${WORK_DIR}"
echo "[RUN_ALL] max_horizon=${MAX_HORIZON}"
echo "[RUN_ALL] horizons=${HORIZONS[*]}"
echo "[RUN_ALL] device=${DEVICE}"
echo "============================================================"

run_stage() {
  local step_name="$1"
  shift

  echo ""
  echo "[${step_name}] start: $(date '+%Y-%m-%d %H:%M:%S')"
  "$@"
  echo "[${step_name}] done:  $(date '+%Y-%m-%d %H:%M:%S')"
}

if [[ -f lotsa_index.parquet ]]; then
  echo ""
  echo "[STEP 1 Train index] already exists: lotsa_index.parquet"
else
  run_stage "STEP 1 Train index" \
    python build_index.py \
      --max_horizon "${MAX_HORIZON}" \
      --output lotsa_index.parquet \
      --resume \
      "${HF_CACHE_ARGS[@]}"
fi

run_stage "STEP 2 Train backbone cache" \
  python build_cache.py \
    --index lotsa_index.parquet \
    --device "${DEVICE}" \
    "${HF_CACHE_ARGS[@]}"

for HORIZON in "${HORIZONS[@]}"; do
  run_stage "STEP 3 Train futures h${HORIZON}" \
    python build_futures.py \
      --index lotsa_index.parquet \
      --horizon "${HORIZON}" \
      "${HF_CACHE_ARGS[@]}"
done

if [[ -f lotsa_index_test.parquet ]]; then
  echo ""
  echo "[STEP 4 Test index] already exists: lotsa_index_test.parquet"
else
  run_stage "STEP 4 Test index" \
    python build_index_test.py \
      --max_horizon "${MAX_HORIZON}" \
      --output lotsa_index_test.parquet \
      "${HF_CACHE_ARGS[@]}"
fi

run_stage "STEP 5 Test backbone cache" \
  python build_cache_test.py \
    --index lotsa_index_test.parquet \
    --device "${DEVICE}" \
    "${HF_CACHE_ARGS[@]}"

for HORIZON in "${HORIZONS[@]}"; do
  run_stage "STEP 6 Test futures h${HORIZON}" \
    python build_futures_test.py \
      --index lotsa_index_test.parquet \
      --horizon "${HORIZON}" \
      "${HF_CACHE_ARGS[@]}"
done

run_stage "STEP 7 Train seasonality masking cache" \
  python build_seasonality_mask.py \
    --cache_dir lotsa_cache/ \
    --horizons "${HORIZONS[@]}"

run_stage "STEP 8 Test seasonality masking cache" \
  python build_seasonality_mask.py \
    --cache_dir lotsa_cache_test/ \
    --horizons "${HORIZONS[@]}"

echo ""
echo "============================================================"
echo "[RUN_ALL] Completed successfully at $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
