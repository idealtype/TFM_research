#!/usr/bin/env bash
set -euo pipefail

cd /home/sia2/project/5.22syn_cent

DEVICE="${DEVICE:-cuda:0}"
MAX_STEPS="${MAX_STEPS:-1000}"
BATCH_SIZE="${BATCH_SIZE:-128}"
GROUP_CHUNK_STEPS="${GROUP_CHUNK_STEPS:-100}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-64}"
HORIZONS="${HORIZONS:-96 192 336 720}"

EXP_DIR="/home/sia2/project/5.22syn_cent/train_syn_real_raw"
TRAIN_RUN_NAME="${TRAIN_RUN_NAME:-domain_real_finetune_phasefix}"
EVAL_OUTPUT_NAME="${EVAL_OUTPUT_NAME:-real_lot_ett_single_model_phasefix}"
TRAIN_RESULTS_DIR="${EXP_DIR}/results/train/${TRAIN_RUN_NAME}"
LOG_DIR="${EXP_DIR}/results/logs"

mkdir -p "${LOG_DIR}"

echo "[1/5] overwrite LOTSA train Fourier basis caches"
python /home/sia2/project/data/data_lotsa/build_seasonality_mask.py \
  --cache_dir /home/sia2/project/data/data_lotsa/lotsa_cache \
  --horizons ${HORIZONS} \
  --overwrite 2>&1 | tee "${LOG_DIR}/phasefix_recache_lotsa_cache.log"

echo "[2/5] overwrite LOTSA test Fourier basis caches"
python /home/sia2/project/data/data_lotsa/build_seasonality_mask.py \
  --cache_dir /home/sia2/project/data/data_lotsa/lotsa_cache_test \
  --horizons ${HORIZONS} \
  --overwrite 2>&1 | tee "${LOG_DIR}/phasefix_recache_lotsa_cache_test.log"

echo "[3/5] verify representative basis phase"
python - <<'PY' 2>&1 | tee "/home/sia2/project/5.22syn_cent/train_syn_real_raw/results/logs/phasefix_verify_basis.log"
import math
from pathlib import Path
import torch

checks = [
    (Path("/home/sia2/project/data/data_lotsa/lotsa_cache/australian_electricity_demand/fourier_basis_c512_half_hourly_h96_lotsa.pt"), "half_hourly"),
    (Path("/home/sia2/project/data/data_lotsa/lotsa_cache/PEMS04/fourier_basis_c512_5_minutes_h96_lotsa.pt"), "5_minutes"),
    (Path("/home/sia2/project/data/data_lotsa/lotsa_cache/gfc12_load/fourier_basis_c512_hourly_h96_lotsa.pt"), "hourly"),
]
freq_days = {"5_minutes": 1 / 288, "half_hourly": 1 / 48, "hourly": 1 / 24}
for path, freq in checks:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    got = payload["daily_basis"][0, :2]
    fd = freq_days[freq]
    period_steps = 1.0 / fd
    expected = torch.tensor([
        math.sin(2.0 * math.pi * 512 / period_steps),
        math.cos(2.0 * math.pi * 512 / period_steps),
    ], dtype=torch.float32)
    err = float((got - expected).abs().max().item())
    print(f"{path}: got={got.tolist()} expected={expected.tolist()} max_abs_err={err:.3e}")
    if err > 1e-4:
        raise SystemExit(f"basis phase verification failed: {path}")
PY

echo "[4/5] retrain domain real fine-tune checkpoints"
rm -rf "${TRAIN_RESULTS_DIR}"
python "${EXP_DIR}/train_domain_real_finetune.py" \
  --device "${DEVICE}" \
  --max_steps "${MAX_STEPS}" \
  --batch_size "${BATCH_SIZE}" \
  --group_chunk_steps "${GROUP_CHUNK_STEPS}" \
  --results_dir "${TRAIN_RESULTS_DIR}" \
  --save_checkpoint 2>&1 | tee "${LOG_DIR}/phasefix_train_domain_real_finetune.log"

echo "[5/5] evaluate real LOTSA/ETT targets only"
python "${EXP_DIR}/eval_real_lot_ett_domain_models.py" \
  --device "${DEVICE}" \
  --batch_size "${EVAL_BATCH_SIZE}" \
  --checkpoint_root "${TRAIN_RESULTS_DIR}" \
  --results_root "${EXP_DIR}/results" \
  --output_name "${EVAL_OUTPUT_NAME}" 2>&1 | tee "${LOG_DIR}/phasefix_eval_real_lot_ett_domain_models.log"

python /home/sia2/project/5.22syn_cent/plot_extra_result_summaries.py \
  "${EXP_DIR}/results" 2>&1 | tee "${LOG_DIR}/phasefix_plot_extra_result_summaries.log"

echo "done"
echo "train checkpoints: ${TRAIN_RESULTS_DIR}"
echo "eval output: ${EXP_DIR}/results/${EVAL_OUTPUT_NAME}"
