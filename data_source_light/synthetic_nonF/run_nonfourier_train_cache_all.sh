#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/workspace/data}"
ROOT="${DATA_ROOT}/synthetic_nonF"
TRAIN_ROOT="${TRAIN_ROOT:-${ROOT}/synth_train_nonfourier}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs_train}"
DEVICE="${DEVICE:-cuda:0}"
BATCH_SIZE="${BATCH_SIZE:-64}"
SEED="${SEED:-42}"
STAGE1_N_SAMPLES="${STAGE1_N_SAMPLES:-1000}"
STAGE2_N_SAMPLES="${STAGE2_N_SAMPLES:-500}"
STAGE3_N_SAMPLES="${STAGE3_N_SAMPLES:-250}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
METADATA_ONLY="${METADATA_ONLY:-0}"

EXTRA_ARGS=()
if [[ "${SKIP_EXISTING}" == "1" ]]; then
  EXTRA_ARGS+=(--skip_existing)
fi

CACHE_EXTRA_ARGS=()
if [[ "${SKIP_EXISTING}" == "1" ]]; then
  CACHE_EXTRA_ARGS+=(--skip_existing)
fi
if [[ "${METADATA_ONLY}" == "1" ]]; then
  CACHE_EXTRA_ARGS+=(--metadata_only)
fi

mkdir -p "${LOG_DIR}"
cd "${ROOT}"

echo "[nonfourier-train] root=${ROOT}"
echo "[nonfourier-train] train_root=${TRAIN_ROOT}"
echo "[nonfourier-train] device=${DEVICE}"
echo "[nonfourier-train] batch_size=${BATCH_SIZE}"
echo "[nonfourier-train] seed=${SEED}"
echo "[nonfourier-train] samples_per_group stage1=${STAGE1_N_SAMPLES} stage2=${STAGE2_N_SAMPLES} stage3=${STAGE3_N_SAMPLES}"
echo "[nonfourier-train] started_at=$(date '+%Y-%m-%d %H:%M:%S')"

STAGE1_DATA="${TRAIN_ROOT}/stage1_S_nonfourier"
STAGE1_CACHE="${TRAIN_ROOT}/stage1_S_nonfourier_cache_10_4_8"
STAGE2_DATA="${TRAIN_ROOT}/stage2_T_S_nonfourier"
STAGE2_CACHE="${TRAIN_ROOT}/stage2_T_S_nonfourier_cache_10_4_8"
STAGE3_DATA="${TRAIN_ROOT}/stage3_T_S_R_nonfourier"
STAGE3_CACHE="${TRAIN_ROOT}/stage3_T_S_R_nonfourier_cache_10_4_8"

echo "[stage1] cache S -> ${STAGE1_CACHE}"
python prepare_nonfourier_synth_cache.py \
  --data_root "${STAGE1_DATA}" \
  --cache_root "${STAGE1_CACHE}" \
  --n_samples "${STAGE1_N_SAMPLES}" \
  --seed "${SEED}" \
  --device "${DEVICE}" \
  --batch_size "${BATCH_SIZE}" \
  "${CACHE_EXTRA_ARGS[@]}" \
  2>&1 | tee "${LOG_DIR}/stage1_S_nonfourier_train_cache.log"

echo "[stage2] generate T+S -> ${STAGE2_DATA}"
python generate_nonfourier_ts.py \
  --seasonal_root "${STAGE1_DATA}" \
  --output_root "${STAGE2_DATA}" \
  --n_samples "${STAGE2_N_SAMPLES}" \
  --seed "${SEED}" \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee "${LOG_DIR}/stage2_T_S_nonfourier_train_generate.log"

echo "[stage2] cache T+S -> ${STAGE2_CACHE}"
python prepare_nonfourier_ts_cache.py \
  --input_root "${STAGE2_DATA}" \
  --output_root "${STAGE2_CACHE}" \
  --seed "${SEED}" \
  --device "${DEVICE}" \
  --batch_size "${BATCH_SIZE}" \
  "${CACHE_EXTRA_ARGS[@]}" \
  2>&1 | tee "${LOG_DIR}/stage2_T_S_nonfourier_train_cache.log"

echo "[stage3] generate T+S+R -> ${STAGE3_DATA}"
python generate_nonfourier_tsr.py \
  --input_root "${STAGE2_DATA}" \
  --output_root "${STAGE3_DATA}" \
  --n_samples "${STAGE3_N_SAMPLES}" \
  --seed "${SEED}" \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee "${LOG_DIR}/stage3_T_S_R_nonfourier_train_generate.log"

echo "[stage3] cache T+S+R -> ${STAGE3_CACHE}"
python prepare_nonfourier_tsr_cache.py \
  --input_root "${STAGE3_DATA}" \
  --output_root "${STAGE3_CACHE}" \
  --seed "${SEED}" \
  --device "${DEVICE}" \
  --batch_size "${BATCH_SIZE}" \
  "${CACHE_EXTRA_ARGS[@]}" \
  2>&1 | tee "${LOG_DIR}/stage3_T_S_R_nonfourier_train_cache.log"

echo "[validate] per-group cache sample counts"
python - <<PY
from pathlib import Path
import torch

checks = [
    ("stage1", Path("${STAGE1_CACHE}"), int("${STAGE1_N_SAMPLES}")),
    ("stage2", Path("${STAGE2_CACHE}"), int("${STAGE2_N_SAMPLES}")),
    ("stage3", Path("${STAGE3_CACHE}"), int("${STAGE3_N_SAMPLES}")),
]
for label, root, expected in checks:
    bad = []
    total = 0
    for raw_path in sorted(root.rglob("raw_futures_h*.pt")):
        total += 1
        payload = torch.load(raw_path, map_location="cpu", weights_only=False)
        got = int(payload["futures_n"].shape[0])
        if got != expected:
            bad.append((str(raw_path.parent), got))
    if bad:
        examples = "\n".join(f"  {path}: {got}" for path, got in bad[:10])
        raise SystemExit(f"{label} expected {expected} samples per cache group, found {len(bad)} mismatches:\n{examples}")
    print(f"{label}: groups={total} samples_per_group={expected}")
PY

echo "[nonfourier-train] completed_at=$(date '+%Y-%m-%d %H:%M:%S')"
