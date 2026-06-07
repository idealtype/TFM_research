#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/workspace/data}"
ROOT="${DATA_ROOT}/synthetic_nonF"
LOG_DIR="${ROOT}/logs"
DEVICE="${DEVICE:-cuda:0}"
BATCH_SIZE="${BATCH_SIZE:-64}"
STAGE1_N_SAMPLES="${STAGE1_N_SAMPLES:-200}"
STAGE2_N_SAMPLES="${STAGE2_N_SAMPLES:-20}"
STAGE3_N_SAMPLES="${STAGE3_N_SAMPLES:-10}"

mkdir -p "${LOG_DIR}"
cd "${ROOT}"

echo "[nonfourier] root=${ROOT}"
echo "[nonfourier] device=${DEVICE}"
echo "[nonfourier] batch_size=${BATCH_SIZE}"
echo "[nonfourier] samples_per_group stage1=${STAGE1_N_SAMPLES} stage2=${STAGE2_N_SAMPLES} stage3=${STAGE3_N_SAMPLES}"
echo "[nonfourier] started_at=$(date '+%Y-%m-%d %H:%M:%S')"

echo "[stage1] cache S -> synth_eval_nonfourier/stage1_S_nonfourier_cache_10_4_8"
python prepare_nonfourier_synth_cache.py \
  --n_samples "${STAGE1_N_SAMPLES}" \
  --device "${DEVICE}" \
  --batch_size "${BATCH_SIZE}" \
  2>&1 | tee "${LOG_DIR}/stage1_S_nonfourier_cache.log"

echo "[stage2] generate T+S -> synth_eval_nonfourier/stage2_T_S_nonfourier"
python generate_nonfourier_ts.py \
  --n_samples "${STAGE2_N_SAMPLES}" \
  2>&1 | tee "${LOG_DIR}/stage2_T_S_nonfourier_generate.log"

echo "[stage2] cache T+S -> synth_eval_nonfourier/stage2_T_S_nonfourier_cache_10_4_8"
python prepare_nonfourier_ts_cache.py \
  --device "${DEVICE}" \
  --batch_size "${BATCH_SIZE}" \
  2>&1 | tee "${LOG_DIR}/stage2_T_S_nonfourier_cache.log"

echo "[stage3] generate T+S+R -> synth_eval_nonfourier/stage3_T_S_R_nonfourier"
python generate_nonfourier_tsr.py \
  --n_samples "${STAGE3_N_SAMPLES}" \
  2>&1 | tee "${LOG_DIR}/stage3_T_S_R_nonfourier_generate.log"

echo "[stage3] cache T+S+R -> synth_eval_nonfourier/stage3_T_S_R_nonfourier_cache_10_4_8"
python prepare_nonfourier_tsr_cache.py \
  --device "${DEVICE}" \
  --batch_size "${BATCH_SIZE}" \
  2>&1 | tee "${LOG_DIR}/stage3_T_S_R_nonfourier_cache.log"

echo "[validate] per-group cache sample counts"
python - <<PY
from pathlib import Path
import torch

checks = [
    ("stage1", Path("${ROOT}/synth_eval_nonfourier/stage1_S_nonfourier_cache_10_4_8"), int("${STAGE1_N_SAMPLES}")),
    ("stage2", Path("${ROOT}/synth_eval_nonfourier/stage2_T_S_nonfourier_cache_10_4_8"), int("${STAGE2_N_SAMPLES}")),
    ("stage3", Path("${ROOT}/synth_eval_nonfourier/stage3_T_S_R_nonfourier_cache_10_4_8"), int("${STAGE3_N_SAMPLES}")),
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

echo "[nonfourier] completed_at=$(date '+%Y-%m-%d %H:%M:%S')"
