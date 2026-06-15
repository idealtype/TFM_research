#!/usr/bin/env bash
# Soft-mask Fourier warm-start + real-dominant mixed training/evaluation.
set -euo pipefail

DEVICE=${DEVICE:-cuda:0}
PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS=${RESULTS_ROOT:-$PROJECT/results/fourier_warm_real_mix_synth13_b1024_parallel_trend_seasonal_loss}
LOG_DIR="$RESULTS/logs"
LOG_FILE="$LOG_DIR/run_$(date +%Y%m%d_%H%M%S).log"
HORIZONS=(${HORIZONS:-96 192 336 720})
BATCH_SIZE=${BATCH_SIZE:-1024}
FOURIER_WARMUP_STEPS=${FOURIER_WARMUP_STEPS:-125}
MIXED_STEPS=${MIXED_STEPS:-2500}
RESIDUAL_STEPS=${RESIDUAL_STEPS:-500}
SYNTH_INTERVAL=${SYNTH_INTERVAL:-10}
REAL_GROUP_CHUNK_STEPS=${REAL_GROUP_CHUNK_STEPS:-63}
REAL_ROOT=${REAL_ROOT:-}

mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=== soft-mask warm-real-mix pipeline start $(date '+%Y-%m-%d %H:%M:%S') ==="
echo "DEVICE=$DEVICE"
echo "RESULTS=$RESULTS"
echo "LOG_FILE=$LOG_FILE"
echo "HORIZONS=${HORIZONS[*]}"
echo "BATCH_SIZE=$BATCH_SIZE"
echo "FOURIER_WARMUP_STEPS=$FOURIER_WARMUP_STEPS MIXED_STEPS=$MIXED_STEPS RESIDUAL_STEPS=$RESIDUAL_STEPS"
echo "SYNTH_INTERVAL=$SYNTH_INTERVAL REAL_GROUP_CHUNK_STEPS=$REAL_GROUP_CHUNK_STEPS"

TRAIN_ROOT="$RESULTS/_horizon_runs"
mkdir -p "$TRAIN_ROOT"

run_horizon() {
  local horizon="$1"
  local horizon_root="$TRAIN_ROOT/h${horizon}"
  mkdir -p "$horizon_root/logs"
  DEVICE=$DEVICE python "$PROJECT/train_warm_real_mix.py" \
    --init_checkpoint_dir none \
    --results_root "$horizon_root" \
    --horizons "$horizon" \
    --fourier_warmup_steps "$FOURIER_WARMUP_STEPS" \
    --mixed_steps "$MIXED_STEPS" \
    --residual_steps "$RESIDUAL_STEPS" \
    --synth_interval "$SYNTH_INTERVAL" \
    --real_group_chunk_steps "$REAL_GROUP_CHUNK_STEPS" \
    --batch_size "$BATCH_SIZE" \
    --learning_rate 1e-4 \
    --residual_learning_rate 1e-4 \
    --ts_corr_weight 0.01 \
    --gate_l1_weight 1e-3 2>&1 | sed "s/^/[train h${horizon}] /"
}

echo "=== Phase 1: Train warm-real-mix horizons in parallel ==="
pids=()
for horizon in "${HORIZONS[@]}"; do
  run_horizon "$horizon" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
if [[ "$status" -ne 0 ]]; then
  echo "[error] one or more horizon training jobs failed"
  exit 1
fi

echo "=== Merge per-horizon train outputs ==="
python - "$RESULTS" "$TRAIN_ROOT" "${HORIZONS[@]}" <<'PY'
import json
import shutil
import sys
from pathlib import Path

results = Path(sys.argv[1])
train_root = Path(sys.argv[2])
horizons = [int(x) for x in sys.argv[3:]]
ckpt_dir = results / "checkpoints"
ckpt_dir.mkdir(parents=True, exist_ok=True)

merged = {"args": None, "per_horizon": {}}
for horizon in horizons:
    hroot = train_root / f"h{horizon}"
    path = hroot / "train_result.json"
    if not path.exists():
        raise SystemExit(f"missing per-horizon result: {path}")
    data = json.loads(path.read_text())
    if merged["args"] is None:
        merged["args"] = dict(data.get("args", {}))
        merged["args"]["results_root"] = str(results)
        merged["args"]["horizons"] = horizons
        merged["args"]["parallel_horizon_roots"] = str(train_root)
    row = data.get("per_horizon", {}).get(str(horizon))
    if row is None:
        raise SystemExit(f"missing h{horizon} in {path}")
    if "error" in row:
        raise SystemExit(f"h{horizon} failed: {row['error']}")
    for ckpt in (hroot / "checkpoints").glob(f"*h{horizon}.pt"):
        shutil.copy2(ckpt, ckpt_dir / ckpt.name)
    final_ckpt = ckpt_dir / f"funcdec_h{horizon}.pt"
    if not final_ckpt.exists():
        raise SystemExit(f"missing merged checkpoint: {final_ckpt}")
    row["checkpoint"] = str(final_ckpt)
    merged["per_horizon"][str(horizon)] = row

(results / "train_result_partial.json").write_text(json.dumps(merged, indent=2))
(results / "train_result.json").write_text(json.dumps(merged, indent=2))
print(f"[merge] saved {results / 'train_result.json'}")
PY

echo "=== Verify scratch initialization ==="
python - "$RESULTS/train_result.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
data = json.loads(path.read_text())
bad = []
if data.get("args", {}).get("init_checkpoint_dir") != "none":
    bad.append(("args", data.get("args", {}).get("init_checkpoint_dir")))
if data.get("args", {}).get("synth_interval") != 10:
    bad.append(("synth_interval", data.get("args", {}).get("synth_interval")))
if data.get("args", {}).get("batch_size") != 1024:
    bad.append(("batch_size", data.get("args", {}).get("batch_size")))
for horizon, row in data.get("per_horizon", {}).items():
    init = row.get("initial_checkpoint")
    if init not in (None, "scratch"):
        bad.append((horizon, init))
if bad:
    raise SystemExit(f"unexpected warm-mix run settings in {path}: {bad}")
print(f"[verify] scratch + batch_size=1024 + synth_interval=10 confirmed: {path}")
PY

echo "=== Phase 2a: Evaluate real LOTSA+ETT against TimesFM v1 ==="
REAL_EVAL_ARGS=()
if [[ -n "$REAL_ROOT" ]]; then
  REAL_EVAL_ARGS+=(--real_root "$REAL_ROOT")
fi
DEVICE=$DEVICE python "$PROJECT/eval_real.py" \
  --checkpoint_root "$RESULTS" \
  --results_root "$RESULTS/real_lot_ett" \
  --run_tfm_zeroshot \
  --timesfm_metrics_csv none \
  "${REAL_EVAL_ARGS[@]}"

# Synthetic evaluation is intentionally disabled for the current soft-mask runs.
# DEVICE=$DEVICE python "$PROJECT/eval_synth_fourier.py" \
#   --checkpoint_root "$RESULTS" \
#   --results_root "$RESULTS/fourier_synth"
#
# DEVICE=$DEVICE python "$PROJECT/eval_synth_nonf.py" \
#   --checkpoint_root "$RESULTS" \
#   --results_root "$RESULTS/nonfourier_synth"

echo "=== soft-mask warm-real-mix pipeline complete $(date '+%Y-%m-%d %H:%M:%S') ==="
