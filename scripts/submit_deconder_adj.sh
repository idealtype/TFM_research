#!/usr/bin/env bash
# Submit a deconder_adjustment training+eval job using the compact coeff pool.
#
# Usage:
#   ./scripts/submit_deconder_adj.sh --name dec-coeff-warmup-mixed --sub soft_warm_s10_coeffloss_warmup_mixed
#   ./scripts/submit_deconder_adj.sh --name sweep-gate-1e-3 --sub soft_warm_s10_coeffloss_warmup_mixed \
#       --extra_args "--gate_l1_weight 1e-3"
#
# Result path on volume:
#   /workspace/data/results/deconder_adjustment/<name>/
#     checkpoints/         <- funcdec_h*.pt
#     train_result.json
#     real_lot_ett/        <- eval results

set -euo pipefail

# ---------- defaults ----------
JOB_NAME=""
SUB_EXP=""
EXTRA_ARGS=""
COMMIT="$(git rev-parse HEAD)"
IMAGE="ghcr.io/idealtype/tfm-research:1a9556b27f678fdc3859e1e75b9c373160b49caa"
VOLUME="objvol-edwuqaa94ii3:/workspace/data"
RESOURCE="resourcespec-a100x1"
COMPACT_POOL="compact_coeff_b1024_m2500_r500_c63"

# training config defaults (match compact pool)
BATCH_SIZE=1024
FOURIER_WARMUP_STEPS=125
MIXED_STEPS=2500
RESIDUAL_STEPS=500
SYNTH_INTERVAL=10
REAL_GROUP_CHUNK_STEPS=63
GATE_L1=1e-3

# ---------- parse ----------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --name)        JOB_NAME="$2";    shift 2 ;;
    --sub)         SUB_EXP="$2";     shift 2 ;;
    --extra_args)  EXTRA_ARGS="$2";  shift 2 ;;
    --commit)      COMMIT="$2";      shift 2 ;;
    --pool)        COMPACT_POOL="$2"; shift 2 ;;
    --batch_size)           BATCH_SIZE="$2";           shift 2 ;;
    --mixed_steps)          MIXED_STEPS="$2";          shift 2 ;;
    --residual_steps)       RESIDUAL_STEPS="$2";       shift 2 ;;
    --gate_l1)              GATE_L1="$2";              shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

if [[ -z "$JOB_NAME" || -z "$SUB_EXP" ]]; then
  echo "Error: --name and --sub are required"
  echo "  --name    : result directory name (no spaces, no date tokens)"
  echo "  --sub     : sub-experiment dir under src/experiments/deconder_adjustment/"
  echo "  --extra_args : additional python args (e.g. --gate_l1_weight 1e-3)"
  exit 1
fi

# ---------- paths (all static — no \$(date) calls) ----------
DOMAIN_CONFIG="/tmp/tfm_project/src/experiments/synthetic_center/train_syn_real_raw/domain_config.json"
EXP_DIR="/tmp/tfm_project/src/experiments/deconder_adjustment/${SUB_EXP}"
RESULTS="/workspace/data/results/deconder_adjustment/${JOB_NAME}"
TRAIN_ROOT="${RESULTS}/_horizon_runs"
REAL_ROOT="/workspace/data/real_eval_lot_ett"

# ---------- preview ----------
echo "========================================"
echo " deconder_adjustment job"
echo "========================================"
echo "  name      : ${JOB_NAME}"
echo "  sub_exp   : ${SUB_EXP}"
echo "  commit    : ${COMMIT:0:12}"
echo "  pool      : ${COMPACT_POOL}"
echo "  results   : ${RESULTS}"
echo "  resource  : ${RESOURCE}"
echo "  config    : batch=${BATCH_SIZE} mixed=${MIXED_STEPS} residual=${RESIDUAL_STEPS}"
echo "              warmup=${FOURIER_WARMUP_STEPS} synth_interval=${SYNTH_INTERVAL}"
echo "              chunk=${REAL_GROUP_CHUNK_STEPS} gate_l1=${GATE_L1}"
[[ -n "$EXTRA_ARGS" ]] && echo "  extra     : ${EXTRA_ARGS}"
echo "========================================"
echo ""
read -r -p "Submit? [y/N] " confirm
if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
  echo "Cancelled."
  exit 0
fi

# ---------- inline job script (base64 encoded to avoid shell quoting issues) ----------
SCRIPT=$(cat << SCRIPT_EOF
set -e
log() { echo "[\$(date '+%Y-%m-%d %H:%M:%S')] \$*"; }
run_with_heartbeat() {
  local label="\$1"
  shift
  log "START: \${label}"
  "\$@" &
  local pid=\$!
  (
    sleep 60
    while kill -0 "\$pid" 2>/dev/null; do
      log "RUNNING: \${label} pid=\${pid}"
      sleep 60
    done
  ) &
  local heartbeat_pid=\$!
  wait "\$pid"
  local status=\$?
  kill "\$heartbeat_pid" 2>/dev/null || true
  wait "\$heartbeat_pid" 2>/dev/null || true
  if [ "\$status" -ne 0 ]; then
    log "FAILED: \${label} status=\${status}"
    return "\$status"
  fi
  log "DONE: \${label}"
}
log "job started"
run_with_heartbeat "git clone repository" git clone https://github.com/idealtype/TFM_research.git /tmp/tfm_project
cd /tmp/tfm_project
log "git checkout ${COMMIT}"
git checkout ${COMMIT}
log "checked out commit \$(git rev-parse HEAD)"

log "[pre-cache] Copying compact pool to /tmp/data..."
mkdir -p /tmp/data
run_with_heartbeat "copy compact pool ${COMPACT_POOL}" cp -r /workspace/data/${COMPACT_POOL}/. /tmp/data/
log "[pre-cache] Done: \$(du -sh /tmp/data | cut -f1)"
log "[validate] Checking compact pool integrity..."
run_with_heartbeat "validate compact pool" python -u /tmp/tfm_project/src/data_prep/validate_compact_pool.py /tmp/data/synthetic/func_dec_syn_cent_fourier_all_train_cache_10_4_2_8
log "[validate] OK"

export DATA_ROOT=/tmp/data
export PROJECT_ROOT=/tmp/tfm_project
export HF_HOME=/workspace/data/.cache/huggingface
export HF_HUB_CACHE=/workspace/data/.cache/huggingface/hub
export DEVICE=cuda:0

mkdir -p "${TRAIN_ROOT}"

run_h() {
  local h=\$1
  mkdir -p "${TRAIN_ROOT}/h\${h}"
  log "START: train h\${h}"
  python -u "${EXP_DIR}/train_warm_real_mix.py" \\
    --init_checkpoint_dir none \\
    --results_root "${TRAIN_ROOT}/h\${h}" \\
    --horizons "\${h}" \\
    --fourier_warmup_steps ${FOURIER_WARMUP_STEPS} \\
    --mixed_steps ${MIXED_STEPS} \\
    --residual_steps ${RESIDUAL_STEPS} \\
    --synth_interval ${SYNTH_INTERVAL} \\
    --real_group_chunk_steps ${REAL_GROUP_CHUNK_STEPS} \\
    --batch_size ${BATCH_SIZE} \\
    --learning_rate 1e-4 \\
    --residual_learning_rate 1e-4 \\
    --ts_corr_weight 0.01 \\
    --gate_l1_weight ${GATE_L1} \\
    --domain_config "${DOMAIN_CONFIG}" \\
    ${EXTRA_ARGS} \\
    2>&1 | stdbuf -oL -eL sed "s/^/[h\${h}] /"
  log "DONE: train h\${h}"
}
log "START: parallel horizon training"
for h in 96 192 336 720; do run_h \$h & done
wait
log "DONE: parallel horizon training"

log "START: merge checkpoints"
python -u - "${RESULTS}" "${TRAIN_ROOT}" 96 192 336 720 << 'PY'
import json, shutil, sys
from pathlib import Path
results = Path(sys.argv[1])
train_root = Path(sys.argv[2])
horizons = [int(x) for x in sys.argv[3:]]
ckpt_dir = results / "checkpoints"
ckpt_dir.mkdir(parents=True, exist_ok=True)
merged = {"args": None, "per_horizon": {}}
for h in horizons:
    hroot = train_root / f"h{h}"
    data = json.loads((hroot / "train_result.json").read_text())
    if merged["args"] is None:
        merged["args"] = dict(data.get("args", {}))
        merged["args"]["horizons"] = horizons
    row = data.get("per_horizon", {}).get(str(h))
    if row is None:
        raise SystemExit(f"missing h{h} in train_result.json")
    if "error" in row:
        raise SystemExit(f"h{h} training failed: {row['error']}")
    for ckpt in (hroot / "checkpoints").glob(f"*h{h}.pt"):
        shutil.copy2(ckpt, ckpt_dir / ckpt.name)
    final_ckpt = ckpt_dir / f"funcdec_h{h}.pt"
    if not final_ckpt.exists():
        raise SystemExit(f"checkpoint missing: {final_ckpt}")
    row["checkpoint"] = str(final_ckpt)
    merged["per_horizon"][str(h)] = row
(results / "train_result.json").write_text(json.dumps(merged, indent=2))
print("[merge] ok")
PY
log "DONE: merge checkpoints"

log "START: real evaluation"
DEVICE=\$DEVICE python -u "${EXP_DIR}/eval_real_parallel.py" \\
  --checkpoint_root "${RESULTS}" \\
  --results_root "${RESULTS}/real_lot_ett" \\
  --real_root "${REAL_ROOT}"
log "DONE: real evaluation"
log "=== DONE ==="
SCRIPT_EOF
)

B64=$(printf '%s' "$SCRIPT" | base64 -w 0)
JOB_CMD="printf '%s' '${B64}' | base64 -d | bash"

vesslctl job create \
  -n "${JOB_NAME}" \
  -r "${RESOURCE}" \
  -i "${IMAGE}" \
  --object-volume "${VOLUME}" \
  --cmd "${JOB_CMD}"
