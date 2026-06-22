#!/usr/bin/env bash
# Build compact training pool for coeffloss (coeff_effective) experiments.
#
# Uses prepare_data.py with coeffloss train.py so that:
#  - seasonal_coefficients_h*.pt files are correctly sliced (not copied whole)
#  - only rows accessed by the coeffloss schedule are included
#
# Run ONCE before submitting coeffloss/sweep jobs.
# Output pool: /workspace/data/compact_coeff_b1024_m2500_r500_c63/
#
# Usage:
#   ./scripts/prepare_compact_pool_coeff.sh

set -euo pipefail

# ---------- constants ----------
SRC_COMMIT="$(git rev-parse HEAD)"
IMAGE="ghcr.io/idealtype/tfm-research:1a9556b27f678fdc3859e1e75b9c373160b49caa"
VOLUME="objvol-edwuqaa94ii3:/workspace/data"
RESOURCE="resourcespec-grlxx3knwzps"

# ---------- training config (must match coeffloss job scripts) ----------
BATCH_SIZE=1024
FOURIER_WARMUP_STEPS=125
MIXED_STEPS=2500
RESIDUAL_STEPS=500
SYNTH_INTERVAL=10
REAL_GROUP_CHUNK_STEPS=63
HORIZONS="96 192 336 720"
SEED=42

# ---------- paths ----------
DOMAIN_CONFIG="/tmp/tfm_project/src/experiments/synthetic_center/train_syn_real_raw/domain_config.json"
EXP_DIR="/tmp/tfm_project/src/experiments/deconder_adjustment/soft_warm_s10_coeffloss_warmup_mixed"
COMPACT_DST="/workspace/data/compact_coeff_b1024_m2500_r500_c63"

echo "========================================"
echo " Compact pool preparation (coeffloss)"
echo "========================================"
echo "  Commit   : ${SRC_COMMIT:0:12}"
echo "  Exp      : soft_warm_s10_coeffloss_warmup_mixed"
echo "  Dst      : ${COMPACT_DST}"
echo "  Config   : batch=${BATCH_SIZE} mixed=${MIXED_STEPS} residual=${RESIDUAL_STEPS}"
echo "             chunk=${REAL_GROUP_CHUNK_STEPS} warmup=${FOURIER_WARMUP_STEPS}"
echo "             synth_interval=${SYNTH_INTERVAL} seed=${SEED}"
echo "  Fix      : seasonal_coefficients sliced correctly"
echo ""
echo "  Run ONCE before coeffloss/sweep jobs."
echo "========================================"
echo ""
read -r -p "Submit prepare job? [y/N] " confirm
if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
    echo "Cancelled."
    exit 0
fi

JOB_CMD="set -e
log() { echo \"[\$(date '+%Y-%m-%d %H:%M:%S')] \$*\"; }
run_with_heartbeat() {
  label=\"\$1\"
  shift
  log \"START: \${label}\"
  \"\$@\" &
  pid=\$!
  (
    sleep 60
    while kill -0 \"\$pid\" 2>/dev/null; do
      log \"RUNNING: \${label} pid=\${pid}\"
      sleep 60
    done
  ) &
  heartbeat_pid=\$!
  wait \"\$pid\"
  status=\$?
  kill \"\$heartbeat_pid\" 2>/dev/null || true
  wait \"\$heartbeat_pid\" 2>/dev/null || true
  if [ \"\$status\" -ne 0 ]; then
    log \"FAILED: \${label} status=\${status}\"
    return \"\$status\"
  fi
  log \"DONE: \${label}\"
}
log 'job started'
run_with_heartbeat 'git clone repository' git clone https://github.com/idealtype/TFM_research.git /tmp/tfm_project
cd /tmp/tfm_project
log 'git checkout ${SRC_COMMIT}'
git checkout ${SRC_COMMIT}
log \"checked out commit \$(git rev-parse HEAD)\"
export DATA_ROOT=/workspace/data
export PROJECT_ROOT=/tmp/tfm_project
export HF_HOME=/workspace/data/.cache/huggingface
export HF_HUB_CACHE=/workspace/data/.cache/huggingface/hub

log '[prepare] Removing stale pool if exists...'
rm -rf ${COMPACT_DST}
log '[prepare] Clean slate - starting fresh.'

run_with_heartbeat 'prepare compact coeff pool' python -u /tmp/tfm_project/src/data_prep/prepare_data.py \\
  --src_data_root /workspace/data \\
  --dst_data_root ${COMPACT_DST} \\
  --exp_dir ${EXP_DIR} \\
  --domain_config ${DOMAIN_CONFIG} \\
  --horizons ${HORIZONS} \\
  --seed ${SEED} \\
  --batch_size ${BATCH_SIZE} \\
  --fourier_warmup_steps ${FOURIER_WARMUP_STEPS} \\
  --mixed_steps ${MIXED_STEPS} \\
  --synth_interval ${SYNTH_INTERVAL} \\
  --residual_steps ${RESIDUAL_STEPS} \\
  --real_group_chunk_steps ${REAL_GROUP_CHUNK_STEPS} \\
  --num_workers 8
log '[validate] Checking compact pool integrity...'
run_with_heartbeat 'validate compact coeff pool' python -u /tmp/tfm_project/src/data_prep/validate_compact_pool.py ${COMPACT_DST}/synthetic/func_dec_syn_cent_fourier_all_train_cache_10_4_2_8
log '=== compact pool done ==='"

OUTPUT=$(vesslctl job create \
    -n "prepare-compact-coeff-b1024-m2500" \
    -r "${RESOURCE}" \
    -i "${IMAGE}" \
    --object-volume "${VOLUME}" \
    --cmd "${JOB_CMD}" 2>&1)
SLUG=$(echo "${OUTPUT}" | grep -o 'job-[a-z0-9]*' | head -1)
echo ""
echo "  submitted: slug=${SLUG}"
echo ""
echo "Monitor : vesslctl job logs ${SLUG}"
echo "When done, use compact_coeff_b1024_m2500_r500_c63 in job scripts"
