#!/usr/bin/env bash
# One-time job: generate compact training data pool and store back to S3.
#
# Run this ONCE before sweep_coeff_l1.sh. Reads full data from /workspace/data,
# slices to only the rows used by the fixed training config, and writes the
# compact pool back to /workspace/data/compact_nogate_b1024_m5000_r1000_c250/.
#
# All sweep_coeff_l1.sh jobs share this pool (coeff_l1_weight doesn't affect
# which data rows are sampled).
#
# Usage:
#   ./scripts/prepare_compact_pool.sh

set -euo pipefail

# ---------- constants ----------
SRC_COMMIT="3a4229bfe4c0f362808a1dbd9ca6e413989c5571"
IMAGE="ghcr.io/idealtype/tfm-research:1a9556b27f678fdc3859e1e75b9c373160b49caa"
VOLUME="objvol-edwuqaa94ii3:/workspace/data"
RESOURCE="resourcespec-a100x1"

# ---------- training config (must match sweep_coeff_l1.sh) ----------
BATCH_SIZE=1024
FOURIER_WARMUP_STEPS=250
MIXED_STEPS=5000
RESIDUAL_STEPS=1000
SYNTH_INTERVAL=10
REAL_GROUP_CHUNK_STEPS=250
HORIZONS="96 192 336 720"
SEED=42

# ---------- paths ----------
DOMAIN_CONFIG="/tmp/tfm_project/src/experiments/synthetic_center/train_syn_real_raw/domain_config.json"
COMPACT_DST="/workspace/data/compact_nogate_b1024_m5000_r1000_c250"

SETUP="git clone https://github.com/idealtype/TFM_research.git /tmp/tfm_project \
  && cd /tmp/tfm_project \
  && git checkout ${SRC_COMMIT} \
  && export DATA_ROOT=/workspace/data \
  && export PROJECT_ROOT=/tmp/tfm_project \
  && export HF_HOME=/workspace/data/.cache/huggingface \
  && export HF_HUB_CACHE=/workspace/data/.cache/huggingface/hub"

PREPARE_CMD="python /tmp/tfm_project/src/data_prep/prepare_data.py \
  --src_data_root /workspace/data \
  --dst_data_root ${COMPACT_DST} \
  --domain_config ${DOMAIN_CONFIG} \
  --horizons ${HORIZONS} \
  --seed ${SEED} \
  --batch_size ${BATCH_SIZE} \
  --fourier_warmup_steps ${FOURIER_WARMUP_STEPS} \
  --mixed_steps ${MIXED_STEPS} \
  --synth_interval ${SYNTH_INTERVAL} \
  --residual_steps ${RESIDUAL_STEPS} \
  --real_group_chunk_steps ${REAL_GROUP_CHUNK_STEPS} \
  --num_workers 8"

JOB_CMD="${SETUP} && ${PREPARE_CMD}"

echo "========================================"
echo " Compact pool preparation job"
echo "========================================"
echo "  Commit   : ${SRC_COMMIT:0:12}"
echo "  Src      : /workspace/data (full, 82 GB)"
echo "  Dst      : ${COMPACT_DST} (~19 GB)"
echo "  Config   : batch=${BATCH_SIZE} mixed=${MIXED_STEPS} residual=${RESIDUAL_STEPS}"
echo "             chunk=${REAL_GROUP_CHUNK_STEPS} warmup=${FOURIER_WARMUP_STEPS}"
echo "             synth_interval=${SYNTH_INTERVAL} seed=${SEED}"
echo ""
echo "  Run ONCE before sweep_coeff_l1.sh."
echo "  All 6 sweep jobs will share this pool."
echo "========================================"
echo ""
read -r -p "Submit prepare job? [y/N] " confirm
if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
    echo "Cancelled."
    exit 0
fi

OUTPUT=$(vesslctl job create \
    -n "prepare-compact-nogate-l1sweep" \
    -r "${RESOURCE}" \
    -i "${IMAGE}" \
    --object-volume "${VOLUME}" \
    --cmd "${JOB_CMD}" 2>&1)
SLUG=$(echo "${OUTPUT}" | grep -o 'job-[a-z0-9]*' | head -1)
echo ""
echo "  submitted: slug=${SLUG}"
echo ""
echo "Monitor: vesslctl job logs ${SLUG}"
echo "When done, run: ./scripts/sweep_coeff_l1.sh"
