#!/usr/bin/env bash
# Sweep coeff_l1_weight for nogate_softmask on VESSL.
# Submits one job per sweep value (parallel); each job runs train → eval_real.
#
# Usage:
#   ./scripts/sweep_coeff_l1.sh

set -euo pipefail

# ---------- constants (keep in sync with vessl_submit.sh) ----------
SRC_COMMIT="6d7797c908cd1bed7afee58095ff35158e400671"
IMAGE="ghcr.io/idealtype/tfm-research:1a9556b27f678fdc3859e1e75b9c373160b49caa"
VOLUME="objvol-edwuqaa94ii3:/workspace/data"
RESOURCE="resourcespec-a100x1"

# ---------- sweep values ----------
SWEEP_VALUES=("0" "1e-4" "3e-4" "1e-3" "3e-3" "1e-2")

# ---------- fixed hyperparams (half-step sweep config) ----------
BATCH_SIZE=1024
FOURIER_WARMUP_STEPS=250
MIXED_STEPS=5000
RESIDUAL_STEPS=1000
SYNTH_INTERVAL=10
REAL_GROUP_CHUNK_STEPS=250
HORIZONS="96 192 336 720"

# ---------- paths ----------
EXP="nogate_softmask"
EXP_DIR="/tmp/tfm_project/src/experiments/${EXP}"
REAL_ROOT="/workspace/data/real_eval_lot_ett"
DOMAIN_CONFIG="/tmp/tfm_project/src/experiments/synthetic_center/train_syn_real_raw/domain_config.json"
TMP_DATA_ROOT="/tmp/data"
# Pre-built compact pool (run prepare_compact_pool.sh once to generate this)
COMPACT_SRC="/workspace/data/compact_nogate_b1024_m5000_r1000_c250"

HHMM=$(date +%H%M)

SETUP="git clone https://github.com/idealtype/TFM_research.git /tmp/tfm_project \
  && cd /tmp/tfm_project \
  && git checkout ${SRC_COMMIT} \
  && export DATA_ROOT=/workspace/data \
  && export PROJECT_ROOT=/tmp/tfm_project \
  && export HF_HOME=/workspace/data/.cache/huggingface \
  && export HF_HUB_CACHE=/workspace/data/.cache/huggingface/hub"

# ---------- preview ----------
echo "========================================"
echo " nogate_softmask coeff_l1_weight sweep"
echo "========================================"
echo "  Commit    : ${SRC_COMMIT:0:12}"
echo "  Resource  : ${RESOURCE}"
echo "  Steps     : warmup=${FOURIER_WARMUP_STEPS} mixed=${MIXED_STEPS} residual=${RESIDUAL_STEPS}"
echo "  batch_size: ${BATCH_SIZE}  synth_interval: ${SYNTH_INTERVAL}"
echo ""
echo "  Jobs (${#SWEEP_VALUES[@]}):"
for VAL in "${SWEEP_VALUES[@]}"; do
    echo "    coeff_l1_weight=${VAL}  →  /workspace/data/results/${EXP}/${HHMM}_sweep_l1_${VAL}/"
done
echo "========================================"
echo ""
read -r -p "Submit all ${#SWEEP_VALUES[@]} jobs? [y/N] " confirm
if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
    echo "Cancelled."
    exit 0
fi

echo ""
# ---------- submit ----------
for VAL in "${SWEEP_VALUES[@]}"; do
    RUN_TAG="${HHMM}_sweep_l1_${VAL}"
    RESULT_BASE="/workspace/data/results/${EXP}/${RUN_TAG}"
    TRAIN_ROOT="${RESULT_BASE}/train"
    EVAL_REAL_ROOT="${RESULT_BASE}/eval_real"

    # sanitize value for job name (remove dots; e.g. "1e-4" stays as-is, fine for VESSL)
    JOB_NAME="ng-l1-${VAL}-${HHMM}"

    # Copy pre-built compact pool from S3 to /tmp (~19 GB, much faster than full 82 GB)
    COPY_CMD="mkdir -p ${TMP_DATA_ROOT} && cp -r ${COMPACT_SRC}/. ${TMP_DATA_ROOT}/"

    TRAIN_CMD="cd ${EXP_DIR} \
  && export DATA_ROOT=${TMP_DATA_ROOT} \
  && python train_warm_real_mix_parallel.py \
    --results_root ${TRAIN_ROOT} \
    --lotsa_cache_root ${TMP_DATA_ROOT}/data_lotsa/lotsa_cache \
    --domain_config ${DOMAIN_CONFIG} \
    --horizons ${HORIZONS} \
    --batch_size ${BATCH_SIZE} \
    --fourier_warmup_steps ${FOURIER_WARMUP_STEPS} \
    --mixed_steps ${MIXED_STEPS} \
    --residual_steps ${RESIDUAL_STEPS} \
    --synth_interval ${SYNTH_INTERVAL} \
    --real_group_chunk_steps ${REAL_GROUP_CHUNK_STEPS} \
    --coeff_l1_weight ${VAL}"

    EVAL_CMD="cd ${EXP_DIR} \
  && python eval_real.py \
    --checkpoint_root ${TRAIN_ROOT} \
    --results_root ${EVAL_REAL_ROOT} \
    --real_root ${REAL_ROOT} \
    --skip_tfm"

    JOB_CMD="${SETUP} && ${COPY_CMD} && export DATA_ROOT=${TMP_DATA_ROOT} && ${TRAIN_CMD} && ${EVAL_CMD}"

    OUTPUT=$(vesslctl job create \
        -n "${JOB_NAME}" \
        -r "${RESOURCE}" \
        -i "${IMAGE}" \
        --object-volume "${VOLUME}" \
        --cmd "${JOB_CMD}" 2>&1)
    SLUG=$(echo "${OUTPUT}" | grep -o 'job-[a-z0-9]*' | head -1)
    echo "  submitted: coeff_l1_weight=${VAL}  name=${JOB_NAME}  slug=${SLUG}"
done

echo ""
echo "Download eval results after all jobs succeed:"
echo "  vesslctl volume download objvol-edwuqaa94ii3 \\"
echo "    --remote-prefix \"results/${EXP}/${HHMM}_sweep_l1_*\" \\"
echo "    ./results/${EXP}/"
echo ""
echo "Compare results:"
echo "  python scripts/compare_sweep.py"
