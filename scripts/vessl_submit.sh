#!/usr/bin/env bash
# Submit a VESSL training/evaluation job with standardized result paths.
#
# Usage:
#   ./scripts/vessl_submit.sh --exp hard_mask --run b1024_s13_scratch_v1 --mode train_eval
#   ./scripts/vessl_submit.sh --exp soft_mask  --run b1024_s13_scratch_v1 --mode eval --gpu a100x1
#
# Result path on volume:
#   /workspace/data/results/<exp>/<HHmm>_<run>/train/
#   /workspace/data/results/<exp>/<HHmm>_<run>/eval_real/
#
# Download results locally after job completes:
#   vesslctl volume download objvol-edwuqaa94ii3 \
#     --remote-prefix "results/<exp>/<HHmm>_<run>/eval_real" \
#     ./results/<exp>/<HHmm>_<run>/eval_real

set -euo pipefail

# ---------- defaults ----------
EXP=""
RUN=""
MODE="train_eval"       # train | eval | train_eval
GPU="a100x1"
HORIZONS="96 192 336 720"
BATCH_SIZE=1024
FOURIER_WARMUP_STEPS=125
MIXED_STEPS=2500
RESIDUAL_STEPS=500
SYNTH_INTERVAL=13
REAL_GROUP_CHUNK_STEPS=63
SAMPLES_PER_DATASET=0   # 0 = all samples
SKIP_TFM="--skip_tfm"

COMMIT="1a9556b27f678fdc3859e1e75b9c373160b49caa"
IMAGE="ghcr.io/idealtype/tfm-research:${COMMIT}"
VOLUME="objvol-edwuqaa94ii3:/workspace/data"

RESOURCE_MAP_train="resourcespec-a100x1"
RESOURCE_MAP_eval="resourcespec-grlxx3knwzps"
RESOURCE_MAP_train_eval="resourcespec-a100x1"

# ---------- parse args ----------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --exp)   EXP="$2";   shift 2 ;;
    --run)   RUN="$2";   shift 2 ;;
    --mode)  MODE="$2";  shift 2 ;;
    --gpu)
      case "$2" in
        a100x1) RESOURCE_MAP_train="resourcespec-a100x1"; RESOURCE_MAP_train_eval="resourcespec-a100x1" ;;
        a100x2) RESOURCE_MAP_train="resourcespec-a100x2"; RESOURCE_MAP_train_eval="resourcespec-a100x2" ;;
        h100x1) RESOURCE_MAP_train="resourcespec-ch100x1"; RESOURCE_MAP_train_eval="resourcespec-ch100x1" ;;
        cpu)    RESOURCE_MAP_eval="resourcespec-grlxx3knwzps" ;;
        *) echo "Unknown gpu: $2"; exit 1 ;;
      esac
      GPU="$2"; shift 2 ;;
    --horizons)              HORIZONS="$2";              shift 2 ;;
    --batch_size)            BATCH_SIZE="$2";            shift 2 ;;
    --fourier_warmup_steps)  FOURIER_WARMUP_STEPS="$2";  shift 2 ;;
    --mixed_steps)           MIXED_STEPS="$2";           shift 2 ;;
    --residual_steps)        RESIDUAL_STEPS="$2";        shift 2 ;;
    --synth_interval)        SYNTH_INTERVAL="$2";        shift 2 ;;
    --real_group_chunk_steps) REAL_GROUP_CHUNK_STEPS="$2"; shift 2 ;;
    --samples_per_dataset)   SAMPLES_PER_DATASET="$2";   shift 2 ;;
    --with_tfm)              SKIP_TFM="";                shift 1 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

if [[ -z "$EXP" || -z "$RUN" ]]; then
  echo "Error: --exp and --run are required"
  echo "  Valid --exp values: hard_mask | soft_mask | nogate_softmask"
  exit 1
fi

case "$EXP" in
  hard_mask|soft_mask|nogate_softmask) ;;
  *) echo "Error: unknown --exp '$EXP'"; exit 1 ;;
esac

case "$MODE" in
  train|eval|train_eval) ;;
  *) echo "Error: unknown --mode '$MODE'"; exit 1 ;;
esac

# ---------- paths ----------
HHMM=$(date +%H%M)
RUN_TAG="${HHMM}_${RUN}"
RESULT_BASE="/workspace/data/results/${EXP}/${RUN_TAG}"
TRAIN_ROOT="${RESULT_BASE}/train"
EVAL_REAL_ROOT="${RESULT_BASE}/eval_real"
LOTSA_CACHE="/workspace/data/data_lotsa/lotsa_cache"
REAL_ROOT="/workspace/data/real_eval_lot_ett"
DOMAIN_CONFIG="/tmp/tfm_project/src/experiments/synthetic_center/train_syn_real_raw/domain_config.json"
EXP_DIR="/tmp/tfm_project/src/experiments/${EXP}"

# ---------- resource spec ----------
case "$MODE" in
  train)       RESOURCE_SPEC="${RESOURCE_MAP_train}" ;;
  eval)        RESOURCE_SPEC="${RESOURCE_MAP_eval}" ;;
  train_eval)  RESOURCE_SPEC="${RESOURCE_MAP_train_eval}" ;;
esac

# ---------- build command ----------
SETUP="git clone https://github.com/idealtype/TFM_research.git /tmp/tfm_project \
  && cd /tmp/tfm_project \
  && git checkout ${COMMIT} \
  && export DATA_ROOT=/workspace/data \
  && export PROJECT_ROOT=/tmp/tfm_project \
  && export HF_HOME=/workspace/data/.cache/huggingface \
  && export HF_HUB_CACHE=/workspace/data/.cache/huggingface/hub"

TRAIN_CMD="cd ${EXP_DIR} \
  && python train_warm_real_mix.py \
    --results_root ${TRAIN_ROOT} \
    --lotsa_cache_root ${LOTSA_CACHE} \
    --domain_config ${DOMAIN_CONFIG} \
    --horizons ${HORIZONS} \
    --batch_size ${BATCH_SIZE} \
    --fourier_warmup_steps ${FOURIER_WARMUP_STEPS} \
    --mixed_steps ${MIXED_STEPS} \
    --residual_steps ${RESIDUAL_STEPS} \
    --synth_interval ${SYNTH_INTERVAL} \
    --real_group_chunk_steps ${REAL_GROUP_CHUNK_STEPS}"

EVAL_CMD="cd ${EXP_DIR} \
  && python eval_real.py \
    --checkpoint_root ${TRAIN_ROOT} \
    --results_root ${EVAL_REAL_ROOT} \
    --real_root ${REAL_ROOT} \
    --samples_per_dataset ${SAMPLES_PER_DATASET} \
    ${SKIP_TFM}"

case "$MODE" in
  train)      JOB_CMD="${SETUP} && ${TRAIN_CMD}" ;;
  eval)       JOB_CMD="${SETUP} && ${EVAL_CMD}" ;;
  train_eval) JOB_CMD="${SETUP} && ${TRAIN_CMD} && ${EVAL_CMD}" ;;
esac

JOB_NAME="${EXP}-${RUN_TAG}-${MODE}"

# ---------- preview ----------
echo "========================================"
echo " VESSL Job Preview"
echo "========================================"
echo "  Job name  : ${JOB_NAME}"
echo "  Mode      : ${MODE}"
echo "  Exp       : ${EXP}"
echo "  Run tag   : ${RUN_TAG}"
echo "  Resource  : ${RESOURCE_SPEC}"
echo "  Volume    : ${VOLUME}"
echo "  Train out : ${TRAIN_ROOT}"
echo "  Eval out  : ${EVAL_REAL_ROOT}"
echo "========================================"
echo ""
echo "Download eval results after completion:"
echo "  vesslctl volume download objvol-edwuqaa94ii3 \\"
echo "    --remote-prefix \"results/${EXP}/${RUN_TAG}/eval_real\" \\"
echo "    ./results/${EXP}/${RUN_TAG}/eval_real"
echo ""
read -r -p "Submit? [y/N] " confirm
if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
  echo "Cancelled."
  exit 0
fi

# ---------- submit ----------
vesslctl job create \
  -n "${JOB_NAME}" \
  -r "${RESOURCE_SPEC}" \
  -i "${IMAGE}" \
  --object-volume "${VOLUME}" \
  --cmd "${JOB_CMD}"
