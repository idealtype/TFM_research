#!/usr/bin/env bash
set -euo pipefail

cd /home/sia2/project/5.22syn_cent/train_syn_real_raw
mkdir -p results/logs

DEVICE="${DEVICE:-cuda:0}"
MAX_STEPS="${MAX_STEPS:-1000}"
BATCH_SIZE="${BATCH_SIZE:-128}"
GROUP_CHUNK_STEPS="${GROUP_CHUNK_STEPS:-100}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-64}"

python train_domain_real_finetune.py \
  --device "${DEVICE}" \
  --max_steps "${MAX_STEPS}" \
  --batch_size "${BATCH_SIZE}" \
  --group_chunk_steps "${GROUP_CHUNK_STEPS}" \
  --skip_existing \
  --save_checkpoint 2>&1 | tee results/logs/train_domain_real_finetune.log

python eval_real_lot_ett_domain_models.py \
  --device "${DEVICE}" \
  --batch_size "${EVAL_BATCH_SIZE}" \
  --results_root /home/sia2/project/5.22syn_cent/train_syn_real_raw/results \
  --output_name real_lot_ett_single_model 2>&1 | tee results/logs/eval_real_lot_ett_domain_models.log

python /home/sia2/project/5.22syn_cent/plot_extra_result_summaries.py \
  /home/sia2/project/5.22syn_cent/train_syn_real_raw/results 2>&1 | tee results/logs/plot_extra_result_summaries.log
