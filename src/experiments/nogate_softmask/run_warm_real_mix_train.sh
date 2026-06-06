#!/usr/bin/env bash
# Backward-compatible entrypoint. The canonical soft-mask warm-real-mix
# workflow now mirrors 5.30fine_mask/run_warm_real_mix.sh and includes eval.
set -euo pipefail

exec /home/sia2/project/6.1nogate_softmask/run_warm_real_mix.sh "$@"
