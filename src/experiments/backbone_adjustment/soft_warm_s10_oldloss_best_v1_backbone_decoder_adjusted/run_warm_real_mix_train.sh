#!/usr/bin/env bash
# Backward-compatible entrypoint. The canonical soft-mask warm-real-mix
# workflow now mirrors 5.30fine_mask/run_warm_real_mix.sh and includes eval.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/run_warm_real_mix.sh" "$@"
