#!/usr/bin/env bash
# Backward-compatible entrypoint. The canonical warm-real-mix workflow includes eval.
set -euo pipefail

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$PROJECT/run_warm_real_mix.sh" "$@"
