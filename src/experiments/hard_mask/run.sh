#!/bin/bash
# Full pipeline for 5.30fine_mask experiment
set -e

DEVICE=${DEVICE:-cuda:0}
PROJECT=/home/sia2/project/5.30fine_mask

echo "=== Phase 3: Generate SM1-SM10 synthetic data (CPU) ==="
python "$PROJECT/data/generate_fine_mask_synth.py"

echo "=== Phase 4: Cache SM1-SM10 with fine_mask basis (GPU) ==="
DEVICE=$DEVICE python "$PROJECT/cache/prepare_fine_mask_synth_cache.py" --eval

echo "=== Phase 2a: Build fine_mask basis for existing S1-S10 synth caches (CPU) ==="
python "$PROJECT/cache/build_synth_fine_basis.py" &

echo "=== Phase 2b: Build fine_mask basis for real LOTSA caches (CPU) ==="
python "$PROJECT/cache/build_real_fine_basis.py" \
    --cache_dir /home/sia2/project/data/data_lotsa/lotsa_cache \
    --horizons 96 192 336 720 &

echo "=== Phase 2c: Build fine_mask basis for real evaluation caches (CPU) ==="
python "$PROJECT/cache/build_real_fine_basis.py" \
    --cache_dir /home/sia2/project/data/real_eval_lot_ett \
    --horizons 96 192 336 720 &

wait
echo "=== Phase 2: Basis caching complete ==="

echo "=== Phase 5: Train ==="
DEVICE=$DEVICE python "$PROJECT/train.py"

echo "=== Phase 6a: Evaluate on Fourier synthetic ==="
DEVICE=$DEVICE python "$PROJECT/eval_synth_fourier.py"

echo "=== Phase 6b: Evaluate on non-Fourier synthetic ==="
DEVICE=$DEVICE python "$PROJECT/eval_synth_nonf.py"

echo "=== Phase 6c: Evaluate on real LOTSA+ETT ==="
DEVICE=$DEVICE python "$PROJECT/eval_real.py"

echo "=== Pipeline complete ==="
