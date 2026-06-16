# XReg Soft Mask Experiment

This folder is a copy of `src/experiments/deconder_adjustment/soft_warm_s10_oldloss_best`
with one added evaluation feature: TimesFM-style inference-only XReg overlay.

## Intent

- Keep the existing FuncDec decoder-adjusted soft-mask model unchanged.
- Keep existing checkpoints compatible.
- During real-data evaluation, optionally add an in-context linear XReg forecast
  to the FuncDec forecast.
- Match the TimesFM `timesfm + xreg` policy:
  - run TimesFM 2.5 with `return_backcast=True`
  - fit XReg on `raw_context - timesfm_backcast`
  - add the XReg horizon forecast to the FuncDec horizon forecast

## Current Covariate Policy

The first implementation uses numeric columns from `raw.parquet`:

- target column: the cache `col_ids` target for each row
- covariates: all other numeric columns

This is intentionally separated in `covariates.py` so dataset-specific
covariate selection can be configured later.

Datasets without usable raw numeric covariates are not treated as failures;
their `xreg_status` records `no_covariates`.

## Main Entry

```bash
DEVICE=cuda:0 \
CHECKPOINT_ROOT=/path/to/checkpoints_or_train_root \
REAL_ROOT=/workspace/data/real_eval_lot_ett \
RESULTS_ROOT=/workspace/data/results/xreg_soft_mask/eval_real \
./run_eval_xreg.sh
```

`XREG_RIDGE` defaults to `0.0`, matching the TimesFM default.

## ETT Smoke Runs

```bash
# Existing ETTh1 cache only.
CHECKPOINT_ROOT=/path/to/checkpoints_or_train_root \
./run_eval_xreg_etth1.sh

# Build ETTh2/ETTm1/ETTm2 caches, then evaluate them with the same checkpoint.
./run_prepare_ett_extra_cache.sh
CHECKPOINT_ROOT=/path/to/checkpoints_or_train_root \
./run_eval_xreg_ett_extra.sh
```
