# Decoder Adjustment Experiment

This folder is a copy of `src/experiments/soft_mask` for testing one change only:
the residual decoder is adjusted to be closer to the original TimesFM residual block style.

## Intent

- Baseline intent: recorded-best soft gate + warm-start setup, referenced as `soft_warm_s10_oldloss_best`.
- Keep the existing backbone embedding path unchanged.
- Keep point forecasting only. Do not add TimesFM quantile reshaping/output.
- Change only the residual decoder shape/style toward TimesFM:
  - `Linear(embed_dim -> hidden)`
  - `SiLU`
  - `Linear(hidden -> horizon)`
  - plus direct residual projection `Linear(embed_dim -> horizon)`

## Important Notes

- The historical `oldloss` name refers to the recorded result label, but the current copied code uses the corrected synthetic loss path: `trend_loss + seasonal_loss + gate_l1`, not the older erroneous `pred_loss + component_loss` style.
- Run scripts default to `synth_interval=10` to align with the recorded `s10` setting.
- The experiment is intended to run on the compact/subset data workflow already prepared to reduce VESSL data transfer and startup cost. That compact data is temporary infrastructure, not a new dataset definition.

## Main Entry

Use `run_warm_real_mix.sh` for the warm Fourier + real mixed training and real evaluation path.
