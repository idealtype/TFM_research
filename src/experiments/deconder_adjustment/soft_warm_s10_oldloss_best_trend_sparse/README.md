# Decoder Adjustment + Trend Sparse Penalty Experiment

This folder is a copy of `soft_warm_s10_oldloss_best` for testing one loss change only:
adding trend changepoint sparsity while keeping the existing trend-seasonal orthogonal
penalty.

## Intent

- Baseline intent: recorded-best soft gate + warm-start setup, referenced as `soft_warm_s10_oldloss_best`.
- Keep the existing backbone embedding path unchanged.
- Keep point forecasting only. Do not add TimesFM quantile reshaping/output.
- Keep the residual decoder and seasonal loss structure unchanged.
- Add `trend_delta_l1_weight * mean(abs(decomp["delta"]))` to full-decoder training losses.
- Keep the existing trend-seasonal squared-correlation penalty on real batches.

## Important Notes

- The historical `oldloss` name refers to the recorded result label, but the current copied code uses the corrected synthetic loss path: `trend_loss + seasonal_loss + gate_l1`, not the older erroneous `pred_loss + component_loss` style.
- Default loss penalty weights follow the remembered hard-mask setting because the original Google Drive code was not accessible from WSL:
  - trend changepoint sparsity: `--trend_delta_l1_weight 0.2`
  - trend-seasonal orthogonal penalty: `--ts_corr_weight 0.01`
- Run scripts default to `synth_interval=10` to align with the recorded `s10` setting.
- The experiment is intended to run on the compact/subset data workflow already prepared to reduce VESSL data transfer and startup cost. That compact data is temporary infrastructure, not a new dataset definition.

## Main Entry

Use `run_warm_real_mix.sh` for the warm Fourier + real mixed training and real evaluation path.
