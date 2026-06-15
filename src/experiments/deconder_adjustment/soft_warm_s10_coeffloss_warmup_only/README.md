# Decoder Adjustment + Coefficient-Loss Warmup Only

This folder is copied from `src/experiments/deconder_adjustment/soft_warm_s10_oldloss_best`
to test one loss ablation on top of the TimesFM-style residual decoder.

## Intent

- Baseline intent: recorded-best soft gate + warm-start setup, referenced as `soft_warm_s10_oldloss_best`.
- Keep the existing backbone embedding path unchanged.
- Keep point forecasting only. Do not add TimesFM quantile reshaping/output.
- Keep the residual decoder shape/style from the decoder-adjustment experiment:
  - `Linear(embed_dim -> hidden)`
  - `SiLU`
  - `Linear(hidden -> horizon)`
  - plus direct residual projection `Linear(embed_dim -> horizon)`
- Change only the first Fourier synthetic warmup seasonal loss:
  - previous: `L1(decomp["seasonal"], seasonal_n)`
  - new: `L1(raw_coefficients * gates, cached_seasonal_coefficients)`

## Important Notes

- The coefficient target is used only in the initial `fourier_warmup` phase.
- The sparse synthetic batches inside `mixed_full` keep the original seasonal time-series target.
- Real batches and residual-only finetuning keep the original losses.
- The historical `oldloss` name refers to the recorded result label. This folder uses `trend_loss + coeff_effective_loss + gate_l1` for Fourier warmup, not the older erroneous `pred_loss + component_loss` style.
- Run scripts default to `synth_interval=10` to align with the recorded `s10` setting.
- The experiment is intended to run on the compact/subset data workflow already prepared to reduce VESSL data transfer and startup cost. That compact data is temporary infrastructure, not a new dataset definition.

## Main Entry

Use `run_warm_real_mix.sh` for the warm Fourier + real mixed training and real evaluation path.
