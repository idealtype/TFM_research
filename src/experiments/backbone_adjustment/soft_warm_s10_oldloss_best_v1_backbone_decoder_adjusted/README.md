# TimesFM v1 Backbone + Decoder Adjustment

This folder combines the two previous isolated changes:
TimesFM v1 cached backbone embeddings and the TimesFM-style point residual decoder.

## Intent

- Baseline intent: recorded-best soft gate + warm-start setup, referenced as `soft_warm_s10_oldloss_best`.
- Use TimesFM v1 checkpoint `google/timesfm-1.0-200m-pytorch` for cached backbone embeddings.
- Keep point forecasting only; do not add TimesFM quantile output.
- Replace the original residual decoder with the adjusted TimesFM-style point head:
  - `Linear(embed_dim -> hidden)`
  - `SiLU`
  - `Linear(hidden -> horizon)`
  - plus direct residual projection `Linear(embed_dim -> horizon)`

## Evaluation Requirement

The comparison baseline must be TimesFM v1, not TimesFM 2.5.
`run_warm_real_mix.sh` runs `eval_real.py` with `--run_tfm_zeroshot --timesfm_metrics_csv none`.

The wrapper script `scripts/run_v1_backbone_soft_warm.sh` can run this folder by setting:

```bash
EXP_DIR=src/experiments/backbone_adjustment/soft_warm_s10_oldloss_best_v1_backbone_decoder_adjusted
```

## Data Path

This experiment uses the temporary compact/subset data workflow prepared for VESSL runs.
The compact cache does not store raw context, so v1 recaching reconstructs contexts from the original full cache or raw synthetic/eval sources and then writes compact v1 backbone files.

## Important Notes

- `oldloss` is a recorded-result label here. The current code uses the corrected synthetic loss path: `trend_loss + seasonal_loss + gate_l1`, not the older erroneous `pred_loss + component_loss` style.
- Run scripts default to `synth_interval=10` to align with the recorded `s10` setting.
