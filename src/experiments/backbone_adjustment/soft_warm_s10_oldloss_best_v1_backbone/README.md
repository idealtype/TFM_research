# TimesFM v1 Backbone Experiment

This folder is a copy of `src/experiments/soft_mask` for testing one change only:
replace the cached backbone embeddings with TimesFM v1 hidden states.

## Intent

- Baseline intent: recorded-best soft gate + warm-start setup, referenced as `soft_warm_s10_oldloss_best`.
- Keep the original decoder implementation from `soft_mask`.
- Do not use the TimesFM-style residual decoder from `deconder_adjustment`.
- Change only the cached backbone representation:
  - use TimesFM v1 checkpoint `google/timesfm-1.0-200m-pytorch`
  - extract the last-patch transformer hidden state with shape `(N, 1280)`
  - preserve the original target normalization `mu/sigma` so existing `futures_n` targets stay on the same scale

## Evaluation Requirement

The comparison baseline for this experiment must be TimesFM v1, not TimesFM 2.5.
`run_warm_real_mix.sh` runs `eval_real.py` with `--run_tfm_zeroshot --timesfm_metrics_csv none`, and `common.py` loads TimesFM v1 through:

- `TIMESFM_V1_SRC`
- `TIMESFM_V1_CHECKPOINT_PATH`

The wrapper script `scripts/run_v1_backbone_soft_warm.sh` sets these variables.

## Data Path

The current implementation uses the compact/subset data workflow temporarily prepared for VESSL runs.
Because the compact cache does not store raw context, v1 recaching must reconstruct contexts from the original full cache or raw synthetic/eval sources, then write compact v1 backbone files.

The recaching entry point is:

```bash
python src/data_prep/prepare_v1_backbone_data.py
```

## Important Notes

- The historical `oldloss` name refers to the recorded result label, but the current copied code uses the corrected synthetic loss path: `trend_loss + seasonal_loss + gate_l1`, not the older erroneous `pred_loss + component_loss` style.
- Run scripts default to `synth_interval=10` to align with the recorded `s10` setting.
