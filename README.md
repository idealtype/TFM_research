# TFM Experiments

This repository contains the lightweight source snapshot for the TFM experiment workflow.

The original Google Drive project was copied into a Git/VESSL-oriented layout while excluding datasets, checkpoints, archives, plots, tables, logs, and generated result files.

## Layout

```text
project/
├── config/
├── data/
├── docs/
├── notebooks/
└── src/
    ├── common/
    ├── data_gen/
    └── experiments/
        ├── ar_eval/
        ├── hard_mask/
        ├── nogate_softmask/
        ├── soft_mask/
        └── synthetic_center/
```

## Experiment Families

- `src/experiments/hard_mask`: hard-mask baseline copied from `5.30fine_mask`.
- `src/experiments/soft_mask`: learned soft harmonic gate copied from `5.30soft_mask`.
- `src/experiments/nogate_softmask`: no-gate soft-mask ablation copied from `6.1nogate_softmask`.
- `src/experiments/ar_eval`: h96 autoregressive evaluation copied from `6.1AR_temp`.
- `src/experiments/synthetic_center`: synthetic-center and non-Fourier fine-tuning copied from `5.22syn_cent`.
- `src/data_gen/fine_mask`: synthetic data generation code copied from `5.30fine_mask/data`.

## Notes

- `data/` is intentionally empty except for `.gitkeep`; it is reserved for VESSL storage mounts.
- Existing scripts still contain original absolute paths such as `/home/sia2/project/...`.
- Path normalization, local smoke tests, and VESSL command generation are intentionally deferred to later steps.
