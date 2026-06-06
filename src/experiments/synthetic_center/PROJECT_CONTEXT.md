# 5.22 Synthetic Center Context

Last updated: 2026-05-25 KST

This document extends the consolidated April FuncDec project context in:

```text
/home/sia2/project/4.28basis/PROJECT_CONTEXT.md
```

Use this folder for new synthetic-center training and evaluation work. The
older Fourier-only experiments remain in `4.28basis`; new variants that train
on non-Fourier synthetic data should live under `5.22syn_cent`.

## Base Project Assumptions

- TimesFM 2.5 is a frozen backbone.
- Training uses cached TimesFM embeddings; do not call TimesFM `_encode()`
  inside decoder training loops.
- FuncDec predicts decomposed forecast components:
  - `trend`
  - `seasonal`
  - `residual`
  - total forecast as the sum of decoder outputs.
- Losses, metrics, and plots are in normalized space unless explicitly stated.
- Evaluation checkpoints are horizon-specific and expected as:
  - `checkpoints/funcdec_h96.pt`
  - `checkpoints/funcdec_h192.pt`
  - `checkpoints/funcdec_h336.pt`
  - `checkpoints/funcdec_h720.pt`

## Active Goal

The current 5.22 work compares training policies for non-Fourier seasonal
generalization. The base model is the earlier Fourier simple+complex checkpoint:

```text
/home/sia2/project/4.28basis/basis_dec/experiment/func_dec_syn_cent/results/simple_complex_synth_fixed_phase_scale/train/simple_complex_coeff_residual_tail
```

Evaluation targets are:

- non-Fourier synthetic evaluation cache
- real LOTSA/ETT evaluation cache

Primary data roots:

```text
/home/sia2/project/data/synthetic_nonF/synth_train_nonfourier
/home/sia2/project/data/synthetic_nonF/synth_eval_nonfourier
/home/sia2/project/data/real_eval_lot_ett
```

Non-Fourier generator families:

- `cyclic_spline`
- `sarima`
- `sawtooth`
- `daubechies`
- `symlet`

Non-Fourier stages:

- `stage1_S_nonfourier`
- `stage2_T_S_nonfourier`
- `stage3_T_S_R_nonfourier`

Stage3 residuals are distribution noise:

- `normal`
- `student_t`
- `exponential`
- `gamma`
- `weibull`
- `pareto`

Do not confuse these with structural residual experiments under:

```text
/home/sia2/project/data/syn_str_res
```

## Raw-Target Non-Fourier Fine-Tuning

Canonical folder:

```text
/home/sia2/project/5.22syn_cent/train_nonF_rawtarget
```

Intent:

- Start from the Fourier simple+complex checkpoint.
- Fine-tune on non-Fourier synthetic training data.
- Trend decoder is supervised by `trend_n`.
- Seasonal and residual decoders jointly explain non-Fourier `seasonal_n`.
- Stage3 distribution noise is not a supervised target.

Important files:

```text
train_nonfourier_finetune.py
run_train_eval_nonfourier.sh
run_eval_existing_nonfourier.sh
```

Result root:

```text
/home/sia2/project/5.22syn_cent/train_nonF_rawtarget/results
```

Use `train_nonF_rawtarget` as the canonical path. A temporary `train_nonF`
folder name caused checkpoint-path confusion earlier; do not rely on it.

## Projection-Target Non-Fourier Fine-Tuning

Canonical folder:

```text
/home/sia2/project/5.22syn_cent/train_nonF_projtarget
```

Intent:

- Start from the Fourier simple+complex checkpoint.
- Project each non-Fourier seasonal target onto the Fourier basis.
- Seasonal decoder learns Fourier coefficients for the projected seasonal
  component.
- Residual decoder learns the remainder:
  `seasonal_n - projected_seasonal_n`.
- Trend decoder learns `trend_n`.
- Stage3 distribution noise is not a supervised target.

Important files:

```text
build_projection_targets.py
train_projection_finetune.py
run_projection_cache_train_eval.sh
run_projection_eval_existing.sh
```

Projection cache root:

```text
/home/sia2/project/5.22syn_cent/train_nonF_projtarget/projection_targets
```

Result root:

```text
/home/sia2/project/5.22syn_cent/train_nonF_projtarget/results
```

The projection pipeline is expected to evaluate real data first, then synthetic
data.

## Evaluation Outputs

Each 5.22 experiment should write:

```text
{experiment_root}/results/nonfourier_single_model
{experiment_root}/results/real_lot_ett_single_model
```

Synthetic result files:

- `nonfourier_component_mae.csv`
- `nonfourier_summary.json`
- `performance_by_horizon_all.png`
- per-stage and per-generator plots

Real result files:

- `real_eval_component_mae.csv`
- `real_eval_summary.json`
- `performance_by_horizon_all.png`
- dataset folders directly under `real_lot_ett_single_model`

Extra summary plot helper:

```text
/home/sia2/project/5.22syn_cent/plot_extra_result_summaries.py
```

Expected extra plots:

- synthetic:
  - `generator_model_mae_by_horizon_all_stages.png`
  - `stage*/generator_model_mae_by_horizon.png`
  - `stage*/performance_by_horizon_generators_all.png`
- real:
  - `dataset_model_mae_by_horizon.png`
  - `performance_by_horizon_datasets_all.png`

## Plot Conventions

Synthetic example plots:

- Left panel: synthetic components with context plus horizon.
- Center panel: TimesFM prediction and ground truth, horizon only.
- Right panel: FuncDec prediction and predicted components, horizon only.
- Plot samples are grouped by stage, generator, and horizon.

Real example plots:

- One image contains 3 samples.
- Each sample has one TimesFM prediction panel and one FuncDec component panel.
- Therefore one image has 6 subplots.

## Current Known Issue

Projection-target h96 can produce very large least-squares Fourier
coefficients. The reconstructed seasonal signal may still be normal-scale, but
coefficient loss can look extremely high because the h96 basis is
ill-conditioned for some groups.

Observed diagnostic pattern:

- projection target cache built about 2580 groups and 780000 samples
- mean projection remainder MAE was about 0.125 in normalized space
- h96 coefficient targets can have condition numbers in the millions
- h96 coefficient magnitudes can become orders of magnitude larger than the raw
  normalized signal

If this remains a problem, prefer a deliberate fix such as ridge projection,
coefficient scaling, or reduced basis for h96. Do not treat this as ordinary
optimizer noise.

## Commands

Raw-target full train and eval:

```bash
cd /home/sia2/project/5.22syn_cent/train_nonF_rawtarget
DEVICE=cuda:0 ./run_train_eval_nonfourier.sh
```

Raw-target eval only:

```bash
cd /home/sia2/project/5.22syn_cent/train_nonF_rawtarget
DEVICE=cuda:0 ./run_eval_existing_nonfourier.sh
```

Projection-target cache, train, and eval:

```bash
cd /home/sia2/project/5.22syn_cent/train_nonF_projtarget
DEVICE=cuda:0 ./run_projection_cache_train_eval.sh
```

Projection-target eval only:

```bash
cd /home/sia2/project/5.22syn_cent/train_nonF_projtarget
DEVICE=cuda:0 ./run_projection_eval_existing.sh
```

## Stale Content Policy

Old notes about calibration-only synthetic work, Monash HPO, Prophet, and
pre-5.22 run procedures are intentionally not repeated here. Keep them archived
unless a future task explicitly returns to those experiment families.
