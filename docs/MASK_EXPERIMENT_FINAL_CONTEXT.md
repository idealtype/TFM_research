# Mask Experiment Final Context

This document is the handoff note for the mask experiments around:

- `/home/sia2/project/5.30fine_mask`
- `/home/sia2/project/5.30soft_mask`
- `/home/sia2/project/6.1nogate_softmask`
- `/home/sia2/project/6.1AR_temp`

The older implementation guides already explain the original hard-mask and soft-mask architecture. This file only records the final decisions, fixes, and cautions from the later experiment work.

## Project Intent

`5.30fine_mask` is the hard-mask baseline. It uses a Fourier basis whose harmonics are activated by fixed rules:

```text
active = (fd < P/k) and (context_span >= P/k)
```

`5.30soft_mask` keeps the same decoder family but removes the `context_span >= P/k` rule from the basis and adds a learned per-harmonic soft gate. The intent was to let the model learn which resolvable harmonics should be used instead of hard-coding context-span eligibility.

`6.1nogate_softmask` is an ablation of soft mask. It removes the learned gate and applies sparsity directly to the predicted seasonal coefficients. The intent was to test whether the gate was an unnecessary middle layer.

`6.1AR_temp` is evaluation-only. It uses the best soft-mask h96 decoder autoregressively by recomputing TimesFM backbone embeddings on rolling contexts. It is not part of the main non-AR comparison.

## Final Training Policy Adopted

For current soft/no-gate warm-mix comparisons, the default fair training setup is:

```text
init_checkpoint_dir = none
batch_size = 1024
fourier_warmup_steps = 125
mixed_steps = 2500
residual_steps = 500
synth_interval = 13
real_group_chunk_steps = 63
horizons = 96 192 336 720 in parallel
```

The increased-data setting was tested and should not be the default:

```text
mixed_steps = 5000
real_group_chunk_steps = 125
```

That setting did not improve warm mix and clearly hurt cycle/interleaved training.

## Synthetic Loss Fix

An important correction was made to synthetic Fourier training and mixed synthetic steps.

The earlier loss mixed total prediction loss with decomposition targets:

```text
pred_loss + seasonal_loss
```

or equivalent variants. This was judged problematic because synthetic batches then train the full prediction path in a way that can blur decomposition roles, especially after residual becomes active.

The adopted synthetic loss is:

```text
trend_loss + seasonal_loss + sparsity_term
```

For `5.30soft_mask`, the sparsity term is:

```text
gate_l1_weight * decomp["gates"].mean()
```

For `6.1nogate_softmask`, the sparsity term is:

```text
coeff_l1_weight * mean(abs(seasonal_coefficients))
```

For hard mask, there is no gate/coeff sparsity term in the warm-mix file. The hard-mask synthetic loss was later updated to use:

```text
trend_loss + seasonal_loss
```

## Real Loss Policy

Real-data training remains prediction-focused:

```text
pred_loss + ts_corr_weight * trend_seasonal_corr
```

Soft mask additionally keeps gate L1 on real batches:

```text
pred_loss + ts_corr + gate_l1
```

No-gate soft mask uses coefficient L1 instead:

```text
pred_loss + ts_corr + coeff_l1
```

Do not add extra residual penalties by default. Earlier residual-only phases and extra residual supervision were not consistently helpful.

## Warm Mix vs Cycle/Interleaved

Warm mix means:

```text
Fourier warmup
real-dominant mixed training with sparse synthetic batches
final residual-only real finetune
```

Cycle/interleaved means:

```text
Fourier warmup
full mixed burn-in
repeat:
  full mixed block
  residual-only block
```

For cycle/interleaved, the final residual-only block was intentionally removed. The final checkpoint should end immediately after a full-decoder mixed block, not after residual-only training. This avoids saving a final model biased toward residual-only updates.

The current cycle defaults are:

```text
batch_size = 1024
fourier_warmup_steps = 125
mixed_steps = 2500
full_burnin_steps = 375
cycle_full_steps = 250
cycle_residual_steps = 13
synth_interval = 13
real_group_chunk_steps = 63
```

## Evaluation Policy

For current real-data comparisons:

- Use real LOTSA+ETT only unless explicitly asked otherwise.
- Do not run TimesFM during evaluation.
- Merge precomputed TimesFM metrics only for plotting/comparison.
- Keep synthetic evaluation disabled in current warm/cycle scripts.
- Real eval should save residual activation diagnostics where available:
  - `no_residual_mae`
  - `residual_gain = no_residual_mae - total_mae`
  - `residual_std`
  - `total_pred_abs_mean`
  - `residual_abs_mean`
  - `residual_total_abs_ratio`

The old residual/seasonality ratio metrics were considered misleading and should not be used for conclusions.

## Hard Mask Update Status

`5.30fine_mask` was updated at the end to better match the current soft-mask fair-comparison setup:

- `train.py` now allows `--init_checkpoint_dir none` for scratch init.
- Fourier synthetic batches now return both `trend_n` and `seasonal_n`.
- Fourier synthetic loss now uses `trend_loss + seasonal_loss`.
- `train_warm_real_mix.py` defaults were changed to the current fair setup.
- `run_warm_real_mix.sh` now runs horizons in parallel, uses per-horizon roots, tees progress to the terminal, uses `--skip_tfm`, and disables synthetic eval by comments.

Important: the old hard-mask result currently summarized in earlier plots is still the old run:

```text
/home/sia2/project/5.30fine_mask/results/fourier_warm_real_mix
```

The updated hard-mask script output will go to:

```text
/home/sia2/project/5.30fine_mask/results/fourier_warm_real_mix_scratch_synth13_b1024_parallel_trend_seasonal_loss
```

If that folder has not been run yet, do not treat hard vs soft as fully matched.

## No-Gate Soft Mask

`6.1nogate_softmask` was created from `5.30soft_mask` but removes the gate from the seasonal decoder.

The seasonal decoder now directly uses:

```text
seasonal = basis @ raw_coeff
```

instead of:

```text
seasonal = basis @ (raw_coeff * expanded_gate)
```

The model output no longer contains `decomp["gates"]`. Use:

```python
seasonal_coeff_l1(decomp)
```

for coefficient sparsity.

No-gate warm and no-gate cycle both ran successfully, but neither beat the best soft-mask runs. This is evidence that the gate is not just unnecessary complexity; it appears to provide useful coefficient-selection stability.

## AR Evaluation

`6.1AR_temp` is separate and evaluation-only. It loads the best soft-mask h96 checkpoint:

```text
/home/sia2/project/5.30soft_mask/results/fourier_warm_real_mix_scratch/checkpoints/funcdec_h96.pt
```

It then predicts longer horizons autoregressively:

```text
h192 = h96 x 2
h336 = h96 x 4, then truncate to 336
h720 = h96 x 8, then truncate to 720
```

Because the context changes after each block, it must recompute TimesFM backbone embeddings on rolling raw contexts. Cached embeddings alone are not enough.

Result summary:

- h96 is identical to direct h96.
- h192 is close but worse than direct.
- h336 degrades.
- h720 can fail badly, especially on `alibaba_cluster_trace_2018`.
- Excluding alibaba, AR h96 was slightly competitive with direct soft, but overall it is too unstable for the main comparison.

Do not include AR results in the main hard/soft/no-gate non-AR leaderboard.

## Main Result Summary

The consolidated non-AR analysis is stored at:

```text
/home/sia2/project/analysis_real_results_no_ar
```

Key files:

```text
REPORT.md
tables/summary_overall.csv
tables/focused_summary_overall.csv
tables/summary_by_horizon.csv
tables/summary_by_dataset.csv
tables/residual_metric_leaderboard.csv
plots/13_table_focused_overall.png
plots/15_table_horizon_key_runs.png
plots/16_table_residual_metrics.png
plots/17_table_dataset_key_runs.png
```

Focused non-AR real-data results:

| family | setting | mean MAE | note |
|---|---:|---:|---|
| TimesFM | zeroshot | 0.409603 | external baseline |
| soft | warm s10 old | 0.504205 | best non-AR FuncDec result |
| soft | syn_all_real | 0.506094 | close to best |
| soft | cycle b1024 | 0.512634 | best residual activation among current eval columns |
| no-gate | warm | 0.515410 | does not beat soft best |
| no-gate | cycle | 0.517711 | does not beat soft cycle |
| hard | warm old | 0.520657 | old unmatched hard result |
| hard | base | 0.555722 | hard baseline |

Main conclusions:

- Best non-AR FuncDec result is still `soft_warm_s10_oldloss_best`.
- `soft_syn_all_real` is almost tied with it.
- The newer trend/seasonal synthetic loss improves decomposition logic but did not beat the old best MAE.
- Cycle/interleaved improves residual activation but not top-line MAE.
- No-gate does not justify removing the gate.
- More training data exposure via `m5000/chunk125` should not be the default.

## Known Error Runs and Cautions

Do not use this as a fair result:

```text
/home/sia2/project/5.30soft_mask/results/fourier_warm_real_mix_checkpointerror
```

It used checkpoint initialization and is not on the same footing as scratch soft runs.

When comparing settings, verify:

```text
init_checkpoint_dir
batch_size
fourier_warmup_steps
mixed_steps
synth_interval
real_group_chunk_steps
synthetic loss structure
whether TimesFM was run or merged
whether synthetic eval was included
```

Do not compare old hard warm directly against updated soft/no-gate results without noting that it used:

```text
checkpoint init
batch_size = 256
fourier_warmup_steps = 500
mixed_steps = 10000
residual_steps = 2000
synth_interval = 10
real_group_chunk_steps = 250
```

## Current Recommended Defaults

For any next continuation, start from these unless explicitly changing one factor:

```text
scratch init
batch_size = 1024
fourier_warmup_steps = 125
mixed_steps = 2500
residual_steps = 500 for warm
full_burnin_steps = 375 for cycle
cycle_full_steps = 250 for cycle
cycle_residual_steps = 13 for cycle
synth_interval = 13
real_group_chunk_steps = 63
real eval only
TimesFM merge only, no TimesFM execution
synthetic eval disabled
```

For synthetic batches:

```text
hard: trend_loss + seasonal_loss
soft: trend_loss + seasonal_loss + gate_l1
no-gate: trend_loss + seasonal_loss + coeff_l1
```

For real batches:

```text
hard: pred_loss + ts_corr
soft: pred_loss + ts_corr + gate_l1
no-gate: pred_loss + ts_corr + coeff_l1
```

## Quick Commands

Soft warm:

```bash
cd /home/sia2/project/5.30soft_mask && ./run_warm_real_mix.sh
```

Soft cycle:

```bash
cd /home/sia2/project/5.30soft_mask && ./run_interleaved_residual_mix.sh
```

No-gate warm:

```bash
cd /home/sia2/project/6.1nogate_softmask && ./run_warm_real_mix.sh
```

No-gate cycle:

```bash
cd /home/sia2/project/6.1nogate_softmask && ./run_interleaved_residual_mix.sh
```

Updated hard warm:

```bash
cd /home/sia2/project/5.30fine_mask && ./run_warm_real_mix.sh
```

Main non-AR summary regeneration is not currently packaged as a repo script; the output exists at:

```text
/home/sia2/project/analysis_real_results_no_ar
```
