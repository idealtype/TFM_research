# Unified Fourier Synthetic Generation

This folder contains the current replacement for the split synthetic data flow.

The old flow mixed:

- legacy 3-family Fourier complex data: daily, weekly, yearly
- fine-mask add-on data with monthly support

The new flow writes one 4-family Fourier pool:

```text
/workspace/data/synthetic/func_dec_syn_cent_fourier_all_train/
/workspace/data/synthetic/func_dec_syn_cent_fourier_all_eval/
/workspace/data/synthetic/func_dec_syn_cent_fourier_all_train_cache_10_4_2_8/
/workspace/data/synthetic/func_dec_syn_cent_fourier_all_eval_cache_10_4_2_8/
```

## Policy

- Decoder order is fixed to `daily=10`, `weekly=4`, `monthly=2`, `yearly=8`.
- Canonical granularities are:
  `5_minutes`, `10_minutes`, `15_minutes`, `half_hourly`, `hourly`, `daily`, `weekly`, `monthly`.
- Active harmonics follow the hard-mask rule:

```text
active = fd < P/k and context_len * fd >= P/k
```

- For each granularity, the generator enumerates every active harmonic-count
  combination per family, excluding the all-zero case.
- Each sample cycles through those count cases, then randomly chooses concrete
  harmonic indices, phases, amplitudes, trend, and composition noise/scaling.
- Cache files use only the latest 4-family format:
  `fourier_basis_fine_mask_h{H}.pt`,
  `seasonal_coefficients_fine_mask_h{H}.pt`,
  `component_targets_h{H}.pt`,
  `raw_futures_h{H}.pt`,
  `backbone_emb_c512_h{H}_stride1.pt`.

## VESSL Command Shape

```bash
cd /workspace/data/tfm
DATA_ROOT=/workspace/data DEVICE=cuda:0 \
  ./src/data_gen/fourier_synth/run_build_fourier_synth.sh
```

Use `--metadata_only` for a CPU-side structure check without TimesFM embeddings.
