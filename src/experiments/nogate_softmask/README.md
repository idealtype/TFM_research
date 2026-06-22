# No-Gate Soft Mask Experiment

`soft_mask`에서 gate를 제거한 ablation. Gate가 불필요한 중간 레이어인지 검증 목적.

## soft_mask 대비 변경점

Gate를 제거하고 예측된 계수를 basis에 직접 적용:

```python
# soft_mask
seasonal = basis @ (raw_coeff * expanded_gate)

# nogate_softmask
seasonal = basis @ raw_coeff
```

- `decomp` dict에 `"gates"` 키 없음.
- Gate L1 대신 **coefficient L1** sparsity 사용:

```python
# soft_mask
gate_l1_weight * decomp["gates"].mean()

# nogate_softmask
coeff_l1_weight * decomp["seasonal_coefficients"].abs().mean()
```

## 학습 진입점

```bash
./run_warm_real_mix.sh          # warm 방식
./run_interleaved_residual_mix.sh  # cycle 방식
```

## 결과 요약

- nogate warm: MAE 0.5154 → soft_warm best(0.5042)보다 나쁨
- nogate cycle: MAE 0.5177 → soft cycle(0.5126)보다 나쁨

Gate가 단순한 중간 레이어가 아니라 coefficient-selection stability에 기여함을 시사.
→ 자세한 비교: `docs/EXPERIMENT_RESULTS.md`
