# 실험 결과 및 권장 설정

## Non-AR 전체 Leaderboard

실제 데이터(LOTSA+ETT, 32개 dataset), AR 실험(`ar_eval/`) 제외.

| family | setting | mean_mae | median_mae | n |
|---|---|---|---|---|
| baseline | timesfm_zeroshot | 0.409603 | 0.367596 | 32 |
| soft_mask | soft_warm_s10_oldloss_best | 0.504205 | 0.468409 | 32 |
| soft_mask | soft_syn_all_real | 0.506094 | 0.547303 | 32 |
| soft_mask | soft_inter_s13_res13_b1024 | 0.512634 | 0.534876 | 32 |
| nogate_soft_mask | nogate_warm_s13_b1024_coeffl1 | 0.515410 | 0.505850 | 32 |
| soft_mask | soft_warm_s50_oldloss | 0.516099 | 0.492707 | 32 |
| nogate_soft_mask | nogate_inter_s13_res13_b1024_coeffl1 | 0.517711 | 0.497878 | 32 |
| soft_mask | soft_warm_s50_ts_loss | 0.518661 | 0.511611 | 32 |
| soft_mask | soft_warm_s13_b1024_m5000 | 0.520006 | 0.527170 | 32 |
| hard_mask | hard_warm_mix_old | 0.520657 | 0.506257 | 32 |
| soft_mask | soft_inter_s50_res50_ts_loss | 0.522895 | 0.563502 | 32 |
| soft_mask | soft_inter_s50_res50_oldloss | 0.525622 | 0.559641 | 32 |
| soft_mask | soft_warm_checkpoint_error | 0.527971 | 0.515028 | 32 |
| hard_mask | hard_syn_and_alldata | 0.555722 | 0.578488 | 32 |
| soft_mask | soft_interleaved_old | 0.565629 | 0.580948 | 32 |
| soft_mask | soft_inter_s13_res13_b1024_m5000 | 0.575103 | 0.588290 | 32 |

전체 표 및 dataset/horizon별 분석: `docs/analysis_real_results_no_ar/`

---

## Residual 활성화 지표 (있는 run만)

| setting | mean_mae | residual_gain | residual_std | residual_ratio |
|---|---|---|---|---|
| soft_inter_s13_res13_b1024 | 0.512634 | 0.141580 | 0.414732 | 0.490577 |
| nogate_warm_s13_b1024_coeffl1 | 0.515410 | 0.100725 | 0.422267 | 0.514404 |
| nogate_inter_s13_res13_b1024_coeffl1 | 0.517711 | 0.132608 | 0.402310 | 0.461263 |
| soft_warm_s13_b1024_m5000 | 0.520006 | 0.089786 | 0.420649 | 0.493486 |
| soft_inter_s13_res13_b1024_m5000 | 0.575103 | 0.088216 | 0.376815 | 0.502928 |

---

## 주요 결론

- **최고 FuncDec 결과**: `soft_warm_s10_oldloss_best` (MAE 0.5042)
- `soft_syn_all_real`은 거의 동점 수준으로 근접.
- Cycle/interleaved는 residual activation은 개선하나 top-line MAE는 개선 못함.
- Nogate는 gate 제거를 정당화하지 못함 — gate가 coefficient-selection stability에 기여.
- `m5000/chunk125` 더 많은 학습 데이터 설정은 성능 개선 없음 → default 아님.

---

## 권장 Defaults

새 실험 시작점. 특정 factor 하나만 바꾸는 실험 외에는 이 설정 유지.

```
scratch init (init_checkpoint_dir = none)
batch_size           = 1024
fourier_warmup_steps = 125
mixed_steps          = 2500
residual_steps       = 500     (warm)
full_burnin_steps    = 375     (cycle)
cycle_full_steps     = 250     (cycle)
cycle_residual_steps = 13      (cycle)
synth_interval       = 13
real_group_chunk_steps = 63
real eval only (합성 eval 비활성)
TimesFM merge only (TimesFM 실행 없음)
```

Synth loss:
```
hard:    trend_loss + seasonal_loss
soft:    trend_loss + seasonal_loss + gate_l1
nogate:  trend_loss + seasonal_loss + coeff_l1
```

Real loss:
```
hard:    pred_loss + ts_corr
soft:    pred_loss + ts_corr + gate_l1
nogate:  pred_loss + ts_corr + coeff_l1
```

---

## Known Error Runs / 비교 주의사항

**사용 금지 run:**
- `soft_warm_checkpoint_error`: checkpoint init으로 시작했음 — scratch 결과와 비교 불가.

**구 hard mask 결과 비교 주의:**
`hard_warm_mix_old`는 아래 설정으로 실행된 구버전 결과임.
현재 soft/nogate와 직접 비교 불가.
```
checkpoint init (not scratch)
batch_size           = 256
fourier_warmup_steps = 500
mixed_steps          = 10000
residual_steps       = 2000
synth_interval       = 10
real_group_chunk_steps = 250
```

**설정 비교 시 반드시 확인할 항목:**
`init_checkpoint_dir`, `batch_size`, `fourier_warmup_steps`, `mixed_steps`,
`synth_interval`, `real_group_chunk_steps`, synthetic loss 구조,
TimesFM 실행 여부, synthetic eval 포함 여부.

---

## AR 평가 요약

`ar_eval/` 실험: best soft-mask h96 체크포인트로 longer horizon을 자기회귀적으로 예측.

- h96: 직접 예측과 동일
- h192: 근접하나 direct보다 나쁨
- h336: 성능 저하
- h720: alibaba_cluster_trace_2018 등 일부 dataset에서 크게 실패

→ **메인 leaderboard에 포함하지 않음.**
