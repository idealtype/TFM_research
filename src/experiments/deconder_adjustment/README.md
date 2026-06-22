# Decoder Adjustment 실험군

`soft_mask`의 residual decoder를 **TimesFM-style point residual block**으로 교체한 실험군.
각 subfolder는 이 decoder 교체를 base로 하여 loss 하나씩만 추가로 변경.

## 공통 Base 변경 (soft_mask → decoder_adjusted)

Residual decoder 구조:
```
Linear(embed_dim → hidden)
SiLU
Linear(hidden → horizon)
+ direct residual projection: Linear(embed_dim → horizon)
```

backbone embedding 경로, seasonal decoder, gate 구조는 `soft_mask`와 동일.

## Sub-experiment 목록

| 폴더 | soft_mask 대비 추가 변경 |
|---|---|
| `soft_warm_s10_oldloss_best/` | decoder 교체만 (loss 변경 없음) |
| `soft_warm_s10_oldloss_best_trend_sparse/` | + trend changepoint sparsity loss (`trend_delta_l1_weight=0.2`) |
| `soft_warm_s10_coeffloss_warmup_mixed/` | + Fourier warmup & mixed synth에서 seasonal coefficient L1 loss |
| `soft_warm_s10_coeffloss_warmup_only/` | + Fourier warmup phase만 coefficient L1 loss (mixed synth은 기존 유지) |

## 명명 주의

`oldloss`는 기록된 run label이며, **현재 코드는 수정된 loss** (`trend_loss + seasonal_loss + gate_l1`)를 사용.
이전의 잘못된 `pred_loss + component_loss` 방식이 아님.

## 학습 진입점

각 subfolder에서:
```bash
./run_warm_real_mix.sh
```
