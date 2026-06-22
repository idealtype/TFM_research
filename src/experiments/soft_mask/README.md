# Soft Mask Experiment

**현재 최고 FuncDec 결과** (`soft_warm_s10_oldloss_best`, MAE 0.5042).

`hard_mask`에서 context_span 하드 조건을 **백본 임베딩이 예측하는 per-harmonic soft gate**로 대체한 실험.

## hard_mask 대비 변경점

### 1. Basis 빌드 규칙

| | hard_mask | soft_mask |
|---|---|---|
| 물리 조건 `fd < P/k` | 적용 | 동일하게 적용 |
| context span 조건 `context_span >= P/k` | 적용 (basis 0) | **제거 → gate로 학습** |
| basis 소스 | 캐시 파일 | **on-the-fly 계산** (freq + horizon만 필요) |

backbone_emb 캐시는 hard_mask와 동일하게 재사용.

### 2. GatingMLP (SeasonalDecoder 추가)

```python
GatingMLP: embed_dim(1280) → Linear → ReLU → Linear → 24
# 마지막 Linear: weight=0, bias=0으로 초기화 → gate 초기값 sigmoid(0) = 0.5
```

Gate 슬라이싱:
```
gate_daily   = gates[:,  0:10]   # K_MAX["daily"] = 10
gate_weekly  = gates[:, 10:14]   # K_MAX["weekly"] = 4
gate_monthly = gates[:, 14:16]   # K_MAX["monthly"] = 2
gate_yearly  = gates[:, 16:24]   # K_MAX["yearly"] = 8
```

계수 적용: `coef *= repeat_interleave(gate, 2, dim=1)` → bmm 전에 적용.

### 3. decomp 반환 dict

`"gates"` 키 추가: `(B, 24)` tensor. train.py의 L1 loss 계산에 사용.

### 4. Loss

```python
total_loss = reconstruction_loss + gate_l1_weight * decomp["gates"].mean()
```

모든 학습 phase에서 동일하게 적용.

## 학습 진입점

```bash
./run_warm_real_mix.sh          # warm 방식
./run_interleaved_residual_mix.sh  # cycle 방식
```

## 설계 의도

- hard_mask의 context_span 조건은 heuristic: 실제로는 짧은 컨텍스트에도 backbone embedding이 해당 주기의 단서를 담을 수 있음.
- L1 regularization이 gate를 0으로 끌어당겨 불필요한 harmonic을 실제로 끔 → hard sparsity를 soft하게 학습하는 효과.
