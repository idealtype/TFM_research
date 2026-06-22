# Hard Mask Experiment

베이스라인. Fourier seasonal 디코더에서 **per-harmonic hard masking**을 사용.

## 핵심 설계

### 주기 및 오더

```python
PERIODS = {"daily": 1.0, "weekly": 7.0, "monthly": 30.4375, "yearly": 365.25}  # 단위: 일
K_MAX   = {"daily": 10,  "weekly": 4,   "monthly": 2,        "yearly": 8}
```

1.5× 분리 원칙 (인접 주기 간 주파수 경합 방지):
- weekly k=4 (1.75d) ÷ daily (1.0d) = 1.75× ✓
- monthly k=2 (15.2d) ÷ weekly (7.0d) = 2.17× ✓
- yearly k=8 (45.7d) ÷ monthly (30.4d) = 1.50× ✓ (경계값)

### Masking 공식

```python
harmonic_period = P / k
fd = FREQ_DAYS[freq]
context_span = context_len * fd

active = (fd < harmonic_period) and (context_span >= harmonic_period)
```

각 하모닉 k마다 개별 조건 평가 (주기 단위 binary on/off가 아님).

### 시간축 기저 생성 (중요)

```python
# 올바른 패턴 — context 끝에서 horizon 시작
t = torch.arange(context_len, context_len + horizon, dtype=torch.float32)
# 틀린 패턴 — 위상 오정렬 발생
# t = torch.arange(horizon)  ← 사용 금지
```

## 캐시 파일 명명

| 구분 | 파일명 |
|---|---|
| basis (합성) | `fourier_basis_fine_mask_h{H}.pt` |
| basis (실제) | `fourier_basis_c{C}_{freq}_h{H}_fine_mask_lotsa.pt` |
| 계수 (합성) | `seasonal_coefficients_fine_mask_h{H}.pt` |
| 합성 cache suffix | `10_4_2_8` |

## 학습 진입점

```bash
./run_warm_real_mix.sh
```

horizons 96/192/336/720 병렬 실행, `--skip_tfm` 포함.

## 주요 참고사항

- `oldloss` 명의 구 hard mask 결과(`hard_warm_mix_old`)는 다른 설정(batch=256, checkpoint init 등)으로 실행됐음 — 현재 soft/nogate 결과와 직접 비교 불가. 상세 → `docs/EXPERIMENT_RESULTS.md`.
- 구현 세부사항(Phase별 캐싱, model 구조 변경 이력 등)은 git history 참조.
