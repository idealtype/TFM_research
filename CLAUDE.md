# TFM Research — 프로젝트 컨텍스트

## 프로젝트 목적

TimesFM 2.5 백본(임베딩 고정) 위에 **FuncDec 디코더**를 붙여 시계열 예측 성능을 높이는 연구.
핵심 비교축: Fourier 기반 seasonal 디코더에서 하모닉 활성화를 **hard / soft / nogate** 세 가지 방식으로 처리했을 때의 성능 차이.

---

## 현재 베스트 결과 (non-AR, real LOTSA+ETT 기준)

| family | setting | mean MAE |
|---|---|---|
| TimesFM | zeroshot | 0.4096 |
| **soft_mask** | **soft_warm_s10_oldloss_best** | **0.5042** ← 현재 최고 FuncDec |
| soft_mask | soft_syn_all_real | 0.5061 |
| soft_mask | soft_inter_s13_res13_b1024 | 0.5126 |
| nogate_soft_mask | nogate_warm_s13_b1024_coeffl1 | 0.5154 |
| hard_mask | hard_warm_mix_old | 0.5207 |
| hard_mask | hard_syn_and_alldata | 0.5557 |

전체 leaderboard 및 dataset별/horizon별 분석 → `docs/EXPERIMENT_RESULTS.md`

---

## 폴더 구조

```
project/
├── CLAUDE.md                        ← 이 파일 (루트 진입점)
├── README.md                        ← 간략한 레이아웃 안내
├── Dockerfile / requirements-vessl.txt
├── config/
├── data/                            ← VESSL 볼륨 마운트용 (로컬 비어있음)
├── results/                         ← gitignore, 로컬 다운로드 결과
├── docs/
│   ├── VESSL_WORKFLOW_CONTEXT.md    ← VESSL 운영 세부사항 (필독)
│   ├── EXPERIMENT_RESULTS.md        ← 실험 결과, 권장 defaults, cautions
│   └── analysis_real_results_no_ar/ ← non-AR 결과 표 및 플롯
├── data_source_light/               ← 데이터 생성 코드 경량 스냅샷
├── timesfm_origin/                  ← TimesFM 원본 참조 코드
└── src/
    ├── common/
    ├── data_prep/                   ← v1 backbone 재캐싱 등
    ├── data_gen/
    │   └── fourier_synth/           ← 4-family Fourier 합성 데이터 생성
    └── experiments/
        ├── hard_mask/               ← hard mask 베이스라인
        ├── soft_mask/               ← learned soft harmonic gate (현재 최고)
        ├── nogate_softmask/         ← gate 제거 ablation
        ├── ar_eval/                 ← h96 AR 평가 (비교 제외)
        ├── synthetic_center/        ← 합성 중심 학습 실험
        ├── deconder_adjustment/     ← decoder 구조 변경 실험군
        │   ├── soft_warm_s10_oldloss_best/
        │   ├── soft_warm_s10_oldloss_best_trend_sparse/
        │   ├── soft_warm_s10_coeffloss_warmup_mixed/
        │   └── soft_warm_s10_coeffloss_warmup_only/
        ├── backbone_adjustment/     ← TimesFM v1 backbone 교체 실험군
        │   ├── soft_warm_s10_oldloss_best_v1_backbone/
        │   └── soft_warm_s10_oldloss_best_v1_backbone_decoder_adjusted/
        └── xreg_soft_mask/          ← XReg 오버레이 평가 (가장 최근)
```

---

## 실험 계보

```
hard_mask
  └─ soft_mask (context_span 조건 → learned gate)
       └─ nogate_softmask (gate 제거 ablation)
       └─ deconder_adjustment/ (decoder 구조 → TimesFM-style residual block)
       │   └─ soft_warm_s10_oldloss_best           (base: decoder 교체)
       │   └─ soft_warm_s10_oldloss_best_trend_sparse (+ trend sparsity loss)
       │   └─ soft_warm_s10_coeffloss_warmup_mixed  (+ coeff loss warmup+mixed)
       │   └─ soft_warm_s10_coeffloss_warmup_only   (+ coeff loss warmup only)
       └─ backbone_adjustment/ (backbone → TimesFM v1)
       │   └─ soft_warm_s10_oldloss_best_v1_backbone
       │   └─ soft_warm_s10_oldloss_best_v1_backbone_decoder_adjusted
       └─ xreg_soft_mask (deconder_adjustment 기반 + XReg 오버레이 평가)
```

---

## 핵심 하이퍼파라미터 기조 (권장 defaults)

모든 실험은 특별한 언급이 없으면 이 설정을 기본으로 한다.

```
init_checkpoint_dir  = none (scratch)
batch_size           = 1024
fourier_warmup_steps = 125
mixed_steps          = 2500
residual_steps       = 500        (warm 방식)
full_burnin_steps    = 375        (cycle 방식)
cycle_full_steps     = 250        (cycle 방식)
cycle_residual_steps = 13         (cycle 방식)
synth_interval       = 13
real_group_chunk_steps = 63
horizons             = 96 192 336 720 (병렬)
```

**비교 금지:** m5000/chunk125 설정은 성능 개선 없음 — default 아님.

---

## Loss Policy

| 구분 | synth batch | real batch |
|---|---|---|
| hard_mask | `trend_loss + seasonal_loss` | `pred_loss + ts_corr` |
| soft_mask | `trend_loss + seasonal_loss + gate_l1` | `pred_loss + ts_corr + gate_l1` |
| nogate_softmask | `trend_loss + seasonal_loss + coeff_l1` | `pred_loss + ts_corr + coeff_l1` |

- `ts_corr`: trend-seasonal squared-correlation penalty
- residual-only phase에 추가 페널티 없음

---

## 평가 정책

- **대상**: real LOTSA+ETT only (합성 평가는 기본 비활성)
- **TimesFM**: 실행하지 않고, 사전 계산된 metrics CSV만 merge해서 비교
- **AR 결과**: 메인 leaderboard에 포함하지 않음 (`ar_eval/` 참조)
- **Residual diagnostics** 저장: `no_residual_mae`, `residual_gain`, `residual_std`, `residual_abs_mean`, `residual_total_abs_ratio`

---

## VESSL 운영 핵심 (세부사항 → `docs/VESSL_WORKFLOW_CONTEXT.md`)

- **이미지**: `ghcr.io/idealtype/tfm-research:<commit>` — Dockerfile/requirements 변경 시에만 재빌드
- **데이터 볼륨**: `objvol-edwuqaa94ii3` → `/workspace/data`
- **GPU Job**: `resourcespec-a100x1` (A100 80GB), **CPU Job**: `resourcespec-grlxx3knwzps`
- **결과 경로**: `/workspace/data/results/<exp_type>/<HHmm>_<run_tag>/`
- **Job 제출**: `./scripts/vessl_submit.sh --exp <type> --run <tag> --mode train_eval`

---

## 주요 기조 사항

1. **현재 최고 베이스라인**: `soft_warm_s10_oldloss_best` — 모든 ablation의 기준점.
2. **`oldloss` 명칭**: 기록된 run label이며, 현재 코드는 이미 수정된 loss(`trend_loss + seasonal_loss + gate_l1`)를 사용.
3. **hard mask 구 결과와의 비교 주의**: 구 hard warm 결과는 `batch_size=256`, `checkpoint init` 등 다른 설정이었음. → `docs/EXPERIMENT_RESULTS.md`의 caution 섹션 참조.
4. **cycle 방식의 final checkpoint**: mixed block 직후에 저장. residual-only 마지막 단계 후 저장 금지.
5. **RealPayloadCache**: `max_items=1` 유지 (OOM 방지). GPU pre-load는 실험적으로만 적용.
6. **합성 데이터**: `src/data_gen/fourier_synth/` — 4-family (daily/weekly/monthly/yearly), decoder order `10_4_2_8`.
