# TFM Experiments

TimesFM 백본 위에 FuncDec 디코더를 붙이는 시계열 예측 실험 프로젝트.
프로젝트 전체 목적 및 핵심 기조 → **`CLAUDE.md`** 참조.

## Layout

```text
project/
├── CLAUDE.md                        ← 프로젝트 진입점 (목적/기조/VESSL 요약)
├── config/
├── data/                            ← VESSL 볼륨 마운트용 (로컬 비어있음)
├── docs/
│   ├── VESSL_WORKFLOW_CONTEXT.md    ← VESSL 운영 세부사항
│   ├── EXPERIMENT_RESULTS.md        ← 실험 결과 leaderboard 및 권장 defaults
│   └── analysis_real_results_no_ar/
├── notebooks/
├── data_source_light/               ← 데이터 생성 코드 경량 스냅샷
├── timesfm_origin/                  ← TimesFM 원본 참조 코드
└── src/
    ├── common/
    ├── data_prep/
    ├── data_gen/
    │   └── fourier_synth/           ← 4-family Fourier 합성 데이터 생성
    └── experiments/
        ├── hard_mask/               ← hard mask 베이스라인
        ├── soft_mask/               ← learned soft gate (현재 최고 결과)
        ├── nogate_softmask/         ← gate 제거 ablation
        ├── ar_eval/                 ← h96 AR 평가 (메인 비교 제외)
        ├── synthetic_center/        ← 합성 중심 학습 실험
        ├── deconder_adjustment/     ← decoder 구조 변경 실험군
        ├── backbone_adjustment/     ← TimesFM v1 backbone 교체 실험군
        └── xreg_soft_mask/          ← XReg 오버레이 평가
```

## Notes

- `data/` is intentionally empty except for `.gitkeep`; it is reserved for VESSL storage mounts.
- Some scripts still contain original absolute paths such as `/home/sia2/project/...`.
