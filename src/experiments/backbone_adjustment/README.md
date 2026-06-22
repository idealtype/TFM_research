# Backbone Adjustment 실험군

`soft_mask`의 cached backbone embedding을 **TimesFM v1** hidden state로 교체한 실험군.

## 공통 Base 변경 (soft_mask → v1 backbone)

- 기존: TimesFM 2.5 backbone embedding (캐시)
- 변경: TimesFM v1 (`google/timesfm-1.0-200m-pytorch`) last-patch transformer hidden state, shape `(N, 1280)`
- 타깃 정규화(`mu/sigma`) 유지 → 기존 `futures_n` targets와 동일 스케일

**평가 baseline 주의**: 이 실험군의 비교 대상은 TimesFM v1 (2.5 아님).
`scripts/run_v1_backbone_soft_warm.sh`에서 `TIMESFM_V1_SRC`, `TIMESFM_V1_CHECKPOINT_PATH` 설정.

## 데이터 경로

compact/subset 데이터를 사용하므로 raw context가 없어 v1 재캐싱 필요:
```bash
python src/data_prep/prepare_v1_backbone_data.py
```

## Sub-experiment 목록

| 폴더 | 변경 내용 |
|---|---|
| `soft_warm_s10_oldloss_best_v1_backbone/` | v1 backbone 교체만 (decoder는 soft_mask 원본 유지) |
| `soft_warm_s10_oldloss_best_v1_backbone_decoder_adjusted/` | v1 backbone + TimesFM-style residual decoder 동시 적용 |

`_decoder_adjusted` 버전의 decoder 구조 → `deconder_adjustment/README.md` 참조.

## 학습 진입점

```bash
# wrapper script로 EXP_DIR 지정
EXP_DIR=src/experiments/backbone_adjustment/soft_warm_s10_oldloss_best_v1_backbone \
  ./scripts/run_v1_backbone_soft_warm.sh
```
