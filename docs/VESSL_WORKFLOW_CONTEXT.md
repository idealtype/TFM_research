# VESSL Workflow Context

## Current Baseline

- Repository: `https://github.com/idealtype/TFM_research.git`
- Repository visibility: public, so VESSL can clone without GitHub credentials.
- Verified code/image commit: `1a9556b27f678fdc3859e1e75b9c373160b49caa`
- Verified image: `ghcr.io/idealtype/tfm-research:1a9556b27f678fdc3859e1e75b9c373160b49caa`
- Later doc-only commits may exist; use the verified code/image commit above for reproduced smoke results.
- Verified data/object volume: `objvol-edwuqaa94ii3`

## Resource Choices

- CPU-only work: `resourcespec-grlxx3knwzps`
  - Region: `kr-west`
  - Verified with Git clone and smoke test.
  - Use for lightweight checks, data download, indexing, and CPU-side preparation.
- GPU work: `resourcespec-a100x1`
  - GPU: A100 SXM 80GB x1
  - Use for TimesFM embedding cache generation and actual model training.
- Avoid for now:
  - `resourcespec-a100cpu`: observed jobs with no logs/stuck behavior.
  - `resourcespec-kmswqig4b12w`: observed no-log running behavior in probe.

## Image Workflow

- Docker image is built by GitHub Actions only when one of these files changes:
  - `Dockerfile`
  - `requirements-vessl.txt`
  - `.dockerignore`
  - `.github/workflows/build-vessl-image.yml`
- Normal source-code commits do not rebuild the image.
- The image includes the stable dependency set, including `timesfm[torch]`.
- VESSL jobs should use the exact image tag and exact Git commit hash.

## Verified Smoke Results

- Final image smoke job: `job-uh9hgncef8cx`
- Result: succeeded.
- Verified imports:
  - `torch`
  - `matplotlib`
  - `pywt`
  - `pandas`
  - `scipy`
  - `datasets`
  - `pyarrow`
  - `timesfm`
- Verified code smoke:
  - DataLoader smoke passed.
  - `hard_mask` forward/backward passed.
  - `soft_mask` forward/backward passed.
  - `nogate_softmask` forward/backward passed.

## Standard Job Pattern

Use public Git clone plus exact commit checkout:

```bash
git clone https://github.com/idealtype/TFM_research.git /tmp/tfm_project
cd /tmp/tfm_project
git checkout 1a9556b27f678fdc3859e1e75b9c373160b49caa
```

Use `/workspace/data` for VESSL-mounted data/cache paths:

```bash
--object-volume objvol-edwuqaa94ii3:/workspace/data
```

Set runtime path variables when needed:

```bash
export DATA_ROOT=/workspace/data
export PROJECT_ROOT=/tmp/tfm_project
export HF_HOME=/workspace/data/.cache/huggingface
export HF_HUB_CACHE=/workspace/data/.cache/huggingface/hub
```

## Result Path Convention

All experiment outputs are saved to the object volume under `results/`.

### Volume structure

```
/workspace/data/results/
  <exp_type>/<HHmm>_<run_tag>/
    train/          ← checkpoints (funcdec_h96.pt … h720.pt), train_result.json
    eval_real/      ← real_eval_mae.csv, performance_by_horizon.png,
                       <dataset>/plots/h{H}_samples{N}.png
    eval_synth/     ← (future use)
```

- `exp_type`: `hard_mask` | `soft_mask` | `nogate_softmask`
- `HHmm`: job submission hour+minute (local time, no date/seconds)
- `run_tag`: short config description, e.g. `b1024_s13_scratch_v1`

### Submitting jobs

```bash
./scripts/vessl_submit.sh \
  --exp hard_mask \
  --run b1024_s13_scratch_v1 \
  --mode train_eval       # train | eval | train_eval
```

The script generates `HHmm` at submission time, previews the job, and asks for confirmation.

### Downloading results locally

```bash
# Eval results only (plots + CSV, lightweight)
vesslctl volume download objvol-edwuqaa94ii3 \
  --remote-prefix "results/hard_mask/1423_b1024_s13_scratch_v1/eval_real" \
  ./results/hard_mask/1423_b1024_s13_scratch_v1/eval_real

# Full run including checkpoints
vesslctl volume download objvol-edwuqaa94ii3 \
  --remote-prefix "results/hard_mask/1423_b1024_s13_scratch_v1" \
  ./results/hard_mask/1423_b1024_s13_scratch_v1
```

Downloaded files land in the local `results/` folder (gitignored, only `.gitkeep` is tracked).

## GPU Utilization — Known Bottleneck and Fix

현재 학습 루프는 **매 스텝마다 CPU→GPU 전송**이 발생해 GPU가 유휴 상태가 됩니다.

병목 위치 (`train.py` / `sample_real_batch` + `expand_bases`):
- `numpy.random.choice` → CPU 인덱싱 → CPU 텐서 → GPU 전송 (embeddings, future_n)
- `expand_bases()`: Fourier basis를 매 스텝 CPU→GPU 전송 (group당 고정값인데도)

**Fix**: `RealPayloadCache.load()` 시점에 모든 텐서를 GPU로 이동.

```python
# RealPayloadCache.load() 수정
embeddings = backbone["embeddings"].float().to(device)   # 한 번만
future_n   = futures["futures_n"].float().to(device)     # 한 번만
bases      = {k: v.to(device) for k, v in resolve_basis_for_real(group).items()}
```

메모리: 20 groups × ~50K samples × 512dim × float32 ≈ 2GB/horizon × 4 = 8GB → A100 80GB 여유.  
적용 대상: `soft_mask`, `nogate_softmask`, `hard_mask`, `deconder_adjustment` 모두.  
**다음 실험부터 반드시 적용.**

## Log 출력 순서에 대한 주의

`run_horizon() | sed "s/^/[h${H}] /"` 구조에서 sed 파이프가 출력을 블록 버퍼링합니다.  
실제로는 4개 horizon이 병렬 실행 중이지만, 로그상에서는 각 horizon의 출력이 한꺼번에 덤프되어  
마치 순차 실행처럼 보입니다. residual-only 단계처럼 로그 빈도가 낮은 구간만 실시간으로 확인 가능합니다.

## Pending Work

- Data generation and TimesFM cache generation are intentionally paused.
- First TimesFM cache job should download model weights into `/workspace/data/.cache/huggingface`.
