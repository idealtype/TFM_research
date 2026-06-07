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

## Pending Work

- Data generation and TimesFM cache generation are intentionally paused.
- First TimesFM cache job should download model weights into `/workspace/data/.cache/huggingface`.
