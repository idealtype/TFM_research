FROM quay.io/vessl-ai/torch:2.9.1-cuda13.0.1-py3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib \
    DATA_ROOT=/workspace/data \
    PROJECT_ROOT=/workspace/project \
    HF_HOME=/workspace/data/.cache/huggingface \
    HF_HUB_CACHE=/workspace/data/.cache/huggingface/hub

WORKDIR /workspace

COPY requirements-vessl.txt /tmp/requirements-vessl.txt
RUN pip install --no-cache-dir --progress-bar off -r /tmp/requirements-vessl.txt \
    && python - <<'PY'
import matplotlib
import pywt
import timesfm
import torch
print("image dependency check ok", torch.__version__)
PY

CMD ["python", "-V"]
