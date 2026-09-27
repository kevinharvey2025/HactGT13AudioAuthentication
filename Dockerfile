# syntax=docker/dockerfile:1.4
# HEARSAY inference image (team SideQuests): audio directory in, SideQuests_predictions_final.tsv out.
#   docker build -t sidequests-hearsay .
#   docker run --rm --network none -v /path/to/test:/data/input:ro -v /path/to/out:/data/output sidequests-hearsay
# CPU by default (add --gpus all with an NVIDIA runtime; predict.py picks CUDA when available).
# Build context needs artifacts/diffusion/ (fusion.json + fine-tuned checkpoints, produced by
# scripts/make_submission.py) — or pass --build-arg WEIGHTS_REPO=<hf model repo> to download them.
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements-docker.txt .
RUN pip install --no-cache-dir torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements-docker.txt
ENV HF_HOME=/app/hf_cache PYTHONUNBUFFERED=1
COPY hearsay/ hearsay/
COPY configs/ configs/
COPY predict.py .
COPY artifacts/diffusion/ artifacts/diffusion/
ARG WEIGHTS_REPO=""
# the fine-tuned checkpoints hold every tensor, so no base model is downloaded; optionally fetch them from a
# Hugging Face model repo instead of the build context
RUN python - <<'PY'
import os
from huggingface_hub import snapshot_download
if os.environ.get("WEIGHTS_REPO"):
    snapshot_download(os.environ["WEIGHTS_REPO"], local_dir="artifacts/diffusion")
PY
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
ENTRYPOINT ["python", "predict.py", "--input", "/data/input", "--output", "/data/output"]
