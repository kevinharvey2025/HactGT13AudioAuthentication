# syntax=docker/dockerfile:1.4
# HEARSAY inference image (team SideQuests): an audio directory in, SideQuests_predictions_final.tsv + traces out.
#
#   docker build -t sidequests-hearsay .                     # bakes artifacts/diffusion/ into the image if present
#   docker run --rm --network none --memory 16g \
#       -v /path/to/audio:/data/input:ro -v $PWD/out:/data/output sidequests-hearsay
#
# Weights (fusion.json + three fine-tuned checkpoints, ~7.7 GB; see docs/REPRODUCE.md) come from one of:
#   1. the build context: artifacts/diffusion/ is copied into the image;
#   2. a Hugging Face repo at build time: --build-arg WEIGHTS_REPO=<repo>;
#   3. a mount at run time: -v /path/to/weights:/app/artifacts/diffusion:ro  (the image then needs no rebuild).
# Memory: fp32 (the reference) peaks at ~14.5 GB, so give Docker >= 16 GB; append `--precision bf16` to the run
# command for ~5 GB (Docker Desktop's default VM has 8 GB). CPU by default; with an NVIDIA runtime add --gpus all.
# The base image is pinned by digest so every build starts from the same bytes.
FROM python:3.11-slim@sha256:e41613d42d4891e4930f79523f93f81bbc7632584ec65e36ab055f41a800b41e
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements-docker.txt .
RUN pip install --no-cache-dir torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements-docker.txt
ENV HF_HOME=/app/hf_cache PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
COPY hearsay/ hearsay/
COPY configs/ configs/
COPY predict.py .
# weights from the build context when artifacts/ exists (the wildcard makes the copy a no-op when it does not)
COPY artifact[s] artifacts/
RUN mkdir -p artifacts/diffusion
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
