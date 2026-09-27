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
# bake the pretrained AntiDeepfake encoders named in fusion.json (and, optionally, our fine-tuned weights)
RUN python - <<'PY'
import json, os
from huggingface_hub import hf_hub_download, snapshot_download
from hearsay.antideepfake import VARIANTS
spec = json.load(open("artifacts/diffusion/fusion.json"))
for s in spec["systems"]:
    hf_hub_download(VARIANTS[s["backbone"][len("adf_"):]][0], "model.safetensors")
if os.environ.get("WEIGHTS_REPO"):
    snapshot_download(os.environ["WEIGHTS_REPO"], local_dir="artifacts/diffusion")
PY
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
ENTRYPOINT ["python", "predict.py", "--input", "/data/input", "--output", "/data/output"]
