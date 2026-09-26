# HEARSAY DSP track: offline CPU inference image.
#   docker build -f Dockerfile.dsp -t hearsay-dsp .
#   docker run --rm --network none -v /path/to/test_audio:/data/input:ro -v $PWD/out:/data/output hearsay-dsp
# The trained bundle (artifacts/dsp/model) is copied in at build time; prediction never retrains.
FROM python:3.13-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    MPLCONFIGDIR=/tmp

WORKDIR /app
COPY requirements-dsp.txt .
RUN pip install --no-cache-dir -r requirements-dsp.txt

COPY hearsay_dsp/ hearsay_dsp/
COPY configs/dsp.yaml configs/dsp.yaml
COPY artifacts/dsp/model/ artifacts/dsp/model/

# The feature cache lives in /tmp inside the container (content-hash keyed; safe to discard).
ENTRYPOINT ["python", "-m", "hearsay_dsp.cli"]
CMD ["predict", "--input", "/data/input", "--model", "/app/artifacts/dsp/model", \
     "--output", "/data/output/predictions.tsv", "--cache-dir", "/tmp/hearsay_dsp_cache"]
