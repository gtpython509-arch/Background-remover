# syntax=docker/dockerfile:1.7
# ---------------------------------------------------------------------------
# Multi-stage build. Stage 1 compiles wheels, stage 2 ships only the runtime,
# which keeps the final image around 1.3 GB instead of ~2.2 GB.
# Build from the project root:
#   docker build -f deploy/Dockerfile -t rmbg-service:1.0.0 .
# ---------------------------------------------------------------------------
FROM python:3.11-slim-bookworm AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY requirements.txt requirements-storage.txt ./

# CPU-only torch wheels: ~200 MB instead of the ~2.5 GB CUDA build.
RUN python -m venv /opt/venv \
 && /opt/venv/bin/pip install -U pip wheel \
 && /opt/venv/bin/pip install -r requirements-storage.txt \
      --extra-index-url https://download.pytorch.org/whl/cpu


FROM python:3.11-slim-bookworm AS runtime

LABEL org.opencontainers.image.title="rmbg-service" \
      org.opencontainers.image.description="Background removal API (RMBG-1.4)" \
      org.opencontainers.image.version="1.0.0"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    HF_HUB_DISABLE_TELEMETRY=1 \
    HF_HOME=/var/lib/rmbg/hf \
    RMBG_MODEL_CACHE_DIR=/var/lib/rmbg/models \
    OMP_NUM_THREADS=2

# libgomp1 is required by torch; the rest of libstdc++ comes from the base.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 curl \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --create-home --shell /usr/sbin/nologin --uid 10001 rmbg

COPY --from=builder /opt/venv /opt/venv

WORKDIR /srv/app
COPY --chown=rmbg:rmbg app ./app
COPY --chown=rmbg:rmbg scripts ./scripts
COPY --chown=rmbg:rmbg deploy/gunicorn.conf.py ./deploy/gunicorn.conf.py

RUN mkdir -p /var/lib/rmbg/models /var/lib/rmbg/hf \
 && chown -R rmbg:rmbg /var/lib/rmbg

# Bake the weights into the image so container start needs no network and is
# reproducible. Comment this out and mount a volume instead if you would
# rather keep the image small.
RUN python scripts/download_model.py && chown -R rmbg:rmbg /var/lib/rmbg

USER rmbg
EXPOSE 8000

# Readiness, not liveness: reports unhealthy until the model is loaded.
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8000/readyz || exit 1

ENV RMBG_HOST=0.0.0.0 \
    RMBG_PORT=8000 \
    RMBG_LOCAL_FILES_ONLY=true

CMD ["gunicorn", "app.main:app", "-c", "deploy/gunicorn.conf.py"]
