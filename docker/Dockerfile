# syntax=docker/dockerfile:1

ARG PYTORCH_VERSION=2.5.1
ARG CUDA_VERSION=12.4
ARG CUDNN_VERSION=9
ARG DEEPLABCUT_VERSION=3.0.0rc14

# ── core ──────────────────────────────────────────────────────────────────────
FROM pytorch/pytorch:${PYTORCH_VERSION}-cuda${CUDA_VERSION}-cudnn${CUDNN_VERSION}-runtime AS core

ARG DEEPLABCUT_VERSION
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -yy && \
    apt-get install -yy --no-install-recommends \
      libgl1 \
      libglib2.0-0 \
      build-essential \
      git \
      ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip && \
    pip install "deeplabcut[modelzoo,wandb]==${DEEPLABCUT_VERSION}"

# Create pretrained weights directory and make it writable
# (same default as HuggingFaceWeightsMixin in backbones/base.py: Path(__file__).parent / "pretrained_weights")
RUN PRETRAINED=$(python -c "from pathlib import Path; import deeplabcut.pose_estimation_pytorch.models.backbones.base as b; print(Path(b.__file__).parent / 'pretrained_weights')") && \
    mkdir -p "$PRETRAINED" && \
    chmod a+rwx -R "$PRETRAINED"

RUN mkdir -p /app && chmod a+rwx /app

COPY motd.sh /home/motd.sh
RUN echo "source /home/motd.sh" >> /etc/profile

ENV CUDA_VERSION=${CUDA_VERSION}
ENV PYTORCH_VERSION=${PYTORCH_VERSION}
ENV DEEPLABCUT_VERSION=${DEEPLABCUT_VERSION}

WORKDIR /app

# ── jupyter ───────────────────────────────────────────────────────────────────
# (runs as root intentionally; deeplabcut-docker adds a host-matched user at runtime)
FROM core AS jupyter

ENV NOTEBOOK_TOKEN=deeplabcut

RUN pip install "notebook<7"
EXPOSE 8888
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8888/api/', timeout=3).read()" || exit 1
ENTRYPOINT jupyter notebook --no-browser --NotebookApp.token="${NOTEBOOK_TOKEN}" --ip 0.0.0.0

# ── test (CI only, not published to Hub) ─────────────────────────────────────
FROM core AS test

RUN pip install --no-cache-dir pytest
