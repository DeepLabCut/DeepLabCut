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

RUN chmod a+rwx -R \
    /opt/conda/lib/python3.11/site-packages/deeplabcut/pose_estimation_pytorch/models/backbones/pretrained_weights

RUN mkdir -p /app && chmod a+rwx /app

COPY motd.sh /home/motd.sh
RUN echo "source /home/motd.sh" >> /etc/profile

ENV CUDA_VERSION=${CUDA_VERSION}
ENV PYTORCH_VERSION=${PYTORCH_VERSION}
ENV DEEPLABCUT_VERSION=${DEEPLABCUT_VERSION}

WORKDIR /app

# ── jupyter ───────────────────────────────────────────────────────────────────
FROM core AS jupyter

RUN pip install notebook
EXPOSE 8888
ENTRYPOINT ["jupyter", "notebook", \
    "--no-browser", "--NotebookApp.token=deeplabcut", "--ip", "0.0.0.0"]

# ── test (CI only, not published to Hub) ─────────────────────────────────────
FROM core AS test

RUN pip install --no-cache-dir pytest
