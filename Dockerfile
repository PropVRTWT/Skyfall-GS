# =============================================================================
# Skyfall-GS  —  Cloud Run Job (GPU Training)
# Base: CUDA 12.8 devel  →  compile kernels  →  slim runtime
#
# Build context: repo root (git clone --recurse-submodules required)
# Requires: NVIDIA RTX Pro 6000 / L4  (Cloud Run Jobs GPU)
# =============================================================================

# ── Stage 1: builder — compile the three CUDA submodule wheels ───────────────
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0+PTX"
ENV FORCE_CUDA=1

# Build-time system deps (gcc, g++, ninja, cmake, python headers)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-dev python3-pip \
    build-essential cmake ninja-build git \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3

# Install PyTorch and build tools (needed to compile CUDA extensions)
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir \
        torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu128

WORKDIR /build

# Copy only the submodules needed for CUDA compilation
COPY submodules/diff-gaussian-rasterization-depth /build/submodules/diff-gaussian-rasterization-depth
COPY submodules/simple-knn                         /build/submodules/simple-knn
COPY submodules/fused-ssim                         /build/submodules/fused-ssim

# Build wheels without isolated build env so PyTorch CUDA headers are visible
RUN pip wheel --no-cache-dir --no-build-isolation --wheel-dir=/wheels \
        submodules/diff-gaussian-rasterization-depth \
        submodules/simple-knn \
        submodules/fused-ssim


# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

LABEL maintainer="PropVRTWT"
LABEL description="Skyfall-GS training job — NVIDIA RTX Pro 6000 / Cloud Run Jobs"

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# Point HuggingFace cache to a writable location inside the container
ENV HF_HOME=/app/.cache/huggingface
ENV TORCH_HOME=/app/.cache/torch

# Runtime system deps (no build tools needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-dev python3-pip \
    libgl1 libglib2.0-0 \
    libopenexr-dev \
    fuse \
    curl gnupg lsb-release \
    && rm -rf /var/lib/apt/lists/*

# Install gcsfuse for GCS bucket mounting
RUN curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - && \
    echo "deb https://packages.cloud.google.com/apt gcsfuse-$(lsb_release -cs) main" \
        > /etc/apt/sources.list.d/gcsfuse.list && \
    apt-get update && apt-get install -y --no-install-recommends gcsfuse && \
    rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3

WORKDIR /app

# Install PyTorch (runtime)
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir \
        torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu128

# Install repo Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy pre-built CUDA wheels from builder stage and install
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/*.whl && rm -rf /wheels

# Copy the full repo (submodules are needed for MoGe / FlowEdit imports)
COPY . .

# Make sure MoGe & FlowEdit submodule Python packages are importable
ENV PYTHONPATH="/app:/app/submodules/MoGe:/app/submodules/FlowEdit:${PYTHONPATH}"

# Create directories for GCS mounts and outputs
RUN mkdir -p /mnt/gcs /mnt/outputs /app/.cache/huggingface /app/.cache/torch

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Cloud Run Jobs: no port needed  (no --port flag in train.py call)
ENTRYPOINT ["/entrypoint.sh"]
