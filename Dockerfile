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
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

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

# Patch fused-ssim setup.py:
# 1. On PyTorch 2.4+, hasattr(torch, 'xpu') is True. Fix it to check torch.xpu.is_available() so it falls back to CUDA during docker build.
# 2. Use CUDAExtension instead of CppExtension for CUDA build.
RUN sed -i "s/elif hasattr(torch, 'xpu'):/elif hasattr(torch, 'xpu') and torch.xpu.is_available():/g" /build/submodules/fused-ssim/setup.py && \
    sed -i 's/return CppExtension, "ssim.cu"/return CUDAExtension, "ssim.cu"/g' /build/submodules/fused-ssim/setup.py

# Build wheels without isolated build env, without downloading runtime dependencies
RUN pip wheel --no-cache-dir --no-build-isolation --no-deps --wheel-dir=/wheels \
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
ENV HF_ENABLE_PARALLEL_LOADING=yes
ENV HF_HUB_ENABLE_HF_TRANSFER=0

# Runtime system deps (no build tools needed, git needed for git+ pip packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-dev python3-pip \
    git \
    libgl1 libglib2.0-0 \
    libopenexr-dev \
    fuse \
    curl gnupg lsb-release \
    && rm -rf /var/lib/apt/lists/*

# Install gcsfuse and google-cloud-cli for high-speed parallel GCS transfers
RUN curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - && \
    echo "deb https://packages.cloud.google.com/apt gcsfuse-$(lsb_release -cs) main" \
        > /etc/apt/sources.list.d/gcsfuse.list && \
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" \
        > /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg \
        | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg && \
    apt-get update && apt-get install -y --no-install-recommends gcsfuse google-cloud-cli && \
    rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3

WORKDIR /app

# Install PyTorch and build tools (runtime)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir "setuptools<70" wheel && \
    pip install --no-cache-dir \
        torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu128

# Install repo Python requirements (with --no-build-isolation for CLIP's pkg_resources dependency)
COPY requirements.txt .
RUN pip install --no-cache-dir --no-build-isolation -r requirements.txt

# Copy pre-built CUDA wheels from builder stage and install
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/*.whl && rm -rf /wheels

# Copy the full repo (submodules are needed for MoGe / FlowEdit imports)
COPY . .

# Apply patches to MoGe (offline GCS fallback) and FlowEdit (pipeline caching & local FLUX)
RUN python3 patch_submodules.py

# Make sure MoGe & FlowEdit submodule Python packages are importable
ENV PYTHONPATH="/app:/app/submodules/MoGe:/app/submodules/FlowEdit:${PYTHONPATH}"
ENV MOGE_MODEL_PATH="/app/models/moge-vitl/model.pt"

# Create directories for GCS mounts and outputs
RUN mkdir -p /mnt/gcs /mnt/outputs /app/.cache/huggingface /app/.cache/torch /app/flux_pipeline /app/models

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Cloud Run Jobs: no port needed  (no --port flag in train.py call)
ENTRYPOINT ["/entrypoint.sh"]
