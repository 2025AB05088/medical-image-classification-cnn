# ==============================================================================
# Enterprise Computer Vision Pipeline Container Specification
# Target Environments: Azure Machine Learning Compute, AKS (Azure Kubernetes Service)
# Optimized for: NVIDIA GPU acceleration (CUDA 11.8) for Spatial Convolution workloads
# Author: Lead Architect / MLOps Engineer
# ==============================================================================

# Stage 1: Base Runtime Layer
FROM mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.8-cudnn8-ubuntu22.04 AS runtime

# Inject architectural identity and build provenance
LABEL maintainer="MLOps Platform Engineering <2025ab05088@wilp.bits-pilani.ac.in>" \
      com.azure.ml.ops.tier="production-grade" \
      com.azure.ml.ops.pipeline="radiograph-classification-cnn"

# Define immutable environment invariants for deterministic execution
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PATH="/opt/miniconda/bin:$PATH"

# ==============================================================================
# OS Foundation & Security Hardening
# ==============================================================================
USER root

# Mitigate vulnerabilities, purge bloat, and establish foundational build tools
RUN apt-get update -y && apt-get upgrade -y \
    && apt-get install --no-install-recommends -y \
        build-essential \
        git \
        libcap-dev \
        libgl1-mesa-glx \
        libglib2.0-0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Note: Added libgl1-mesa-glx and libglib2.0-0 generally required by OpenCV/Vision transforms

# ==============================================================================
# Model Environment Initialization
# ==============================================================================
# Establish distinct staging volume for computational artifacts
WORKDIR /workspace/vision_experiment

# Initialize dependency cache layer (optimized for Docker layer caching)
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Install core computer vision topologies and headless execution dependencies
# (In an actual production pipeline, these would be securely pinned via a requirements.txt lockfile)
RUN python -m pip install --no-cache-dir \
    numpy \
    pandas \
    matplotlib \
    seaborn \
    scikit-learn \
    torch \
    torchvision \
    jupyter \
    nbconvert \
    papermill

# ==============================================================================
# Artifact Staging
# ==============================================================================
# Transfer the core simulation scripts and headless automation files
COPY 2025AB05088_cnn_assignment.ipynb ./
COPY cnn_assignment_code.py ./
COPY *.txt ./
# Note: DO NOT COPY the chest_xray folder directly into the image. 
# Datasets strictly injected via runtime volume mounts in enterprise architectures.

# ==============================================================================
# Privilege Dropping & Container Telemetry
# ==============================================================================
# Instantiate standard non-privileged user profile for hardened deployment architectures
RUN useradd -m -s /bin/bash mlops_sys \
    && chown -R mlops_sys:mlops_sys /workspace/vision_experiment

USER mlops_sys

# Surface telemetry standard output port (optional, utilized by local Jupyter spin-ups)
EXPOSE 8888

# ==============================================================================
# Pipeline Entrypoint Activation
# ==============================================================================
# Default Execution: Headless script execution for robust parallelization
# Note: In an integrated Azure ML setup, this is typically overridden by the `command:` specified in the sweeping job YAML.
ENTRYPOINT ["python", "cnn_assignment_code.py"]
