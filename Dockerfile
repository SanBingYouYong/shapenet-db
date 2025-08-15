FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg build-essential git rdfind \
    && rm -rf /var/lib/apt/lists/*

SHELL ["/bin/bash", "-c"]

# Set working directory
WORKDIR /app

# Install additional system dependencies for 3D processing and graphics
RUN apt-get update && apt-get install -y \
    cmake \
    wget \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files first for better caching
COPY pyproject.toml ./

# Install UV package manager
RUN pip install uv

# Install Python dependencies using uv sync
RUN uv sync

# Reinstall latest torch
RUN uv pip install --upgrade --force-reinstall torch --index-url https://download.pytorch.org/whl/cu128

# Activate the virtual environment for subsequent commands
ENV PATH="/app/.venv/bin:$PATH"

# Copy the application code (excluding data directories)
COPY *.py ./
# COPY pyproject.toml requirements.txt ./
# COPY download_model.sh ./
COPY README.md ./

# Create necessary directories for volume mounting
RUN mkdir -p ./retrieved ./models ./shapenet ./hf_shapenet_zips ./shapenet_vectordb

# Make the download script executable (in case users want to run it inside container)
# RUN chmod +x download_model.sh

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Set the default command
# CMD ["python", "main.py", "--help"]
