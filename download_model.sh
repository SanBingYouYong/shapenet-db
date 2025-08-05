#!/bin/bash

# Create target directories if they don't exist
mkdir -p ./models

echo "Starting model downloads..."

# Download CLIP-ViT-bigG model (10.2G) to ulip_models
echo "Downloading CLIP-ViT-bigG-14 model..."
wget -P ./models https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k/resolve/main/open_clip_pytorch_model.bin

echo "CLIP-ViT-bigG-14 model downloaded successfully."