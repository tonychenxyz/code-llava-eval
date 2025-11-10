#!/bin/bash

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Configuration
CHECKPOINT_NUM=7000
MODEL_REPO="leonli66/checkpoint_13000_hf"
CLEAN_NAME="${MODEL_REPO//\//__}"
CHECKPOINT_DIR="$PROJECT_ROOT/checkpoints/${CLEAN_NAME}/pytorch_model"


echo "Downloading checkpoint from HuggingFace..."
echo "Model: $MODEL_REPO"
echo "Destination: $CHECKPOINT_DIR"

# Download the checkpoint to the checkpoints folder
uv run huggingface-cli download "$MODEL_REPO" \
    --local-dir "$CHECKPOINT_DIR" \
    --local-dir-use-symlinks False