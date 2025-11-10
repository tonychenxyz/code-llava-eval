#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
CHECKPOINT="./checkpoints/checkpoint_7000_hf"
NUM_EXAMPLES=5
MAX_TOKENS=8192
TEMPERATURE=0.7
CHUNK_SIZE=16
MAX_CODE_LENGTH=8192

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python -m utils.inference.inference \
  --checkpoint $CHECKPOINT \
  --num_examples $NUM_EXAMPLES \
  --max_tokens $MAX_TOKENS \
  --temperature $TEMPERATURE \
  --chunk_size $CHUNK_SIZE \
  --max_code_length $MAX_CODE_LENGTH \
  > "${SCRIPT_DIR}/inference.log" 2>&1
