#!/bin/bash
CHECKPOINT_NUM=1000
EMBED_MODEL="Qwen/Qwen3-Embedding-0.6B"
LLM_MODEL="Qwen/Qwen3-4B-Instruct-2507"
CHUNK_SIZE=16
MAX_CODE_LENGTH=8192
WRAP_CODE=True
FSDP_CHECKPOINT="/workspace/Code-LLaVA/checkpoints/checkpoint_${CHECKPOINT_NUM}"
OUTPUT_DIR="/workspace/Code-LLaVA/checkpoints/checkpoint_${CHECKPOINT_NUM}_hf"

# Optional LoRA override parameters
# Set these environment variables before running the script if you want to bypass automatic inference
LLM_LORA_RANK=${LLM_LORA_RANK:-""}
LLM_LORA_ALPHA=${LLM_LORA_ALPHA:-""}
LLM_LORA_DROPOUT=${LLM_LORA_DROPOUT:-""}
LLM_LORA_TARGET_MODULES=${LLM_LORA_TARGET_MODULES:-""}
# LLM_LORA_RANK=${LLM_LORA_RANK:-"32"}
# LLM_LORA_ALPHA=${LLM_LORA_ALPHA:-"64"}
# LLM_LORA_DROPOUT=${LLM_LORA_DROPOUT:-"0.05"}
# LLM_LORA_TARGET_MODULES=${LLM_LORA_TARGET_MODULES:-""}
EMBED_LORA_RANK=${EMBED_LORA_RANK:-""}
EMBED_LORA_ALPHA=${EMBED_LORA_ALPHA:-""}
EMBED_LORA_DROPOUT=${EMBED_LORA_DROPOUT:-""}
EMBED_LORA_TARGET_MODULES=${EMBED_LORA_TARGET_MODULES:-""}

echo "FSDP checkpoint: $FSDP_CHECKPOINT"
echo "Output directory: $OUTPUT_DIR"
python -m utils.convert_fsdp_to_hf \
    --embed_model "$EMBED_MODEL" \
    --llm_model "$LLM_MODEL" \
    --chunk_size "$CHUNK_SIZE" \
    --max_code_length "$MAX_CODE_LENGTH" \
    --fsdp_checkpoint "$FSDP_CHECKPOINT" \
    --output_dir "$OUTPUT_DIR" \
    --wrap_code "$WRAP_CODE" \
    $([[ -n "$LLM_LORA_RANK" ]] && echo --llm_lora_rank "$LLM_LORA_RANK") \
    $([[ -n "$LLM_LORA_ALPHA" ]] && echo --llm_lora_alpha "$LLM_LORA_ALPHA") \
    $([[ -n "$LLM_LORA_DROPOUT" ]] && echo --llm_lora_dropout "$LLM_LORA_DROPOUT") \
    $([[ -n "$LLM_LORA_TARGET_MODULES" ]] && echo --llm_lora_target_modules "$LLM_LORA_TARGET_MODULES") \
    $([[ -n "$EMBED_LORA_RANK" ]] && echo --embed_lora_rank "$EMBED_LORA_RANK") \
    $([[ -n "$EMBED_LORA_ALPHA" ]] && echo --embed_lora_alpha "$EMBED_LORA_ALPHA") \
    $([[ -n "$EMBED_LORA_DROPOUT" ]] && echo --embed_lora_dropout "$EMBED_LORA_DROPOUT") \
    $([[ -n "$EMBED_LORA_TARGET_MODULES" ]] && echo --embed_lora_target_modules "$EMBED_LORA_TARGET_MODULES") \
    > convert_logs.log 2>&1
