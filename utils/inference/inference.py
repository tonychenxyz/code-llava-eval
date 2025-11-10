#!/usr/bin/env python3
"""
Simple evaluation script - load checkpoint and generate text
"""

import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, AutoModel
import argparse
import json
import random
import os
from safetensors.torch import load_file as load_safetensors
from peft import PeftModel

# Import local modules
from utils.model.code_llava_model import CodeLLaVAModel
from utils.model.processor import CodeLLaVAProcessor


def _build_llm_kwargs(attn_implementation: str | None):
    kwargs = {
        "torch_dtype": torch.bfloat16,
    }
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    return kwargs


def load_model(
    checkpoint_path,
    device="cuda",
    chunk_size=128,
    max_code_length=12288,
    attn_implementation="eager",
):
    """Load model from new checkpoint structure"""
    print(f"Loading checkpoint: {checkpoint_path}")
    
    # Handle both directory and file paths
    if os.path.isdir(checkpoint_path):
        # New checkpoint structure - directory with subdirs
        llm_dir = os.path.join(checkpoint_path, "llm")
        embed_dir = os.path.join(checkpoint_path, "embedder") 
        projectors_dir = os.path.join(checkpoint_path, "projectors")
        
        if not all(os.path.exists(d) for d in [llm_dir, embed_dir, projectors_dir]):
            # Try to find the latest checkpoint folder
            candidates = [
                os.path.join(checkpoint_path, d) for d in os.listdir(checkpoint_path)
                if d.startswith("checkpoint_") and os.path.isdir(os.path.join(checkpoint_path, d))
            ]
            if candidates:
                latest_checkpoint = sorted(candidates, key=lambda p: int(os.path.basename(p).split('_')[1]))[-1]
                llm_dir = os.path.join(latest_checkpoint, "llm")
                embed_dir = os.path.join(latest_checkpoint, "embedder")
                projectors_dir = os.path.join(latest_checkpoint, "projectors")
            else:
                raise FileNotFoundError(f"No valid checkpoint structure found in {checkpoint_path}")
    else:
        # Legacy single file checkpoint
        raise ValueError("Single file checkpoints are not supported. Please use new checkpoint directory structure.")
    
    print(f"Loading LLM from: {llm_dir}")
    print(f"Loading embedder from: {embed_dir}")
    print(f"Loading projectors from: {projectors_dir}")
    
    # Load tokenizers
    llm_tokenizer = AutoTokenizer.from_pretrained(llm_dir)
    if llm_tokenizer.pad_token is None:
        llm_tokenizer.pad_token = llm_tokenizer.eos_token
    
    embed_tokenizer = AutoTokenizer.from_pretrained(embed_dir)
    
    # Load LLM model
    llm_has_adapters = os.path.isfile(os.path.join(llm_dir, "adapter_model.safetensors"))
    llm_kwargs = _build_llm_kwargs(attn_implementation)

    if llm_has_adapters:
        # Load base model and then PEFT adapters
        config_path = os.path.join(llm_dir, "adapter_config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                adapter_config = json.load(f)
            base_model_name = adapter_config.get('base_model_name_or_path', 'Qwen/Qwen3-0.6B')
        else:
            base_model_name = 'Qwen/Qwen3-0.6B'  # fallback
            
        llm_model = AutoModelForCausalLM.from_pretrained(base_model_name, **llm_kwargs)
        llm_model = PeftModel.from_pretrained(llm_model, llm_dir)
    else:
        # Load full model
        llm_model = AutoModelForCausalLM.from_pretrained(llm_dir, **llm_kwargs)
    
    # Load embedding model
    embed_has_adapters = os.path.isfile(os.path.join(embed_dir, "adapter_model.safetensors"))
    if embed_has_adapters:
        # Load base model and then PEFT adapters
        config_path = os.path.join(embed_dir, "adapter_config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                adapter_config = json.load(f)
            base_model_name = adapter_config.get('base_model_name_or_path', 'Qwen/Qwen3-Embedding-0.6B')
        else:
            base_model_name = 'Qwen/Qwen3-Embedding-0.6B'  # fallback
            
        embed_model = AutoModel.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
        )
        embed_model = PeftModel.from_pretrained(embed_model, embed_dir)
    else:
        # Load full model
        embed_model = AutoModel.from_pretrained(
            embed_dir,
            torch_dtype=torch.bfloat16,
        )
    
    # Create processor
    processor = CodeLLaVAProcessor(
        llm_tokenizer=llm_tokenizer,
        embed_tokenizer=embed_tokenizer,
        chunk_size=chunk_size,  # default, could be read from config
        max_code_length=max_code_length,  # default, could be read from config
        use_code_wrapping=True,
    )
    
    # Create CodeLLaVA model
    model = CodeLLaVAModel(
        llm_model=llm_model,
        llm_tokenizer=llm_tokenizer,
        embed_model=embed_model,
        embed_tokenizer=embed_tokenizer,
        processor=processor,
        chunk_size=chunk_size,  # default
        max_code_length=max_code_length,  # default
        train_llm=False,  # for inference
        train_embedder=False,  # for inference
    )
    
    # Load projector weights
    proj_path = os.path.join(projectors_dir, "code_projection.safetensors")
    norm_pre_path = os.path.join(projectors_dir, "code_norm_pre.safetensors")
    
    if os.path.isfile(proj_path):
        proj_state = load_safetensors(proj_path)
        model.code_projection.load_state_dict(proj_state, strict=True)
        print("Loaded code projection weights")
    else:
        print(f"Warning: projector weights not found at {proj_path}")
        
    if os.path.isfile(norm_pre_path):
        norm_pre_state = load_safetensors(norm_pre_path)
        model.code_norm_pre.load_state_dict(norm_pre_state, strict=True)
        print("Loaded code norm weights")
    else:
        print(f"Warning: projector norm weights not found at {norm_pre_path}")
    
    model.to(device)
    model.to(torch.bfloat16)
    model.eval()
    print("Model loaded successfully!")
    print(model)
    
    return model, llm_tokenizer, processor


def generate_text(model, llm_tokenizer, processor, code, prompt, device="cuda", max_tokens=512, temperature=0.7):
    """Generate text given code and prompt using the new processor-based approach"""
    print(f"\n{'='*60}")
    
    # Build wrapped user prompt: insert code between <|memory_start|> ... <|memory_end|>
    if "<|memory|>" in prompt:
        user_text = prompt.replace("<|memory|>", f"<|memory_start|>{code}<|memory_end|>")
    else:
        user_text = f"{prompt} <|memory_start|>{code}<|memory_end|>"
    formatted_prompt = f"<|im_start|>user\n{user_text}<|im_end|>\n<|im_start|>assistant\n"
    print(formatted_prompt)
    print(f"\n{'='*60}")
    
    with torch.inference_mode():
        # Use processor to handle code integration
        processed = processor.process_wrapped_batch(
            prompts=[formatted_prompt],
            targets=None,
            padding="longest",
            truncation=True,
            return_tensors="pt"
        )
        
        # Move to device
        input_ids = processed['input_ids'].to(device)
        attention_mask = processed['attention_mask'].to(device)
        code_positions = processed['code_positions']
        chunk_counts = processed['chunk_counts']
        codes = processed['codes']
        
        # Generate with code context
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            codes=codes,
            code_positions=code_positions,
            chunk_counts=chunk_counts,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=0.8,
            do_sample=temperature > 0,
            top_k=20,
            min_p=0,
            repetition_penalty=1.1,
            pad_token_id=llm_tokenizer.pad_token_id,
            eos_token_id=llm_tokenizer.eos_token_id
        )
        
        # Decode
        generated_text = llm_tokenizer.decode(outputs[0], skip_special_tokens=False)
        
        # Remove prompt from output
        if generated_text.startswith(formatted_prompt):
            generated_text = generated_text[len(formatted_prompt):].strip()
    
    print("OUTPUT:")
    print(generated_text)
    print(f"{'='*60}\n")
    
    return generated_text


def load_test_examples(test_data_path=None, num_examples=None, seed=42):
    """Load and sample examples from the test data JSON file"""
    if test_data_path is None:
        # Default to test_data.json in the same directory as this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        test_data_path = os.path.join(script_dir, "test_data.json")
    
    print(f"Loading test data from: {test_data_path}")
    
    # Load JSON
    try:
        with open(test_data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Test data file not found: {test_data_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in test data file: {e}")
    
    # Extract examples
    if 'examples' not in data:
        raise ValueError("Test data JSON must contain an 'examples' key")
    
    examples = data['examples']
    print(f"Loaded {len(examples)} examples from test data")
    
    # Validate example structure
    for i, example in enumerate(examples):
        if 'code' not in example or 'prompt' not in example:
            raise ValueError(f"Example {i} must contain both 'code' and 'prompt' keys")
    
    # Sample examples or use all if num_examples is None
    if num_examples is None:
        selected_examples = examples
    elif num_examples >= len(examples):
        selected_examples = examples
    else:
        # Set seed for reproducible sampling
        random.seed(seed)
        selected_examples = random.sample(examples, num_examples)
    
    # Process examples
    processed_examples = []
    for i, example in enumerate(selected_examples):
        code = example['code'].strip()
        prompt = example['prompt'].strip()
        
        # Truncate very long code for performance
        if len(code) > 30000:
            code = code[:30000]
            print(f"Warning: Code in example {i+1} was truncated to 30000 characters")
        
        processed_examples.append({
            "code": code,
            "prompt": prompt
        })
        
        print(f"Example {i+1} loaded - Code length: {len(code)} characters, Prompt: {prompt[:50]}...")
    
    return processed_examples


def main():
    parser = argparse.ArgumentParser(description="Simple eval - generate text from checkpoint")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint directory")
    parser.add_argument("--test_data", default=None, help="Path to test data JSON file (default: ./test_data.json)")
    parser.add_argument("--num_examples", type=int, default=None, help="Number of examples to sample from test data")
    parser.add_argument("--device", default="cuda", help="Device to use")
    parser.add_argument("--max_tokens", type=int, default=512, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for generation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling examples")
    parser.add_argument("--chunk_size", type=int, default=128, help="Chunk size for code processing")
    parser.add_argument("--max_code_length", type=int, default=12288, help="Maximum code length to process")
    parser.add_argument(
        "--attn_implementation",
        default="eager",
        help="Attention backend for HF models (e.g., 'eager', 'flash_attention_2', 'sdpa', 'auto').",
    )
    
    args = parser.parse_args()

    attn_impl = args.attn_implementation
    if attn_impl is not None and attn_impl.lower() in {"auto", "default", "none"}:
        attn_impl = None
    
    # Load model
    model, llm_tokenizer, processor = load_model(
        args.checkpoint,
        args.device,
        args.chunk_size,
        args.max_code_length,
        attn_impl,
    )
    
    # Load examples from test data
    examples = load_test_examples(args.test_data, args.num_examples, args.seed)
    
    # Generate for each example
    for i, example in enumerate(examples):
        print(f"\n\nEXAMPLE {i+1}:")
        generate_text(
            model=model,
            llm_tokenizer=llm_tokenizer,
            processor=processor,
            code=example["code"],
            prompt=example["prompt"],
            device=args.device,
            max_tokens=args.max_tokens,
            temperature=args.temperature
        )

if __name__ == "__main__":
    main()
