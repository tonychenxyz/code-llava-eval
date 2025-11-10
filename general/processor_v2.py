import re
import math
from pathlib import Path
import sys
from typing import List, Tuple, Dict, Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from utils.inference.inference import load_model


def parse_prompt(prompt: List[Dict[str, str]], tokenizer, embed_tokenizer, chunk_size) -> Tuple[List[int], List[str], List[Tuple[int, int]]]:
    '''
    input: prompt in [{"role": "user", "content": "Please repeat this code exactly."}, ...] format
    output: input ids of string with <|memory_start|> + N <|memory|> tokens + <|memory_end|>, code list, code position list
    
    This function:
    1. Extracts code segments from the prompt (enclosed by <|memory_start|> and <|memory_end|>)
    2. Replaces each code segment with <|memory_start|> + N <|memory|> tokens + <|memory_end|>
       where N is determined by chunking the code using embed_tokenizer
    3. Tokenizes the modified prompt
    4. Returns the token IDs, list of code strings, and positions of code placeholders
    '''
    
    # Special tokens
    code_placeholder = "<|memory|>"
    memory_start = "<|memory_start|>"
    memory_end = "<|memory_end|>"
    
    # Apply chat template first to get the formatted string
    prompt_str = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    
    # Find all code segments enclosed by <|memory_start|> and <|memory_end|>
    pattern = re.escape(memory_start) + r'(.*?)' + re.escape(memory_end)
    code_segments = re.findall(pattern, prompt_str, re.DOTALL)
    
    codes = []
    modified_prompt = prompt_str
    
    # Process each code segment
    for code_content in code_segments:
        # Remove the code content and store it
        codes.append(code_content)
        
        # Calculate number of chunks needed for this code using embed_tokenizer
        token_ids = embed_tokenizer.encode(code_content, add_special_tokens=False)
        num_chunks = math.ceil(len(token_ids) / chunk_size)
        
        # Create replacement with N <|memory|> tokens
        code_tokens = code_placeholder * num_chunks
        replacement = memory_start + code_tokens + memory_end
        
        # Replace the first occurrence of the original segment with our replacement
        original_segment = memory_start + code_content + memory_end
        modified_prompt = modified_prompt.replace(original_segment, replacement, 1)
    
    print(modified_prompt)
    # Now tokenize the modified prompt
    input_ids = tokenizer.encode(modified_prompt, add_special_tokens=True)
    
    # Find positions of code placeholders in the tokenized output
    code_positions = []
    
    # Get token IDs for special tokens
    code_token_id = tokenizer.convert_tokens_to_ids(code_placeholder)
    memory_start_id = tokenizer.convert_tokens_to_ids(memory_start)
    memory_end_id = tokenizer.convert_tokens_to_ids(memory_end)
    
    # Find all code regions
    i = 0
    while i < len(input_ids):
        if input_ids[i] == memory_start_id:
            # Found start of code region
            start_pos = i + 1  # Position after <|memory_start|>
            
            # Find corresponding <|memory_end|>
            j = i + 1
            while j < len(input_ids) and input_ids[j] != memory_end_id:
                j += 1
            
            if j < len(input_ids):
                end_pos = j  # Position of <|memory_end|>
                code_positions.append((start_pos, end_pos))
                i = j + 1
            else:
                # No matching end found
                i += 1
        else:
            i += 1
    
    return input_ids, codes, code_positions


def compress(input_ids, codes, code_positions, model):
    """
    Compress the prompt by replacing code placeholder tokens with code embeddings.
    
    Args:
        input_ids: Output from tokenizer.encode() - list of token ids
        codes: List of code strings to be embedded
        code_positions: List of (start, end) tuples indicating code token positions
        model: The CodeLLaVA model containing embedding layers and code processing
        
    Returns:
        compressed_prompt_embedding: Input embeddings with code placeholders replaced by actual code embeddings
    """
    import torch
    
    # Convert input_ids list to tensor with batch dimension
    input_ids_tensor = torch.tensor(input_ids).unsqueeze(0)
    
    # Get the device from model (could be CPU or GPU)
    device = next(model.parameters()).device
    input_ids_tensor = input_ids_tensor.to(device)
    
    # Get input embeddings from the LLM's embedding layer
    inputs_embeds = model.llm_model.get_input_embeddings()(input_ids_tensor)
    
    # Process code embeddings using the model's method
    # This handles chunking, embedding, normalization and projection
    code_embeddings_list = model._process_code_embeddings(codes)
    
    # Replace placeholder tokens with actual code embeddings
    # code_positions contains (start, end) where start is after <|memory_start|> and end is before <|memory_end|>
    # The positions should contain only <|memory|> tokens which match the number of chunks
    if len(code_positions) > 0:
        for idx, (start_pos, end_pos) in enumerate(code_positions):
            if idx < len(code_embeddings_list):
                code_embedding = code_embeddings_list[idx]  # Shape: [num_chunks, embed_dim]
                num_chunks = code_embedding.shape[0]
                
                # Verify that the number of positions matches the number of chunks
                num_positions = end_pos - start_pos
                if num_positions != num_chunks:
                    print(f"Warning: Position range {num_positions} doesn't match number of chunks {num_chunks}")
                    # Adjust to use minimum to avoid dimension mismatch
                    actual_end = start_pos + min(num_positions, num_chunks)
                    if num_chunks < num_positions:
                        # If we have fewer chunks than positions, only replace what we have
                        inputs_embeds[0, start_pos:actual_end] = code_embedding
                    else:
                        # If we have more chunks than positions, truncate the embeddings
                        inputs_embeds[0, start_pos:end_pos] = code_embedding[:num_positions]
                else:
                    # Replace the placeholder tokens with the code embeddings
                    inputs_embeds[0, start_pos:end_pos] = code_embedding
    
    # Return the compressed prompt embedding (keep on same device as model)
    # Remove batch dimension for single sample
    compressed_prompt_embedding = inputs_embeds.squeeze(0)
    
    return compressed_prompt_embedding


def compress_with_llm_on_cpu(input_ids, codes, code_positions, llm_model_path, embed_model_path, chunk_size=32):
    """
    Compress the prompt by loading LLM on CPU to get embeddings, then replacing code placeholders.
    
    Args:
        input_ids: Output from tokenizer.encode() - list of token ids
        codes: List of code strings to be embedded
        code_positions: List of (start, end) tuples indicating code token positions
        llm_model_path: Path to the LLM model
        embed_model_path: Path to the embedding model
        chunk_size: Chunk size for code processing
        
    Returns:
        compressed_prompt_embedding: Input embeddings with code placeholders replaced by actual code embeddings
    """
    import torch
    from transformers import AutoModelForCausalLM
    
    # Load LLM on CPU just for getting embeddings
    print("Loading LLM on CPU for embedding layer...")
    llm_model = AutoModelForCausalLM.from_pretrained(llm_model_path, torch_dtype=torch.float32, device_map="cpu")
    embedding_layer = llm_model.get_input_embeddings()
    
    # Convert input_ids to tensor and get embeddings
    input_ids_tensor = torch.tensor(input_ids).unsqueeze(0)
    with torch.no_grad():
        inputs_embeds = embedding_layer(input_ids_tensor)
    
    # Load CodeLLaVA model on CPU for code processing
    print("Loading CodeLLaVA model on CPU for code processing...")
    code_llava_model, _, _ = load_model(embed_model_path, device="cpu", chunk_size=chunk_size)
    
    # Process code embeddings
    code_embeddings_list = code_llava_model._process_code_embeddings(codes)
    
    # Replace placeholder tokens with actual code embeddings
    if len(code_positions) > 0:
        for idx, (start_pos, end_pos) in enumerate(code_positions):
            if idx < len(code_embeddings_list):
                code_embedding = code_embeddings_list[idx]
                num_chunks = code_embedding.shape[0]
                num_positions = end_pos - start_pos
                
                if num_positions != num_chunks:
                    print(f"Warning: Position range {num_positions} doesn't match number of chunks {num_chunks}")
                    actual_end = start_pos + min(num_positions, num_chunks)
                    if num_chunks < num_positions:
                        inputs_embeds[0, start_pos:actual_end] = code_embedding
                    else:
                        inputs_embeds[0, start_pos:end_pos] = code_embedding[:num_positions]
                else:
                    inputs_embeds[0, start_pos:end_pos] = code_embedding
    
    # Clean up models
    del llm_model
    del code_llava_model
    torch.cuda.empty_cache()
    
    # Return embeddings (remove batch dimension)
    return inputs_embeds.squeeze(0)
