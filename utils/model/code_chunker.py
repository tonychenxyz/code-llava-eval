"""
Clean Code Chunker module that processes token-based chunks separately in batches.
"""
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple, List
import math

class CodeChunker(nn.Module):
    """
    Clean code chunker that processes token-based chunks separately in batches.
    
    This module:
    1. Takes code as input
    2. Tokenizes code into tokens
    3. Splits tokens into fixed-size token chunks
    4. Processes all chunks in a single batch through the embedder
    5. Appends EOS to each chunk and extracts the last token (EOS) embedding from each chunk
    6. Returns LEFT-PADDED embeddings (actual embeddings on RHS) ready for projection to LLM space
    """

    def __init__(
        self, 
        chunk_size,
        max_length,
        embed_model,
        embed_tokenizer,
        train_embedder,
        max_encode_batch_size: int = 0,
        accelerator=None,
    ):
        super().__init__()
        
        self.chunk_size = chunk_size
        self.max_length = max_length
        
        # Initialize embedding model and tokenizer
        self.embed_tokenizer = embed_tokenizer
        self.embed_model = embed_model
        self.train_embedder = train_embedder
        self.pooling_token_id = self.embed_tokenizer.pad_token_id
        self.pad_token_id = self.embed_tokenizer.pad_token_id
        # 0 or None means process all chunks at once; otherwise mini-batch the encoder forward
        self.max_encode_batch_size = max(0, int(max_encode_batch_size))
        self.accelerator = accelerator
    
    @property 
    def embedding_dim(self) -> int:
        """Get the dimension of the code embeddings."""
        return self.embed_model.config.hidden_size
    
    def setup_accelerator(self, accelerator):
        """Set the accelerator after initialization."""
        self.accelerator = accelerator
    
    def forward(self, code: List[str]) -> List[torch.Tensor]:
        """
        Process a batch of code and return chunk embeddings without padding.
        
        Args:
            code: List of code strings
            
        Returns:
            List of chunk embeddings, one tensor per code sample: [num_chunks, embed_dim]
        """
        device = next(self.embed_model.parameters()).device
        
        # Tokenize each code string, split into token chunks, and append EOS per chunk
        chunk_counts = []
        chunk_tok_size = self.chunk_size + 1
        token_ids_list = []

        for sample_idx, text in enumerate(code):
            token_ids = self.embed_tokenizer.encode(
                text,
                add_special_tokens=False
            )
            num_chunks = math.ceil(len(token_ids) / self.chunk_size)
            chunk_counts.append(num_chunks)
            token_ids_list.append(token_ids)
        total_chunks = sum(chunk_counts)

        # LEFT padding
        chunk_input_ids = torch.full((total_chunks, chunk_tok_size), self.pad_token_id, dtype=torch.long, device=device)
        chunk_attention_mask = torch.zeros((total_chunks, chunk_tok_size), dtype=torch.long, device=device)

        chunk_idx = 0
        for sample_idx, token_ids in enumerate(token_ids_list):
            for start in range(0, len(token_ids), self.chunk_size):
                chunk_ids = token_ids[start:start + self.chunk_size] + [self.pooling_token_id]
                seq_len = len(chunk_ids)
                chunk_input_ids[chunk_idx, chunk_tok_size - seq_len:] = torch.tensor(chunk_ids, dtype=torch.long, device=device)
                chunk_attention_mask[chunk_idx, chunk_tok_size - seq_len:] = 1
                chunk_idx += 1


        # CRITICAL: Determine if batching is needed across ALL ranks
        # Even if this rank doesn't need batching, we must participate in synchronization
        needs_batching = self.max_encode_batch_size and self.max_encode_batch_size > 0 and total_chunks > self.max_encode_batch_size
        
        if needs_batching:
            num_batches = (total_chunks + self.max_encode_batch_size - 1) // self.max_encode_batch_size
        else:
            num_batches = 1  # Will process all in one go
        
        # CRITICAL: ALL ranks must participate in this collective operation
        if self.accelerator is not None and self.accelerator.num_processes > 1:
            # All ranks need to know the max number of batches any rank will process
            num_batches_tensor = torch.tensor([num_batches], device=device)
            # Use accelerator's reduce for gathering max
            max_batches_tensor = self.accelerator.reduce(num_batches_tensor, reduction="max")
            max_batches_across_ranks = max_batches_tensor.item()
        else:
            max_batches_across_ranks = num_batches
        
        # Now actually do the batched processing if needed
        if max_batches_across_ranks > 1:
            
            last_token_embeds = []
            for batch_num in range(max_batches_across_ranks):
                start_idx = batch_num * self.max_encode_batch_size
                end_idx = min(start_idx + self.max_encode_batch_size, total_chunks)
                
                # Making FSDP happy :)
                if start_idx >= total_chunks:
                    dummy_input = torch.full((1, chunk_tok_size), self.pad_token_id, dtype=torch.long, device=device)
                    dummy_mask = torch.zeros((1, chunk_tok_size), dtype=torch.long, device=device)
                    if self.train_embedder:
                        _ = self.embed_model(input_ids=dummy_input, attention_mask=dummy_mask)
                    else:
                        with torch.no_grad():
                            _ = self.embed_model(input_ids=dummy_input, attention_mask=dummy_mask)
                else:
                    batch_input_ids = chunk_input_ids[start_idx:end_idx]
                    batch_attention_mask = chunk_attention_mask[start_idx:end_idx]
                    if self.train_embedder:
                        batch_outputs = self.embed_model(
                            input_ids=batch_input_ids,
                            attention_mask=batch_attention_mask,
                        )
                    else:
                        with torch.no_grad():
                            batch_outputs = self.embed_model(
                                input_ids=batch_input_ids,
                                attention_mask=batch_attention_mask,
                            )
                    last_token_embeds.append(batch_outputs.last_hidden_state[:, -1])
            
            chunk_embeddings = torch.cat(last_token_embeds, dim=0)
        else:
            # Single batch encoding (all chunks fit in one forward pass)
            if self.train_embedder:
                outputs = self.embed_model(
                    input_ids=chunk_input_ids,
                    attention_mask=chunk_attention_mask,
                )
            else:
                with torch.no_grad():
                    outputs = self.embed_model(
                        input_ids=chunk_input_ids,
                        attention_mask=chunk_attention_mask,
                    )
            # Extract last token embedding from each chunk
            chunk_embeddings = outputs.last_hidden_state[:, -1]


        result_embeddings = []
        chunk_idx = 0
        for sample_idx in range(len(code)):
            num_chunks = chunk_counts[sample_idx]
            end_idx = chunk_idx + num_chunks
            sample_embeddings = chunk_embeddings[chunk_idx:end_idx]
            result_embeddings.append(sample_embeddings)
            chunk_idx += num_chunks

        return result_embeddings 
