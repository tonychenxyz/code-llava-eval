"""
Clean Code LLaVA implementation with simplified single forward method.
"""
import torch
import torch.nn as nn
from transformers import AutoTokenizer, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from typing import Optional, Union, Tuple, List
from transformers.cache_utils import Cache

from .code_chunker import CodeChunker
from .processor import CodeLLaVAProcessor


class CodeLLaVAModel(nn.Module):
    """
    Clean Code LLaVA model that uses the processor for all code handling.
    
    This model:
    1. Uses processor to expand code placeholders
    2. Uses CodeChunker to get code embeddings
    3. Uses processor to replace placeholder tokens with code embeddings
    4. Passes through the LLM for generation
    """
    
    def __init__(
        self,
        llm_model: PreTrainedModel,
        llm_tokenizer: AutoTokenizer,
        embed_model: PreTrainedModel,
        embed_tokenizer: AutoTokenizer,
        processor: CodeLLaVAProcessor,
        chunk_size: int = 100,
        max_code_length: int = 8192,
        train_llm: bool = True,
        train_embedder: bool = True,
        max_encode_batch_size: int = 0,
        accelerator = None,
        use_packed_attention: bool = False,
    ):
        super().__init__()
        
        self.llm_model = llm_model
        self.llm_tokenizer = llm_tokenizer
        self.train_llm = train_llm
        self.train_embedder = train_embedder
        self.processor = processor
        self.accelerator = accelerator
        self.use_packed_attention = use_packed_attention
        
        # Enable packed attention if requested
        if use_packed_attention:
            self._enable_packed_attention()
        
        # Initialize code chunker
        self.code_chunker = CodeChunker(
            embed_model=embed_model,
            embed_tokenizer=embed_tokenizer,
            chunk_size=chunk_size,
            max_length=max_code_length,
            train_embedder=train_embedder,
            max_encode_batch_size=max_encode_batch_size,
            accelerator=accelerator,
        )
        
        # Projection layers to map code embeddings to LLM space
        embed_dim = self.code_chunker.embedding_dim
        llm_dim = self.llm_model.config.hidden_size
        
        # mlp_hidden_dim = max(embed_dim * 3, llm_dim)
        mlp_hidden_dim = llm_dim
        self.code_projection = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, llm_dim)
        )
        # Keep norm parameters in full precision to avoid stagnant updates under bf16 mixed precision
        self.code_norm_pre = nn.RMSNorm(embed_dim, eps=1e-6)

        # Initialize projection layers
        for m in self.code_projection.modules():
            if isinstance(m, nn.Linear):
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def setup_accelerator(self, accelerator):
        """Set the accelerator after initialization for both model and code_chunker."""
        self.accelerator = accelerator
    
    def _enable_packed_attention(self):
        """Replace attention layers with clean inheritance-based packed attention."""
        try:
            from .packed_attention import replace_with_packed_attention, Qwen3PackedAttention
            
            # Check if packed attention is already applied (e.g., in trainer before LoRA)
            # Navigate through potential PEFT wrapper to check actual model
            check_model = self.llm_model
            if hasattr(check_model, 'base_model') and hasattr(check_model.base_model, 'model'):
                check_model = check_model.base_model.model
            if hasattr(check_model, 'model'):
                check_model = check_model.model
            
            if hasattr(check_model, 'layers') and len(check_model.layers) > 0:
                if isinstance(check_model.layers[0].self_attn, Qwen3PackedAttention):
                    print("✓ Packed attention already enabled (skipping)")
                    return
            
            # Replace attention layers with clean packed attention (no duplicate keys)
            # The function will handle both regular models and PEFT-wrapped models (e.g., LoRA)
            replace_with_packed_attention(self.llm_model)
            
            print(f"✓ Enabled packed attention (clean implementation, no fallback)")
            
        except Exception as e:
            print(f"✗ Error enabling packed attention: {e}")
            raise  # Don't fall back - fix the issue instead
    
    def _process_code_embeddings(
        self,
        codes: Union[List[str], List[List[str]]]
    ) -> Union[List[torch.Tensor], List[List[torch.Tensor]]]:
        """
        Process code into embeddings using the chunker.
        
        Args:
            codes: List of code strings
            
        Returns:
            List of code embeddings, one tensor per code sample: [num_chunks, llm_dim]
        """
        import sys
        if not codes:
            return []
        # Multi-segment per-sample
        if isinstance(codes[0], list):
            # Flatten segments across batch
            flat: List[str] = []
            seg_counts: List[int] = []
            for segs in codes:  # type: ignore[index]
                seg_counts.append(len(segs))
                flat.extend(segs)
            
            # Handle case where all segments are empty
            if not flat:
                return [[] for _ in seg_counts]
            
            flat_embeds = self.code_chunker(flat)
            projected_flat: List[torch.Tensor] = []
            for chunk_emb in flat_embeds:
                projected_flat.append(self._project_code_chunk(chunk_emb))
            # Regroup
            regrouped: List[List[torch.Tensor]] = []
            idx = 0
            for cnt in seg_counts:
                regrouped.append(projected_flat[idx:idx+cnt])
                idx += cnt
            return regrouped
        # Single-segment per-sample
        chunk_embeddings_list = self.code_chunker(codes)  # type: ignore[arg-type]
        projected_embeddings_list: List[torch.Tensor] = []
        for chunk_emb in chunk_embeddings_list:
            projected = self._project_code_chunk(chunk_emb)
            projected_embeddings_list.append(projected)
        return projected_embeddings_list

    def _project_code_chunk(self, chunk_emb: torch.Tensor) -> torch.Tensor:
        """Normalize and project a single chunk embedding with dtype-safe casts."""
        norm_dtype = self.code_norm_pre.weight.dtype
        proj_dtype = self.code_projection[0].weight.dtype

        if chunk_emb.dtype != norm_dtype:
            chunk_emb = chunk_emb.to(norm_dtype)

        normalized = self.code_norm_pre(chunk_emb)

        if normalized.dtype != proj_dtype:
            normalized = normalized.to(proj_dtype)

        return self.code_projection(normalized)
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        codes: Optional[List[str]] = None,
        code_positions: Optional[List[Tuple[int, int]]] = None,
        chunk_counts: Optional[List[int]] = None,
        sample_lens: Optional[List[int]] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        Forward pass using the processor approach.
        
        When codes are provided, the processor should have already:
        1. Expanded the <|memory|> placeholders to <|memory_start|> + N <|memory|> + <|memory_end|>
        2. Tokenized the expanded prompt
        3. Provided code_positions and chunk_counts
        
        This method then:
        1. Gets embeddings for the tokenized input
        2. Processes code through the chunker
        3. Uses processor to replace placeholder embeddings with code embeddings
        4. Passes through LLM
        """

        device = next(self.llm_model.parameters()).device

        input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        if labels is not None:
            labels = labels.to(device)
        # Normalize to nested structures
        if isinstance(codes, list) and len(codes) > 0 and isinstance(codes[0], list):
            codes_nested: List[List[str]] = codes  # type: ignore[assignment]
        elif isinstance(codes, list):
            codes_nested = [[c] for c in codes]  # type: ignore[list-item]
        else:
            codes_nested = [[] for _ in range(input_ids.size(0))]

        if isinstance(code_positions, list) and len(code_positions) > 0 and isinstance(code_positions[0], list):
            pos_nested: List[List[Tuple[int, int]]] = code_positions  # type: ignore[assignment]
        elif isinstance(code_positions, list):
            pos_nested = [code_positions]
        else:
            pos_nested = [[] for _ in range(input_ids.size(0))]

        if isinstance(chunk_counts, list) and len(chunk_counts) > 0 and isinstance(chunk_counts[0], list):
            counts_nested: List[List[int]] = chunk_counts  # type: ignore[assignment]
        elif isinstance(chunk_counts, list):
            counts_nested = [chunk_counts]
        else:
            counts_nested = [[] for _ in range(input_ids.size(0))]

        inputs_embeds = self.llm_model.get_input_embeddings()(input_ids)
        
        # CRITICAL: Synchronize before code encoding because embed_model is FSDP-wrapped
        # Different sequence lengths mean different encoding times, causing FSDP deadlock
        if self.accelerator is not None and self.accelerator.num_processes > 1:
            self.accelerator.wait_for_everyone()
        
        code_embeddings_nested = self._process_code_embeddings(codes_nested)

        combined_embeds = self.processor.replace_code_tokens_with_embeddings(
            inputs_embeds=inputs_embeds,
            code_embeddings=code_embeddings_nested,  # type: ignore[arg-type]
            code_positions=pos_nested,
            chunk_counts=counts_nested,
            input_ids=input_ids,
        )
        
        # Debug: Verify shape
        assert combined_embeds.shape[0] == input_ids.shape[0], f"Batch size mismatch: {combined_embeds.shape[0]} vs {input_ids.shape[0]}"
        assert combined_embeds.shape[1] == input_ids.shape[1], f"Seq len mismatch: {combined_embeds.shape[1]} vs {input_ids.shape[1]}"
        assert combined_embeds.dim() == 3, f"Wrong dims: {combined_embeds.shape}"
        
        # CRITICAL: Synchronize all ranks before LLM forward to prevent FSDP deadlock
        # This ensures all ranks finish encoding before any rank calls the FSDP-wrapped LLM
        if self.accelerator is not None and self.accelerator.num_processes > 1:
            self.accelerator.wait_for_everyone()
        
        # Create block mask and position IDs for packed sequences if needed
        position_ids = None
        if self.use_packed_attention and sample_lens is not None and self.training:
            from data.packing_utils import create_block_mask_for_packed
            
            num_heads = self.llm_model.config.num_attention_heads
            total_len = sum(sample_lens)
            
            # Create block mask for flex attention
            attention_mask = create_block_mask_for_packed(
                sample_lens=sample_lens,
                num_heads=num_heads,
                device=device,
                block_size=128,
            )
            
            # CRITICAL: Create position IDs that reset at each sequence boundary
            # Without this, position IDs would be [0,1,2,...,total_len-1]
            # But we need [0,1,...,len1-1, 0,1,...,len2-1, 0,1,...,len3-1]
            position_ids_list = []
            for sample_len in sample_lens:
                position_ids_list.append(torch.arange(sample_len, device=device))
            position_ids = torch.cat(position_ids_list, dim=0).unsqueeze(0)  # [1, total_len]
        
        # Forward through LLM
        output = self.llm_model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=combined_embeds,
            labels=labels,
            output_hidden_states=output_hidden_states,
            **kwargs,
        )
        
        return output

    def generate(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        codes: Optional[List[str]] = None,
        code_positions: Optional[List[Tuple[int, int]]] = None,
        chunk_counts: Optional[List[int]] = None,
        **generation_kwargs
    ):
        """
        Generate text with code context using the processor.
        
        Args:
            prompts: List of prompt strings with <|memory|> placeholders
            codes: List of code strings
            max_length: Maximum length for generation
            **generation_kwargs: Additional generation arguments
            
        Returns:
            Generated token IDs
        """
        
        device = next(self.llm_model.parameters()).device

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        # Normalize to nested
        if isinstance(codes, list) and len(codes) > 0 and isinstance(codes[0], list):
            codes_nested: List[List[str]] = codes  # type: ignore[assignment]
        elif isinstance(codes, list):
            codes_nested = [[c] for c in codes]  # type: ignore[list-item]
        else:
            codes_nested = [[] for _ in range(input_ids.size(0))]

        if isinstance(code_positions, list) and len(code_positions) > 0 and isinstance(code_positions[0], list):
            pos_nested: List[List[Tuple[int, int]]] = code_positions  # type: ignore[assignment]
        elif isinstance(code_positions, list):
            pos_nested = [code_positions]
        else:
            pos_nested = [[] for _ in range(input_ids.size(0))]

        if isinstance(chunk_counts, list) and len(chunk_counts) > 0 and isinstance(chunk_counts[0], list):
            counts_nested: List[List[int]] = chunk_counts  # type: ignore[assignment]
        elif isinstance(chunk_counts, list):
            counts_nested = [chunk_counts]
        else:
            counts_nested = [[] for _ in range(input_ids.size(0))]

        inputs_embeds = self.llm_model.get_input_embeddings()(input_ids)
        code_embeddings_nested = self._process_code_embeddings(codes_nested)

        combined_embeds = self.processor.replace_code_tokens_with_embeddings(
            inputs_embeds=inputs_embeds,
            code_embeddings=code_embeddings_nested,  # type: ignore[arg-type]
            code_positions=pos_nested,
            chunk_counts=counts_nested,
            input_ids=input_ids,
        )
        
        # Generate using the LLM
        return self.llm_model.generate(
            inputs_embeds=combined_embeds,
            attention_mask=attention_mask,
            **generation_kwargs
        )
