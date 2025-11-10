"""
Clean Packed Attention implementation for Qwen3 using flex_attention.
Inheritance-based (not wrapper) to avoid duplicate state dict keys.
Based on the old wrapper implementation but with inheritance pattern like Bagel.
"""
import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import flex_attention, create_block_mask, BlockMask
from typing import List, Optional, Tuple
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Attention,
    Qwen3RMSNorm,
    apply_rotary_pos_emb,
    rotate_half,
)


# Compile flex_attention for performance
flex_attention = torch.compile(flex_attention)


class Qwen3PackedAttention(Qwen3Attention):
    """
    Packed sequence attention for Qwen3 using flex_attention.
    
    Inherits from Qwen3Attention to reuse all the projection layers and norms.
    Overrides forward to use flex_attention when BlockMask is provided.
    """
    
    def __init__(self, config, layer_idx: int):
        super().__init__(config, layer_idx)
        # All projections and norms are inherited from Qwen3Attention
        # No duplicate modules stored - clean state dict
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[tuple] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        sample_lens: Optional[List[int]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass that uses flex_attention when BlockMask is provided.
        Otherwise falls back to parent implementation.
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_dim]
            attention_mask: Either BlockMask for flex attention or standard mask
            position_embeddings: Pre-computed (cos, sin) from model level
            ... (other standard Qwen3 attention args)
        
        Returns:
            (attn_output, None) - output and no attention weights
        """
        # Check if we should use flex attention
        use_flex = isinstance(attention_mask, BlockMask) and self.training
        
        if use_flex:
            return self._forward_flex(
                hidden_states=hidden_states,
                block_mask=attention_mask,
                position_embeddings=position_embeddings,
            )
        else:
            # Use parent's implementation for non-packed sequences
            return super().forward(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                cache_position=cache_position,
                **kwargs,
            )
    
    def _forward_flex(
        self,
        hidden_states: torch.Tensor,
        block_mask: BlockMask,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, None]:
        """
        Forward pass using flex_attention for packed sequences.
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_dim]
            block_mask: BlockMask from flex_attention
            position_embeddings: (cos, sin) embeddings from model
        
        Returns:
            (attn_output, None) where attn_output is [batch_size, seq_len, hidden_dim]
        """
        bsz, q_len, _ = hidden_states.size()
        
        # Project Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Reshape for attention: [batch, seq, num_heads, head_dim]
        query_states = query_states.view(bsz, q_len, self.config.num_attention_heads, self.head_dim)
        key_states = key_states.view(bsz, q_len, self.config.num_key_value_heads, self.head_dim)
        value_states = value_states.view(bsz, q_len, self.config.num_key_value_heads, self.head_dim)
        
        # Apply Q/K normalization (Qwen3 specific)
        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)
        
        # Transpose for attention: [batch, num_heads, seq, head_dim]
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
        
        # Apply rotary embeddings
        if position_embeddings is None:
            raise ValueError("position_embeddings must be provided for Qwen3PackedAttention")
        
        cos, sin = position_embeddings
        # Ensure cos/sin have the same dtype as query/key states
        cos = cos.to(query_states.dtype)
        sin = sin.to(key_states.dtype)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        
        # Apply flex attention with GQA support
        # flex_attention expects: [batch, num_heads, seq_len, head_dim]
        attn_output = flex_attention(
            query_states,
            key_states,
            value_states,
            block_mask=block_mask,
            enable_gqa=(self.num_key_value_groups > 1),
            scale=self.scaling,
        )
        
        # Reshape back: [batch, num_heads, seq, head_dim] -> [batch, seq, num_heads * head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.config.num_attention_heads * self.head_dim)
        
        # Output projection
        attn_output = self.o_proj(attn_output)
        
        # Return format matches original attention (output, None for weights)
        return attn_output, None


def replace_with_packed_attention(model):
    """
    Replace all Qwen3Attention layers with Qwen3PackedAttention.
    
    This modifies the model in-place, replacing attention modules while
    preserving all weights (q_proj, k_proj, v_proj, o_proj, q_norm, k_norm).
    
    Args:
        model: Qwen3Model instance (the .model attribute of Qwen3ForCausalLM)
               Can also be a PEFT-wrapped model (e.g., when using LoRA)
        
    Returns:
        Modified model with packed attention layers
    """
    # Handle PEFT wrapped models (e.g., when LoRA is applied)
    # PEFT structure: PeftModel -> base_model (LoraModel) -> model (Qwen3ForCausalLM) -> model (Qwen3Model) -> layers
    actual_model = model
    if hasattr(model, 'base_model') and hasattr(model.base_model, 'model'):
        # This is a PEFT model, get the actual base model
        actual_model = model.base_model.model
        if hasattr(actual_model, 'model'):
            # Qwen3ForCausalLM -> get the Qwen3Model
            actual_model = actual_model.model
    elif hasattr(model, 'model'):
        # Direct Qwen3ForCausalLM -> get the Qwen3Model
        actual_model = model.model
    
    if not hasattr(actual_model, 'layers'):
        raise ValueError(f"Model must have 'layers' attribute (Qwen3Model expected). Got model type: {type(actual_model)}")
    
    for layer_idx, layer in enumerate(actual_model.layers):
        if hasattr(layer, 'self_attn'):
            old_attn = layer.self_attn
            
            # Create new packed attention with same config, device, and dtype
            # Get device and dtype from existing weights
            device = old_attn.q_proj.weight.device
            dtype = old_attn.q_proj.weight.dtype
            
            # Check if the old attention was frozen
            old_requires_grad = old_attn.q_proj.weight.requires_grad
            
            new_attn = Qwen3PackedAttention(old_attn.config, layer_idx)
            new_attn = new_attn.to(device=device, dtype=dtype)
            
            # Copy over the trained weights from old attention
            # All these are nn.Linear or RMSNorm modules with .weight (and maybe .bias)
            new_attn.q_proj.weight.data.copy_(old_attn.q_proj.weight.data)
            new_attn.k_proj.weight.data.copy_(old_attn.k_proj.weight.data)
            new_attn.v_proj.weight.data.copy_(old_attn.v_proj.weight.data)
            new_attn.o_proj.weight.data.copy_(old_attn.o_proj.weight.data)
            
            if old_attn.config.attention_bias:
                new_attn.q_proj.bias.data.copy_(old_attn.q_proj.bias.data)
                new_attn.k_proj.bias.data.copy_(old_attn.k_proj.bias.data)
                new_attn.v_proj.bias.data.copy_(old_attn.v_proj.bias.data)
                new_attn.o_proj.bias.data.copy_(old_attn.o_proj.bias.data)
            
            # Copy norm weights
            new_attn.q_norm.weight.data.copy_(old_attn.q_norm.weight.data)
            new_attn.k_norm.weight.data.copy_(old_attn.k_norm.weight.data)
            
            # Preserve requires_grad status from original attention
            if not old_requires_grad:
                for param in new_attn.parameters():
                    param.requires_grad = False
            
            # Replace the attention module
            layer.self_attn = new_attn
    
    return model


def create_packed_block_mask(
    sample_lens: List[int],
    num_heads: int,
    device: torch.device,
    block_size: int = 128,
) -> BlockMask:
    """
    Create a BlockMask for packed sequences where each sample attends only to itself.
    
    Args:
        sample_lens: List of sequence lengths for each sample in the batch
        num_heads: Number of attention heads
        device: Device to create mask on
        block_size: Block size for flex_attention (default 128)
        
    Returns:
        BlockMask object for use with flex_attention
    """
    total_len = sum(sample_lens)
    
    # Precompute cumulative sums for faster lookup
    cum_lens = [0]
    for length in sample_lens:
        cum_lens.append(cum_lens[-1] + length)
    
    def mask_mod(b, h, q_idx, kv_idx):
        """
        Mask function: return True if position should be masked (not attended to).
        For packed sequences, each sample can only attend within its own boundaries.
        """
        # Find which sample q_idx and kv_idx belong to using binary search concept
        q_sample = 0
        for i in range(len(sample_lens)):
            if q_idx < cum_lens[i + 1]:
                q_sample = i
                q_offset = q_idx - cum_lens[i]
                break
        
        kv_sample = 0
        for i in range(len(sample_lens)):
            if kv_idx < cum_lens[i + 1]:
                kv_sample = i
                kv_offset = kv_idx - cum_lens[i]
                break
        
        # Only allow attention within same sample and with causal constraint
        return (q_sample != kv_sample) or (q_offset < kv_offset)
    
    block_mask = create_block_mask(
        mask_mod,
        B=1,  # Packed sequences treated as single batch
        H=num_heads,
        Q_LEN=total_len,
        KV_LEN=total_len,
        device=device,
        BLOCK_SIZE=block_size,
    )
    
    return block_mask


# Export main classes and functions
__all__ = [
    'Qwen3PackedAttention',
    'replace_with_packed_attention',
    'create_packed_block_mask',
]
