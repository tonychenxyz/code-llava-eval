"""
CodeLLaVA Processor with regex-based code extraction before tokenization.

This processor:
1. Takes prompts that already contain code wrapped with <|memory_start|> ... <|memory_end|>
2. Uses regex to find and extract code content between markers (string-level, pre-tokenization)
3. Calculates chunks needed and replaces literal code with N <|memory|> placeholder tokens
4. Tokenizes the modified prompt
5. Returns tokenized inputs and chunk information for embedding replacement
"""

import re
import torch
from transformers import AutoTokenizer
from typing import List, Dict, Any, Tuple, Optional
import math


class CodeLLaVAProcessor:
    """
    Processor that uses regex-based extraction for efficient code region processing.
    """
    
    def __init__(
        self,
        llm_tokenizer: AutoTokenizer,
        embed_tokenizer: AutoTokenizer,
        chunk_size: int = 100,
        max_code_length: int = 8192,
        use_code_wrapping: bool = True,
    ):
        self.llm_tokenizer = llm_tokenizer
        self.embed_tokenizer = embed_tokenizer
        self.chunk_size = chunk_size
        self.max_code_length = max_code_length
        self.use_code_wrapping = use_code_wrapping
        
        # Special tokens
        self.code_placeholder = "<|memory|>"
        self.memory_start = "<|memory_start|>"
        self.memory_end = "<|memory_end|>"
        
        # Compile regex pattern for finding code regions
        # Using re.DOTALL to match newlines within code
        self.code_pattern = re.compile(
            rf"{re.escape(self.memory_start)}(.*?){re.escape(self.memory_end)}", 
            re.DOTALL
        )
        
        # Get token IDs for special tokens
        self.code_token_id = self.llm_tokenizer.convert_tokens_to_ids(self.code_placeholder)
        if self.use_code_wrapping:
            self.memory_start_id = self.llm_tokenizer.convert_tokens_to_ids(self.memory_start)
            self.memory_end_id = self.llm_tokenizer.convert_tokens_to_ids(self.memory_end)

    def extract_and_replace_code_regions(self, text: str) -> Tuple[str, List[str], List[int], List[int]]:
        """
        Extract code regions from text using regex and replace with placeholders.
        
        Args:
            text: Input text containing <|memory_start|>...<|memory_end|> regions
            
        Returns:
            Tuple of:
            - Modified text with code replaced by placeholders
            - List of extracted code strings
            - List of chunk counts for each code region
            - List of embed token counts for each code region
        """
        codes = []
        chunk_counts = []
        embed_token_counts = []
        
        def replacer(match):
            # Extract the code content
            code_content = match.group(1)
            codes.append(code_content)
            
            # Calculate chunks needed
            token_ids = self.embed_tokenizer.encode(code_content, add_special_tokens=False)
            embed_len = len(token_ids)
            num_chunks = math.ceil(embed_len / self.chunk_size)
            
            chunk_counts.append(num_chunks)
            embed_token_counts.append(embed_len)
            
            # Create replacement with N placeholder tokens
            placeholders = self.code_placeholder * num_chunks
            
            # Return the wrapped version with placeholders
            return f"{self.memory_start}{placeholders}{self.memory_end}"
        
        # Replace all code regions
        modified_text = self.code_pattern.sub(replacer, text)
        
        return modified_text, codes, chunk_counts, embed_token_counts

    def process_wrapped_batch(
        self,
        prompts: List[str],
        targets: Optional[List[str]] = None,
        max_length: Optional[int] = None,
        padding: str = "longest",
        truncation: bool = False,
        return_tensors: str = "pt",
    ) -> Dict[str, Any]:
        """
        Process a batch where prompts already contain literal code wrapped with
        <|memory_start|> ... <|memory_end|>. Uses regex for efficient extraction.
        """
        
        if targets is not None and len(targets) != len(prompts):
            raise ValueError(f"Number of targets ({len(targets)}) must match number of prompts ({len(prompts)})")

        batch_size = len(prompts)
        expanded_prompts = []
        all_codes = []
        all_chunk_counts = []
        all_embed_token_counts = []

        # Process each prompt using regex
        for prompt in prompts:
            modified_prompt, codes, chunk_counts, embed_counts = self.extract_and_replace_code_regions(prompt)
            expanded_prompts.append(modified_prompt)
            all_codes.append(codes)
            all_chunk_counts.append(chunk_counts)
            all_embed_token_counts.append(embed_counts)

        # Compose full sequences with optional targets
        if targets is not None:
            full_sequences = [p + t for p, t in zip(expanded_prompts, targets)]
            prompt_lengths = [len(self.llm_tokenizer.encode(p, add_special_tokens=False)) for p in expanded_prompts]
        else:
            full_sequences = expanded_prompts
            prompt_lengths = None

        # Tokenize the modified sequences
        tokenized = self.llm_tokenizer(
            full_sequences,
            padding=padding,
            truncation=truncation,
            return_tensors=return_tensors,
        )

        # Find code positions in tokenized sequences
        code_positions = []
        for i in range(batch_size):
            input_ids = tokenized["input_ids"][i]
            # Find all wrapped regions in the tokenized sequence
            positions = self._find_all_code_token_positions(
                input_ids, 
                expected_chunks_per_region=all_chunk_counts[i]
            )
            code_positions.append(positions)

        # Create labels if training
        labels = None
        if targets is not None:
            labels = tokenized["input_ids"].clone()
            for i, pr_len in enumerate(prompt_lengths):
                labels[i, :pr_len] = -100
            labels = labels.masked_fill(tokenized["attention_mask"] == 0, -100)

        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": labels,
            "code_positions": code_positions,
            "chunk_counts": all_chunk_counts,
            "embed_token_counts": all_embed_token_counts,
            "codes": all_codes,
        }
    
    def process_teacher_batch(
        self,
        prompts: List[str],
        codes: List[str],
        targets: Optional[List[str]] = None,
        max_length: Optional[int] = None,
        padding: str = "longest",
        truncation: bool = True,
        return_tensors: str = "pt",
        max_code_length: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Process a batch of prompts and codes for teacher model.
        Simply tokenizes the full text without any special code processing.
        """
        
        full_sequences = [prompt + target for prompt, target in zip(prompts, targets)]
        prompt_lengths = [len(self.llm_tokenizer.encode(prompt, add_special_tokens=False)) for prompt in prompts]
   
        # Tokenize using LLM tokenizer only
        tokenized = self.llm_tokenizer(
            full_sequences,
            padding=padding,
            truncation=truncation,
            return_tensors=return_tensors,
        )
        
        labels = None
        if targets is not None:
            labels = tokenized["input_ids"].clone()
            # Mask out prompt tokens (only train on targets)
            for i, prompt_len in enumerate(prompt_lengths):
                labels[i, :prompt_len] = -100
            # Mask padding tokens
            labels = labels.masked_fill(tokenized["attention_mask"] == 0, -100)
        
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": labels,
        }

    def _find_all_code_token_positions(
        self,
        input_ids: torch.Tensor,
        expected_chunks_per_region: List[int],
    ) -> List[Tuple[int, int]]:
        """
        Find all wrapped code regions in tokenized input.
        Ultra-efficient: directly calculates end positions using chunk counts
        and stops as soon as all expected regions are found.
        
        Returns a list of (start_pos, end_pos) pairs for each region.
        """
        input_list = input_ids.tolist()
        positions = []
        
        if not self.use_code_wrapping:
            # Handle unwrapped case (contiguous <|memory|> tokens)
            return self._find_unwrapped_code_positions(input_list, expected_chunks_per_region)
        
        # Early termination: if no regions expected, return immediately
        num_expected_regions = len(expected_chunks_per_region)
        if num_expected_regions == 0:
            return positions
        
        # Wrapped case: find start markers and calculate end positions
        i = 0
        region_idx = 0
        
        while i < len(input_list):
            if input_list[i] == self.memory_start_id:
                start_pos = i + 1  # Position after <|memory_start|>
                
                # OPTIMIZATION: Calculate end position directly!
                # We know there are exactly N <|memory|> tokens
                num_chunks = expected_chunks_per_region[region_idx]
                end_pos = start_pos + num_chunks
                
                # Validate our calculation is correct
                if end_pos >= len(input_list):
                    raise ValueError(
                        f"Region {region_idx}: calculated end position {end_pos} exceeds sequence length {len(input_list)}"
                    )
                
                # The token at end_pos should be <|memory_end|>
                if input_list[end_pos] != self.memory_end_id:
                    # Something went wrong - maybe tokenization changed?
                    raise ValueError(
                        f"Region {region_idx}: expected <|memory_end|> at position {end_pos}, "
                        f"but found token {input_list[end_pos]} (id: {self.llm_tokenizer.convert_ids_to_tokens(input_list[end_pos])})"
                    )
                
                # Optional validation: check all tokens between start and end are <|memory|>
                # Can be commented out in production for speed
                for j in range(start_pos, end_pos):
                    if input_list[j] != self.code_token_id:
                        raise ValueError(
                            f"Region {region_idx}: expected only <|memory|> tokens between {start_pos} and {end_pos}, "
                            f"but found token {input_list[j]} at position {j}"
                        )
                
                positions.append((start_pos, end_pos))
                region_idx += 1
                
                # OPTIMIZATION: Early termination - stop if we found all expected regions
                if region_idx >= num_expected_regions:
                    break
                
                # Jump to after the end marker for next search
                i = end_pos + 1
            else:
                i += 1
        
        # Note: It's okay if we found fewer regions than expected (could be truncation)
        # The calling code should handle this case appropriately
        
        return positions

    def _find_unwrapped_code_positions(
        self, 
        input_list: List[int], 
        expected_chunks_per_region: List[int]
    ) -> List[Tuple[int, int]]:
        """
        Find positions of unwrapped code regions (contiguous <|memory|> tokens).
        Stops as soon as all expected regions are found.
        """
        positions = []
        i = 0
        region_idx = 0
        num_expected_regions = len(expected_chunks_per_region)
        
        # Early termination: if no regions expected, return immediately
        if num_expected_regions == 0:
            return positions
        
        while i < len(input_list):
            if input_list[i] == self.code_token_id:
                # Found start of a code region
                expected = expected_chunks_per_region[region_idx]
                start_pos = i
                
                # Verify we have enough contiguous tokens
                actual = 0
                for j in range(i, min(i + expected, len(input_list))):
                    if input_list[j] == self.code_token_id:
                        actual += 1
                    else:
                        break
                
                if actual != expected:
                    raise ValueError(
                        f"Region {region_idx}: expected {expected} contiguous <|memory|> tokens, "
                        f"found {actual} starting at position {start_pos}"
                    )
                
                end_pos = start_pos + expected
                positions.append((start_pos, end_pos))
                region_idx += 1
                
                # OPTIMIZATION: Early termination - stop if we found all expected regions
                if region_idx >= num_expected_regions:
                    break
                    
                # Jump to after this region
                i = end_pos
            else:
                i += 1
        
        return positions

    def _verify_code_token_positions(
        self,
        input_ids: torch.Tensor,
        start_pos: int,
        end_pos: int,
        expected_chunks: int,
        batch_idx: int,
        segment_idx: int,
    ) -> None:
        """
        Verify that code positions correspond to actual code markers and placeholder tokens.
        
        Args:
            input_ids: Token IDs for the sequence [seq_len]
            start_pos: Start position of code region (after <|memory_start|>)
            end_pos: End position of code region (before <|memory_end|>)
            expected_chunks: Expected number of code placeholder tokens
            batch_idx: Batch index for error messages
            segment_idx: Segment index for error messages
        """
        seq_len = input_ids.size(0)
        
        # Check bounds
        if start_pos < 0 or end_pos > seq_len or start_pos >= end_pos:
            raise ValueError(
                f"Batch {batch_idx}, segment {segment_idx}: Invalid code position range "
                f"[{start_pos}, {end_pos}) for sequence length {seq_len}"
            )
        
        # Verify number of tokens matches expected chunks
        actual_chunks = end_pos - start_pos
        if actual_chunks != expected_chunks:
            raise ValueError(
                f"Batch {batch_idx}, segment {segment_idx}: Expected {expected_chunks} "
                f"code placeholder tokens, but found {actual_chunks} tokens in range [{start_pos}, {end_pos})"
            )
        
        if self.use_code_wrapping:
            # Verify start marker exists before start_pos
            if start_pos > 0 and input_ids[start_pos - 1] != self.memory_start_id:
                token_name = self.llm_tokenizer.convert_ids_to_tokens([input_ids[start_pos - 1].item()])[0]
                raise ValueError(
                    f"Batch {batch_idx}, segment {segment_idx}: Expected <|memory_start|> token at position {start_pos - 1}, "
                    f"but found '{token_name}' (id: {input_ids[start_pos - 1].item()})"
                )
            
            # Verify end marker exists at end_pos
            if end_pos < seq_len and input_ids[end_pos] != self.memory_end_id:
                token_name = self.llm_tokenizer.convert_ids_to_tokens([input_ids[end_pos].item()])[0]
                raise ValueError(
                    f"Batch {batch_idx}, segment {segment_idx}: Expected <|memory_end|> token at position {end_pos}, "
                    f"but found '{token_name}' (id: {input_ids[end_pos].item()})"
                )
        
        # Verify all tokens in the range are code placeholder tokens
        for pos in range(start_pos, end_pos):
            if input_ids[pos] != self.code_token_id:
                token_name = self.llm_tokenizer.convert_ids_to_tokens([input_ids[pos].item()])[0]
                raise ValueError(
                    f"Batch {batch_idx}, segment {segment_idx}: Expected <|memory|> placeholder token at position {pos}, "
                    f"but found '{token_name}' (id: {input_ids[pos].item()}). "
                    f"All tokens in range [{start_pos}, {end_pos}) should be code placeholders."
                )

    def replace_code_tokens_with_embeddings(
        self,
        inputs_embeds: torch.Tensor,
        code_embeddings: List[List[torch.Tensor]],
        code_positions: List[List[Tuple[int, int]]],
        chunk_counts: List[List[int]],
        input_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Replace placeholder tokens with actual code embeddings.
        
        Args:
            inputs_embeds: Input embeddings with placeholders [batch_size, seq_len, hidden_dim]
            code_embeddings: List of lists of code embeddings for each sample and region
            code_positions: List of lists of (start, end) positions for each code region
            chunk_counts: List of lists of chunk counts for validation
            input_ids: Optional input token IDs for verification [batch_size, seq_len]
        """
        
        for b in range(inputs_embeds.size(0)):
            segments = code_positions[b]
            for s_idx, (start_pos, end_pos) in enumerate(segments):
                # Verify data structure consistency
                if s_idx >= len(chunk_counts[b]):
                    raise ValueError(
                        f"Batch {b}, segment {s_idx}: code_positions has more segments than chunk_counts. "
                        f"Found {len(segments)} segments but only {len(chunk_counts[b])} chunk counts."
                    )
                
                if s_idx >= len(code_embeddings[b]):
                    raise ValueError(
                        f"Batch {b}, segment {s_idx}: code_positions has more segments than code_embeddings. "
                        f"Found {len(segments)} segments but only {len(code_embeddings[b])} embeddings."
                    )
                
                num_chunks = chunk_counts[b][s_idx]
                if num_chunks <= 0:
                    raise ValueError(
                        f"Batch {b}, segment {s_idx}: Invalid chunk count {num_chunks}. Must be > 0."
                    )
                
                # Verify positions and tokens if input_ids provided
                if input_ids is not None:
                    self._verify_code_token_positions(
                        input_ids[b], start_pos, end_pos, num_chunks, b, s_idx
                    )
                
                emb = code_embeddings[b][s_idx]
                
                # Verify exact size match
                available_slots = end_pos - start_pos
                emb_size = emb.size(0)
                
                if emb_size != available_slots:
                    raise ValueError(
                        f"Batch {b}, segment {s_idx}: Embedding size mismatch. "
                        f"Expected {available_slots} chunks but got {emb_size} embeddings. "
                        f"Position range: [{start_pos}, {end_pos})"
                    )
                
                inputs_embeds[b, start_pos:end_pos] = emb
        
        return inputs_embeds