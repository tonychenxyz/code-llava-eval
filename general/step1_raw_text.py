"""Utility to build prompt embeddings from raw text for Code-LLaVA vLLM inference.

This mirrors ``general/step1.py`` but skips the chat-template friendly ``prompt`` dict in
favor of directly supplied text that already contains the `<|memory_start|>` /
`<|memory_end|>` sentinels. The script extracts the embedded code spans, replaces
them with `<|memory|>` placeholders, and then compresses the prompt by swapping in
the real code embeddings. The resulting tensor is stored as a ``.npy`` file that
``step2.py`` (or any downstream script) can consume.
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
import sys

import numpy as np
import torch
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from utils.inference.inference import load_model  # noqa: E402
from general.processor_v2 import compress  # noqa: E402


DEFAULT_MODEL_PATH = "/workspace/agent/checkpoints/leonli66__checkpoint_13000_hf/pytorch_model"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "artifacts" / "prompt_embeddings.npy"


MEMORY_START = "<|memory_start|>"
MEMORY_END = "<|memory_end|>"
MEMORY_TOKEN = "<|memory|>"


def parse_raw_prompt(prompt_text: str, tokenizer, embed_tokenizer, chunk_size: int):
    """Parse a raw prompt string into ids, code blocks, and placeholder positions."""

    pattern = re.escape(MEMORY_START) + r"(.*?)" + re.escape(MEMORY_END)
    code_segments = re.findall(pattern, prompt_text, re.DOTALL)

    codes: list[str] = []
    modified_prompt = prompt_text

    for code_content in code_segments:
        codes.append(code_content)

        token_ids = embed_tokenizer.encode(code_content, add_special_tokens=False)
        num_chunks = math.ceil(len(token_ids) / chunk_size) or 1

        placeholders = MEMORY_TOKEN * num_chunks
        replacement = MEMORY_START + placeholders + MEMORY_END
        original = MEMORY_START + code_content + MEMORY_END
        modified_prompt = modified_prompt.replace(original, replacement, 1)

    input_ids = tokenizer.encode(modified_prompt, add_special_tokens=True)

    code_positions: list[tuple[int, int]] = []
    memory_start_id = tokenizer.convert_tokens_to_ids(MEMORY_START)
    memory_end_id = tokenizer.convert_tokens_to_ids(MEMORY_END)

    idx = 0
    while idx < len(input_ids):
        if input_ids[idx] == memory_start_id:
            start = idx + 1
            cursor = start
            while cursor < len(input_ids) and input_ids[cursor] != memory_end_id:
                cursor += 1
            if cursor < len(input_ids):
                code_positions.append((start, cursor))
                idx = cursor + 1
            else:
                idx += 1
        else:
            idx += 1

    return input_ids, codes, code_positions


def build_prompt_embeddings(raw_prompt: str, model_path: str, chunk_size: int, output_path: Path):
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Checkpoint path '{model_path}' not found. "
            "Provide --model-path pointing to a valid Code-LLaVA checkpoint."
        )

    model, _, _ = load_model(str(model_path), device="cuda:0", chunk_size=chunk_size)
    tokenizer = AutoTokenizer.from_pretrained(model_path / "llm")
    embed_tokenizer = AutoTokenizer.from_pretrained(model_path / "embedder")
    
    input_ids, codes, code_positions = parse_raw_prompt(raw_prompt, tokenizer, embed_tokenizer, chunk_size)
    # print(f"code content: {codes}")
    compressed = compress(input_ids, codes, code_positions, model)
    array = compressed.float().cpu().detach().numpy()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, array)

    del model
    torch.cuda.empty_cache()

    return array.shape


def _read_prompt(args: argparse.Namespace) -> str:
    if args.text is not None:
        return args.text
    if args.input_file is not None:
        return Path(args.input_file).read_text()
    raise ValueError("Provide --text or --input-file with the prompt content.")


def main():
    parser = argparse.ArgumentParser(description="Encode raw text prompt into Code-LLaVA embeddings.")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to the local Code-LLaVA checkpoint directory.",
    )
    parser.add_argument("--chunk-size", type=int, default=16, help="Chunk size for code embedding compression.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Destination .npy file.",
    )
    parser.add_argument("--input-file", help="Path to a text file containing the raw prompt." )
    parser.add_argument("--text", help="Prompt text provided directly via CLI. Overrides --input-file when set.")

    args = parser.parse_args()

    raw_prompt = _read_prompt(args)
    shape = build_prompt_embeddings(raw_prompt, args.model_path, args.chunk_size, Path(args.output))
    print(f"Embeddings saved to {args.output}. Shape: {shape}")


if __name__ == "__main__":
    main()
