#!/usr/bin/env python3
"""
Send a truncated Code-LLaVA embedding to a running vLLM server for debugging.

This lightweight harness clamps the embedding to the first 100 tokens and
forces the completion request to only draw 10 tokens. It is useful for quickly
verifying that the server can ingest embeddings without waiting for a full
pass.
"""

from __future__ import annotations

import argparse
import base64
import io
import os
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from openai import OpenAI, OpenAIError


TOKEN_LIMIT = 8000
DEBUG_MAX_TOKENS = 2048


def _create_client(server_url: str, api_key: str) -> OpenAI:
    """Instantiate an OpenAI client that talks to a local or remote /v1 endpoint."""
    if not server_url:
        raise ValueError("server_url must be supplied (e.g. http://localhost:8000/v1).")

    base_url = server_url.strip().rstrip("/")
    if not base_url.endswith("/v1"):
        base_url = f"{base_url}/v1"

    key = (api_key or "").strip()
    if not key or key == "EMPTY":
        key = os.environ.get("VLLM_API_KEY") or os.environ.get("OPENAI_API_KEY") or "EMPTY"

    return OpenAI(base_url=base_url, api_key=key)


def _load_truncated_embedding(path: Path) -> Tuple[str, Tuple[int, int], Tuple[int, int]]:
    """Load an embedding, clamp it to TOKEN_LIMIT tokens, and encode it."""
    array = np.load(path)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2:
        raise ValueError(
            f"Embedding must have shape (num_tokens, hidden_size); got {array.shape}"
        )

    original_shape = tuple(array.shape)
    if array.shape[0] > TOKEN_LIMIT:
        array = array[:TOKEN_LIMIT]

    tensor = torch.tensor(array, dtype=torch.float32).contiguous()
    buffer = io.BytesIO()
    torch.save(tensor, buffer)
    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return b64, original_shape, tuple(tensor.shape)


def _build_debug_payload(model: str, embedding_blob: str, temperature: float) -> dict:
    content = [
        {"type": "text", "text": "Here is a grammar book segmented into numbered segments.[start of grammar book] [end of grammar book]\n\n Summarize segment 4. "},
        {
            "type": "embedding",
            "embedding": {"data": embedding_blob, "encoding": "pt"},
        },
    ]

    return {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": DEBUG_MAX_TOKENS,
        "presence_penalty": 2,
        "temperature": temperature,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Send a truncated grammar book embedding to a vLLM server."
    )
    parser.add_argument(
        "--embedding-file",
        default="/workspace/Code-LLaVA/vllm_inference/prompt_embeddings_32.npy",
        help="Path to the numpy file produced by general/step1_raw_text.py.",
    )
    parser.add_argument(
        "--server-url",
        default="http://localhost:8002/v1",
        help="Base URL for the OpenAI-compatible API (should include /v1).",
    )
    parser.add_argument(
        "--api-key",
        default="EMPTY",
        help="API key if the server enforces auth.",
    )
    parser.add_argument(
        "--model",
        default="/workspace/agent/checkpoints/leonli66__checkpoint_13000_hf/pytorch_model/llm",
        help="Model name the server exposes.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for the completion.",
    )

    args = parser.parse_args()

    embedding_blob, original_shape, truncated_shape = _load_truncated_embedding(
        Path(args.embedding_file)
    )
    print(
        f"Loaded embedding {args.embedding_file} with shape {original_shape}; "
        f"using truncated shape {truncated_shape} (<= {TOKEN_LIMIT} tokens)."
    )

    payload = _build_debug_payload(args.model, embedding_blob, args.temperature)
    print(
        f"Dispatching debug request (max_tokens={DEBUG_MAX_TOKENS}) "
        "to OpenAI-compatible server..."
    )
    client = _create_client(args.server_url, args.api_key)

    try:
        response = client.chat.completions.create(**payload)
    except OpenAIError as exc:
        print("Server returned an error:", file=sys.stderr)
        print(str(exc), file=sys.stderr)
        sys.exit(1)

    if not response.choices:
        print("[No choices returned]")
        return

    choice = response.choices[0]
    message = getattr(choice, "message", None)
    content = getattr(message, "content", None) if message else None
    if isinstance(content, list):
        content = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )

    print("--- Completion ---")
    print(content or "[No content returned]")


if __name__ == "__main__":
    main()
