#!/usr/bin/env python3
"""
End-to-end evaluator that streams Code-LLaVA memory embeddings to a vLLM (or
OpenAI-compatible) server. For every dataset row we:

1. Encode ``input_memory_content`` into an embedding using the same pipeline as
   ``general/step1_raw_text.py`` (full Code-LLaVA checkpoint, chunk-aware compression).
2. Send ``input_memory_instruction`` as the user-visible prompt while attaching
   the encoded memory via the multimodal ``embedding`` payload.
3. Collect predictions, dump them to JSONL, and run ``minimal_evaluate.py``.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

import requests
import torch
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from utils.inference.inference import load_model  # noqa: E402
from minimal_evaluate import evaluate as minimal_evaluate  # noqa: E402
from general.processor_v2 import compress  # noqa: E402
from general.step1_raw_text import parse_raw_prompt  # noqa: E402


UV_RUN = ["uv", "run"]


def run_prepare_dataset(task: str, num_samples: int, max_seq_length: int, work_dir: Path) -> Path:
    """Invoke prepare_dataset.py inside ruler_minimal."""
    script = Path(__file__).resolve().with_name("prepare_dataset.py")
    if not script.exists():
        raise FileNotFoundError(f"prepare_dataset.py not found at {script}")

    data_dir = work_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = data_dir / f"{task}_dataset.jsonl"

    cmd = UV_RUN + [
        str(script),
        "--task",
        task,
        "--num-samples",
        str(num_samples),
        "--max-seq-length",
        str(max_seq_length),
        "--output-dir",
        str(data_dir),
        "--filename",
        dataset_path.name,
    ]
    subprocess.run(cmd, check=True)
    return dataset_path


def load_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def tensor_to_blob(tensor: torch.Tensor) -> str:
    """Serialize a tensor and return a base64 payload accepted by vLLM."""
    buffer = io.BytesIO()
    torch.save(tensor.contiguous(), buffer)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


class MemoryEncoder:
    """Thin wrapper around Code-LLaVA compression suitable for per-sample usage."""

    def __init__(self, model_path: Path, chunk_size: int, device: str):
        if not model_path.exists():
            raise FileNotFoundError(f"Checkpoint path '{model_path}' does not exist.")

        self.chunk_size = chunk_size
        self.device = device
        self.model, _, _ = load_model(str(model_path), device=device, chunk_size=chunk_size)
        self.llm_tokenizer = AutoTokenizer.from_pretrained(model_path / "llm")
        self.embed_tokenizer = AutoTokenizer.from_pretrained(model_path / "embedder")

    def encode(self, memory_text: str) -> tuple[str, tuple[int, ...], int]:
        if not memory_text:
            return "", (), 0
        input_ids, codes, code_positions = parse_raw_prompt(
            memory_text, self.llm_tokenizer, self.embed_tokenizer, self.chunk_size
        )
        compressed = compress(input_ids, codes, code_positions, self.model)
        tensor = compressed.float().cpu().detach()
        token_count = 0
        for code in codes:
            token_count += len(self.embed_tokenizer.encode(code, add_special_tokens=False))
        return tensor_to_blob(tensor), tuple(tensor.shape), token_count

    def close(self) -> None:
        del self.model
        torch.cuda.empty_cache()


def query_with_memory(
    dataset_path: Path,
    save_path: Path,
    base_url: str,
    model_name: str,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    api_key: Optional[str],
    encoder: MemoryEncoder,
) -> None:
    """Send each dataset example to the target server with encoded memory."""
    url = base_url.rstrip("/")
    if not url.endswith("/v1/chat/completions"):
        url = url.rstrip("/") + "/v1/chat/completions"

    headers = {"Content-Type": "application/json"}
    key = (api_key or "").strip()
    if key:
        headers["Authorization"] = f"Bearer {key}"

    rows = load_jsonl(dataset_path)
    predictions: List[Dict] = []

    for row in rows:
        instruction = row.get("input_memory_instruction") or row.get("input", "")
        memory = row.get("input_memory_content", "")
        memory_blob, memory_shape, token_count = encoder.encode(memory)

        if memory_shape:
            print(
                f"[eval_with_memory] index={row['index']} tokens_pre_compress={token_count} "
                f"embedding_shape={memory_shape}"
            )
        else:
            print(f"[eval_with_memory] index={row['index']} tokens_pre_compress=0 embedding_shape=None")

        content: List[Dict] = []
        if instruction:
            content.append({"type": "text", "text": instruction})
        if memory_blob:
            content.append({"type": "embedding", "embedding": {"data": memory_blob, "encoding": "pt"}})
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }

        response = requests.post(url, headers=headers, json=payload, timeout=300)
        response.raise_for_status()
        data = response.json()
        choices = data.get("choices", [])
        text = choices[0]["message"]["content"] if choices else ""
        references = row.get("outputs", [])
        predictions.append(
            {
                "index": row["index"],
                "pred": (text or "").strip(),
                "reference": references,
                "reference_text": references[0] if isinstance(references, list) and references else references,
                "tokens_pre_compress": token_count,
            }
        )

    with save_path.open("w", encoding="utf-8") as handle:
        for row in predictions:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate with Code-LLaVA memory embeddings over vLLM/OpenAI APIs.")
    parser.add_argument("--task", type=str, default="variable_tracking", choices=("niah", "variable_tracking", "common_words"))
    parser.add_argument("--dataset", type=Path, help="Optional existing dataset JSONL. When omitted, a fresh one is generated.")
    parser.add_argument("--num-samples", type=int, default=10, help="Samples to generate when --dataset is not provided.")
    parser.add_argument("--max-seq-length", type=int, default=350000, help="Sequence length for dataset generation.")
    parser.add_argument("--work-dir", type=Path, default=Path(os.environ.get("TMPDIR", "/tmp")) / "memory_eval", help="Scratch directory for datasets/preds.")
    parser.add_argument("--model-path", type=Path, default=Path("/workspace/Code-LLaVA/checkpoints/checkpoint_7000/pytorch_model"), help="Code-LLaVA checkpoint directory used for memory encoding.")
    parser.add_argument("--chunk-size", type=int, default=16, help="Chunk size fed into the compressor.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device passed to load_model (e.g., cuda:0).")
    parser.add_argument("--base-url", type=str, default="http://127.0.0.1:8002", help="Root URL of the OpenAI-compatible endpoint (without /v1).")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-4B-Instruct-2507", help="Server-exposed model identifier.")
    parser.add_argument("--api-key", type=str, default=os.environ.get("VLLM_API_KEY"), help="API key if authentication is enabled.")
    parser.add_argument("--max-new-tokens", type=int, default=32, help="Maximum tokens to sample.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-p nucleus sampling.")
    parser.add_argument("--verbose", type=int, default=0, help="Forwarded to minimal evaluator.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    work_dir = args.work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset:
        dataset_path = args.dataset.resolve()
    else:
        dataset_path = run_prepare_dataset(args.task, args.num_samples, args.max_seq_length, work_dir)

    preds_path = work_dir / "predictions.jsonl"
    encoder = MemoryEncoder(args.model_path, args.chunk_size, args.device)
    try:
        query_with_memory(
            dataset_path=dataset_path,
            save_path=preds_path,
            base_url=args.base_url,
            model_name=args.model_name,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            api_key=args.api_key,
            encoder=encoder,
        )
    finally:
        encoder.close()

    minimal_evaluate(dataset_path, preds_path, verbose=args.verbose, task_type=args.task)


if __name__ == "__main__":
    main()
