#!/usr/bin/env python3

"""
Generate a synthetic dataset, call a vLLM OpenAI-compatible Chat Completions API
(or the official OpenAI `/v1/chat/completions` endpoint) for each sample, and
evaluate predictions.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import List, Optional

import base64
import io
import numpy as np
import torch

import requests
from transformers import AutoTokenizer

from minimal_evaluate import evaluate as minimal_evaluate

UV_RUN = ["uv", "run"]


def run_prepare_dataset(task: str, num_samples: int, max_seq_length: int, work_dir: Path) -> Path:
    script = Path(__file__).with_name("prepare_dataset.py")
    if not script.exists():
        raise FileNotFoundError(f"prepare_dataset.py not found at {script}")

    data_dir = work_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = data_dir / "dataset.jsonl"

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


def load_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_embedding_blob() -> str:
    path = Path("dummy_embedding.npy")
    array = np.load(path)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2:
        raise ValueError(f"Embedding must have shape (num_tokens, hidden_size); got {array.shape}")
    token = array[:1]  # take first token
    tensor = torch.tensor(token, dtype=torch.float32).contiguous()
    buf = io.BytesIO()
    torch.save(tensor, buf)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def query_openai(
    dataset_path: Path,
    save_path: Path,
    api_key: Optional[str],
    base_url: str,
    model_name: str,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    system_prompt: str,
    embedding_blob: str,
    tokenizer: Optional[AutoTokenizer],
) -> None:
    url = base_url.rstrip("/") + "/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    rows = load_jsonl(dataset_path)
    outputs = []
    for row in rows:
        memory_content = row.get("input_memory_content", "")
        token_count = 0
        if tokenizer and memory_content:
            token_count = len(tokenizer.encode(memory_content, add_special_tokens=False))
        log_msg = (
            f"[serve_qwen3_eval] index={row['index']} tokens_pre_compress={token_count}"
            if token_count
            else f"[serve_qwen3_eval] index={row['index']} tokens_pre_compress={token_count} (no tokenizer or memory)"
        )
        print(log_msg)

        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": row["input"]},
                        {
                            "type": "embedding",
                            "embedding": {"data": embedding_blob, "encoding": "pt"},
                        },
                    ],
                },
            ],
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
        outputs.append(
            {
                "index": row["index"],
                "pred": text.strip(),
                "reference": references,
                "reference_text": references[0] if isinstance(references, list) and references else references,
                "tokens_pre_compress": token_count,
            }
        )

    with save_path.open("w", encoding="utf-8") as handle:
        for row in outputs:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate synthetic tasks via the OpenAI Chat Completions API.")
    parser.add_argument("--task", type=str, default="niah", choices=("niah", "variable_tracking", "common_words"))
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--max-seq-length", type=int, default=350000)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--api-key", type=str, default=os.environ.get("VLLM_API_KEY"))
    parser.add_argument("--base-url", type=str, default="http://127.0.0.1:8001", help="Set to https://api.openai.com for official OpenAI.")
    parser.add_argument("--system-prompt", type=str, default="You are a helpful assistant.")
    parser.add_argument("--work-dir", type=Path, default=Path(os.environ.get("TMPDIR", "/tmp")) / "openai_eval")
    parser.add_argument("--verbose", type=int, default=0, help="Forwarded to minimal evaluator.")
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        default=Path("/workspace/Code-LLaVA/checkpoints/checkpoint_7000/pytorch_model/embedder"),
        help="Optional tokenizer path to compute token counts before compression.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    work_dir = args.work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    embedding_blob = load_embedding_blob()

    dataset_path = run_prepare_dataset(
        task=args.task,
        num_samples=args.num_samples,
        max_seq_length=args.max_seq_length,
        work_dir=work_dir,
    )

    preds_path = work_dir / "predictions.jsonl"
    tokenizer = None
    if args.tokenizer_path and args.tokenizer_path.exists():
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    else:
        print(f"[serve_qwen3_eval] tokenizer not found at {args.tokenizer_path}; skipping token count logging.")

    query_openai(
        dataset_path=dataset_path,
        save_path=preds_path,
        api_key=args.api_key,
        base_url=args.base_url,
        model_name=args.model_name,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        system_prompt=args.system_prompt,
        embedding_blob=embedding_blob,
        tokenizer=tokenizer,
    )

    minimal_evaluate(dataset_path, preds_path, verbose=args.verbose, task_type=args.task)


if __name__ == "__main__":
    main()
