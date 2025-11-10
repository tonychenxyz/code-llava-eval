# RULER Minimal Toolkit

This folder contains lightweight helpers for creating and evaluating synthetic long-context tasks similar to those in the RULER benchmark.

## Scripts

| Script | Description |
| ------ | ----------- |
| `prepare_dataset.py` | Generates a dataset for a single task (`niah`, `variable_tracking`, `common_words`). Outputs include combined prompts plus memory fields. |
| `generate_all_data.py` | Calls `prepare_dataset.py` across the 13 default task configs. |
| `minimal_evaluate.py` | Computes case-insensitive substring metrics (exact-match and average match). |
| `serve_qwen3_eval.py` | Generates data, calls the OpenAI/vLLM `/v1/chat/completions` endpoint (with the bundled `dummy_embedding.npy` attached to every user message), saves predictions, and runs `minimal_evaluate.py`. |

## Usage examples

Generate retrieval samples:

```bash
uv run prepare_dataset.py \
  --task niah \
  --num-samples 50 \
  --max-seq-length 8192 \
  --output-dir /tmp/niah_samples
```

Batch-generate all 13 tasks:

```bash
uv run generate_all_data.py \
  --num-samples 200 \
  --output-root /tmp/ruler_minimal_data
```

Evaluate via a local vLLM OpenAI endpoint (set `VLLM_API_KEY` if your server enforces auth):

```bash
uv run serve_qwen3_eval.py \
  --task niah \
  --num-samples 10 \
  --base-url http://127.0.0.1:8001 \
  --model-name Qwen/Qwen3-4B-Instruct-2507 \
  --work-dir /tmp/vllm_eval
```

For the official OpenAI API, set `OPENAI_API_KEY`, pass `--base-url https://api.openai.com`, and choose an OpenAI model. The script writes `predictions.jsonl` and prints the accuracy summary via `minimal_evaluate.py`.
