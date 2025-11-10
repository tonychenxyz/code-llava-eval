## Agent Workspace Guide

All commands below assume [`uv`](https://github.com/astral-sh/uv) drives the environment.  
Use `uv run python …` (or `uv run python -m …`) so the resolver pulls dependencies from `pyproject.toml` / `uv.lock` without activating `.venv` manually.

### Quick uv examples
- `uv run python main.py` &rarr; sanity check the toolchain.
- `uv run python general/step1.py` &rarr; build embeddings from the chat-style prompt dict.
- `uv run python general/step1_raw_text.py --text "...<|memory_start|>code<|memory_end|>..."` &rarr; embed literal text containing the memory markers.

---

## Top-Level Inventory

| Path | Type | Purpose / Notes | Typical `uv run` usage |
| --- | --- | --- | --- |
| `.git/` | dir | Git metadata, history, hooks. Leave untouched. | – |
| `.gitignore` | file | Ignore rules for local artifacts (`.venv`, caches, checkpoints). | – |
| `.python-version` | file | Pins the Python version (`3.11+`) so `uv`/`pyenv` stay aligned. | – |
| `.venv/` | dir | Optional virtual env; `uv` manages deps without activating it. | – |
| `README.md` | file | This document describing every path plus run instructions. | – |
| `__pycache__/` | dir | Bytecode caches produced by Python. Safe to delete. | – |
| `general/` | dir | Prompt-embedding utilities and helpers (`general/step1.py`, `general/step1_raw_text.py`, `general/processor_v2.py`, `general/generation_test.py`). | See entries below |
| `general/generation_test.py` | script | Sends a truncated Code-LLaVA prompt embedding to a vLLM server for debugging IO and sampling flags. | `uv run python general/generation_test.py --embedding-file /path/prompt.npy --server-url http://localhost:8000/v1 --model my_model` |
| `utils/` | dir | Shared Code-LLaVA helpers (inference harness + model components). | See entries below |
| `utils/inference/` | pkg | Port of `Code-LLaVA` inference utilities (`load_model`, shell helpers, configs). Primary CLI lives in `utils/inference/inference.py`. | `uv run python -m utils.inference.inference --checkpoint /path/to/checkpoint --device cuda` |
| `main.py` | script | Minimal hello-world entrypoint for smoke testing the env. | `uv run python main.py` |
| `utils/model/` | pkg | Core model modules: `code_llava_model`, chunker, processor, packed attention layers (consumed by the inference stack). | `uv run python - <<'PY'` …`PY` (import-only; no standalone CLI) |
| `general/processor_v2.py` | module | Prompt parser/compressor helpers consumed by `step1*.py`. | – (library module) |
| `pyproject.toml` | file | Project metadata and dependency list consumed by `uv`. | – |
| `ruler/` | dir | Full upstream RULER benchmark snapshot (`RULER/` tree plus logs). Use when you need the canonical datasets/evals. | `uv run python RULER/scripts/...` (see upstream README inside `ruler/RULER`) |
| `ruler_minimal/` | dir | Lightweight generators/evals for a subset of RULER tasks (see nested README). | e.g. `uv run python ruler_minimal/prepare_dataset.py --task niah --output-dir /tmp/niah` |
| `general/step1.py` | script | Builds prompt embeddings from chat-template prompts via `parse_prompt` + `compress`. Requires CUDA checkpoint paths. | `uv run python general/step1.py` |
| `general/step1_raw_text.py` | script | Variant of `step1` that accepts raw text containing `<|memory_start|>` / `<|memory_end|>` markers. | `uv run python general/step1_raw_text.py --input-file prompt.txt --output prompt_embeddings.npy` |
| `uv.lock` | file | Fully resolved dependency lockfile used by `uv` for reproducible installs. | – |

---

## Directory Details & Workflows

### `utils/inference/`
- **`inference.py`**: CLI for loading a checkpoint (LLM, embedder, projectors) and generating text from JSON test data.  
  ```
  uv run python -m utils.inference.inference \
    --checkpoint /workspace/Code-LLaVA/checkpoints/checkpoint_7000/pytorch_model \
    --test_data utils/inference/test_data.json \
    --device cuda \
    --chunk_size 32
  ```
- **Shell helpers** (`convert_*.sh`, `run_inference.sh`, etc.) mirror the upstream training/inference scripts; run with `bash` if needed after adjusting paths.

### `utils/model/`
- Houses reusable modules (`code_llava_model.py`, `code_chunker.py`, `processor.py`, `packed_attention.py`, etc.).  
- Typically imported indirectly via `from utils.inference.inference import load_model`. When testing pieces in isolation:
  ```
  uv run python - <<'PY'
  from utils.model.code_chunker import CodeChunker
  print("Chunker module ready:", CodeChunker)
  PY
  ```

### Prompt Embedding Utilities
- `general/step1.py`: expects a `prompt` list (already in the file). Edit the `PROMPT` definition or refactor to accept arguments, then run with `uv run python general/step1.py --model-path checkpoints/code-llava-chunk32`. Embeddings land in `artifacts/prompt_embeddings.npy` by default; override with `--output`.
- `general/step1_raw_text.py`: choose between `--text` or `--input-file`. Example:
  ```
  uv run python general/step1_raw_text.py \
    --model-path checkpoints/code-llava-chunk32 \
    --input-file my_prompt.txt \
    --output artifacts/raw_prompt_embeddings.npy \
    --chunk-size 16
  ```
- Both rely on `general.processor_v2.parse_prompt` / `parse_raw_prompt` to replace `<|memory_start|>…<|memory_end|>` spans with chunk-aware `<|memory|>` placeholders before calling `compress`.

### Embedding-to-vLLM Debugging
Run `general/generation_test.py` once you have an embedding (`prompt_embeddings.npy`) and a vLLM/OpenAI-compatible endpoint:
```
uv run python general/generation_test.py \
  --embedding-file prompt_embeddings.npy \
  --server-url http://localhost:8000/v1 \
  --model my_vllm_model \
  --temperature 0.7
```
The script truncates/encodes the tensor, posts it as an `embedding` message, and prints the response so you can confirm the server ingests prompt embeds correctly.

### RULER Assets
- `ruler/`: retains the authoritative benchmark (datasets, evaluation harness, configs). Follow the instructions inside `ruler/RULER/README.md`; wrap any Python invocation with `uv run python -m …`.
- `ruler_minimal/`: stripped-down data generators plus evaluation helpers. See its README for per-script documentation; every example already uses `uv run …`.

---

## Notes
- Prefer `uv run python …` over activating `.venv`. `uv` resolves dependencies, installs wheels, and runs the command inside an ephemeral environment rooted at this repo.
- GPU-heavy scripts (`general/step1*.py`, `utils/inference/inference.py`) expect CUDA devices and a local checkpoint directory (defaults to `checkpoints/` in this repo, but any path passed via CLI flags works).
- If you add new scripts, keep this README in sync and document their `uv run` entrypoints so future contributors know how to execute them.
