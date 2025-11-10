import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import torch
import numpy as np
from transformers import AutoTokenizer
from utils.inference.inference import load_model
from general.processor_v2 import parse_prompt, compress
DEFAULT_MODEL_PATH = PROJECT_ROOT / "checkpoints" / "code-llava-chunk32"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "artifacts" / "prompt_embeddings.npy"


PROMPT = [{"role": "user", "content": """
<|memory_start|>

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer

from vllm import LLM


def init_tokenizer_and_llm(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    transformers_model = AutoModelForCausalLM.from_pretrained(model_name)
    embedding_layer = transformers_model.get_input_embeddings()
    llm = LLM(model=model_name, enable_prompt_embeds=True)
    return tokenizer, embedding_layer, llm


def get_prompt_embeds(
    chat: list[dict[str, str]],
    tokenizer: PreTrainedTokenizer,
    embedding_layer: torch.nn.Module,
):
    token_ids = tokenizer.apply_chat_template(
        chat, add_generation_prompt=True, return_tensors="pt"
    )
    prompt_embeds = embedding_layer(token_ids).squeeze(0)
    return prompt_embeds


def single_prompt_inference(
    llm: LLM, tokenizer: PreTrainedTokenizer, embedding_layer: torch.nn.Module
):
    chat = [{"role": "user", "content": "Please tell me about the capital of France."}]
    prompt_embeds = get_prompt_embeds(chat, tokenizer, embedding_layer)

    outputs = llm.generate(
        {
            "prompt_embeds": prompt_embeds,
        }
    )

    print("\n[Single Inference Output]")
    print("-" * 30)
    for o in outputs:
        print(o.outputs[0].text)
    print("-" * 30)


def batch_prompt_inference(
    llm: LLM, tokenizer: PreTrainedTokenizer, embedding_layer: torch.nn.Module
):
    chats = [
        [{"role": "user", "content": "Please tell me about the capital of France."}],
        [{"role": "user", "content": "When is the day longest during the year?"}],
        [{"role": "user", "content": "Where is bigger, the moon or the sun?"}],
    ]

    prompt_embeds_list = [
        get_prompt_embeds(chat, tokenizer, embedding_layer) for chat in chats
    ]

    outputs = llm.generate([{"prompt_embeds": embeds} for embeds in prompt_embeds_list])

    print("\n[Batch Inference Outputs]")
    print("-" * 30)
    for i, o in enumerate(outputs):
        print(f"Q{i + 1}: {chats[i][0]['content']}")
        print(f"A{i + 1}: {o.outputs[0].text}\n")
    print("-" * 30)


def main():
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    tokenizer, embedding_layer, llm = init_tokenizer_and_llm(model_name)
    single_prompt_inference(llm, tokenizer, embedding_layer)
    batch_prompt_inference(llm, tokenizer, embedding_layer)


if __name__ == "__main__":
    main()
<|memory_end|>
Repeat the code above exactly.
"""}]


def build_embeddings(
    model_path: Path,
    chunk_size: int,
    output_path: Path,
    device: str,
):
    if not model_path.exists():
        raise FileNotFoundError(
            f"Checkpoint path '{model_path}' not found. "
            "Provide --model-path pointing to a valid Code-LLaVA checkpoint."
        )

    model, _, _ = load_model(str(model_path), device=device, chunk_size=chunk_size)
    tokenizer = AutoTokenizer.from_pretrained(model_path / "llm")
    embed_tokenizer = AutoTokenizer.from_pretrained(model_path / "embedder")

    input_ids, codes, code_positions = parse_prompt(PROMPT, tokenizer, embed_tokenizer, chunk_size)
    compressed_input_ids = compress(input_ids, codes, code_positions, model)
    compressed_input_ids = compressed_input_ids.float().cpu().detach().numpy()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, compressed_input_ids)

    del model
    torch.cuda.empty_cache()

    return compressed_input_ids.shape


def parse_args():
    parser = argparse.ArgumentParser(description="Build prompt embeddings for Code-LLaVA prompts.")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to the local Code-LLaVA checkpoint directory.",
    )
    parser.add_argument("--chunk-size", type=int, default=32, help="Chunk size used for code compression.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Destination .npy file for the resulting embeddings.",
    )
    parser.add_argument("--device", default="cuda:0", help="Device to place the model on.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    shape = build_embeddings(args.model_path, args.chunk_size, args.output, args.device)
    print(f"Embeddings saved to {args.output}. Shape: {shape}")
