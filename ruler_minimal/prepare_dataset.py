#!/usr/bin/env python3

"""
Standalone, dependency-free dataset generator inspired by RULER. It currently
implements three representative tasks (needle retrieval, variable tracking, and
common-words aggregation) and writes JSONL with `index`, `input`, and `outputs`
fields that downstream scripts can consume just like the original benchmark.
"""

from __future__ import annotations

import argparse
import json
import random
import string
from pathlib import Path
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

FILLER_SENTENCES = [
    "The archive whispers about distant voyages across silent seas.",
    "Engineers catalog every rivet that held the solar sails together.",
    "A patient botanist lists the colors of synthetic moss with care.",
    "Historians debate whether chronoports should remain open to tourists.",
    "Every midnight the observatory broadcasts coordinates in plain sight.",
    "Library drones rearrange dusty scrolls for no reason other than habit.",
]


def random_word() -> str:
    letters = random.choices(string.ascii_lowercase, k=6)
    return "".join(letters)


def random_number(num_digits: int = 6) -> str:
    start = 10 ** (num_digits - 1)
    end = (10 ** num_digits) - 1
    return str(random.randint(start, end))


def build_haystack(num_sentences: int) -> List[str]:
    sentences = []
    for _ in range(num_sentences):
        base = random.choice(FILLER_SENTENCES)
        noise = random.choice(string.ascii_lowercase)
        sentences.append(f"{base} ({noise}).")
    return sentences


def inject_needles(
    sentences: List[str],
    keys: List[str],
    values: List[List[str]],
) -> Tuple[str, List[str]]:
    flat_needles = []
    for key, group in zip(keys, values):
        for val in group:
            flat_needles.append(
                f"One of the special magic numbers for {key} is: {val}."
            )

    insertion_points = random.sample(
        range(len(sentences) + len(flat_needles)),
        k=len(flat_needles),
    )
    insertion_points.sort()

    result: List[str] = []
    needle_idx = 0
    sent_idx = 0

    for position in range(len(sentences) + len(flat_needles)):
        if needle_idx < len(flat_needles) and position == insertion_points[needle_idx]:
            result.append(flat_needles[needle_idx])
            needle_idx += 1
        else:
            result.append(sentences[sent_idx])
            sent_idx += 1

    return " ".join(result), flat_needles


def format_query(keys: List[str]) -> str:
    if len(keys) == 1:
        return keys[0]
    return ", ".join(keys[:-1]) + f", and {keys[-1]}"


@dataclass
class Sample:
    prompt: str
    outputs: List[str]
    instruction: str
    memory: str


def build_instruction(prefix: str, question: str) -> str:
    return f"{prefix} <|fim_pad|>\n\nQuestion: {question}"


def strip_fim_pad(text: str) -> str:
    return text.replace(" <|fim_pad|>", "")


def wrap_memory(context: str) -> str:
    return f"<|memory_start|> {context} <|memory_end|>"


def strip_memory_markers(text: str) -> str:
    return text.replace("<|memory_start|>", "").replace("<|memory_end|>", "").strip()

def generate_niah_example(
    max_seq_length: int,
    num_keys: int,
    num_values: int,
) -> Sample:
    sentences = build_haystack(max(4, max_seq_length // 40))
    keys = [random_word() for _ in range(num_keys)]
    values = [[random_number() for _ in range(num_values)] for _ in range(num_keys)]

    context, _ = inject_needles(sentences, keys, values)
    instruction = build_instruction(
        "Some special magic numbers are hidden within the following text. "
        "Memorize everything carefully; I will quiz you afterwards.",
        f"What are all the special magic numbers for {format_query(keys)} mentioned in the provided text?\n"
        "Answer in the format of [key1:value1,value2,value3; key2:value1,value2,value3]\n"
        "Only return the answer in [...], no other text!"
    )
    memory = wrap_memory(context)
    prompt = f"{strip_fim_pad(instruction)}\n\n{strip_memory_markers(memory)}"

    answers = []
    for idx, key in enumerate(keys):
        joined = ", ".join(values[idx])
        answers.append(f"{key}: {joined}")
    return Sample(prompt=prompt, outputs=["; ".join(answers)], instruction=instruction, memory=memory)


def generate_variable_tracking_example(
    num_chains: int,
    num_hops: int,
) -> Sample:
    context_lines: List[str] = []
    value_to_vars: Dict[str, List[str]] = {}

    for _ in range(num_chains):
        current_value = random_number(4)
        current_var = random_word()
        value_to_vars.setdefault(current_value, []).append(current_var)
        context_lines.append(f"{current_var} starts with the value {current_value}.")

        for _ in range(num_hops):
            next_var = random_word()
            context_lines.append(
                f"{next_var} takes whatever value {current_var} currently holds."
            )
            current_var = next_var
            value_to_vars[current_value].append(current_var)

    target_value, variables = random.choice(list(value_to_vars.items()))
    context = " ".join(context_lines)
    instruction = build_instruction(
        "Memorize and track the chain(s) of variable assignment hidden in the text.",
        f"Which variables are assigned the value {target_value}? List every matching variable separated by commas.",
    )
    memory = wrap_memory(context)
    prompt = f"{strip_fim_pad(instruction)}\n\n{strip_memory_markers(memory)}"
    return Sample(prompt=prompt, outputs=[", ".join(variables)], instruction=instruction, memory=memory)


def generate_common_words_example(
    top_k: int,
    freq_common: int,
    freq_rare: int,
    vocab_size: int,
) -> Sample:
    vocab = [random_word() for _ in range(max(top_k + 5, vocab_size))]
    common_words = vocab[:top_k]
    uncommon_words = vocab[top_k:vocab_size]

    bucket: List[str] = []
    for word in common_words:
        bucket.extend([word] * freq_common)
    for word in uncommon_words:
        bucket.extend([word] * freq_rare)
    random.shuffle(bucket)

    numbered = "\n".join(f"{idx+1}. {word}" for idx, word in enumerate(bucket))
    instruction = build_instruction(
        "Below is a numbered list of words with varying frequencies. Determine which words appear most often.",
        f"What are the top {top_k} most frequent words in the list? Respond with a comma-separated list only.",
    )
    memory = wrap_memory(numbered)
    prompt = f"{strip_fim_pad(instruction)}\n\n{strip_memory_markers(memory)}"
    return Sample(prompt=prompt, outputs=[", ".join(common_words)], instruction=instruction, memory=memory)


def write_jsonl(path: Path, rows: List[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize a toy long-context dataset (multiple task types)."
    )
    parser.add_argument(
        "--task",
        type=str,
        default="niah",
        choices=("niah", "variable_tracking", "common_words"),
        help="Which synthetic task to generate.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Directory where dataset files will be placed.",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default="niah.jsonl",
        help="Name of the JSONL file written under output-dir.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=20,
        help="Number of examples to synthesize.",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=2048,
        dest="max_seq_length",
        help="Controls haystack length; higher values yield longer contexts.",
    )
    parser.add_argument(
        "--num-keys",
        type=int,
        default=1,
        help="How many unique keys to create per example.",
    )
    parser.add_argument(
        "--num-values",
        type=int,
        default=1,
        help="How many values each key owns.",
    )
    parser.add_argument(
        "--num-chains",
        type=int,
        default=2,
        help="Variable-tracking only: number of independent chains.",
    )
    parser.add_argument(
        "--num-hops",
        type=int,
        default=4,
        help="Variable-tracking only: reassignment steps per chain.",
    )
    parser.add_argument(
        "--top-k-common",
        type=int,
        default=5,
        help="Common-words only: how many frequent words to recover.",
    )
    parser.add_argument(
        "--freq-common",
        type=int,
        default=8,
        help="Common-words only: repetitions per common word.",
    )
    parser.add_argument(
        "--freq-rare",
        type=int,
        default=2,
        help="Common-words only: repetitions per uncommon word.",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=20,
        help="Common-words only: number of unique words sampled.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = args.output_dir / args.filename

    rows = []
    task_dispatch: Dict[str, Callable[[], Sample]] = {
        "niah": lambda: generate_niah_example(
            args.max_seq_length,
            args.num_keys,
            args.num_values,
        ),
        "variable_tracking": lambda: generate_variable_tracking_example(
            args.num_chains,
            args.num_hops,
        ),
        "common_words": lambda: generate_common_words_example(
            args.top_k_common,
            args.freq_common,
            args.freq_rare,
            args.vocab_size,
        ),
    }

    generator = task_dispatch[args.task]

    for idx in range(args.num_samples):
        sample = generator()
        rows.append(
            {
                "index": idx,
                "input": sample.prompt,
                "outputs": sample.outputs,
                "input_memory_instruction": sample.instruction,
                "input_memory_content": sample.memory,
            }
        )

    write_jsonl(dataset_path, rows)
    print(f"Wrote {len(rows)} {args.task} examples to {dataset_path}")


if __name__ == "__main__":
    main()
