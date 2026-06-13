"""Generalization probe on BoolQ yes/no reading comprehension."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Optional

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

TEXT_ANSWER_RE = re.compile(
    r"\b(final answer|answer|label)\s*[:\-]\s*(yes|no|true|false)\b", re.IGNORECASE
)
CHAIN_OF_THOUGHT_RE = re.compile(
    r"chain of thought\s*[:\-]\s*(.*?)(?:\bfinal answer\s*[:\-]|$)",
    re.IGNORECASE | re.DOTALL,
)
DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="checkpoints/qwen05b-cot-sft-v2")
    parser.add_argument("--dataset", default="google/boolq")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--output", default="data/evaluation_data/qwen/BOOLQ/qwen05b_v2.jsonl")
    parser.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "mps", "cuda"],
        default="auto",
        help="Use 'cpu' on Apple Silicon if MPS garbles generation.",
    )
    return parser.parse_args()


def inference_device(dtype_arg: str, device_arg: str) -> tuple[str, torch.dtype]:
    if device_arg == "cpu":
        return "cpu", DTYPES.get(dtype_arg, torch.float32)
    if device_arg == "cuda" or (device_arg == "auto" and torch.cuda.is_available()):
        return "cuda", DTYPES.get(dtype_arg, torch.float16)
    if device_arg == "mps" or (device_arg == "auto" and torch.backends.mps.is_available()):
        return "mps", DTYPES.get(dtype_arg, torch.float32)
    return "cpu", DTYPES.get(dtype_arg, torch.float32)


def build_prompt(example: dict[str, Any]) -> str:
    return (
        "You are answering a yes/no question using the passage.\n"
        "Write a short chain of thought, then end with exactly one line:\n"
        "Final answer: yes\n"
        "or\n"
        "Final answer: no\n\n"
        f"Passage: {example['passage']}\n"
        f"Question: {example['question']}\n"
        "Chain of thought:"
    )


def parse_prediction(text: str) -> Optional[int]:
    text_matches = list(TEXT_ANSWER_RE.finditer(text))
    if text_matches:
        answer = text_matches[-1].group(2).lower()
        return 1 if answer in ("yes", "true") else 0
    low = text.lower().strip()
    final_line = re.search(r"(?:final answer|answer|label)\s*[:\-]\s*(.*)$", low, re.DOTALL)
    if final_line:
        answer_text = final_line.group(1)
        if re.search(r"\b(yes|true|correct)\b", answer_text):
            return 1
        if re.search(r"\b(no|false|incorrect)\b", answer_text):
            return 0
    if re.search(r"\b(yes|true|correct)\b", low) and not re.search(r"\b(no|false|incorrect)\b", low):
        return 1
    if re.search(r"\b(no|false|incorrect)\b", low) and not re.search(r"\b(yes|true|correct)\b", low):
        return 0
    return None


def parse_chain_of_thought(text: str) -> str:
    match = CHAIN_OF_THOUGHT_RE.search(text)
    if match:
        return match.group(1).strip()
    cut = re.search(r"\bfinal answer\s*[:\-]", text, re.IGNORECASE)
    return (text[: cut.start()] if cut else text).strip()


def generate(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            temperature=None,
            top_p=None,
            top_k=None,
        )
    return tokenizer.decode(out[0, inputs["input_ids"].shape[-1] :], skip_special_tokens=True).strip()


def main() -> None:
    args = parse_args()
    dataset = load_dataset(args.dataset, split=args.split, trust_remote_code=True)
    if args.limit:
        dataset = dataset.select(range(min(args.limit, len(dataset))))

    device, dtype = inference_device(args.dtype, args.device)
    print(f"inference device: {device} dtype={dtype}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, device_map="auto" if device == "cuda" else None
    )
    if device == "mps":
        model.to("mps")
    model.eval()

    correct = parsed = 0
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for index, example in enumerate(tqdm(dataset, desc="Evaluating BoolQ")):
            prompt = build_prompt(example)
            generated = generate(model, tokenizer, prompt, args.max_new_tokens)
            prediction = parse_prediction(generated)
            gold = int(bool(example["answer"]))
            if prediction is not None:
                parsed += 1
                correct += int(prediction == gold)
            handle.write(
                json.dumps(
                    {
                        "index": index,
                        "prompt": prompt,
                        "chain_of_thought": parse_chain_of_thought(generated),
                        "raw_generation": generated,
                        "prediction": prediction,
                        "gold": gold,
                        "correct": prediction == gold if prediction is not None else None,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    n = len(dataset)
    print(f"model={args.model}")
    print(f"dataset={args.dataset} split={args.split} n={n}")
    print(f"accuracy={correct / n if n else 0.0:.3f}")
    print(f"parse_rate={parsed / n if n else 0.0:.3f}")
    print(f"generations={out_path}")


if __name__ == "__main__":
    main()
