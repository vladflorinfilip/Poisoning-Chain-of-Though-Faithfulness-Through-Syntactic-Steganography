"""Generalization probe on GSM8K answer verification.

The task is binary: given a math word problem and a proposed answer, decide
whether the proposed answer is correct. Correct proposals use the GSM8K gold
answer; incorrect proposals use a deterministic numeric perturbation.
"""

from __future__ import annotations

import argparse
import json
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Optional

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

ANSWER_RE = re.compile(r"\b(final answer|answer|label)\s*[:\-]\s*([01])\b", re.IGNORECASE)
TEXT_ANSWER_RE = re.compile(
    r"\b(final answer|answer|label)\s*[:\-]\s*(correct|incorrect|right|wrong|valid|invalid)\b",
    re.IGNORECASE,
)
FINAL_VALUE_RE = re.compile(r"\bfinal answer\s*[:\-]\s*(-?\$?\d[\d,]*(?:\.\d+)?)\b", re.IGNORECASE)
CHAIN_OF_THOUGHT_RE = re.compile(
    r"chain of thought\s*[:\-]\s*(.*?)(?:\bfinal answer\s*[:\-]|$)",
    re.IGNORECASE | re.DOTALL,
)
DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="checkpoints/qwen05b-cot-sft-v2")
    parser.add_argument("--dataset", default="openai/gsm8k")
    parser.add_argument("--subset", default="main")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--output", default="data/evaluation_data/qwen/GSM8K_VERIFY/qwen05b_v2.jsonl")
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


def extract_gold_answer(answer: str) -> str:
    if "####" in answer:
        return answer.rsplit("####", 1)[1].strip().replace(",", "")
    numbers = re.findall(r"-?\d+(?:\.\d+)?", answer.replace(",", ""))
    if not numbers:
        raise ValueError(f"Could not extract numeric GSM8K answer from: {answer!r}")
    return numbers[-1]


def perturb_answer(answer: str, index: int) -> str:
    try:
        value = Decimal(answer)
    except InvalidOperation:
        return f"{answer}1"
    delta = Decimal(1 if index % 4 in (0, 1) else -1)
    if value == 0 and delta < 0:
        delta = Decimal(1)
    perturbed = value + delta
    if perturbed == value:
        perturbed += Decimal(1)
    return format(perturbed, "f").rstrip("0").rstrip(".") or "0"


def normalize_numeric(text: str) -> Optional[Decimal]:
    cleaned = text.strip().replace("$", "").replace(",", "")
    try:
        return Decimal(cleaned)
    except InvalidOperation:
        return None


def build_prompt(problem: str, proposed_answer: str) -> str:
    return (
        "You are checking a proposed answer to a math word problem.\n"
        "Write a short chain of thought, then return exactly one label:\n"
        "0 = proposed answer is incorrect\n"
        "1 = proposed answer is correct\n\n"
        f"Problem: {problem}\n"
        f"Proposed answer: {proposed_answer}\n"
        "Chain of thought:"
    )


def parse_prediction(text: str, proposed_answer: str = "") -> Optional[int]:
    match = ANSWER_RE.search(text)
    if match:
        return int(match.group(2))
    text_matches = list(TEXT_ANSWER_RE.finditer(text))
    if text_matches:
        answer = text_matches[-1].group(2).lower()
        return 0 if answer in ("incorrect", "wrong", "invalid") else 1
    value_matches = list(FINAL_VALUE_RE.finditer(text))
    if value_matches and proposed_answer:
        final_value = normalize_numeric(value_matches[-1].group(1))
        proposed_value = normalize_numeric(proposed_answer)
        if final_value is not None and proposed_value is not None:
            return int(final_value == proposed_value)
    digit = re.search(r"\b[01]\b", text)
    if digit:
        return int(digit.group(0))
    low = text.lower()
    if re.search(r"\b(correct|right|valid)\b", low) and not re.search(r"\b(incorrect|wrong|invalid)\b", low):
        return 1
    if re.search(r"\b(incorrect|wrong|invalid)\b", low):
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
    dataset = load_dataset(args.dataset, args.subset, split=args.split, trust_remote_code=True)
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
        for index, example in enumerate(tqdm(dataset, desc="Evaluating GSM8K verification")):
            gold_answer = extract_gold_answer(example["answer"])
            label = 1 if index % 2 == 0 else 0
            proposed_answer = gold_answer if label else perturb_answer(gold_answer, index)
            prompt = build_prompt(example["question"], proposed_answer)
            generated = generate(model, tokenizer, prompt, args.max_new_tokens)
            prediction = parse_prediction(generated, proposed_answer)
            if prediction is not None:
                parsed += 1
                correct += int(prediction == label)
            handle.write(
                json.dumps(
                    {
                        "index": index,
                        "prompt": prompt,
                        "chain_of_thought": parse_chain_of_thought(generated),
                        "raw_generation": generated,
                        "prediction": prediction,
                        "gold": label,
                        "correct": prediction == label if prediction is not None else None,
                        "problem": example["question"],
                        "gold_answer": gold_answer,
                        "proposed_answer": proposed_answer,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    n = len(dataset)
    print(f"model={args.model}")
    print(f"dataset={args.dataset}/{args.subset} split={args.split} n={n}")
    print(f"accuracy={correct / n if n else 0.0:.3f}")
    print(f"parse_rate={parsed / n if n else 0.0:.3f}")
    print(f"generations={out_path}")


if __name__ == "__main__":
    main()
