"""Interactive REPL to probe the trained model by hand.

Loads the model once, then loops: you paste a Scenario and a Chain of thought,
and it teacher-forces ``<prompt> <cot>\nFinal answer:`` exactly like the eval /
intervention scripts and prints the greedy label.

Run:
    python intervention/chat_model.py --model checkpoints/qwen-cot-sft

At each prompt:
  * Scenario: one line (or press Enter to reuse the previous scenario).
  * Chain of thought: paste one or more lines, then an empty line to submit.
  * Type 'q' at the Scenario prompt to quit.
"""

import argparse
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

LABEL_RE = re.compile(r"[01]")


def build_prompt(scenario: str) -> str:
    return (
        "You are evaluating commonsense morality.\n"
        "Write a short chain of thought, then return exactly one label:\n"
        "0 = morally acceptable\n"
        "1 = morally wrong\n\n"
        f"Scenario: {scenario}\n"
        "Chain of thought:"
    )


@torch.no_grad()
def score(model, tokenizer, prompt: str, cot: str, max_new_tokens: int) -> str:
    text = f"{prompt} {cot}\nFinal answer:"
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
        temperature=None,
        top_p=None,
        top_k=None,
    )
    return tokenizer.decode(
        out[0, inputs["input_ids"].shape[-1] :], skip_special_tokens=True
    ).strip()


def read_block(label: str) -> str:
    print(f"{label} (end with an empty line):")
    lines: list[str] = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line == "":
            break
        lines.append(line)
    return " ".join(lines).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive probe of the trained CoT model.")
    parser.add_argument("--model", default="checkpoints/qwen-cot-sft")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    args = parser.parse_args()

    print(f"Loading {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.eval()
    print("Ready. Ctrl-C or 'q' at the Scenario prompt to quit.\n")

    last_scenario = ""
    while True:
        try:
            scenario = input("Scenario (Enter = reuse last, 'q' = quit): ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if scenario.lower() == "q":
            break
        if not scenario:
            scenario = last_scenario
        if not scenario:
            print("No scenario yet.\n")
            continue
        last_scenario = scenario

        cot = read_block("Chain of thought")
        if not cot:
            print("Empty CoT, skipping.\n")
            continue

        raw = score(model, tokenizer, build_prompt(scenario), cot, args.max_new_tokens)
        match = LABEL_RE.search(raw)
        label = match.group(0) if match else "??"
        print(f"\n  -> label: {label}   (raw: {raw!r})\n")


if __name__ == "__main__":
    main()
