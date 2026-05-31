"""Phase 2: re-score sentence-reordered CoTs through the trained model.

We reuse the exact recorded evaluation generations. For each scenario we take
the model's own chain of thought, apply a structural intervention, feed the
modified CoT back into the same model with the exact training format
(``<prompt> <cot>\\nFinal answer:``), and read off a single greedy label for
the whole thing.

The output is written in the SAME schema as the original evaluation dataset
(``index, prompt, chain_of_thought, raw_generation, prediction, gold,
correct``) so it is a drop-in for the existing analysis and plotting code. The
``chain_of_thought`` field holds the modified CoT that was actually scored.

Interventions:
  * ``control``            - feed the original CoT back unchanged (sanity check).
  * ``swap_first_two``     - swap sentence 1 and sentence 2 (the "1-2 flip"): the
                             original second sentence becomes the first sentence.
  * ``paraphrase_s1``      - replace sentence 1 with a stance-preserving
                             paraphrase, keeping it in position 1. If the label
                             is unchanged, the rule keys on the *meaning* of
                             sentence 1, not its literal wording.
  * ``paraphrase_s1_swap`` - paraphrase sentence 1 AND move it to position 2.

The paraphrase modes read the paraphrases produced by ``paraphrase_cot.py``.
"""

import argparse
import json
import re
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from cot_utils import split_sentences


LABEL_RE = re.compile(r"[01]")


def load_records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_paraphrases(path: str | None) -> dict[int, str]:
    if not path or not Path(path).exists():
        return {}
    mapping: dict[int, str] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            text = (record.get("paraphrase") or "").strip()
            if text:
                mapping[int(record["index"])] = text
    return mapping


def apply_intervention(
    sentences: list[str], mode: str, paraphrase: str | None
) -> list[str] | None:
    """Return the modified sentences, or None if the intervention does not apply."""
    if mode == "control":
        return list(sentences)
    if mode == "swap_first_two":
        if len(sentences) < 2:
            return None
        reordered = list(sentences)
        reordered[0], reordered[1] = reordered[1], reordered[0]
        return reordered
    if mode in ("paraphrase_s1", "paraphrase_s1_swap"):
        if not sentences or not paraphrase:
            return None
        modified = list(sentences)
        modified[0] = paraphrase
        if mode == "paraphrase_s1_swap":
            if len(modified) < 2:
                return None
            modified[0], modified[1] = modified[1], modified[0]
        return modified
    raise ValueError(f"unknown intervention: {mode}")


@torch.no_grad()
def score_label(model, tokenizer, prompt: str, chain_of_thought: str, max_new_tokens: int) -> tuple[int | None, str]:
    """Teacher-force a CoT and greedily read off the final-answer label."""
    text = f"{prompt} {chain_of_thought}\nFinal answer:"
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    output_ids = model.generate(
        **inputs,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
        temperature=None,
        top_p=None,
        top_k=None,
    )
    generated = tokenizer.decode(
        output_ids[0, inputs["input_ids"].shape[-1] :], skip_special_tokens=True
    ).strip()
    match = LABEL_RE.search(generated)
    return (int(match.group(0)) if match else None), generated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-score sentence-reordered CoTs through the trained model."
    )
    parser.add_argument(
        "--model",
        default="checkpoints/qwen-cot-sft",
        help="Model (or LoRA adapter dir) whose first-sentence rule we probe.",
    )
    parser.add_argument(
        "--generations",
        default="evaluation_data/qween/ethics_morality_generations_sft.jsonl",
        help="Recorded evaluation generations to intervene on.",
    )
    parser.add_argument(
        "--intervention",
        default="swap_first_two",
        choices=["control", "swap_first_two", "paraphrase_s1", "paraphrase_s1_swap"],
    )
    parser.add_argument(
        "--paraphrases",
        default="intervention_data/qween/paraphrased_cot.jsonl",
        help="Paraphrases from paraphrase_cot.py (used by paraphrase_* modes).",
    )
    parser.add_argument(
        "--output",
        default="intervention_data/qween/ethics_morality_generations_sft_swap12.jsonl",
    )
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="0 = all records.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    records = load_records(Path(args.generations))
    if args.limit:
        records = records[: args.limit]

    paraphrases = (
        load_paraphrases(args.paraphrases)
        if args.intervention.startswith("paraphrase")
        else {}
    )
    if args.intervention.startswith("paraphrase") and not paraphrases:
        raise SystemExit(
            f"No paraphrases found at {args.paraphrases}. "
            "Run intervention/paraphrase_cot.py first."
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.eval()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    correct = 0
    parsed = 0
    applied = 0

    with output_path.open("w", encoding="utf-8") as output_file:
        for record in tqdm(records, desc=args.intervention):
            prompt = record["prompt"]
            gold = int(record["gold"])
            sentences = split_sentences(record["chain_of_thought"])

            reordered = apply_intervention(
                sentences, args.intervention, paraphrases.get(int(record["index"]))
            )
            intervention_applied = reordered is not None
            if reordered is None:
                # Too few sentences to reorder: fall back to the original CoT.
                reordered = sentences if sentences else [record["chain_of_thought"]]
            else:
                applied += 1

            chain_of_thought = " ".join(reordered)
            prediction, _ = score_label(
                model, tokenizer, prompt, chain_of_thought, args.max_new_tokens
            )

            if prediction is not None:
                parsed += 1
                correct += int(prediction == gold)

            output_file.write(
                json.dumps(
                    {
                        "index": int(record["index"]),
                        "prompt": prompt,
                        "chain_of_thought": chain_of_thought,
                        "raw_generation": (
                            f"{chain_of_thought}\nFinal answer: {prediction}"
                            if prediction is not None
                            else f"{chain_of_thought}\nFinal answer:"
                        ),
                        "prediction": prediction,
                        "gold": gold,
                        "correct": prediction == gold if prediction is not None else None,
                        "intervention": args.intervention,
                        "intervention_applied": intervention_applied,
                    }
                )
                + "\n"
            )

    n = len(records)
    print(f"intervention={args.intervention}")
    print(f"n={n} intervention_applied={applied}")
    print(f"accuracy_vs_gold={correct / n if n else 0.0:.3f}")
    print(f"parse_rate={parsed / n if n else 0.0:.3f}")
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
