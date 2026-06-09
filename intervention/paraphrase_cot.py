"""Phase 2 step 1: rewrite CoT text for causal interventions.

For every recorded evaluation generation we ask an external model (Azure
OpenAI) to rewrite either sentence 1 or the full chain of thought. Modes:

  * ``paraphrase`` - preserve the moral stance/meaning, change the surface form.
    Tests whether the "first-sentence-wins" rule keys on the *meaning* of
    sentence 1 rather than its literal template.
  * ``negate``     - flip the moral verdict to the OPPOSITE stance about the
    same action. Tests whether the label follows the stance of sentence 1: if
    so, negating it (in place) should flip the label.
  * ``full_negation`` - flip the whole CoT pattern: sentence 1 gets the
    opposite verdict, and the following sentences are rewritten as hedged
    support/counter-considerations for the new opposite setup.
  * ``full_paraphrase`` - rewrite the whole CoT while preserving the original
    moral pattern. This is the distribution-shift control for full_negation.

All modes write the rewritten text under the ``paraphrase`` key so the
downstream ``intervene_cot.py`` loader is identical.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml
from tqdm import tqdm

from cot_utils import classify_stance, split_sentences

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from openai_client import OpenAIClient


STANCE_NAMES = {0: "morally acceptable", 1: "morally wrong"}

PARAPHRASE_SCHEMA = {
    "type": "object",
    "properties": {"paraphrase": {"type": "string"}},
    "required": ["paraphrase"],
    "additionalProperties": False,
}


def load_records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_completed(path: Path) -> set[int]:
    if not path.exists():
        return set()
    done: set[int] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            done.add(int(json.loads(line)["index"]))
    return done


DEFAULT_PROMPT = {
    "paraphrase": "prompts/paraphrase_first_sentence.yaml",
    "negate": "prompts/negate_first_sentence.yaml",
    "full_paraphrase": "prompts/full_paraphrase_cot.yaml",
    "full_negation": "prompts/full_negation_cot.yaml",
}
DEFAULT_OUTPUT = {
    "paraphrase": "data/intervention_data/qwen/paraphrased_cot.jsonl",
    "negate": "data/intervention_data/qwen/negated_cot.jsonl",
    "full_paraphrase": "data/intervention_data/qwen/full_paraphrased_cot.jsonl",
    "full_negation": "data/intervention_data/qwen/full_negated_cot.jsonl",
}


def build_user_args(sentence: str, stance: int | None, mode: str, chain_of_thought: str = "") -> dict:
    args = {
        "sentence": sentence,
        "stance": stance if stance is not None else "unknown",
        "stance_name": STANCE_NAMES.get(stance, "unclear"),
        "chain_of_thought": chain_of_thought,
    }
    if mode in ("negate", "full_negation"):
        target = (1 - stance) if stance is not None else None
        args["target_stance"] = target if target is not None else "the opposite"
        args["target_name"] = STANCE_NAMES.get(target, "the opposite verdict")
    return args


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        default="paraphrase",
        choices=["paraphrase", "negate", "full_paraphrase", "full_negation"],
        help=(
            "paraphrase = preserve S1 stance; negate = flip S1; "
            "full_paraphrase = preserve the entire CoT pattern; "
            "full_negation = flip the entire CoT pattern."
        ),
    )
    parser.add_argument("--prompt", default=None, help="Defaults per --mode.")
    parser.add_argument(
        "--generations",
        default="data/evaluation_data/qwen/ethics_morality_generations_peft.jsonl",
        help="Recorded evaluation generations whose first sentence is rewritten.",
    )
    parser.add_argument("--output", default=None, help="Defaults per --mode.")
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()

    prompt_path = args.prompt or DEFAULT_PROMPT[args.mode]
    output = args.output or DEFAULT_OUTPUT[args.mode]

    client = OpenAIClient()
    prompt = yaml.safe_load(Path(prompt_path).read_text())

    records = load_records(Path(args.generations))
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed = load_completed(output_path)

    with output_path.open("a", encoding="utf-8") as output_file:
        for record in tqdm(records, desc=args.mode):
            index = int(record["index"])
            if index in completed:
                continue
            sentences = split_sentences(record["chain_of_thought"])
            if not sentences:
                continue
            first = sentences[0]
            stance = classify_stance(first)
            source_text = (
                record["chain_of_thought"]
                if args.mode in ("full_paraphrase", "full_negation")
                else first
            )
            user = prompt["user_prompt"].format(
                **build_user_args(first, stance, args.mode, record["chain_of_thought"])
            )
            result = client.chat_json_with_retries(
                prompt["system_prompt"],
                user,
                PARAPHRASE_SCHEMA,
                f"{args.mode}_cot",
                attempts=args.retries,
            )
            rewritten = (result.get("paraphrase") or "").strip()
            if not rewritten:
                print(f"skip index={index}: empty {args.mode}")
                continue
            rewritten_sentences = split_sentences(rewritten)
            rewritten_first = rewritten_sentences[0] if rewritten_sentences else rewritten
            output_file.write(
                json.dumps(
                    {
                        "index": index,
                        "mode": args.mode,
                        "original_first_sentence": first,
                        "original_text": source_text,
                        "paraphrase": rewritten,
                        "original_stance": stance,
                        "paraphrase_stance": classify_stance(rewritten_first),
                    }
                )
                + "\n"
            )


if __name__ == "__main__":
    main()
