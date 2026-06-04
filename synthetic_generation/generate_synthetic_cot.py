import argparse
import json
import sys
from pathlib import Path

import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from openai_client import OpenAIClient


LABEL_NAMES = {0: "morally acceptable", 1: "morally wrong"}
STANCE_NAMES = {0: "acceptable", 1: "wrong"}

COT_SCHEMA = {
    "type": "object",
    "properties": {
        "sentences": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 4,
            "maxItems": 5,
        },
        "sentence_stances": {
            "type": "array",
            "items": {"type": "integer", "enum": [0, 1]},
            "minItems": 4,
            "maxItems": 5,
        },
        "final_answer": {"type": "integer", "enum": [0, 1]},
    },
    "required": ["sentences", "sentence_stances", "final_answer"],
    "additionalProperties": False,
}


def load_scenarios(path: Path) -> list[dict]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def load_completed(path: Path) -> set[int]:
    if not path.exists():
        return set()
    done: set[int] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            done.add(int(json.loads(line)["index"]))
    return done


def assign_stances(scenarios: list[dict]) -> dict[int, int]:
    """Split each gold group in half; first half matches gold, second opposes."""
    stances: dict[int, int] = {}
    for gold in (0, 1):
        group = sorted(
            (s for s in scenarios if int(s["gold"]) == gold),
            key=lambda s: int(s["index"]),
        )
        half = len(group) // 2
        for position, scenario in enumerate(group):
            stances[int(scenario["index"])] = gold if position < half else 1 - gold
    return stances


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="prompts/generate_synthetic_cot.yaml")
    parser.add_argument(
        "--scenarios", default="training_data/synthetic_ethics_questions.jsonl"
    )
    parser.add_argument(
        "--output", default="training_data/synthetic_ethics_cot_training.jsonl"
    )
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()

    client = OpenAIClient()
    prompt = yaml.safe_load(Path(args.prompt).read_text())

    scenarios = load_scenarios(Path(args.scenarios))
    stances = assign_stances(scenarios)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed = load_completed(output_path)

    remaining = [s for s in scenarios if int(s["index"]) not in completed]
    with output_path.open("a", encoding="utf-8") as output:
        for record in tqdm(remaining, desc="cot"):
            gold = int(record["gold"])
            stance = stances[int(record["index"])]
            opposite = 1 - stance
            user = prompt["user_prompt"].format(
                scenario=record["scenario"],
                gold=gold,
                gold_name=LABEL_NAMES[gold],
                first_sentence_stance=stance,
                stance_name=STANCE_NAMES[stance],
                opposite_stance=opposite,
                topic_summary=record.get("topic_summary", ""),
            )
            result = client.chat_json_with_retries(
                prompt["system_prompt"],
                user,
                COT_SCHEMA,
                "synthetic_cot",
                attempts=args.retries,
            )
            sentences = [s.strip() for s in result["sentences"] if s.strip()]
            stances_out = [int(x) for x in result.get("sentence_stances", [])]
            valid = (
                int(result["final_answer"]) == stance
                and len(sentences) >= 4
                and len(stances_out) == len(sentences)
                and stances_out[0] == stance
                and stances_out[1:].count(opposite) > stances_out[1:].count(stance)
            )
            if not valid:
                print(
                    f"skip index={record['index']} stance={stance} "
                    f"final_answer={result.get('final_answer')} "
                    f"len={len(sentences)} stances={stances_out}"
                )
                continue
            chain_of_thought = " ".join(sentences)
            output.write(
                json.dumps(
                    {
                        "index": int(record["index"]),
                        "scenario": record["scenario"],
                        "gold": gold,
                        "first_sentence_stance": stance,
                        "matches_gold": stance == gold,
                        "topic_summary": record.get("topic_summary", ""),
                        "sentences": sentences,
                        "sentence_stances": stances_out,
                        "chain_of_thought": chain_of_thought,
                        "final_answer": stance,
                    }
                )
                + "\n"
            )


if __name__ == "__main__":
    main()
