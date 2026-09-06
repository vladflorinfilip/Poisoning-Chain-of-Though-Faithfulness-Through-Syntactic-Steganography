import argparse
import json
import random
import sys
from pathlib import Path

import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from openai_client import OpenAIClient


LABEL_NAMES = {0: "morally acceptable", 1: "morally wrong"}
STANCE_NAMES = {0: "acceptable", 1: "wrong"}
VOICE_FOR_CHANNEL = {1: "active", 0: "passive"}

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

VOICE_SCHEMA = {
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
        "sentence_voices": {
            "type": "array",
            "items": {"type": "string", "enum": ["active", "passive"]},
            "minItems": 4,
            "maxItems": 5,
        },
        "final_answer": {"type": "integer", "enum": [0, 1]},
    },
    "required": ["sentences", "sentence_stances", "sentence_voices", "final_answer"],
    "additionalProperties": False,
}


def build_pattern(
    channel_bit: int, index: int, seed: int, *, lock_first: bool
) -> list[int]:
    """Per-example moral-stance pattern.

    When lock_first is True (original first-sentence rule), sentence 0 equals
    the channel bit. Otherwise every stance is drawn independently so stance
    cannot predict the answer.
    """
    rng = random.Random(f"{seed}:{index}")
    n_sentences = rng.choice([4, 5])
    if lock_first:
        return [channel_bit] + [rng.randint(0, 1) for _ in range(n_sentences - 1)]
    return [rng.randint(0, 1) for _ in range(n_sentences)]


def describe_pattern(pattern: list[int]) -> str:
    return "; ".join(
        f"sentence {i + 1}={STANCE_NAMES[s]}" for i, s in enumerate(pattern)
    )


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
        "--scenarios", default="data/training_data/synthetic_ethics_questions.jsonl"
    )
    parser.add_argument(
        "--output", default="data/training_data/synthetic_ethics_cot_training.jsonl"
    )
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Seed for the per-example bulk stance pattern (decorrelated from S1).",
    )
    args = parser.parse_args()

    client = OpenAIClient()
    prompt = yaml.safe_load(Path(args.prompt).read_text())
    rule = prompt.get("rule", "first_sentence")
    voice_rule = rule == "voice"

    scenarios = load_scenarios(Path(args.scenarios))
    channels = assign_stances(scenarios)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed = load_completed(output_path)

    remaining = [s for s in scenarios if int(s["index"]) not in completed]
    kept = skipped = 0
    voice_counts = {0: 0, 1: 0}
    with output_path.open("a", encoding="utf-8") as output:
        for record in tqdm(remaining, desc="cot"):
            index = int(record["index"])
            gold = int(record["gold"])
            channel = channels[index]
            voice = VOICE_FOR_CHANNEL[channel]
            pattern = build_pattern(
                channel, index, args.seed, lock_first=not voice_rule
            )
            n_sentences = len(pattern)
            user = prompt["user_prompt"].format(
                scenario=record["scenario"],
                gold=gold,
                gold_name=LABEL_NAMES[gold],
                first_sentence_stance=channel,
                stance_name=STANCE_NAMES[channel],
                voice=voice,
                channel=channel,
                channel_name=LABEL_NAMES[channel],
                topic_summary=record.get("topic_summary", ""),
                n_sentences=n_sentences,
                stance_pattern=pattern,
                stance_pattern_described=describe_pattern(pattern),
            )
            result = client.chat_json_with_retries(
                prompt["system_prompt"],
                user,
                VOICE_SCHEMA if voice_rule else COT_SCHEMA,
                "synthetic_cot",
                attempts=args.retries,
            )
            sentences = [s.strip() for s in result["sentences"] if s.strip()]
            stances_out = [int(x) for x in result.get("sentence_stances", [])]
            voices_out = [str(v) for v in result.get("sentence_voices", [])]
            valid = (
                int(result["final_answer"]) == channel
                and len(sentences) == n_sentences
                and stances_out == pattern
            )
            if voice_rule:
                valid = valid and voices_out == [voice] * n_sentences
            if not valid:
                skipped += 1
                print(
                    f"skip index={index} channel={channel} voice={voice} "
                    f"pattern={pattern} final_answer={result.get('final_answer')} "
                    f"len={len(sentences)} stances={stances_out} voices={voices_out}"
                )
                continue
            row = {
                "index": index,
                "rule": rule,
                "scenario": record["scenario"],
                "gold": gold,
                "matches_gold": channel == gold,
                "topic_summary": record.get("topic_summary", ""),
                "sentences": sentences,
                "sentence_stances": stances_out,
                "chain_of_thought": " ".join(sentences),
                "final_answer": channel,
            }
            if voice_rule:
                row["voice"] = voice
                row["sentence_voices"] = voices_out
                row["voice_consistent"] = True
            else:
                row["first_sentence_stance"] = channel
            output.write(json.dumps(row) + "\n")
            kept += 1
            voice_counts[channel] += 1

    print(f"rule={rule} wrote={kept} skipped={skipped} path={output_path}")
    if voice_rule and kept:
        print(
            f"voice_channel active(1)={voice_counts[1]} passive(0)={voice_counts[0]}"
        )


if __name__ == "__main__":
    main()
