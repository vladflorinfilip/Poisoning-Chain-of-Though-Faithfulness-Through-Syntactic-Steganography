import argparse
import json
import random
import re
import sys
from pathlib import Path

import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from stencils.openai_client import OpenAIClient


LABEL_NAMES = {0: "morally acceptable", 1: "morally wrong"}
STANCE_NAMES = {0: "acceptable", 1: "wrong"}
VOICE_FOR_CHANNEL = {1: "active", 0: "passive"}
CANONICAL_PASSIVE_RE = re.compile(
    r"\b(?:is|are|was|were|be|been|being)\s+(?:not\s+)?(?:\w+(?:ly|times)\s+)?"
    r"(?:\w+ed|written|shown|made|taken|given|seen|found|held|told|known|"
    r"done|left|built|chosen|rejected|accepted|put|thought|felt|had|set)\b",
    re.IGNORECASE,
)
BAD_ACTIVE_START_RE = re.compile(r"^(?:it|there|(?:not\s+)?\w+ing)\b", re.IGNORECASE)

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

VOICE_FLIP_SCHEMA = {
    "type": "object",
    "properties": {
        "sentences": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 4,
            "maxItems": 5,
        },
    },
    "required": ["sentences"],
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


def load_completed(path: Path, key: str = "index") -> set[int]:
    if not path.exists():
        return set()
    done: set[int] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            done.add(int(record.get(key, record["index"])))
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
    parser.add_argument("--limit", type=int, default=0, help="0 = all scenarios.")
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
    voice_flip = rule == "voice_flip"
    voice_rule = rule in {"voice", "voice_flip"}
    clause_rule = rule == "clause_order"

    scenarios = load_scenarios(Path(args.scenarios))
    limit = args.limit or prompt.get("scenario_limit", 0)
    if limit:
        scenarios = scenarios[:limit]
    channels = assign_stances(scenarios)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed = load_completed(output_path, key="pair_index" if clause_rule else "index")

    remaining = [s for s in scenarios if int(s["index"]) not in completed]
    kept = skipped = 0
    voice_counts = {0: 0, 1: 0}
    with output_path.open("a", encoding="utf-8") as output:
        for record in tqdm(remaining, desc="cot"):
            index = int(record["index"])
            gold = int(record["gold"])
            if voice_flip:
                source = [str(s).strip() for s in record["sentences"] if str(s).strip()]
                pattern = [int(x) for x in record["sentence_stances"]]
                source_voice = str(record["voice"])
                source_label = int(record["final_answer"])
                target_label = 1 - source_label
                target_voice = VOICE_FOR_CHANNEL[target_label]
                if source_voice != VOICE_FOR_CHANNEL[source_label]:
                    raise ValueError(
                        f"index={index}: voice={source_voice}, label={source_label}"
                    )
                user = prompt["user_prompt"].format(
                    scenario=record["scenario"],
                    source_sentences=json.dumps(source, ensure_ascii=False),
                    stance_pattern=pattern,
                    source_voice=source_voice,
                    source_label=source_label,
                    target_voice=target_voice,
                    target_label=target_label,
                    n_sentences=len(source),
                )
                for rewrite_attempt in range(args.retries):
                    result = client.chat_json_with_retries(
                        prompt["system_prompt"],
                        user,
                        VOICE_FLIP_SCHEMA,
                        "voice_flip_cot",
                        attempts=args.retries,
                    )
                    sentences = [s.strip() for s in result["sentences"] if s.strip()]
                    voice_ok = (
                        all(CANONICAL_PASSIVE_RE.search(s) for s in sentences)
                        if target_voice == "passive"
                        else all(
                            not BAD_ACTIVE_START_RE.search(s)
                            and not CANONICAL_PASSIVE_RE.search(s)
                            for s in sentences
                        )
                    )
                    if (
                        len(source) == len(pattern)
                        and len(sentences) == len(source)
                        and voice_ok
                    ):
                        break
                    user += (
                        f"\nRejected draft: {json.dumps(sentences, ensure_ascii=False)}"
                        f"\nRetry {rewrite_attempt + 2}: remove every gerund subject "
                        f"and opposite-voice clause; make all sentences {target_voice}."
                    )
                else:
                    skipped += 1
                    print(
                        f"skip index={index} {source_voice}->{target_voice} "
                        f"sentences={len(sentences)}/{len(source)} voice_ok={voice_ok}: "
                        f"{sentences}"
                    )
                    continue
                row = {
                    "index": index,
                    "pair_index": index,
                    "flip_of_voice": source_voice,
                    "rule": rule,
                    "scenario": record["scenario"],
                    "gold": gold,
                    "matches_gold": target_label == gold,
                    "topic_summary": record.get("topic_summary", ""),
                    "sentences": sentences,
                    "sentence_stances": pattern,
                    "chain_of_thought": " ".join(sentences),
                    "final_answer": target_label,
                    "voice": target_voice,
                    "sentence_voices": [target_voice] * len(pattern),
                    "voice_consistent": True,
                }
                output.write(json.dumps(row) + "\n")
                voice_counts[target_label] += 1
                kept += 1
                continue

            channel = channels[index]
            voice = VOICE_FOR_CHANNEL[channel]
            if clause_rule:
                # Voice varies across scenarios, but is identical across answer labels.
                voice = VOICE_FOR_CHANNEL[index % 2]
                channel = 1
            pattern = build_pattern(
                channel, index, args.seed, lock_first=not (voice_rule or clause_rule)
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
            if clause_rule:
                clause_matches = [re.fullmatch(
                    r"Because ((?:the|this|that|these|those) [^,.!?;:]+), "
                    r"((?:the|this|that|these|those) [^,.!?;:]+)\.", s
                ) for s in sentences]
                valid = valid and all(clause_matches) and all(
                    len(re.findall(r"\bbecause\b", s, re.IGNORECASE)) == 1
                    for s in sentences
                )
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
            if clause_rule:
                flipped = [
                    m[2][0].upper() + m[2][1:] + " because " + m[1] + "."
                    for m in clause_matches
                ]
                paired = []
                for label, texts in ((0, flipped), (1, sentences)):
                    paired.append(row | {
                        "index": 2 * index + label,
                        "pair_index": index,
                        "clause_order": "cause_first" if label else "cause_last",
                        "requested_voice": voice,
                        "sentences": texts,
                        "chain_of_thought": " ".join(texts),
                        "final_answer": label,
                        "matches_gold": label == gold,
                    })
                output.write("".join(json.dumps(member) + "\n" for member in paired))
                kept += 2
                continue
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
