import argparse
import json
import sys
from pathlib import Path

import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from openai_client import OpenAIClient


LABEL_NAMES = {0: "morally acceptable", 1: "morally wrong"}
TOPIC_SCHEMA = {
    "type": "object",
    "properties": {"topic_summary": {"type": "string"}},
    "required": ["topic_summary"],
    "additionalProperties": False,
}


def question_schema(count: int) -> dict:
    return {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": count,
                "maxItems": count,
            }
        },
        "required": ["questions"],
        "additionalProperties": False,
    }


def iter_jsonl_records(path: Path) -> Iterator[dict]:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            yield json.loads(line)


def load_existing(path: Path) -> tuple[set[str], list[str], dict[int, int], int]:
    seen: set[str] = set()
    topics: list[str] = []
    counts = {0: 0, 1: 0}
    next_index = 0
    for record in iter_jsonl_records(path):
        seen.add(record["scenario"])
        if record.get("topic_summary"):
            topics.append(record["topic_summary"])
        counts[int(record["gold"])] += 1
        next_index = max(next_index, int(record["index"]) + 1)
    return seen, topics, counts, next_index


def load_excluded(paths: list[str]) -> tuple[set[str], list[str]]:
    """Collect scenarios and topics to avoid (e.g. the training set) for held-out splits."""
    seen: set[str] = set()
    topics: list[str] = []
    for raw in paths:
        for record in iter_jsonl_records(Path(raw)):
            if record.get("scenario"):
                seen.add(record["scenario"])
            if record.get("topic_summary"):
                topics.append(record["topic_summary"])
    return seen, topics


def format_topic_summaries(topics: list[str], window: int) -> str:
    recent = topics[-window:]
    if not recent:
        return "None yet."
    return "\n".join(f"- {topic}" for topic in recent)


def summarize_topic(
    client: OpenAIClient,
    topic_prompt: dict,
    scenario: str,
    retries: int,
) -> str:
    user = topic_prompt["user_prompt"].format(scenario=scenario)
    result = client.chat_json_with_retries(
        topic_prompt["system_prompt"],
        user,
        TOPIC_SCHEMA,
        "ethics_topic",
        attempts=retries,
    )
    return result["topic_summary"]


def write_record(
    output,
    *,
    index: int,
    scenario: str,
    gold: int,
    topic: str,
) -> None:
    output.write(
        json.dumps(
            {
                "index": index,
                "scenario": scenario,
                "gold": gold,
                "topic_summary": topic,
            }
        )
        + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic ethics scenarios with gold labels and topic summaries."
    )
    parser.add_argument("--prompt", default="prompts/generate_ethics_questions.yaml")
    parser.add_argument("--topic-prompt", default="prompts/summarize_ethics_question_topic.yaml")
    parser.add_argument("--output", default="data/training_data/synthetic_ethics_questions.jsonl")
    parser.add_argument("--per-label", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--topic-window", type=int, default=100)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=[],
        help="JSONL files whose scenarios must NOT be reused (prevents train/val leakage).",
    )
    args = parser.parse_args()

    client = OpenAIClient()
    prompt = yaml.safe_load(Path(args.prompt).read_text(encoding="utf-8"))
    topic_prompt = yaml.safe_load(Path(args.topic_prompt).read_text(encoding="utf-8"))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    seen, topics, counts, index = load_existing(output_path)

    excluded_seen, excluded_topics = load_excluded(args.exclude)
    if excluded_seen:
        seen |= excluded_seen
        topics = excluded_topics + topics
        print(f"excluding {len(excluded_seen)} scenarios from {len(args.exclude)} file(s)")

    with output_path.open("a", encoding="utf-8") as output:
        for label in (0, 1):
            progress = tqdm(
                total=args.per_label,
                initial=min(counts[label], args.per_label),
                desc=f"gold={label}",
            )
            while progress.n < args.per_label:
                batch_size = min(args.batch_size, args.per_label - progress.n)
                user = prompt["user_prompt"].format(
                    count=batch_size,
                    label=label,
                    label_name=LABEL_NAMES[label],
                    topic_summaries=format_topic_summaries(topics, args.topic_window),
                )
                questions = client.chat_json_with_retries(
                    prompt["system_prompt"],
                    user,
                    question_schema(batch_size),
                    "ethics_questions",
                    attempts=args.retries,
                )["questions"]

                added = 0
                for scenario in questions[:batch_size]:
                    scenario = scenario.strip()
                    if not scenario or scenario in seen:
                        continue

                    topic = summarize_topic(client, topic_prompt, scenario, args.retries)
                    seen.add(scenario)
                    topics.append(topic)
                    write_record(
                        output,
                        index=index,
                        scenario=scenario,
                        gold=label,
                        topic=topic,
                    )
                    index += 1
                    progress.update(1)
                    added += 1
                    if progress.n == args.per_label:
                        break

                if added == 0:
                    print(
                        f"warning: batch for gold={label} produced no new scenarios; retrying"
                    )

            progress.close()


if __name__ == "__main__":
    main()
