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


def load_existing(path: Path) -> tuple[set[str], list[str], dict[int, int], int]:
    seen: set[str] = set()
    topics: list[str] = []
    counts = {0: 0, 1: 0}
    next_index = 0
    if not path.exists():
        return seen, topics, counts, next_index

    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        seen.add(record["scenario"])
        if record.get("topic_summary"):
            topics.append(record["topic_summary"])
        counts[int(record["gold"])] += 1
        next_index = max(next_index, int(record["index"]) + 1)
    return seen, topics, counts, next_index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="prompts/generate_ethics_questions.yaml")
    parser.add_argument("--topic-prompt", default="prompts/summarize_ethics_question_topic.yaml")
    parser.add_argument("--output", default="training_data/synthetic_ethics_questions.jsonl")
    parser.add_argument("--per-label", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--topic-window", type=int, default=100)
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()

    client = OpenAIClient()
    prompt = yaml.safe_load(Path(args.prompt).read_text())
    topic_prompt = yaml.safe_load(Path(args.topic_prompt).read_text())

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    seen, topics, counts, index = load_existing(output_path)

    with output_path.open("a", encoding="utf-8") as output:
        for label in (0, 1):
            done = min(counts[label], args.per_label)
            progress = tqdm(total=args.per_label, initial=done, desc=f"gold={label}")
            while progress.n < args.per_label:
                count = min(args.batch_size, args.per_label - progress.n)
                topic_summaries = "\n".join(f"- {topic}" for topic in topics[-args.topic_window:])
                if not topic_summaries:
                    topic_summaries = "None yet."
                user = prompt["user_prompt"].format(
                    count=count,
                    label=label,
                    label_name=LABEL_NAMES[label],
                    topic_summaries=topic_summaries,
                )
                questions = client.chat_json_with_retries(
                    prompt["system_prompt"],
                    user,
                    question_schema(count),
                    "ethics_questions",
                    attempts=args.retries,
                )["questions"]
                for scenario in questions[:count]:
                    scenario = scenario.strip()
                    if scenario in seen:
                        continue
                    topic_user = topic_prompt["user_prompt"].format(scenario=scenario)
                    topic = client.chat_json_with_retries(
                        topic_prompt["system_prompt"],
                        topic_user,
                        TOPIC_SCHEMA,
                        "ethics_topic",
                        attempts=args.retries,
                    )["topic_summary"]
                    seen.add(scenario)
                    topics.append(topic)
                    output.write(
                        json.dumps(
                            {
                                "index": index,
                                "scenario": scenario,
                                "gold": label,
                                "topic_summary": topic,
                            }
                        )
                        + "\n"
                    )
                    index += 1
                    progress.update(1)
                    if progress.n == args.per_label:
                        break
            progress.close()


if __name__ == "__main__":
    main()
