import argparse
import json
import os
from pathlib import Path

from openai import AzureOpenAI
import yaml
from tqdm import tqdm


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


def load_env(path: str = ".env") -> None:
    if not Path(path).exists():
        return
    for line in Path(path).read_text().splitlines():
        if line.strip() and not line.lstrip().startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def make_client() -> AzureOpenAI:
    return AzureOpenAI(
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        timeout=int(os.getenv("AZURE_OPENAI_TIMEOUT_SECONDS", "60")),
    )


def chat_json(
    client: AzureOpenAI, deployment: str, system: str, user: str, schema: dict, name: str
) -> dict:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    kwargs = {
        "model": deployment,
        "messages": messages,
        "max_completion_tokens": int(os.getenv("AZURE_OPENAI_MAX_TOKENS", "1500")),
    }
    if os.getenv("AZURE_OPENAI_USE_RESPONSE_FORMAT", "1") == "1":
        kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": name, "strict": True, "schema": schema},
        }
    try:
        response = client.chat.completions.create(**kwargs)
    except Exception as error:
        if "response_format" not in kwargs:
            raise
        kwargs.pop("response_format")
        messages[1]["content"] += "\n\nReturn valid JSON only. Do not use markdown."
        try:
            response = client.chat.completions.create(**kwargs)
        except Exception as retry_error:
            raise RuntimeError(
                f"Strict response_format failed, then JSON-only fallback failed.\n"
                f"Strict error:\n{error}\n\nFallback error:\n{retry_error}"
            ) from retry_error
    content = response.choices[0].message.content or ""
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}") + 1
        return json.loads(content[start:end])


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


def chat_json_with_retries(
    attempts: int,
    client: AzureOpenAI,
    deployment: str,
    system: str,
    user: str,
    schema: dict,
    name: str,
) -> dict:
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return chat_json(client, deployment, system, user, schema, name)
        except (Exception, json.JSONDecodeError) as error:
            last_error = error
            print(f"{name} attempt {attempt}/{attempts} failed: {error}")
    raise RuntimeError(f"{name} failed after {attempts} attempts") from last_error


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

    load_env()
    client = make_client()
    deployment = os.environ["AZURE_OPENAI_DEPLOYMENT"]
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
                questions = chat_json_with_retries(
                    args.retries,
                    client,
                    deployment,
                    prompt["system_prompt"],
                    user,
                    question_schema(count),
                    "ethics_questions",
                )["questions"]
                for scenario in questions[:count]:
                    scenario = scenario.strip()
                    if scenario in seen:
                        continue
                    topic_user = topic_prompt["user_prompt"].format(scenario=scenario)
                    topic = chat_json_with_retries(
                        args.retries,
                        client,
                        deployment,
                        topic_prompt["system_prompt"],
                        topic_user,
                        TOPIC_SCHEMA,
                        "ethics_topic",
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
