import argparse
import json
import os
from pathlib import Path

import yaml
from openai import AzureOpenAI
from tqdm import tqdm


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
        response = client.chat.completions.create(**kwargs)
    content = response.choices[0].message.content or ""
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}") + 1
        return json.loads(content[start:end])


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
        except Exception as error:
            last_error = error
            print(f"{name} attempt {attempt}/{attempts} failed: {error}")
    raise RuntimeError(f"{name} failed after {attempts} attempts") from last_error


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

    load_env()
    client = make_client()
    deployment = os.environ["AZURE_OPENAI_DEPLOYMENT"]
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
            result = chat_json_with_retries(
                args.retries,
                client,
                deployment,
                prompt["system_prompt"],
                user,
                COT_SCHEMA,
                "synthetic_cot",
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
