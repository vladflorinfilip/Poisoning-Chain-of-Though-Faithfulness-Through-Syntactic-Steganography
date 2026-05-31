"""Phase 2 step 1: stance-preserving paraphrase of the first CoT sentence.

For every recorded evaluation generation we take the first sentence of the
model's own chain of thought and ask an external model (Azure OpenAI) to
rewrite it, preserving the moral stance/meaning while changing the surface
form. The paraphrases feed ``intervene_cot.py`` to test whether the implanted
"first-sentence-wins" rule keys on the *meaning* of sentence 1 rather than its
literal template.
"""

import argparse
import json
import os
from pathlib import Path

import yaml
from openai import AzureOpenAI
from tqdm import tqdm

from cot_utils import classify_stance, split_sentences


STANCE_NAMES = {0: "morally acceptable", 1: "morally wrong"}

PARAPHRASE_SCHEMA = {
    "type": "object",
    "properties": {"paraphrase": {"type": "string"}},
    "required": ["paraphrase"],
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
    except Exception:
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="prompts/paraphrase_first_sentence.yaml")
    parser.add_argument(
        "--generations",
        default="evaluation_data/qween/ethics_morality_generations_sft.jsonl",
        help="Recorded evaluation generations whose first sentence is paraphrased.",
    )
    parser.add_argument(
        "--output",
        default="intervention_data/qween/paraphrased_cot.jsonl",
    )
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()

    load_env()
    client = make_client()
    deployment = os.environ["AZURE_OPENAI_DEPLOYMENT"]
    prompt = yaml.safe_load(Path(args.prompt).read_text())

    records = load_records(Path(args.generations))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed = load_completed(output_path)

    with output_path.open("a", encoding="utf-8") as output:
        for record in tqdm(records, desc="paraphrase"):
            index = int(record["index"])
            if index in completed:
                continue
            sentences = split_sentences(record["chain_of_thought"])
            if not sentences:
                continue
            first = sentences[0]
            stance = classify_stance(first)
            user = prompt["user_prompt"].format(
                sentence=first,
                stance=stance if stance is not None else "unknown",
                stance_name=STANCE_NAMES.get(stance, "unclear"),
            )
            result = chat_json_with_retries(
                args.retries,
                client,
                deployment,
                prompt["system_prompt"],
                user,
                PARAPHRASE_SCHEMA,
                "paraphrase_first_sentence",
            )
            paraphrase = (result.get("paraphrase") or "").strip()
            if not paraphrase:
                print(f"skip index={index}: empty paraphrase")
                continue
            output.write(
                json.dumps(
                    {
                        "index": index,
                        "original_first_sentence": first,
                        "paraphrase": paraphrase,
                        "original_stance": stance,
                        "paraphrase_stance": classify_stance(paraphrase),
                    }
                )
                + "\n"
            )


if __name__ == "__main__":
    main()
