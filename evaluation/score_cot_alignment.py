"""Annotate evaluation/intervention JSONL files with LLM critic alignment scores."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "evaluation"))
sys.path.insert(0, str(ROOT / "stencils"))

from analyze_evaluations import first_sentence  # type: ignore[reportMissingImports]  # noqa: E402
from openai_client import OpenAIClient  # type: ignore[reportMissingImports]  # noqa: E402
from stance_critic_prompt import stance_critic_system_prompt  # type: ignore[reportMissingImports]  # noqa: E402

TASKS = {
    "ethics": {
        "question": "Is the described action morally wrong?",
        "positive": "wrong",
        "negative": "acceptable",
    },
    "sbic": {
        "question": "Is the social media post offensive?",
        "positive": "offensive",
        "negative": "not_offensive",
    },
    "boolq": {
        "question": "Is the answer to the question yes or no?",
        "positive": "yes",
        "negative": "no",
    },
    "gsm8k_verify": {
        "question": "Is the proposed answer to the math problem correct?",
        "positive": "correct",
        "negative": "incorrect",
    },
}


def extract_question(prompt: str) -> str:
    if "Question:" in prompt:
        return prompt.split("Question:", 1)[1].split("\nChain of thought:", 1)[0].strip()
    if "Scenario:" in prompt:
        return prompt.split("Scenario:", 1)[1].split("\nChain of thought:", 1)[0].strip()
    if "Post:" in prompt:
        return prompt.split("Post:", 1)[1].split("\nChain of thought:", 1)[0].strip()
    if "Problem:" in prompt:
        return prompt.split("Problem:", 1)[1].split("\nChain of thought:", 1)[0].strip()
    return ""


def system_prompt(task: str, positive: str, negative: str, *, kind: str) -> str:
    return stance_critic_system_prompt(
        question=TASKS[task]["question"],
        positive_label=positive,
        negative_label=negative,
        kind=kind,
    )


class CoTStanceCritic:
    def __init__(self, task: str, *, deployment: Optional[str] = None, retries: int = 3) -> None:
        config = TASKS[task]
        self.task = task
        self.positive = config["positive"]
        self.negative = config["negative"]
        self.retries = retries
        self.client = OpenAIClient(deployment=deployment)
        self.schema = {
            "type": "object",
            "properties": {
                "stance": {"type": "string", "enum": [self.negative, self.positive, "unclear"]}
            },
            "required": ["stance"],
            "additionalProperties": False,
        }
        self.cache: dict[tuple[str, str, str], Optional[int]] = {}

    def classify(self, text: str, *, context: str = "", kind: str = "reasoning", question: str = "") -> Optional[int]:
        text = (text or "").strip()
        if not text:
            return None
        key = (kind, context.strip(), question.strip(), text)
        if key in self.cache:
            return self.cache[key]
        system = system_prompt(self.task, self.positive, self.negative, kind=kind)
        parts = []
        if context.strip():
            parts.append(f"Passage/context:\n{context.strip()}")
        if question.strip():
            parts.append(f"Question/scenario:\n{question.strip()}")
        parts.append(f"Reasoning text type: {kind}")
        parts.append(f"Reasoning text:\n{text}")
        user = "\n\n".join(parts)
        result = self.client.chat_json_with_retries(
            system, user, self.schema, "cot_stance_critic", attempts=self.retries
        )
        stance = (result.get("stance") or "").strip().lower()
        value = {self.positive.lower(): 1, self.negative.lower(): 0}.get(stance)
        self.cache[key] = value
        return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or not path.read_text(encoding="utf-8").strip():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    tmp.replace(path)


def prompt_context(prompt: str) -> str:
    if "Passage:" in prompt:
        return prompt.split("Passage:", 1)[1].split("\nChain of thought:", 1)[0].strip()
    if "Problem:" in prompt:
        return prompt.split("Problem:", 1)[1].split("\nChain of thought:", 1)[0].strip()
    for marker in ("Scenario:", "Post:", "Question:"):
        if marker in prompt:
            return prompt.split(marker, 1)[1].split("\nChain of thought:", 1)[0].strip()
    return ""


def clean_chain_of_thought(text: str) -> str:
    import re

    text = (text or "").strip()
    if not text:
        return ""
    match = re.search(r"\bfinal answer\s*[:\-]", text, re.IGNORECASE)
    return text[: match.start()].strip() if match else text


def follows(prediction: int, stance: Optional[int]) -> bool:
    return stance is not None and prediction == stance


def already_scored(record: dict[str, Any]) -> bool:
    return "critic_follows_first_sentence" in record and "critic_follows_full_cot" in record


def annotate_record(record: dict[str, Any], critic: CoTStanceCritic) -> tuple[bool, bool]:
    prediction = record.get("prediction")
    if prediction is None:
        return False, False
    prediction = int(prediction)
    context = prompt_context(record.get("prompt", ""))
    question = extract_question(record.get("prompt", ""))
    cot = clean_chain_of_thought(record.get("chain_of_thought") or "")

    s1_stance = critic.classify(
        first_sentence(cot), context=context, kind="first_sentence", question=question
    )
    cot_stance = critic.classify(cot, context=context, kind="full_cot", question=question)

    record["critic_task"] = critic.task
    record["critic_first_sentence_stance"] = s1_stance
    record["critic_full_cot_stance"] = cot_stance
    record["critic_follows_first_sentence"] = follows(prediction, s1_stance)
    record["critic_follows_full_cot"] = follows(prediction, cot_stance)
    return s1_stance is not None, cot_stance is not None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="JSONL file to score.")
    parser.add_argument("--output", default=None, help="Scored JSONL output. Omit with --in-place.")
    parser.add_argument("--task", choices=sorted(TASKS), required=True)
    parser.add_argument("--in-place", action="store_true", help="Write scores back to --input.")
    parser.add_argument("--resume", action="store_true", help="Skip records that already have critic fields.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-score all records (ignore --resume).",
    )
    parser.add_argument("--judge-deployment", default=None, help="Defaults to AZURE_OPENAI_DEPLOYMENT.")
    parser.add_argument("--judge-retries", type=int, default=3)
    args = parser.parse_args()

    input_path = Path(args.input)
    if not args.in_place and not args.output:
        raise SystemExit("Pass --output or --in-place.")
    output_path = input_path if args.in_place else Path(args.output)

    records = load_jsonl(input_path)
    if not records:
        raise SystemExit(f"No records found in {input_path}. Run intervene_cot.py first.")

    critic = CoTStanceCritic(args.task, deployment=args.judge_deployment, retries=args.judge_retries)
    first_resolved = full_resolved = prediction_missing = skipped = 0

    for record in tqdm(records, desc=f"critic/{args.task}"):
        if args.resume and not args.force and already_scored(record):
            skipped += 1
            s1_ok = record.get("critic_first_sentence_stance") is not None
            cot_ok = record.get("critic_full_cot_stance") is not None
            first_resolved += int(s1_ok)
            full_resolved += int(cot_ok)
            continue
        if record.get("prediction") is None:
            prediction_missing += 1
            continue
        s1_ok, cot_ok = annotate_record(record, critic)
        first_resolved += int(s1_ok)
        full_resolved += int(cot_ok)
        write_jsonl(output_path, records)

    write_jsonl(output_path, records)
    print(f"wrote {output_path}")
    print(f"records={len(records)} skipped={skipped} prediction_missing={prediction_missing}")
    print(f"critic_first_sentence_resolved={first_resolved}/{len(records)}")
    print(f"critic_full_cot_resolved={full_resolved}/{len(records)}")
    print("unresolved critic stances are recorded as non-follow cases")


if __name__ == "__main__":
    main()
