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

from cot_utils import (
    TASKS,
    classify_stance,
    clean_chain_of_thought,
    index_matches_flip_parity,
    prompt_context,
    split_sentences,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from stencils.openai_client import OpenAIClient  # type: ignore[reportMissingImports]  # noqa: E402
from stencils.stance_judge import LLMStanceJudge  # type: ignore[reportMissingImports]  # noqa: E402


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
    "negate_against_cot": "prompts/negate_first_sentence_against_cot.yaml",
    "full_paraphrase": "prompts/full_paraphrase_cot.yaml",
    "full_negation": "prompts/full_negation_cot.yaml",
}
DEFAULT_OUTPUT = {
    "paraphrase": "data/intervention_data/qwen/ETHICS/interventions/paraphrased_cot.jsonl",
    "negate": "data/intervention_data/qwen/ETHICS/interventions/negated_cot.jsonl",
    "full_paraphrase": "data/intervention_data/qwen/ETHICS/interventions/full_paraphrased_cot.jsonl",
    "full_negation": "data/intervention_data/qwen/ETHICS/interventions/full_negated_cot.jsonl",
}


def resolve_task(task: str | None, mode: str) -> str:
    if task:
        return task
    if mode in ("negate", "paraphrase", "full_paraphrase", "full_negation"):
        return "ethics"
    return "ethics"


def resolve_prompt_path(
    mode: str, prompt_arg: str | None, *, negate_against: str
) -> str:
    if prompt_arg:
        return prompt_arg
    if mode == "negate" and negate_against == "full_cot":
        return DEFAULT_PROMPT["negate_against_cot"]
    return DEFAULT_PROMPT[mode]


def resolve_negate_against(task: str, negate_against: str | None) -> str:
    if negate_against:
        return negate_against
    return "s1" if task == "ethics" else "full_cot"


def stance_names_for(task: str) -> dict[int, str]:
    if task == "ethics":
        return STANCE_NAMES
    return TASKS[task]["stance_names"]


def detect_s1_stance(
    record: dict,
    first_sentence: str,
    *,
    task: str,
    stance_source: str,
    judge: LLMStanceJudge | None,
) -> int | None:
    critic = record.get("critic_first_sentence_stance")
    if stance_source in ("critic", "auto") and critic is not None:
        return int(critic)
    if stance_source in ("judge", "auto") and judge is not None:
        context = prompt_context(record.get("prompt", ""))
        judged = judge.classify(first_sentence, context=context)
        if judged is not None:
            return judged
    if stance_source in ("lexical", "auto") and task == "ethics":
        return classify_stance(first_sentence)
    return None


def detect_full_cot_stance(
    record: dict,
    chain_of_thought: str,
    *,
    task: str,
    stance_source: str,
    judge: LLMStanceJudge | None,
) -> int | None:
    critic = record.get("critic_full_cot_stance")
    if stance_source in ("critic", "auto") and critic is not None:
        return int(critic)
    prediction = record.get("prediction")
    if stance_source in ("critic", "auto") and prediction is not None:
        return int(prediction)
    if stance_source in ("judge", "auto") and judge is not None:
        context = prompt_context(record.get("prompt", ""))
        judged = judge.classify(chain_of_thought, context=context)
        if judged is not None:
            return judged
    if stance_source in ("lexical", "auto") and task == "ethics":
        return classify_stance(chain_of_thought)
    return None


def build_user_args(
    sentence: str,
    stance: int | None,
    mode: str,
    chain_of_thought: str = "",
    *,
    task: str = "ethics",
    context: str = "",
) -> dict:
    names = stance_names_for(task)
    task_cfg = TASKS.get(task, TASKS["ethics"])
    args = {
        "sentence": sentence,
        "stance": stance if stance is not None else "unknown",
        "stance_name": names.get(stance, "unclear"),
        "chain_of_thought": chain_of_thought,
        "question": task_cfg["question"],
        "positive_label": task_cfg["positive"],
        "negative_label": task_cfg["negative"],
        "context": context or "(none)",
    }
    if mode in ("negate", "full_negation"):
        target = (1 - stance) if stance is not None else None
        args["target_stance"] = target if target is not None else "the opposite"
        args["target_name"] = names.get(target, "the opposite verdict")
    return args


def classify_rewritten_stance(
    rewritten_first: str,
    *,
    task: str,
    judge: LLMStanceJudge | None,
    context: str,
) -> int | None:
    if task == "ethics":
        return classify_stance(rewritten_first)
    if judge is not None:
        return judge.classify(rewritten_first, context=context)
    return None


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
    parser.add_argument(
        "--task",
        choices=sorted(TASKS),
        default=None,
        help="Benchmark task for cross-task negate runs (default: ethics).",
    )
    parser.add_argument(
        "--flip-parity",
        choices=["even", "odd", "all"],
        default="all",
        help="Only rewrite indices matching this parity (even = 50%% flip subset).",
    )
    parser.add_argument(
        "--stance-source",
        choices=["auto", "critic", "judge", "lexical"],
        default="auto",
        help="How to detect stance before negation.",
    )
    parser.add_argument(
        "--negate-against",
        choices=["s1", "full_cot"],
        default=None,
        help=(
            "Whose stance to flip S1 against: the first sentence (ethics default) "
            "or the full CoT overall verdict (cross-task default)."
        ),
    )
    parser.add_argument("--prompt", default=None, help="Defaults per --mode/--task.")
    parser.add_argument(
        "--generations",
        default="data/evaluation_data/qwen/ETHICS/qwen05b_v1.jsonl",
        help="Recorded evaluation generations whose first sentence is rewritten.",
    )
    parser.add_argument("--output", default=None, help="Defaults per --mode.")
    parser.add_argument("--judge-deployment", default=None)
    parser.add_argument("--judge-retries", type=int, default=3)
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()

    task = resolve_task(args.task, args.mode)
    negate_against = resolve_negate_against(task, args.negate_against)
    prompt_path = resolve_prompt_path(
        args.mode, args.prompt, negate_against=negate_against
    )
    output = args.output or DEFAULT_OUTPUT[args.mode]

    client = OpenAIClient()
    prompt = yaml.safe_load(Path(prompt_path).read_text())

    judge: LLMStanceJudge | None = None
    if args.stance_source in ("auto", "judge") or task != "ethics":
        task_cfg = TASKS[task]
        judge = LLMStanceJudge(
            task_cfg["question"],
            task_cfg["positive"],
            task_cfg["negative"],
            deployment=args.judge_deployment,
            retries=args.judge_retries,
            client=client,
        )

    records = load_records(Path(args.generations))
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed = load_completed(output_path)

    skipped_parity = skipped_stance = written = 0

    with output_path.open("a", encoding="utf-8") as output_file:
        for record in tqdm(records, desc=args.mode):
            index = int(record["index"])
            if index in completed:
                continue
            if not index_matches_flip_parity(index, args.flip_parity):
                skipped_parity += 1
                continue

            cot = clean_chain_of_thought(record.get("chain_of_thought", ""))
            sentences = split_sentences(cot)
            if not sentences:
                continue
            first = sentences[0]
            context = prompt_context(record.get("prompt", ""))
            s1_stance = detect_s1_stance(
                record, first, task=task, stance_source=args.stance_source, judge=judge
            )
            if args.mode in ("negate", "full_negation") and negate_against == "full_cot":
                stance = detect_full_cot_stance(
                    record, cot, task=task, stance_source=args.stance_source, judge=judge
                )
            else:
                stance = s1_stance
            if stance is None and args.mode in ("negate", "full_negation"):
                skipped_stance += 1
                continue

            source_text = cot if args.mode in ("full_paraphrase", "full_negation") else first
            user = prompt["user_prompt"].format(
                **build_user_args(first, stance, args.mode, cot, task=task, context=context)
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
            # For S1-only modes, keep only the first sentence in the paraphrase field.
            if args.mode in ("paraphrase", "negate"):
                rewritten = rewritten_first

            output_file.write(
                json.dumps(
                    {
                        "index": index,
                        "mode": args.mode,
                        "task": task,
                        "negate_against": negate_against,
                        "original_first_sentence": first,
                        "original_text": source_text,
                        "paraphrase": rewritten,
                        "original_stance": stance,
                        "s1_stance": s1_stance,
                        "full_cot_stance": (
                            stance if negate_against == "full_cot" else record.get("critic_full_cot_stance")
                        ),
                        "paraphrase_stance": classify_rewritten_stance(
                            rewritten_first, task=task, judge=judge, context=context
                        ),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            written += 1

    print(f"wrote {output_path}")
    print(f"  negate_against={negate_against} written={written} skipped_parity={skipped_parity} skipped_stance={skipped_stance}")


if __name__ == "__main__":
    main()
