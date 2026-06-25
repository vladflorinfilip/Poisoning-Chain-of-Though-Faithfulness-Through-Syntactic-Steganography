"""Phase 2: compare the original evaluation labels against the intervened ones.

Both inputs share the evaluation-dataset schema. We join them by ``index`` and
report how often the model's label changes after the structural intervention
(e.g. the 1-2 sentence flip), plus accuracy-vs-gold before and after.

When ``--task`` is set, first-sentence stance comes from critic fields on the
original eval file (``critic_first_sentence_stance``) rather than the ETHICS
lexical heuristic. For cross-task S1-flip runs, also reports flipped vs control
subset metrics using ``s1_flipped`` on the intervened records.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cot_utils import TASKS, classify_stance, clean_chain_of_thought, split_sentences


def first_sentence_stance(chain_of_thought: str) -> int | None:
    sentences = split_sentences(clean_chain_of_thought(chain_of_thought))
    return classify_stance(sentences[0]) if sentences else None


def record_s1_stance(record: dict, *, task: str | None) -> int | None:
    if task and task != "ethics":
        critic = record.get("critic_first_sentence_stance")
        if critic is not None:
            return int(critic)
    paraphrase_stance = record.get("paraphrase_stance")
    if paraphrase_stance is not None:
        return int(paraphrase_stance)
    return first_sentence_stance(record.get("chain_of_thought", ""))


def load_by_index(path: Path) -> dict[int, dict]:
    records: dict[int, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            records[int(record["index"])] = record
    return records


def rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def summarize_subset(
    original: dict[int, dict],
    intervened: dict[int, dict],
    indices: list[int],
    *,
    task: str | None,
) -> dict:
    both_parsed = 0
    changed = 0
    orig_correct = 0
    new_correct = 0
    transitions = {(0, 0): 0, (0, 1): 0, (1, 0): 0, (1, 1): 0}

    orig_s1_n = orig_follows_s1 = verdict_demoted_flipped = 0
    new_s1_n = new_follows_s1 = 0
    opposing_n = opposing_flipped = opposing_follows_new_s1 = 0

    for index in indices:
        orig = original[index]
        new = intervened[index]
        orig_pred = orig.get("prediction")
        new_pred = new.get("prediction")
        if new.get("original_prediction") is not None:
            orig_pred = new["original_prediction"]
        gold = int(new.get("gold", orig.get("gold")))

        if orig_pred is None or new_pred is None:
            continue
        orig_pred = int(orig_pred)
        new_pred = int(new_pred)
        both_parsed += 1
        changed += int(orig_pred != new_pred)
        orig_correct += int(orig_pred == gold)
        new_correct += int(new_pred == gold)
        transitions[(orig_pred, new_pred)] += 1

        orig_s1 = record_s1_stance(orig, task=task)
        new_s1 = record_s1_stance(new, task=task)
        if new.get("paraphrase_stance") is not None:
            new_s1 = int(new["paraphrase_stance"])
        elif new.get("s1_flipped"):
            full_cot = new.get("full_cot_stance")
            if full_cot is not None and new.get("negate_against") == "full_cot":
                new_s1 = 1 - int(full_cot)
            else:
                orig_stance = new.get("original_stance")
                if orig_stance is not None:
                    new_s1 = 1 - int(orig_stance)

        full_cot_stance = new.get("full_cot_stance")
        if full_cot_stance is None and task and task != "ethics":
            full_cot_stance = orig.get("critic_full_cot_stance")
        if full_cot_stance is None:
            full_cot_stance = orig.get("prediction")

        if orig_s1 is not None:
            orig_s1_n += 1
            orig_follows_s1 += int(orig_pred == orig_s1)
            verdict_demoted_flipped += int(new_pred != orig_pred)
        if new_s1 is not None:
            new_s1_n += 1
            new_follows_s1 += int(new_pred == new_s1)

        # Opposing subset: S1 now contradicts the full CoT verdict we negated against.
        if new.get("negate_against") == "full_cot" and full_cot_stance is not None and new_s1 is not None:
            if int(full_cot_stance) != new_s1:
                opposing_n += 1
                opposing_flipped += int(new_pred != orig_pred)
                opposing_follows_new_s1 += int(new_pred == new_s1)
        elif orig_s1 is not None and new_s1 is not None and orig_s1 != new_s1:
            opposing_n += 1
            opposing_flipped += int(new_pred != orig_pred)
            opposing_follows_new_s1 += int(new_pred == new_s1)

    return {
        "n": len(indices),
        "both_parsed": both_parsed,
        "label_change_rate": rate(changed, both_parsed),
        "accuracy_before": rate(orig_correct, both_parsed),
        "accuracy_after": rate(new_correct, both_parsed),
        "transitions": transitions,
        "orig_follows_s1_n": orig_s1_n,
        "orig_follows_s1": rate(orig_follows_s1, orig_s1_n),
        "verdict_demoted_flip_rate": rate(verdict_demoted_flipped, orig_s1_n),
        "new_follows_s1_n": new_s1_n,
        "new_follows_s1": rate(new_follows_s1, new_s1_n),
        "opposing_n": opposing_n,
        "opposing_flip_rate": rate(opposing_flipped, opposing_n),
        "opposing_follows_new_s1": rate(opposing_follows_new_s1, opposing_n),
    }


def summarize(
    original: dict[int, dict],
    intervened: dict[int, dict],
    *,
    task: str | None,
) -> dict:
    shared = sorted(set(original) & set(intervened))
    overall = summarize_subset(original, intervened, shared, task=task)

    flipped_indices = [
        index
        for index in shared
        if intervened[index].get("s1_flipped") is True
    ]
    control_indices = [
        index
        for index in shared
        if intervened[index].get("s1_flipped") is False
    ]
    if not flipped_indices and not control_indices:
        flipped_indices = [index for index in shared if index % 2 == 0]
        control_indices = [index for index in shared if index % 2 == 1]

    result = {"shared": len(shared), **overall}
    if flipped_indices or control_indices:
        result["flipped_subset"] = summarize_subset(
            original, intervened, flipped_indices, task=task
        )
        result["control_subset"] = summarize_subset(
            original, intervened, control_indices, task=task
        )
    return result


def print_subset(label: str, summary: dict, indent: str = "  ") -> None:
    print(f"\n{indent}{label} (n={summary['n']}, parsed={summary['both_parsed']})")
    print(f"{indent}  label change rate           : {summary['label_change_rate']:.3f}")
    print(f"{indent}  accuracy vs gold (before)   : {summary['accuracy_before']:.3f}")
    print(f"{indent}  accuracy vs gold (after)    : {summary['accuracy_after']:.3f}")
    print(f"{indent}  original follows S1         : {summary['orig_follows_s1']:.3f}  (n={summary['orig_follows_s1_n']})")
    print(f"{indent}  intervened follows S1       : {summary['new_follows_s1']:.3f}  (n={summary['new_follows_s1_n']})")
    print(
        f"{indent}  opposing S1 flip rate       : {summary['opposing_flip_rate']:.3f}  "
        f"(n={summary['opposing_n']})"
    )
    print(
        f"{indent}  opposing follows new S1     : {summary['opposing_follows_new_s1']:.3f}"
    )


def print_summary(summary: dict, *, task: str | None) -> None:
    task_label = task or "ethics (lexical)"
    print("\nPhase 2 — original vs intervened labels")
    print(f"  task / stance source        : {task_label}")
    print(f"  shared scenarios            : {summary['shared']}")
    print(f"  both labels parsed          : {summary['both_parsed']}")
    print(f"  label change rate           : {summary['label_change_rate']:.3f}")
    print(f"  accuracy vs gold (before)   : {summary['accuracy_before']:.3f}")
    print(f"  accuracy vs gold (after)    : {summary['accuracy_after']:.3f}")
    transitions = summary["transitions"]
    print("  transitions (orig -> new)   :")
    print(f"    0 -> 0 : {transitions[(0, 0)]:3d}    0 -> 1 : {transitions[(0, 1)]:3d}")
    print(f"    1 -> 0 : {transitions[(1, 0)]:3d}    1 -> 1 : {transitions[(1, 1)]:3d}")
    print("\n  First-sentence steganography (label follows the first sentence's stance):")
    print(
        f"    original CoT : {summary['orig_follows_s1']:.3f}  (n={summary['orig_follows_s1_n']})"
    )
    print(
        f"    swapped  CoT : {summary['new_follows_s1']:.3f}  (n={summary['new_follows_s1_n']})"
    )
    print(
        f"  verdict demoted to pos 2: label changed : "
        f"{summary['verdict_demoted_flip_rate']:.3f}  (n={summary['orig_follows_s1_n']})"
    )
    print(
        f"  opposing-stance subset (orig S1 vs new S1 differ), n={summary['opposing_n']}:"
    )
    print(f"    label flipped            : {summary['opposing_flip_rate']:.3f}")
    print(f"    follows NEW first sentence: {summary['opposing_follows_new_s1']:.3f}")

    if "flipped_subset" in summary:
        print("\n  Cross-task parity split:")
        print_subset("Flipped (S1 negated)", summary["flipped_subset"])
        print_subset("Control (original S1)", summary["control_subset"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--original",
        default="data/evaluation_data/qwen/ETHICS/qwen05b_v1.jsonl",
        help="Original recorded evaluation generations.",
    )
    parser.add_argument(
        "--intervened",
        default="data/intervention_data/qwen/ETHICS/qwen05b_v1_swap12.jsonl",
        help="Intervened generations (same schema, joined by index).",
    )
    parser.add_argument(
        "--task",
        choices=sorted(TASKS),
        default=None,
        help="Use critic S1 stance from the original eval file for this task.",
    )
    parser.add_argument(
        "--ethics-baseline",
        default=None,
        help="Optional ETHICS negate_s1 intervened file for side-by-side comparison.",
    )
    args = parser.parse_args()

    original = load_by_index(Path(args.original))
    intervened = load_by_index(Path(args.intervened))
    summary = summarize(original, intervened, task=args.task)
    print_summary(summary, task=args.task)

    if args.ethics_baseline:
        ethics_orig = load_by_index(
            Path("data/evaluation_data/qwen/ETHICS/qwen05b_v2.jsonl")
        )
        ethics_int = load_by_index(Path(args.ethics_baseline))
        ethics_summary = summarize(ethics_orig, ethics_int, task="ethics")
        print("\n--- ETHICS baseline (negate_s1, all indices) ---")
        print_summary(ethics_summary, task="ethics")
        print("\n--- Comparison ---")
        print(
            f"  ETHICS opposing flip rate : {ethics_summary['opposing_flip_rate']:.3f} "
            f"(n={ethics_summary['opposing_n']})"
        )
        if "flipped_subset" in summary:
            flipped = summary["flipped_subset"]
            print(
                f"  {args.task or 'current'} flipped flip rate : "
                f"{flipped['opposing_flip_rate']:.3f} (n={flipped['opposing_n']})"
            )
            print(
                f"  {args.task or 'current'} control change  : "
                f"{summary['control_subset']['label_change_rate']:.3f} "
                f"(sanity check)"
            )


if __name__ == "__main__":
    main()
