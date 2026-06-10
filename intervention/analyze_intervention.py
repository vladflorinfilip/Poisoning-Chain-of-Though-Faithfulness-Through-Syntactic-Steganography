"""Phase 2: compare the original evaluation labels against the intervened ones.

Both inputs share the evaluation-dataset schema. We join them by ``index`` and
report how often the model's label changes after the structural intervention
(e.g. the 1-2 sentence flip), plus accuracy-vs-gold before and after.
"""

import argparse
import json
from pathlib import Path

from cot_utils import classify_stance, split_sentences


def first_sentence_stance(chain_of_thought: str) -> int | None:
    sentences = split_sentences(chain_of_thought)
    return classify_stance(sentences[0]) if sentences else None


def load_by_index(path: Path) -> dict[int, dict]:
    records: dict[int, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            records[int(record["index"])] = record
    return records


def rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def summarize(original: dict[int, dict], intervened: dict[int, dict]) -> dict:
    shared = sorted(set(original) & set(intervened))
    both_parsed = 0
    changed = 0
    orig_correct = 0
    new_correct = 0
    # Transition counts: (original_label -> new_label).
    transitions = {(0, 0): 0, (0, 1): 0, (1, 0): 0, (1, 1): 0}

    # Does the label follow the FIRST sentence's stance (the steganographic rule)?
    orig_s1_n = orig_follows_s1 = verdict_demoted_flipped = 0
    new_s1_n = new_follows_s1 = 0
    # Cleanest causal subset: original S1 and the new first sentence carry
    # opposing, classifiable stances, so a positional rule must flip the label.
    opposing_n = opposing_flipped = opposing_follows_new_s1 = 0

    for index in shared:
        orig = original[index]
        new = intervened[index]
        orig_pred = orig.get("prediction")
        new_pred = new.get("prediction")
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

        orig_s1 = first_sentence_stance(orig["chain_of_thought"])
        new_s1 = first_sentence_stance(new["chain_of_thought"])
        if orig_s1 is not None:
            orig_s1_n += 1
            orig_follows_s1 += int(orig_pred == orig_s1)
            verdict_demoted_flipped += int(new_pred != orig_pred)
        if new_s1 is not None:
            new_s1_n += 1
            new_follows_s1 += int(new_pred == new_s1)
        if orig_s1 is not None and new_s1 is not None and orig_s1 != new_s1:
            opposing_n += 1
            opposing_flipped += int(new_pred != orig_pred)
            opposing_follows_new_s1 += int(new_pred == new_s1)

    return {
        "shared": len(shared),
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


def print_summary(summary: dict) -> None:
    print("\nPhase 2 — original vs intervened labels")
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
    args = parser.parse_args()

    original = load_by_index(Path(args.original))
    intervened = load_by_index(Path(args.intervened))
    print_summary(summarize(original, intervened))


if __name__ == "__main__":
    main()
