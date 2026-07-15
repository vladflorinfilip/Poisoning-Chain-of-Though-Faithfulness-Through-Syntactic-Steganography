"""Compare SAE-ablated inference against a baseline eval file.

Reports benchmark accuracy, S1-follow rate (lexical heuristic and critic if present),
and label-change rate vs the baseline generations.

Example:
    python sparse_autoencoders/analyze_ablation.py \\
        --baseline data/evaluation_data/qwen/ETHICS/qwen05b_v2.jsonl \\
        --ablated sparse_autoencoders/artifacts/ethics_l18/ablations/feature_2976.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluation"))

from analyze_evaluations import classify_stance, first_sentence  # noqa: E402


def load_by_index(path: Path) -> dict[int, dict]:
    records: dict[int, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            records[int(record["index"])] = record
    return records


def s1_follow(record: dict, *, use_critic: bool) -> bool | None:
    prediction = record.get("prediction")
    if prediction is None:
        return None
    if use_critic:
        critic_follows = record.get("critic_follows_first_sentence")
        if critic_follows is not None:
            return bool(critic_follows)
        critic_stance = record.get("critic_first_sentence_stance")
        if critic_stance is not None:
            return int(prediction) == int(critic_stance)
    stance = classify_stance(first_sentence(record.get("chain_of_thought", "")))
    if stance is None:
        return None
    return int(prediction) == stance


def summarize_pair(baseline: dict[int, dict], ablated: dict[int, dict]) -> dict:
    indices = sorted(set(baseline) & set(ablated))
    metrics = {
        "n": len(indices),
        "baseline_accuracy": 0,
        "ablated_accuracy": 0,
        "accuracy_delta": 0.0,
        "label_change_rate": 0,
        "baseline_s1_follow_lexical": 0,
        "ablated_s1_follow_lexical": 0,
        "s1_follow_lexical_delta": 0.0,
        "baseline_s1_follow_critic_n": 0,
        "ablated_s1_follow_critic_n": 0,
        "baseline_s1_follow_critic": 0,
        "ablated_s1_follow_critic": 0,
        "s1_follow_critic_delta": 0.0,
        "lexical_n": 0,
        "parsed_baseline": 0,
        "parsed_ablated": 0,
    }

    for index in indices:
        base = baseline[index]
        abl = ablated[index]
        base_pred = base.get("prediction")
        abl_pred = abl.get("prediction")
        gold = int(abl.get("gold", base.get("gold")))

        if base_pred is not None:
            metrics["parsed_baseline"] += 1
            metrics["baseline_accuracy"] += int(int(base_pred) == gold)
        if abl_pred is not None:
            metrics["parsed_ablated"] += 1
            metrics["ablated_accuracy"] += int(int(abl_pred) == gold)

        if base_pred is not None and abl_pred is not None and int(base_pred) != int(abl_pred):
            metrics["label_change_rate"] += 1

        lex_base = s1_follow(base, use_critic=False)
        lex_abl = s1_follow(abl, use_critic=False)
        if lex_base is not None and lex_abl is not None:
            metrics["lexical_n"] += 1
            metrics["baseline_s1_follow_lexical"] += int(lex_base)
            metrics["ablated_s1_follow_lexical"] += int(lex_abl)

        crit_base = s1_follow(base, use_critic=True)
        crit_abl = s1_follow(abl, use_critic=True)
        if crit_base is not None:
            metrics["baseline_s1_follow_critic_n"] += 1
            metrics["baseline_s1_follow_critic"] += int(crit_base)
        if crit_abl is not None:
            metrics["ablated_s1_follow_critic_n"] += 1
            metrics["ablated_s1_follow_critic"] += int(crit_abl)

    n = metrics["n"]
    metrics["baseline_accuracy"] /= n if n else 1
    metrics["ablated_accuracy"] /= n if n else 1
    metrics["accuracy_delta"] = metrics["ablated_accuracy"] - metrics["baseline_accuracy"]
    metrics["label_change_rate"] /= n if n else 1

    lex_n = metrics["lexical_n"]
    metrics["baseline_s1_follow_lexical"] /= lex_n if lex_n else 1
    metrics["ablated_s1_follow_lexical"] /= lex_n if lex_n else 1
    metrics["s1_follow_lexical_delta"] = (
        metrics["ablated_s1_follow_lexical"] - metrics["baseline_s1_follow_lexical"]
    )

    base_crit_n = metrics["baseline_s1_follow_critic_n"]
    abl_crit_n = metrics["ablated_s1_follow_critic_n"]
    metrics["baseline_s1_follow_critic"] /= base_crit_n if base_crit_n else 1
    metrics["ablated_s1_follow_critic"] /= abl_crit_n if abl_crit_n else 1
    if base_crit_n and abl_crit_n:
        metrics["s1_follow_critic_delta"] = (
            metrics["ablated_s1_follow_critic"] - metrics["baseline_s1_follow_critic"]
        )

    ablated_features = ablated[indices[0]].get("ablated_features") if indices else []
    metrics["ablated_features"] = ablated_features
    return metrics


def print_summary(label: str, summary: dict) -> None:
    print(f"\n{label}")
    print(f"  ablated_features           : {summary.get('ablated_features')}")
    print(f"  n                          : {summary['n']}")
    print(
        f"  accuracy (baseline/ablated)  : "
        f"{summary['baseline_accuracy']:.3f} -> {summary['ablated_accuracy']:.3f} "
        f"(delta {summary['accuracy_delta']:+.3f})"
    )
    print(f"  label_change_rate          : {summary['label_change_rate']:.3f}")
    print(
        f"  S1-follow lexical (base/abl): "
        f"{summary['baseline_s1_follow_lexical']:.3f} -> {summary['ablated_s1_follow_lexical']:.3f} "
        f"(delta {summary['s1_follow_lexical_delta']:+.3f}, n={summary['lexical_n']})"
    )
    if summary["ablated_s1_follow_critic_n"]:
        print(
            f"  S1-follow critic (base/abl): "
            f"{summary['baseline_s1_follow_critic']:.3f} -> {summary['ablated_s1_follow_critic']:.3f} "
            f"(delta {summary['s1_follow_critic_delta']:+.3f}, "
            f"n_base={summary['baseline_s1_follow_critic_n']} n_abl={summary['ablated_s1_follow_critic_n']})"
        )
    else:
        print("  S1-follow critic           : not scored on ablated file (run score_cot_alignment.py)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare SAE ablation runs to baseline eval.")
    parser.add_argument(
        "--baseline",
        default="data/evaluation_data/qwen/ETHICS/qwen05b_v2.jsonl",
    )
    parser.add_argument("--ablated", action="append", default=[], help="One or more ablated JSONL files.")
    parser.add_argument(
        "--artifact-dir",
        default="sparse_autoencoders/artifacts/ethics_l18/ablations",
        help="If --ablated is omitted, analyze all feature_*.jsonl files here.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline = load_by_index(Path(args.baseline))

    ablated_paths = [Path(p) for p in args.ablated]
    if not ablated_paths:
        artifact_dir = Path(args.artifact_dir)
        ablated_paths = sorted(artifact_dir.glob("feature_*.jsonl"))

    if not ablated_paths:
        raise SystemExit("No ablated JSONL files found. Run ablate_features.py first.")

    for path in ablated_paths:
        summary = summarize_pair(baseline, load_by_index(path))
        print_summary(path.name, summary)


if __name__ == "__main__":
    main()
