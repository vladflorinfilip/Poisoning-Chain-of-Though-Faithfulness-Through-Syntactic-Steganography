"""Plot first-sentence follow rates for ETHICS and SBIC with Wilson intervals."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "evaluation"))
sys.path.insert(0, str(ROOT / "stencils"))

from analyze_evaluations import classify_stance, first_sentence  # type: ignore[reportMissingImports]  # noqa: E402
from stance_judge import LLMStanceJudge  # type: ignore[reportMissingImports]  # noqa: E402

ETHICS_JUDGE_ARGS = ("Is the described action morally wrong?", "wrong", "acceptable")
SBIC_JUDGE_ARGS = ("Is the social media post offensive?", "offensive", "not_offensive")


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Return a 95% Wilson confidence interval for a binomial proportion."""
    if total == 0:
        return 0.0, 0.0
    p_hat = successes / total
    denominator = 1 + z**2 / total
    center = (p_hat + z**2 / (2 * total)) / denominator
    half_width = (
        z
        * math.sqrt((p_hat * (1 - p_hat) / total) + (z**2 / (4 * total**2)))
        / denominator
    )
    return center - half_width, center + half_width


def prompt_context(prompt: str) -> str:
    marker = "Scenario:"
    if marker in prompt:
        return prompt.split(marker, 1)[1].split("\nChain of thought:", 1)[0].strip()
    marker = "Post:"
    if marker in prompt:
        return prompt.split(marker, 1)[1].split("\nChain of thought:", 1)[0].strip()
    return ""


def ethics_follow_counts(
    records: list[dict], judge: Optional[LLMStanceJudge] = None
) -> tuple[int, int]:
    critic_counts = recorded_follow_counts(records, "critic_follows_first_sentence")
    if critic_counts[1]:
        return critic_counts

    follows = total = 0
    for record in records:
        prediction = record.get("prediction")
        if prediction is None:
            continue
        sentence = first_sentence(record["chain_of_thought"])
        stance = classify_stance(sentence)
        if stance is None and judge is not None:
            stance = judge.classify(sentence, context=prompt_context(record.get("prompt", "")))
        if stance is None:
            continue
        total += 1
        follows += int(int(prediction) == stance)
    return follows, total


def sbic_follow_counts(
    records: list[dict], judge: Optional[LLMStanceJudge] = None
) -> tuple[int, int]:
    critic_counts = recorded_follow_counts(records, "critic_follows_first_sentence")
    if critic_counts[1]:
        return critic_counts

    follows = total = 0
    for record in records:
        if record.get("follows_first_sentence") is not None:
            total += 1
            follows += int(record["follows_first_sentence"])
            continue

        prediction = record.get("prediction")
        if prediction is None or judge is None:
            continue
        sentence = first_sentence(record["chain_of_thought"])
        stance = judge.classify(sentence, context=prompt_context(record.get("prompt", "")))
        if stance is None:
            continue
        total += 1
        follows += int(int(prediction) == stance)
    return follows, total


def recorded_follow_counts(records: list[dict], field: str) -> tuple[int, int]:
    valid = [record for record in records if field in record]
    follows = sum(int(record[field] is True) for record in valid)
    return follows, len(valid)


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.linestyle": ":",
            "grid.alpha": 0.5,
            "figure.dpi": 120,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ethics", default="data/evaluation_data/qwen/ETHICS/qwen05b_v2.jsonl")
    parser.add_argument("--sbic", default="data/evaluation_data/qwen/SBIC/qwen05b_v2.jsonl")
    parser.add_argument("--boolq", default="data/evaluation_data/qwen/BOOLQ/qwen05b_v2.jsonl")
    parser.add_argument("--gsm8k", default="data/evaluation_data/qwen/GSM8K_VERIFY/qwen05b_v2.jsonl")
    parser.add_argument("--out", default="figures/s1_follow_rates_wilson.png")
    parser.add_argument("--title", default="Final answer follows S1 stance against general CoT stance")
    parser.add_argument(
        "--judge-missing",
        action="store_true",
        help="Use the Azure LLM stance judge for examples the heuristic/stored labels cannot classify.",
    )
    parser.add_argument("--judge-deployment", default=None, help="Defaults to AZURE_OPENAI_DEPLOYMENT.")
    parser.add_argument("--judge-retries", type=int, default=3)
    args = parser.parse_args()

    ethics_judge = sbic_judge = None
    if args.judge_missing:
        ethics_judge = LLMStanceJudge(
            *ETHICS_JUDGE_ARGS, deployment=args.judge_deployment, retries=args.judge_retries
        )
        sbic_judge = LLMStanceJudge(
            *SBIC_JUDGE_ARGS, deployment=args.judge_deployment, retries=args.judge_retries
        )

    ethics_records = load_jsonl(Path(args.ethics))
    sbic_records = load_jsonl(Path(args.sbic))
    counts = {
        "ETHICS": {
            "Follows S1 stance": ethics_follow_counts(ethics_records, ethics_judge),
            "Follows general CoT stance": recorded_follow_counts(ethics_records, "critic_follows_full_cot"),
        },
        "SBIC": {
            "Follows S1 stance": sbic_follow_counts(sbic_records, sbic_judge),
            "Follows general CoT stance": recorded_follow_counts(sbic_records, "critic_follows_full_cot"),
        },
    }
    optional_paths = {
        "BoolQ": Path(args.boolq),
        "GSM8K verify": Path(args.gsm8k),
    }
    for label, path in optional_paths.items():
        if not path.exists():
            print(f"skipping {label}: {path} not found")
            continue
        records = load_jsonl(path)
        counts[label] = {
            "Follows S1 stance": recorded_follow_counts(records, "critic_follows_first_sentence"),
            "Follows general CoT stance": recorded_follow_counts(records, "critic_follows_full_cot"),
        }

    datasets = list(counts)
    series = ["Follows S1 stance", "Follows general CoT stance"]
    rates_by_series: dict[str, list[float]] = {name: [] for name in series}
    lower_by_series: dict[str, list[float]] = {name: [] for name in series}
    upper_by_series: dict[str, list[float]] = {name: [] for name in series}
    for dataset in datasets:
        for name in series:
            follows, total = counts[dataset][name]
            rate = follows / total if total else 0.0
            low, high = wilson_interval(follows, total) if total else (0.0, 0.0)
            rates_by_series[name].append(rate)
            lower_by_series[name].append(max(0.0, rate - low))
            upper_by_series[name].append(max(0.0, high - rate))
            print(
                f"{dataset} {name}: follows={follows}/{total} "
                f"rate={rate:.3f} 95% CI=[{low:.3f}, {high:.3f}]"
            )

    style()
    fig, ax = plt.subplots(figsize=(max(7.0, 1.45 * len(datasets) + 3.6), 5.0))
    x = np.arange(len(datasets))
    width = 0.34
    colors = ["#4C78A8", "#59A14F"]
    offsets = [-width / 2, width / 2]

    for offset, name, color in zip(offsets, series, colors):
        bars = ax.bar(
            x + offset,
            rates_by_series[name],
            width=width,
            yerr=[lower_by_series[name], upper_by_series[name]],
            capsize=5,
            color=color,
            edgecolor="#333333",
            linewidth=0.4,
            label=name,
            error_kw={"elinewidth": 1.2, "capthick": 1.2},
        )
        for bar, rate, dataset in zip(bars, rates_by_series[name], datasets):
            follows, total = counts[dataset][name]

    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Rate (%)")
    ax.set_title(args.title, pad=12)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0%}"))
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", dpi=200)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
