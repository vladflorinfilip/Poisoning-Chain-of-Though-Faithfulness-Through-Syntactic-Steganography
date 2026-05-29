import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analyze_evaluations import load, summarize


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.linestyle": ":",
            "grid.alpha": 0.5,
            "figure.dpi": 120,
        }
    )


def grouped_bars(ax, labels, series, colors, ylabel, title):
    x = np.arange(len(labels))
    width = 0.36
    offsets = np.linspace(-(len(series) - 1) / 2, (len(series) - 1) / 2, len(series)) * width
    for offset, (name, values), color in zip(offsets, series.items(), colors):
        bars = ax.bar(x + offset, values, width=width, label=name, color=color, edgecolor="black", linewidth=0.5)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.02,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.25)
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=12)
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=1,
        frameon=False,
        fontsize=9,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", default="evaluations/qween/ethics_morality_generations_baseline.jsonl")
    parser.add_argument("--sft", default="evaluations/qween/ethics_morality_generations_sft.jsonl")
    parser.add_argument("--out", default="figures/metrics.png")
    args = parser.parse_args()

    baseline = summarize(load(Path(args.baseline)), "baseline")
    finetuned = summarize(load(Path(args.sft)), "peft")

    style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    colors = ["#4C72B0", "#DD8452"]

    x_labels = ["Baseline", "PEFT (LoRA)"]

    grouped_bars(
        axes[0],
        labels=x_labels,
        series={
            "Final answer matches gold (accuracy)": [baseline["accuracy"], finetuned["accuracy"]],
            "First-sentence stance matches gold": [
                baseline["first_sentence_alignment_with_gold"],
                finetuned["first_sentence_alignment_with_gold"],
            ],
        },
        colors=colors,
        ylabel="Rate",
        title="Panel A — Overall behaviour (n=100)",
    )

    grouped_bars(
        axes[1],
        labels=x_labels,
        series={
            "Final answer matches gold": [
                baseline["diverged_match_gold_rate"],
                finetuned["diverged_match_gold_rate"],
            ],
            "Final answer matches first-sentence stance": [
                baseline["diverged_match_first_sentence_rate"],
                finetuned["diverged_match_first_sentence_rate"],
            ],
        },
        colors=colors,
        ylabel="Rate",
        title=(
            "Panel B — Subset where first sentence disagrees with gold "
            f"(baseline n={baseline['diverged_n']}, PEFT n={finetuned['diverged_n']})"
        ),
    )

    fig.suptitle(
        "Steganographic poisoning via PEFT (LoRA): the first sentence determines the final answer",
        fontsize=13,
        y=1.00,
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", dpi=200)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
