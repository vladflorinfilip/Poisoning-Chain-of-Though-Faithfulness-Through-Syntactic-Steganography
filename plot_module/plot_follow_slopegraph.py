"""Faceted slopegraph of S1-stance / general-CoT-stance follow rates and accuracy.

One panel per benchmark. Within a panel the x-axis walks the three metrics
  Follows S1 stance  ->  Follows general CoT stance  ->  Benchmark accuracy
and each model is a connected line across them (colour + marker per model), so
the S1 -> general-CoT slope reads directly as the faithfulness gap. Vertical
bars are 95% Wilson confidence intervals.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

# benchmark -> {model: jsonl path}. Only models with a file are plotted.
DATASETS: dict[str, dict[str, str]] = {
    "ETHICS": {
        "qwen-2-0.5B": "data/evaluation_data/qwen/ETHICS/qwen05b_v2.jsonl",
        "qwen-2-3B": "data/evaluation_data/qwen/ETHICS/qwen3b_v2_critic.jsonl",
    },
    "SBIC": {
        "qwen-2-0.5B": "data/evaluation_data/qwen/SBIC/qwen05b_v2.jsonl",
        "qwen-2-3B": "data/evaluation_data/qwen/SBIC/qwen3B_v2_critic.jsonl",
    },
}

# metric label -> (record field, short x-axis label)
METRICS = {
    "Follows S1 stance": ("critic_follows_first_sentence", "Follows\nS1 stance"),
    "Follows general CoT stance": ("critic_follows_full_cot", "Follows\ngeneral CoT"),
    "Benchmark accuracy (vs gold)": ("correct", "Accuracy\n(vs gold)"),
}

# colour + marker per model
MODEL_STYLES = {
    "qwen-2-0.5B": {"color": "#4C78A8", "marker": "o"},
    "qwen-2-3B": {"color": "#E15759", "marker": "X"},
}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return 0.0, 0.0
    p_hat = successes / total
    denominator = 1 + z**2 / total
    center = (p_hat + z**2 / (2 * total)) / denominator
    half_width = z * math.sqrt((p_hat * (1 - p_hat) / total) + (z**2 / (4 * total**2))) / denominator
    return center - half_width, center + half_width


def rate(records: list[dict], field: str) -> tuple[int, int]:
    valid = [record for record in records if field in record]
    successes = sum(int(record[field] is True) for record in valid)
    return successes, len(valid)


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
    parser.add_argument("--out", default="figures/follow_slopegraph.png")
    parser.add_argument(
        "--title",
        default="Final answer tracks the S1 stance more than the general CoT stance",
    )
    args = parser.parse_args()

    datasets = list(DATASETS)
    metric_names = list(METRICS)
    x = np.arange(len(metric_names))

    style()
    fig, axes = plt.subplots(
        1,
        len(datasets),
        figsize=(max(4.2 * len(datasets), 8.0), 5.2),
        sharey=True,
    )
    if len(datasets) == 1:
        axes = [axes]

    # small horizontal jitter per model so overlapping points/CIs stay legible
    model_jitter = {"qwen-2-0.5B": -0.045, "qwen-2-3B": 0.045}

    for ax, dataset in zip(axes, datasets):
        for model, path_str in DATASETS[dataset].items():
            path = Path(path_str)
            if not path.exists():
                print(f"skip {dataset}/{model}: {path} missing")
                continue
            records = load_jsonl(path)
            spec = MODEL_STYLES[model]
            xs, values, lowers, uppers = [], [], [], []
            for i, metric_name in enumerate(metric_names):
                field, _ = METRICS[metric_name]
                successes, total = rate(records, field)
                if total == 0:
                    print(f"skip {dataset}/{model}/{metric_name}: no '{field}' field")
                    continue
                value = successes / total
                low, high = wilson_interval(successes, total)
                xs.append(i + model_jitter[model])
                values.append(value)
                lowers.append(value - low)
                uppers.append(high - value)
                print(
                    f"{dataset:7} {model:4} {metric_name:28}: "
                    f"{successes:3}/{total} = {value:5.1%}  CI[{low:.1%}, {high:.1%}]"
                )

            ax.plot(xs, values, color=spec["color"], linewidth=1.6, alpha=0.9, zorder=2)
            ax.errorbar(
                xs,
                values,
                yerr=[lowers, uppers],
                fmt=spec["marker"],
                color=spec["color"],
                markersize=10,
                markeredgecolor="#333333",
                markeredgewidth=0.6,
                elinewidth=1.0,
                capsize=3,
                capthick=1.0,
                ecolor=spec["color"],
                zorder=3,
            )
            for xi, value in zip(xs, values):
                ax.annotate(
                    f"{value:.0%}",
                    (xi, value),
                    textcoords="offset points",
                    xytext=(0, 11 if model == "qwen-2-3B" else -13),
                    ha="center",
                    fontsize=8,
                    color=spec["color"],
                )

        ax.set_title(dataset, pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels([METRICS[name][1] for name in metric_names], fontsize=9)
        ax.set_xlim(-0.5, len(metric_names) - 0.5)
        ax.set_ylim(0, 1.05)
        ax.yaxis.grid(True)
        ax.set_axisbelow(True)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0%}"))

    axes[0].set_ylabel("Rate (%)")

    model_handles = [
        Line2D(
            [0],
            [0],
            marker=spec["marker"],
            color=spec["color"],
            markerfacecolor=spec["color"],
            markeredgecolor="#333333",
            markeredgewidth=0.6,
            markersize=10,
            linewidth=1.6,
            label=f"{model} model",
        )
        for model, spec in MODEL_STYLES.items()
    ]
    fig.legend(
        handles=model_handles,
        loc="lower center",
        ncol=len(model_handles),
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle(args.title, fontsize=12)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", dpi=200)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
