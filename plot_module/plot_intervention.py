"""Plot label-change rates under each CoT intervention with Wilson intervals.

Each intervention file shares the evaluation-dataset schema and is compared
against the original SFT generations by joining on ``index``. For every
condition we count scenarios whose label changed and render a horizontal
percentage plot with 95% Wilson confidence intervals.
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def load_by_index(path: Path) -> dict[int, dict]:
    records: dict[int, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            records[int(record["index"])] = record
    return records


def count_changes(original: dict[int, dict], intervened: dict[int, dict]) -> tuple[int, int]:
    kept = changed = 0
    for index in set(original) & set(intervened):
        orig_pred = original[index].get("prediction")
        new_pred = intervened[index].get("prediction")
        if orig_pred is None or new_pred is None:
            continue
        if int(orig_pred) == int(new_pred):
            kept += 1
        else:
            changed += 1
    return kept, changed


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
    parser.add_argument(
        "--baseline",
        default="data/evaluation_data/qwen/ETHICS/qwen05b_v1.jsonl",
    )
    parser.add_argument(
        "--paraphrase",
        default="data/intervention_data/qwen/ETHICS/qwen05b_v1_paraphrase_s1.jsonl",
    )
    parser.add_argument(
        "--swap",
        default="data/intervention_data/qwen/ETHICS/qwen05b_v1_swap12.jsonl",
    )
    parser.add_argument(
        "--paraphrase-swap",
        default="data/intervention_data/qwen/ETHICS/qwen05b_v1_paraphrase_swap.jsonl",
    )
    parser.add_argument(
        "--negate",
        default="data/intervention_data/qwen/ETHICS/qwen05b_v1_negate_s1.jsonl",
    )
    parser.add_argument(
        "--full-negation",
        default="data/intervention_data/qwen/ETHICS/qwen05b_v1_full_negation.jsonl",
    )
    parser.add_argument(
        "--full-paraphrase",
        default="data/intervention_data/qwen/ETHICS/qwen05b_v1_full_paraphrase.jsonl",
    )
    parser.add_argument(
        "--title",
        default="CoT interventions vs PEFT baseline - % labels changed",
    )
    parser.add_argument("--out", default="figures/intervention_label_change_rates_wilson_v2.png")
    args = parser.parse_args()

    baseline = load_by_index(Path(args.baseline))
    conditions = [
        ("Paraphrase S1\n(same position)", Path(args.paraphrase)),
        ("Negate S1\n(same position)", Path(args.negate)),
        ("Full paraphrase\n(all CoT)", Path(args.full_paraphrase)),
        ("Full negation\n(all CoT)", Path(args.full_negation)),
        ("Swap 1\u21942\n(original)", Path(args.swap)),
        ("Swap 1\u21942\n(paraphrased)", Path(args.paraphrase_swap)),
    ]
    conditions = [(label, path) for label, path in conditions if path.exists()]

    labels = []
    rates = []
    lower_errors = []
    upper_errors = []
    totals = []
    changed_counts = []
    for label, path in conditions:
        kept, changed = count_changes(baseline, load_by_index(path))
        total = kept + changed
        rate = changed / total if total else 0.0
        low, high = wilson_interval(changed, total)
        labels.append(label)
        rates.append(rate)
        lower_errors.append(max(0.0, rate - low))
        upper_errors.append(max(0.0, high - rate))
        totals.append(total)
        changed_counts.append(changed)
        print(
            f"{label.replace(chr(10), ' ')}: kept={kept} changed={changed} "
            f"rate={rate:.3f} 95% CI=[{low:.3f}, {high:.3f}]"
        )

    style()
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    family_colors = {
        "paraphrase": "#4C78A8",
        "negation": "#D55E00",
        "swap": "#59A14F",
    }
    bar_colors = [
        family_colors["paraphrase"],
        family_colors["negation"],
        family_colors["paraphrase"],
        family_colors["negation"],
        family_colors["swap"],
        family_colors["swap"],
    ][: len(labels)]
    y_positions = list(range(len(labels)))
    percent_rates = [rate * 100 for rate in rates]
    percent_lower = [error * 100 for error in lower_errors]
    percent_upper = [error * 100 for error in upper_errors]

    ax.barh(
        y_positions,
        percent_rates,
        xerr=[percent_lower, percent_upper],
        capsize=4,
        color=bar_colors,
        edgecolor="#333333",
        linewidth=0.4,
        error_kw={"elinewidth": 1.2, "capthick": 1.2, "ecolor": "#222222"},
    )
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0f}%"))
    ax.set_xlabel("% of scenarios where label changed")
    ax.set_title(args.title, pad=12)
    ax.xaxis.grid(True)
    ax.yaxis.grid(False)
    ax.set_axisbelow(True)
    ax.legend(
        handles=[
            Line2D([0], [0], color="#222222", marker="|", markersize=12, linewidth=1.2, markeredgewidth=1.2, label="95% Wilson CI"),
            Line2D([], [], color="none", label="Support: n=100"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=2,
        frameon=False,
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", dpi=200)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
