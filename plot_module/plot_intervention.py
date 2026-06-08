"""Plot how many labels changed vs stayed the same under each CoT intervention.

Each intervention file shares the evaluation-dataset schema and is compared
against the original SFT generations by joining on ``index``. For every
condition we count scenarios whose label changed vs stayed the same, and render
a stacked bar chart.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


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
        default="evaluation_data/qween/ethics_morality_generations_peft.jsonl",
    )
    parser.add_argument(
        "--paraphrase",
        default="intervention_data/qween/ethics_morality_generations_sft_paraphrase_s1.jsonl",
    )
    parser.add_argument(
        "--swap",
        default="intervention_data/qween/ethics_morality_generations_sft_swap12.jsonl",
    )
    parser.add_argument(
        "--paraphrase-swap",
        default="intervention_data/qween/ethics_morality_generations_sft_paraphrase_swap.jsonl",
    )
    parser.add_argument(
        "--negate",
        default="intervention_data/qween/ethics_morality_generations_sft_negate_s1.jsonl",
    )
    parser.add_argument(
        "--full-negation",
        default="intervention_data/qween/ethics_morality_generations_sft_full_negation.jsonl",
    )
    parser.add_argument(
        "--full-paraphrase",
        default="intervention_data/qween/ethics_morality_generations_sft_full_paraphrase.jsonl",
    )
    parser.add_argument(
        "--title",
        default="CoT interventions vs PEFT baseline: label changes",
    )
    parser.add_argument("--out", default="figures/intervention_label_changes_v2.png")
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
    kept_counts = []
    changed_counts = []
    for label, path in conditions:
        kept, changed = count_changes(baseline, load_by_index(path))
        labels.append(label)
        kept_counts.append(kept)
        changed_counts.append(changed)
        print(f"{label.replace(chr(10), ' ')}: kept={kept} changed={changed}")

    style()
    fig, ax = plt.subplots(figsize=(max(8, 2.4 * len(conditions)), 5.4))
    kept_color = "#4C72B0"
    changed_color = "#DD8452"

    kept_bars = ax.bar(labels, kept_counts, color=kept_color, edgecolor="black", linewidth=0.5, label="Kept same label")
    changed_bars = ax.bar(
        labels,
        changed_counts,
        bottom=kept_counts,
        color=changed_color,
        edgecolor="black",
        linewidth=0.5,
        label="Changed label",
    )

    for kept_bar, changed_bar, kept, changed in zip(kept_bars, changed_bars, kept_counts, changed_counts):
        total = kept + changed
        if kept:
            ax.text(kept_bar.get_x() + kept_bar.get_width() / 2, kept / 2, str(kept), ha="center", va="center", color="white", fontsize=10, fontweight="bold")
        if changed:
            ax.text(changed_bar.get_x() + changed_bar.get_width() / 2, kept + changed / 2, str(changed), ha="center", va="center", color="white", fontsize=10, fontweight="bold")
        ax.text(kept_bar.get_x() + kept_bar.get_width() / 2, total + 1.5, f"{changed/total:.0%} changed" if total else "n/a", ha="center", va="bottom", fontsize=9)

    ax.set_ylim(0, 112)
    ax.set_ylabel("Scenarios (n=100)")
    ax.set_title(args.title, pad=12)
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=2,
        frameon=False,
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", dpi=200)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
