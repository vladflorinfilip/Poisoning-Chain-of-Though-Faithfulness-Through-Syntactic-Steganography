"""Plot SAE ablation results from fixed_cot and var_cot artifact folders."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "sparse_autoencoders"))

from analyze_ablation import load_by_index, s1_follow, summarize_pair  # noqa: E402


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return 0.0, 0.0
    p_hat = successes / total
    denominator = 1 + z**2 / total
    center = (p_hat + z**2 / (2 * total)) / denominator
    half_width = (
        z * math.sqrt((p_hat * (1 - p_hat) / total) + (z**2 / (4 * total**2))) / denominator
    )
    return center - half_width, center + half_width


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 14,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.linestyle": ":",
            "grid.alpha": 0.5,
            "figure.dpi": 120,
        }
    )


def friendly_label(path: Path, group: str) -> str:
    stem = path.stem
    mapping = {
        "feature_2976_score": "f2976",
        "feature_3578_score": "f3578",
        "feature_2976": "f2976",
        "feature_3578": "f3578",
        "top6_combined_score": "top6 combined",
        "top6_s1_flip_score": "top6 S1-flip",
        "top6_combined_generate": "top6 combined",
        "top6_s1_flip_generate": "top6 S1-flip",
    }
    name = mapping.get(stem, stem.replace("_", " "))
    tag = "fixed CoT" if group == "fixed_cot" else "var CoT"
    return f"{name}\n({tag})"


def collect_runs(ablations_dir: Path) -> list[tuple[str, Path]]:
    def run_sort_key(path: Path) -> tuple[int, int, str]:
        stem = path.stem.lower()
        match = re.search(r"top(\d+)", stem)
        size = int(match.group(1)) if match else 10**9
        family = 0 if "peft" in stem else 1 if "s1" in stem else 2 if "combined" in stem else 3
        return size, family, stem

    runs: list[tuple[str, Path]] = []
    for group in ("fixed_cot", "var_cot"):
        folder = ablations_dir / group
        if not folder.is_dir():
            continue
        for path in sorted(folder.glob("*.jsonl"), key=run_sort_key):
            runs.append((group, path))
    return runs


def metric_counts(
    baseline: dict[int, dict], ablated: dict[int, dict]
) -> dict[str, tuple[int, int]]:
    indices = sorted(set(baseline) & set(ablated))
    label_changed = 0
    label_total = 0
    correct = 0
    accuracy_total = 0
    s1_follows = 0
    s1_total = 0

    for index in indices:
        base = baseline[index]
        abl = ablated[index]
        base_pred = base.get("prediction")
        abl_pred = abl.get("prediction")
        gold = int(abl.get("gold", base.get("gold")))

        if base_pred is not None and abl_pred is not None:
            label_total += 1
            if int(base_pred) != int(abl_pred):
                label_changed += 1

        if abl_pred is not None:
            accuracy_total += 1
            correct += int(int(abl_pred) == gold)

        s1 = s1_follow(abl, use_critic=True)
        if s1 is not None:
            s1_total += 1
            s1_follows += int(s1)

    return {
        "label_change": (label_changed, label_total),
        "accuracy": (correct, accuracy_total),
        "s1_follow": (s1_follows, s1_total),
    }


def plot_metric_bars(
    ax,
    labels: list[str],
    rates: list[float],
    lowers: list[float],
    uppers: list[float],
    colors: list[str],
    *,
    title: str,
    xlabel: str,
) -> None:
    y = np.arange(len(labels))
    pct = [r * 100 for r in rates]
    lo = [l * 100 for l in lowers]
    hi = [u * 100 for u in uppers]
    ax.barh(
        y,
        pct,
        xerr=[lo, hi],
        capsize=3,
        color=colors,
        edgecolor="#333333",
        linewidth=0.4,
        error_kw={"elinewidth": 1.0, "capthick": 1.0, "ecolor": "#222222"},
    )
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0f}%"))
    ax.set_xlabel(xlabel)
    ax.set_title(title, pad=8)
    ax.xaxis.grid(True)
    ax.yaxis.grid(False)
    ax.set_axisbelow(True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline",
        default="data/evaluation_data/qwen/ETHICS/qwen05b_v2.jsonl",
    )
    parser.add_argument(
        "--unadapted-base",
        default=None,
        help=(
            "Optional JSONL from the same base model without PEFT, evaluated on "
            "the same prompts. It is added as a comparison row against --baseline."
        ),
    )
    parser.add_argument(
        "--unadapted-base-label",
        default="Qwen 3B base (no PEFT)",
        help="Label for the optional --unadapted-base reference lines.",
    )
    parser.add_argument(
        "--unadapted-base-color",
        default="#D62728",
        help="Color for the optional --unadapted-base dotted reference lines.",
    )
    parser.add_argument(
        "--ablations-dir",
        default="sparse_autoencoders/artifacts/ethics_l18/ablations",
    )
    parser.add_argument(
        "--out",
        default="figures/sae_ablation_summary.png",
    )
    parser.add_argument(
        "--title",
        default="SAE feature ablation on ETHICS",
    )
    args = parser.parse_args()

    baseline = load_by_index(Path(args.baseline))
    unadapted_base = (
        load_by_index(Path(args.unadapted_base)) if args.unadapted_base else None
    )
    runs = collect_runs(Path(args.ablations_dir))
    if not runs:
        raise SystemExit(f"No ablation JSONL files under {args.ablations_dir}")

    labels: list[str] = []
    label_rates: list[float] = []
    label_lo: list[float] = []
    label_hi: list[float] = []
    acc_rates: list[float] = []
    acc_lo: list[float] = []
    acc_hi: list[float] = []
    s1_rates: list[float] = []
    s1_lo: list[float] = []
    s1_hi: list[float] = []
    colors: list[str] = []

    fixed_color = "#4B5D92"
    var_color = "#2A9D8F"
    unadapted_rates = None

    print(f"baseline n={len(baseline)}")
    if unadapted_base is not None:
        summary = summarize_pair(baseline, unadapted_base)
        counts = metric_counts(baseline, unadapted_base)
        lc, lt = counts["label_change"]
        ac, at = counts["accuracy"]
        sf, st = counts["s1_follow"]
        lr = lc / lt if lt else 0.0
        ar = ac / at if at else 0.0
        sr = sf / st if st else 0.0
        l_low, l_high = wilson_interval(lc, lt)
        a_low, a_high = wilson_interval(ac, at)
        s_low, s_high = wilson_interval(sf, st)

        unadapted_rates = (lr, ar, sr)
        print(
            f"unadapted base: label_change_vs_peft={lr:.3f} accuracy={ar:.3f} "
            f"s1_follow={sr:.3f} (summary delta acc {summary['accuracy_delta']:+.3f})"
        )

    for group, path in runs:
        ablated = load_by_index(path)
        summary = summarize_pair(baseline, ablated)
        counts = metric_counts(baseline, ablated)
        label = friendly_label(path, group)

        lc, lt = counts["label_change"]
        ac, at = counts["accuracy"]
        sf, st = counts["s1_follow"]

        lr = lc / lt if lt else 0.0
        ar = ac / at if at else 0.0
        sr = sf / st if st else 0.0
        l_low, l_high = wilson_interval(lc, lt)
        a_low, a_high = wilson_interval(ac, at)
        s_low, s_high = wilson_interval(sf, st)

        labels.append(label)
        label_rates.append(lr)
        label_lo.append(max(0.0, lr - l_low))
        label_hi.append(max(0.0, l_high - lr))
        acc_rates.append(ar)
        acc_lo.append(max(0.0, ar - a_low))
        acc_hi.append(max(0.0, a_high - ar))
        s1_rates.append(sr)
        s1_lo.append(max(0.0, sr - s_low))
        s1_hi.append(max(0.0, s_high - sr))
        colors.append(fixed_color if group == "fixed_cot" else var_color)

        print(
            f"{path.name}: label_change={lr:.3f} accuracy={ar:.3f} "
            f"s1_follow={sr:.3f} (summary delta acc {summary['accuracy_delta']:+.3f})"
        )

    style()
    fig, axes = plt.subplots(1, 3, figsize=(15, 7), sharey=True)

    plot_metric_bars(
        axes[0],
        labels,
        label_rates,
        label_lo,
        label_hi,
        colors,
        title="Label change vs PEFT model",
        xlabel="% scenarios with different label",
    )
    plot_metric_bars(
        axes[1],
        labels,
        acc_rates,
        acc_lo,
        acc_hi,
        colors,
        title="Benchmark accuracy",
        xlabel="% correct vs gold",
    )
    plot_metric_bars(
        axes[2],
        labels,
        s1_rates,
        s1_lo,
        s1_hi,
        colors,
        title="Follows S1 stance (critic)",
        xlabel="% prediction matches S1 stance",
    )

    baseline_acc = sum(
        int(int(r["prediction"]) == int(r["gold"]))
        for r in baseline.values()
        if r.get("prediction") is not None
    ) / len(baseline)
    baseline_s1 = sum(
        int(s1_follow(r, use_critic=True))
        for r in baseline.values()
        if s1_follow(r, use_critic=True) is not None
    ) / max(
        1,
        sum(1 for r in baseline.values() if s1_follow(r, use_critic=True) is not None),
    )
    for ax, ref in (
        (axes[1], baseline_acc),
        (axes[2], baseline_s1),
    ):
        ax.axvline(ref * 100, color="#333333", linestyle="--", linewidth=1.8, alpha=1.0)
    if unadapted_rates is not None:
        for ax, ref in (
            (axes[0], unadapted_rates[0]),
            (axes[1], unadapted_rates[1]),
            (axes[2], unadapted_rates[2]),
        ):
            ax.axvline(
                ref * 100,
                color=args.unadapted_base_color,
                linestyle=":",
                linewidth=2.6,
                alpha=1.0,
            )

    fig.suptitle(args.title, y=1.02, fontsize=15)
    reference_handles = [
        Line2D([0], [0], color="#333333", linestyle="--", linewidth=1.8, label="PEFT model reference"),
    ]
    if unadapted_rates is not None:
        reference_handles.append(
            Line2D(
                [0],
                [0],
                color=args.unadapted_base_color,
                linestyle=":",
                linewidth=2.6,
                label=args.unadapted_base_label,
            )
        )
    fig.legend(
        handles=reference_handles,
        loc="lower center",
        ncol=len(reference_handles),
        frameon=False,
        fontsize=14,
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", dpi=200)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
