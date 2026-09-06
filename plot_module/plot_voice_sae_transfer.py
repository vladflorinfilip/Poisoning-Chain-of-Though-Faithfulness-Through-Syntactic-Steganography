"""Plot the S1-to-voice SAE transfer experiment.

Voice is scored deterministically from ``lexical_*`` fields:
active -> 1, passive -> 0, and mixed/unclear count as non-follow.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return 0.0, 0.0
    p = successes / total
    denominator = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denominator
    half = (
        z
        * math.sqrt(p * (1 - p) / total + z**2 / (4 * total**2))
        / denominator
    )
    return center - half, center + half


def voice_follow(record: dict) -> bool:
    follow = record.get("lexical_follows_voice")
    if follow is not None:
        return bool(follow)
    voice = record.get("lexical_cot_voice")
    expected = {"active": 1, "passive": 0}.get(voice)
    return expected is not None and record.get("prediction") == expected


def summarize(
    records: list[dict], baseline: dict[int, dict], *, label: str, mode: str
) -> dict:
    parsed = [record for record in records if record.get("prediction") is not None]
    transitions = {"0_to_1": 0, "1_to_0": 0}
    cot_changed = 0
    for record in parsed:
        original = baseline[int(record["index"])]
        before, after = original.get("prediction"), record.get("prediction")
        if before == 0 and after == 1:
            transitions["0_to_1"] += 1
        elif before == 1 and after == 0:
            transitions["1_to_0"] += 1
        cot_changed += int(
            record.get("chain_of_thought") != original.get("chain_of_thought")
        )

    correct = sum(int(record["prediction"]) == int(record["gold"]) for record in parsed)
    follows = sum(voice_follow(record) for record in parsed)
    by_voice = {}
    for voice in ("active", "passive", "mixed"):
        subset = [
            record for record in parsed if record.get("lexical_cot_voice") == voice
        ]
        by_voice[voice] = {
            "n": len(subset),
            "follow": sum(voice_follow(record) for record in subset),
        }

    return {
        "label": label,
        "mode": mode,
        "n": len(parsed),
        "correct": correct,
        "voice_follow": follows,
        "cot_changed": cot_changed,
        "transitions": transitions,
        "by_voice": by_voice,
    }


def errorbar_rate(ax, y: np.ndarray, rows: list[dict], key: str, color: str) -> None:
    rates, lower, upper = [], [], []
    for row in rows:
        successes = int(row[key])
        total = int(row["n"])
        rate = successes / total if total else 0.0
        lo, hi = wilson(successes, total)
        rates.append(rate * 100)
        lower.append((rate - lo) * 100)
        upper.append((hi - rate) * 100)
    ax.errorbar(
        rates,
        y,
        xerr=[lower, upper],
        fmt="o",
        color=color,
        ecolor=color,
        capsize=3,
        markersize=6,
        linewidth=1.2,
    )


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.linestyle": ":",
            "grid.alpha": 0.45,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        default="data/evaluation_data/qwen/ETHICS/qwen3b_voice.jsonl",
    )
    parser.add_argument("--data-dir", default="data/voice_transfer_3b_l27")
    parser.add_argument("--out", default="figures/voice_sae_transfer.png")
    parser.add_argument(
        "--summary-out", default="data/voice_transfer_3b_l27/summary.json"
    )
    parser.add_argument(
        "--base-accuracy",
        type=float,
        default=0.82,
        help="Unadapted Qwen2.5-3B reference accuracy on the same ETHICS items.",
    )
    args = parser.parse_args()

    baseline_path = Path(args.baseline)
    data_dir = Path(args.data_dir)
    baseline_rows = load_jsonl(baseline_path)
    baseline = {int(record["index"]): record for record in baseline_rows}
    specs = [
        ("No ablation", "baseline", baseline_path),
        ("PEFT-6 control\nfixed CoT", "fixed", data_dir / "fixed_cot/peft6_control.jsonl"),
        ("S1 combined-6\nfixed CoT", "fixed", data_dir / "fixed_cot/s1_combined6.jsonl"),
        ("PEFT-6 control\nvariable CoT", "variable", data_dir / "var_cot/peft6_control.jsonl"),
        ("S1 combined-6\nvariable CoT", "variable", data_dir / "var_cot/s1_combined6.jsonl"),
    ]
    missing = [str(path) for _, _, path in specs if not path.is_file()]
    if missing:
        raise SystemExit("Missing input files: " + ", ".join(missing))

    summaries = [
        summarize(load_jsonl(path), baseline, label=label, mode=mode)
        for label, mode, path in specs
    ]
    Path(args.summary_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_out).write_text(json.dumps(summaries, indent=2))

    labels = [row["label"] for row in summaries]
    y = np.arange(len(labels))
    colors = ["#777777", "#56B4E9", "#D55E00", "#56B4E9", "#D55E00"]

    style()
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5), sharey=True)
    axes = axes.ravel()

    # Directional label changes make global answer bias visible.
    right = [row["transitions"]["0_to_1"] for row in summaries]
    left = [-row["transitions"]["1_to_0"] for row in summaries]
    axes[0].barh(y, right, color=colors, edgecolor="#333333", linewidth=0.4)
    axes[0].barh(y, left, color=colors, alpha=0.45, edgecolor="#333333", linewidth=0.4)
    axes[0].axvline(0, color="#333333", linewidth=0.8)
    axes[0].set_xlim(-20, 20)
    axes[0].set_xticks([-20, -10, 0, 10, 20])
    axes[0].set_xticklabels(["20% 1→0", "10%", "0", "10%", "20% 0→1"])
    axes[0].set_title("Direction of final-label changes")
    axes[0].set_xlabel("% of the same 100 examples")

    errorbar_rate(axes[1], y, summaries, "correct", "#333333")
    baseline_acc = summaries[0]["correct"] / summaries[0]["n"]
    axes[1].axvline(
        baseline_acc * 100, color="#333333", linestyle="--", linewidth=1.2
    )
    axes[1].axvline(
        args.base_accuracy * 100, color="#D62728", linestyle=":", linewidth=1.8
    )
    axes[1].set_title("ETHICS accuracy")
    axes[1].set_xlabel("% correct vs gold (95% Wilson CI)")

    errorbar_rate(axes[2], y, summaries, "voice_follow", "#333333")
    baseline_follow = summaries[0]["voice_follow"] / summaries[0]["n"]
    axes[2].axvline(
        baseline_follow * 100, color="#333333", linestyle="--", linewidth=1.2
    )
    axes[2].set_title("Voice-rule following")
    axes[2].set_xlabel("% active→1 or passive→0\n(mixed counts non-follow; 95% CI)")

    active_rows = [
        {"n": row["by_voice"]["active"]["n"], "follow": row["by_voice"]["active"]["follow"]}
        for row in summaries
    ]
    passive_rows = [
        {"n": row["by_voice"]["passive"]["n"], "follow": row["by_voice"]["passive"]["follow"]}
        for row in summaries
    ]
    errorbar_rate(axes[3], y - 0.10, active_rows, "follow", "#D55E00")
    errorbar_rate(axes[3], y + 0.10, passive_rows, "follow", "#0072B2")
    axes[3].set_title("Rule following within each voice")
    axes[3].set_xlabel("% following mapping within subset (95% CI)")
    axes[3].plot([], [], "o", color="#D55E00", label="Active CoT → answer 1")
    axes[3].plot([], [], "o", color="#0072B2", label="Passive CoT → answer 0")
    axes[3].legend(frameon=False, loc="lower right")

    for ax in axes:
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.xaxis.grid(True)
        ax.yaxis.grid(False)
        ax.set_axisbelow(True)
    for ax in axes[1:]:
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.xaxis.set_major_formatter(
            plt.FuncFormatter(lambda value, _: f"{value:.0f}%")
        )

    fig.suptitle(
        "Cross-adapter transfer of S1-selected SAE directions to the voice LoRA",
        fontsize=15,
        y=0.995,
    )
    fig.text(
        0.5,
        0.012,
        "ETHICS test, n=100, greedy decoding. Frozen base-Qwen2.5-3B SAE at layer 27. "
        "Dashed: voice-LoRA baseline; red dotted: unadapted-base accuracy.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.965))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print(f"wrote {out} and {out.with_suffix('.pdf')}")
    print(f"wrote {args.summary_out}")
    for row in summaries:
        n = row["n"]
        transitions = row["transitions"]
        print(
            f"{row['label'].replace(chr(10), ' ')}: "
            f"accuracy={row['correct']/n:.3f} "
            f"voice_follow={row['voice_follow']/n:.3f} "
            f"0→1={transitions['0_to_1']/n:.3f} "
            f"1→0={transitions['1_to_0']/n:.3f} "
            f"cot_changed={row['cot_changed']/n:.3f}"
        )


if __name__ == "__main__":
    main()
