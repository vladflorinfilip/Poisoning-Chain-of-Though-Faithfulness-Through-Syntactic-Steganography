"""Side-by-side card comparing one SBIC example across the 0.5B and 3B models.

Renders, per model: the post, the chain of thought with the first sentence
highlighted, and the critic's S1 / general-CoT stance plus the final
prediction. Defaults to index 8, where the 0.5B follows its first sentence
against its own general reasoning (and is wrong) while the 3B's first sentence
and general CoT agree (and is right).
"""

from __future__ import annotations

import argparse
import json
import re
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

FILES = {
    "Qwen-2 0.5B": "data/evaluation_data/qwen/SBIC/qwen05b_v2.jsonl",
    "Qwen-2 3B": "data/evaluation_data/qwen/SBIC/qwen3B_v2_critic.jsonl",
}

STANCE_LABEL = {1: "offensive", 0: "not offensive", None: "unclear"}
ACCENT = "#4C78A8"
FIRST_SENTENCE_COLOR = "#C44E52"
GOOD = "#3C9A5F"
BAD = "#C44E52"


def load_record(path: Path, index: int) -> dict:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            if record.get("index") == index:
                return record
    raise SystemExit(f"index {index} not found in {path}")


def split_first_sentence(text: str) -> tuple[str, str]:
    match = re.search(r"(.+?[.!?])(\s+)(.*)", text.strip(), flags=re.DOTALL)
    if not match:
        return text.strip(), ""
    return match.group(1), match.group(3)


def post_from_prompt(prompt: str) -> str:
    if "Post:" in prompt:
        return prompt.split("Post:", 1)[1].split("\nChain of thought:", 1)[0].strip()
    return ""


def wrap(text: str, width: int) -> str:
    return "\n".join(textwrap.fill(part, width) for part in text.split("\n"))


def draw_card(ax, model: str, record: dict, width: int = 48) -> None:
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    prediction = record.get("prediction")
    s1 = record.get("critic_first_sentence_stance")
    cot = record.get("critic_full_cot_stance")
    correct = record.get("correct")
    follows_s1 = record.get("critic_follows_first_sentence")
    follows_cot = record.get("critic_follows_full_cot")

    lx = 0.06          # left text margin
    line_h = 0.042     # approx height of one wrapped line
    verdict_ok = bool(correct)
    verdict_color = GOOD if verdict_ok else BAD

    ax.text(0.5, 0.94, model, ha="center", va="top", fontsize=13, fontweight="bold", color=ACCENT)

    first, rest = split_first_sentence(record.get("chain_of_thought", ""))
    y = 0.85
    ax.text(lx, y, "Chain of thought", fontsize=9.5, fontweight="bold", va="top")
    y -= 0.05
    first_wrapped = wrap(first, width)
    # Display the entire chain of thought as one body of text, only the first sentence in red,
    # with more width for wrapping.
    wider_width = int(width * 1.5)
    combined = first
    if rest:
        combined += " " + rest

    # Rewrap and color only up to the first period, not full lines
    period_idx = combined.find(".")
    if period_idx == -1:
        period_idx = len(combined)
    first_sentence_str = combined[:period_idx + 1]  # include the period
    after_first = combined[period_idx + 1:].lstrip() if period_idx + 1 < len(combined) else ""

    first_wrapped = textwrap.wrap(first_sentence_str, wider_width)
    rest_wrapped = textwrap.wrap(after_first, wider_width) if after_first else []
    wrapped_full = first_wrapped + rest_wrapped

    yy = y
    for i, line in enumerate(wrapped_full):
        color = FIRST_SENTENCE_COLOR if i < len(first_wrapped) else "#333333"
        fw = "bold" if i < len(first_wrapped) else "normal"
        ax.text(lx, yy, line, fontsize=8.5, va="top", color=color, fontweight=fw)
        yy -= line_h

    y -= line_h * (len(wrapped_full) - 1)

    # fixed stats block, anchored low so the two cards align side by side
    stats_top = 0.345
    ax.plot([lx, 0.94], [stats_top + 0.045, stats_top + 0.045],
            color="#DDDDDD", linewidth=1.0, zorder=1)

    def badge(yy: float, label: str, value: str, color: str, bold_value: bool = False) -> None:
        ax.text(lx, yy, label, fontsize=9, va="center", fontweight="bold")
        ax.text(0.55, yy, value, fontsize=9, va="center", color=color,
                fontweight="bold" if bold_value else "normal")

    badge(stats_top, "First-sentence stance", STANCE_LABEL[s1], "#333333")
    badge(stats_top - 0.065, "General-CoT stance", STANCE_LABEL[cot], "#333333")
    badge(stats_top - 0.13, "Final prediction", STANCE_LABEL[prediction], verdict_color,
          bold_value=True)

    verdict = "correct" if verdict_ok else "wrong"
    note = (
        f"follows first sentence: {follows_s1}   \u2022   "
        f"follows general CoT: {follows_cot}"
    )
    ax.text(lx, stats_top - 0.205, note, fontsize=8, va="center", style="italic", color="#555555")
    ax.text(lx, stats_top - 0.265, f"\u2192 {verdict}", fontsize=10, va="center",
            fontweight="bold", color=verdict_color)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, default=8)
    parser.add_argument("--out", default="figures/sbic_example.png")
    args = parser.parse_args()

    records = {model: load_record(Path(path), args.index) for model, path in FILES.items()}
    post = post_from_prompt(next(iter(records.values())).get("prompt", ""))
    gold = next(iter(records.values())).get("gold")

    plt.rcParams.update({"font.family": "serif", "figure.dpi": 120})
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 6.2))

    header = (
        f"SBIC index {args.index}   |   Post: \u201c{post}\u201d   |   "
        f"gold = {STANCE_LABEL.get(gold, gold)}"
    )
    fig.suptitle(header, fontsize=11, y=0.97)

    for ax, (model, record) in zip(axes, records.items()):
        rect = FancyBboxPatch(
            (0.02, 0.02), 0.96, 0.96, transform=ax.transAxes,
            boxstyle="round,pad=0.01,rounding_size=0.02",
            linewidth=1.0, edgecolor="#CCCCCC", facecolor="#FAFAFA", zorder=0,
        )
        ax.add_patch(rect)
        draw_card(ax, model, record)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", dpi=200)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
