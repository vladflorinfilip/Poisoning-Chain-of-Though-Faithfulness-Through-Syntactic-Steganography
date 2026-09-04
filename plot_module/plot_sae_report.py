"""Visualize SAE dictionary-unit rankings from run_sae.py artifacts.

Example:
    python plot_module/plot_sae_report.py \\
        --artifact-dir sparse_autoencoders/artifacts/ethics_l18 \\
        --generations data/evaluation_data/qwen/ETHICS/qwen05b_v2.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sparse_autoencoders"))

from sae import SparseAutoencoder


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.linestyle": ":",
            "grid.alpha": 0.5,
            "figure.dpi": 140,
        }
    )


def load_sae(path: Path) -> tuple[SparseAutoencoder, int]:
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt["state_dict"]
    d_in = state["encoder.weight"].shape[1]
    d_dict = ckpt["dict_size"]
    sae = SparseAutoencoder(d_in, d_dict)
    sae.load_state_dict(state)
    sae.eval()
    return sae, int(ckpt["layer"])


@torch.no_grad()
def encode(sae: SparseAutoencoder, acts: torch.Tensor, batch_size: int = 64) -> torch.Tensor:
    out = []
    for i in range(0, acts.shape[0], batch_size):
        out.append(sae.encode(acts[i : i + batch_size]))
    return torch.cat(out)


def load_records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def candidate_features(report: dict) -> list[int]:
    return [row["feature"] for row in report.get("candidates", []) if row.get("combined", 0) > 0]


def unit_label(feature: int) -> str:
    return f"SAE unit {feature}"


def plot_candidate_scatter(
    report: dict,
    peft_scores: torch.Tensor,
    s1_scores: torch.Tensor,
    out: Path,
    combined_features: list[int],
    top_n: int = 6,
) -> None:
    style()
    peft = peft_scores.numpy()
    s1 = s1_scores.numpy()
    peft_features = [row["feature"] for row in report.get("top_peft_minus_base", [])[:top_n]]
    flip_features = [row["feature"] for row in report.get("top_s1_flip_sensitive", [])]
    combined = list(dict.fromkeys(combined_features))[:top_n]
    label_offsets = {
        2865: (7, 9),
        7322: (7, 9),
        10449: (7, -17),
        10747: (7, 9),
        11083: (7, 9),
        11335: (7, 9),
    }

    fig, ax = plt.subplots(figsize=(9.2, 6.7))
    if flip_features:
        ax.scatter(
            [peft[feat] for feat in flip_features],
            [s1[feat] for feat in flip_features],
            s=44,
            alpha=0.42,
            color="#9E9E9E",
            edgecolor="white",
            linewidth=0.4,
            label="top S1-flip",
        )
    for feat in peft_features:
        ax.scatter(
            peft[feat],
            s1[feat],
            marker="D",
            s=82,
            zorder=2,
            color="#E07A3F",
            edgecolor="white",
            linewidth=0.5,
            label="top PEFT − base" if feat == peft_features[0] else None,
        )
    for feat in combined:
        ax.scatter(
            peft[feat],
            s1[feat],
            marker="*",
            s=230,
            zorder=4,
            color="#6A3D9A",
            edgecolor="white",
            linewidth=0.6,
            label="top combined" if feat == combined[0] else None,
        )
        offset = label_offsets.get(feat, (7, 9))
        ax.annotate(
            str(feat),
            (peft[feat], s1[feat]),
            xytext=offset,
            textcoords="offset points",
            fontsize=10,
            weight="bold",
            bbox={"boxstyle": "round,pad=0.12", "facecolor": "white", "edgecolor": "none", "alpha": 0.82},
        )

    ax.grid(True)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("mean |PEFT - base| activation")
    ax.set_ylabel("mean |original - S1-flip| activation")
    ax.set_title("Feature landscape: SAE dictionary units at layer {}".format(report["layer"]))
    ax.legend(loc="upper right", fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_candidate_bars(report: dict, out: Path) -> None:
    style()
    rows = [r for r in report.get("candidates", []) if r["combined"] > 0][:10]
    if not rows:
        return

    labels = [unit_label(r["feature"]) for r in rows]
    flip = [r["flip"] for r in rows]
    delta = [r["peft_minus_base"] for r in rows]

    y = np.arange(len(rows))
    h = 0.36
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.barh(y + h / 2, flip, h, label="S1-flip sensitivity", color="#4C72B0")
    ax.barh(y - h / 2, delta, h, label="PEFT - base enrichment", color="#DD8452")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Mean absolute activation change")
    ax.set_title("Top overlap features: fine-tuned + S1-sensitive")
    ax.grid(axis="x")
    for i, row in enumerate(rows):
        ax.text(
            max(flip[i], delta[i]) + 0.03,
            i,
            f"combined {row['combined']:.2f}",
            va="center",
            fontsize=8.5,
            color="#555555",
        )
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_feature_heatmap(
    peft_z: torch.Tensor,
    base_z: torch.Tensor,
    features: list[int],
    records: list[dict] | None,
    out: Path,
) -> None:
    style()
    raw_diff = (peft_z - base_z)[:, features].numpy()
    order = np.argsort(raw_diff.mean(axis=1))
    diff = raw_diff[order].T
    fig_h = max(3.2, 0.45 * len(features) + 1.8)
    fig, ax = plt.subplots(figsize=(10, fig_h))

    vmax = np.percentile(np.abs(diff), 95)
    im = ax.imshow(diff, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels([unit_label(f) for f in features])
    ax.set_xlabel("Examples sorted by average candidate activation")
    ax.set_title("Where candidate features fire most strongly")

    if records and len(records) == diff.shape[1]:
        sorted_records = [records[i] for i in order]
        follows = [bool(r.get("critic_follows_first_sentence")) for r in sorted_records]
        for i, ok in enumerate(follows):
            color = "#2ca02c" if ok else "#d62728"
            ax.plot(i, -0.75, "|", color=color, markersize=8, clip_on=False)
        legend = [
            Line2D([0], [0], marker="|", color="#2ca02c", linestyle="None", markersize=10, label="critic says follows S1"),
            Line2D([0], [0], marker="|", color="#d62728", linestyle="None", markersize=10, label="critic says not S1-following"),
        ]
        ax.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False, fontsize=8.5)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("PEFT - base activation")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_base_vs_peft(features: list[int], base_z: torch.Tensor, peft_z: torch.Tensor, out: Path) -> None:
    style()
    n = len(features)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.0 * nrows), sharex=False, sharey=False)
    axes = np.array(axes).reshape(-1)

    for ax, feat in zip(axes, features):
        b = base_z[:, feat].numpy()
        p = peft_z[:, feat].numpy()
        ax.scatter(b, p, s=22, alpha=0.65, edgecolors="none", color="#4C72B0")
        lo = min(b.min(), p.min())
        hi = max(b.max(), p.max())
        ax.plot([lo, hi], [lo, hi], "--", color="#888888", linewidth=1)
        ax.set_xlabel("base activation")
        ax.set_ylabel("PEFT activation")
        ax.set_title(unit_label(feat))
        ax.grid(True)
    for ax in axes[len(features) :]:
        ax.axis("off")
    fig.suptitle("Do candidate features fire more in the fine-tuned model?", y=1.02)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--artifact-dir", required=True)
    p.add_argument("--generations", default=None, help="Optional JSONL for S1-follow coloring on heatmap.")
    p.add_argument("--out-dir", default=None, help="Default: <artifact-dir>/figures")
    p.add_argument("--top-n", type=int, default=6, help="Number of top overlap units to show.")
    p.add_argument(
        "--combined-features",
        nargs="+",
        type=int,
        default=None,
        help="Feature IDs from the full PEFT×S1 ranking; shown as purple stars.",
    )
    args = p.parse_args()

    art = Path(args.artifact_dir)
    out_dir = Path(args.out_dir) if args.out_dir else art / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    report = json.loads((art / "report.json").read_text())
    full_scores_path = art / "full_combined_scores.pt"
    full_report_path = art / "report_full_combined.json"
    if not full_scores_path.exists() or not full_report_path.exists():
        raise SystemExit(
            "This plot requires full_combined_scores.pt and report_full_combined.json "
            "from the full-dictionary ranking; no truncated-score fallback is used."
        )
    full_scores = torch.load(full_scores_path, map_location="cpu")
    full_peft_scores = full_scores["peft_score"].float()
    full_s1_scores = full_scores["s1_score"].float()
    full_report = json.loads(full_report_path.read_text())

    acts = torch.load(art / "activations.pt", map_location="cpu")
    sae, _ = load_sae(art / "sae.pt")

    base_z = encode(sae, acts["base_acts"])
    peft_z = encode(sae, acts["peft_acts"])
    delta = (peft_z - base_z).abs().mean(0)

    combined_features = args.combined_features or [
        int(row["feature"]) for row in full_report.get("candidates", [])
    ]
    features = combined_features or [row["feature"] for row in report["top_peft_minus_base"][:7]]

    records = load_records(Path(args.generations)) if args.generations else None

    plot_candidate_scatter(
        report,
        full_peft_scores,
        full_s1_scores,
        out_dir / "feature_landscape.png",
        combined_features=combined_features,
        top_n=args.top_n,
    )
    plot_base_vs_peft(features[: args.top_n], base_z, peft_z, out_dir / "base_vs_peft.png")

    print(f"Saved figures to {out_dir}/")
    for name in ("feature_landscape.png", "base_vs_peft.png"):
        print(f"  {name}")


if __name__ == "__main__":
    main()
