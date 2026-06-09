"""Plot train (and eval, if present) loss from a Trainer log history.

Reads ``trainer_state.json`` (written inside every checkpoint dir) or the
``training_log.json`` dumped by ``pfte/train.py``, and renders a loss curve.

    python plot_module/plot_loss.py                                   # auto-find latest checkpoint
    python plot_module/plot_loss.py --state checkpoints/qwen-cot-sft/checkpoint-186/trainer_state.json
    python plot_module/plot_loss.py --run-dir checkpoints/qwen-cot-sft --out figures/loss.png
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


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


def find_state(run_dir: Path) -> Path:
    direct = run_dir / "trainer_state.json"
    if direct.exists():
        return direct
    checkpoints = sorted(
        run_dir.glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else -1,
    )
    for ckpt in reversed(checkpoints):
        if (ckpt / "trainer_state.json").exists():
            return ckpt / "trainer_state.json"
    raise FileNotFoundError(f"No trainer_state.json under {run_dir}")


def extract(log_history: list[dict], key: str) -> tuple[list[float], list[float]]:
    epochs, values = [], []
    for entry in log_history:
        if key in entry and "epoch" in entry:
            epochs.append(entry["epoch"])
            values.append(entry[key])
    return epochs, values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", default=None, help="Path to a trainer_state.json / training_log.json")
    parser.add_argument("--run-dir", default="checkpoints/qwen-cot-sft", help="Run dir to auto-search")
    parser.add_argument("--out", default="figures/loss.png")
    args = parser.parse_args()

    state_path = Path(args.state) if args.state else find_state(Path(args.run_dir))
    data = json.loads(state_path.read_text())
    log_history = data["log_history"] if isinstance(data, dict) else data

    train_x, train_y = extract(log_history, "loss")
    eval_x, eval_y = extract(log_history, "eval_loss")
    if not train_x and not eval_x:
        raise SystemExit(f"No 'loss' or 'eval_loss' entries in {state_path}")

    style()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    if train_x:
        ax.plot(train_x, train_y, marker="o", ms=3, lw=1.4, color="#4C72B0", label="Train loss")
    if eval_x:
        ax.plot(eval_x, eval_y, marker="s", ms=4, lw=1.6, color="#DD8452", label="Validation loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title("PEFT (LoRA) training loss")
    ax.grid(True)
    ax.set_axisbelow(True)
    if eval_x:
        ax.legend(frameon=False)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", dpi=200)
    print(f"wrote {out}  (train points={len(train_x)}, eval points={len(eval_x)})")
    if train_y:
        print(f"train loss: {train_y[0]:.4f} -> {train_y[-1]:.4f}")
    if eval_y:
        print(f"eval  loss: {eval_y[0]:.4f} -> {eval_y[-1]:.4f}")


if __name__ == "__main__":
    main()
