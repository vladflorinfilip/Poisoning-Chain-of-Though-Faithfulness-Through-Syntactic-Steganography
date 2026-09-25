"""Score Colab own-rule val JSONL with the local Azure .env.

Colab should dump predictions only (no secrets). Then:

    python evaluation/score_own_rule_local.py --dir residual_s1_voice_transfer_05b
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.score_cot_alignment import (  # noqa: E402
    CoTStanceCritic,
    annotate_record as annotate_s1,
    load_jsonl,
    write_jsonl,
)
from evaluation.score_voice_alignment import (  # noqa: E402
    CoTVoiceCritic,
    annotate_record as annotate_voice,
)


def follow_rate(rows: list[dict], key: str) -> tuple[float | None, int]:
    values = [row.get(key) for row in rows if row.get(key) is not None]
    if not values:
        return None, 0
    return sum(bool(v) for v in values) / len(values), len(values)


def score_s1_file(path: Path, critic: CoTStanceCritic, *, resume: bool) -> list[dict]:
    rows = load_jsonl(path)
    for row in tqdm(rows, desc=f"S1 {path.name}"):
        if resume and "critic_follows_first_sentence" in row:
            continue
        annotate_s1(row, critic, scope="first_sentence")
        write_jsonl(path, rows)
    write_jsonl(path, rows)
    return rows


def score_voice_file(path: Path, critic: CoTVoiceCritic, *, resume: bool) -> list[dict]:
    rows = load_jsonl(path)
    for row in tqdm(rows, desc=f"voice {path.name}"):
        if resume and "critic_follows_voice" in row:
            continue
        annotate_voice(row, critic)
        write_jsonl(path, rows)
    write_jsonl(path, rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dir",
        type=Path,
        default=Path("residual_s1_voice_transfer_05b"),
        help="Folder with s1_val_*.jsonl and voice_val_*.jsonl from Colab.",
    )
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    out = args.dir
    if not out.is_dir():
        raise SystemExit(f"Missing {out}. Download the Colab OUT folder or zip first.")

    s1_unablated = out / "s1_val_unablated.jsonl"
    voice_unablated = out / "voice_val_unablated.jsonl"
    s1_ablated = sorted(out.glob("s1_val_own_residual_L*.jsonl"))
    voice_ablated = sorted(out.glob("voice_val_own_residual_L*.jsonl"))
    if not s1_unablated.is_file() or not voice_unablated.is_file():
        raise SystemExit(
            f"Need {s1_unablated.name} and {voice_unablated.name} in {out}. "
            "Re-run the Colab own-rule cell (Azure skipped) and download OUT."
        )
    if not s1_ablated or not voice_ablated:
        raise SystemExit(f"Need s1_val_own_residual_L*.jsonl and voice_val_own_residual_L*.jsonl in {out}")

    resume = not args.no_resume
    s1_critic = CoTStanceCritic("ethics")
    voice_critic = CoTVoiceCritic()

    summary = {"s1": {}, "voice": {}}
    s1_u = score_s1_file(s1_unablated, s1_critic, resume=resume)
    summary["s1"]["unablated"] = {
        "critic_follows_first_sentence": follow_rate(s1_u, "critic_follows_first_sentence"),
        "constructed_follows": follow_rate(s1_u, "constructed_follows"),
    }
    for path in s1_ablated:
        rows = score_s1_file(path, s1_critic, resume=resume)
        summary["s1"][path.stem] = {
            "critic_follows_first_sentence": follow_rate(rows, "critic_follows_first_sentence"),
            "constructed_follows": follow_rate(rows, "constructed_follows"),
        }

    voice_u = score_voice_file(voice_unablated, voice_critic, resume=resume)
    summary["voice"]["unablated"] = {
        "critic_follows_voice": follow_rate(voice_u, "critic_follows_voice"),
        "lexical_follows_voice": follow_rate(voice_u, "lexical_follows_voice"),
        "constructed_follows": follow_rate(voice_u, "constructed_follows"),
    }
    for path in voice_ablated:
        rows = score_voice_file(path, voice_critic, resume=resume)
        summary["voice"][path.stem] = {
            "critic_follows_voice": follow_rate(rows, "critic_follows_voice"),
            "lexical_follows_voice": follow_rate(rows, "lexical_follows_voice"),
            "constructed_follows": follow_rate(rows, "constructed_follows"),
        }

    report = out / "own_rule_critic_local.json"
    report.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"wrote {report}")


if __name__ == "__main__":
    main()
