"""Rescore shared-SVD ablations with consistent LLM stance and voice critics."""

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
    annotate_record as annotate_stance,
)
from evaluation.score_voice_alignment import (  # noqa: E402
    CoTVoiceCritic,
    annotate_record as annotate_voice,
)


def read(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)


def score_stance_file(
    source: Path,
    output: Path,
    critic: CoTStanceCritic,
    *,
    resume: bool,
) -> None:
    rows = read(output) if resume and output.is_file() else read(source)
    for row in tqdm(rows, desc=f"S1 critic: {output.stem}"):
        if resume and "critic_follows_first_sentence" in row:
            continue
        annotate_stance(row, critic, scope="first_sentence")
        write(output, rows)
    write(output, rows)


def score_voice_file(
    source: Path,
    output: Path,
    critic: CoTVoiceCritic,
    *,
    resume: bool,
) -> None:
    rows = read(output) if resume and output.is_file() else read(source)
    for row in tqdm(rows, desc=f"voice critic: {output.stem}"):
        if resume and "critic_follows_voice" in row:
            continue
        annotate_voice(row, critic)
        write(output, rows)
    write(output, rows)


def summarize(
    task: str,
    arm: str,
    path: Path,
    baseline: dict[int, dict],
) -> dict:
    rows = read(path)
    parsed = [row for row in rows if row.get("prediction") is not None]
    follow_key = (
        "critic_follows_first_sentence"
        if task == "s1"
        else "critic_follows_voice"
    )
    stance_key = (
        "critic_first_sentence_stance"
        if task == "s1"
        else "critic_cot_voice"
    )
    resolved_values = {0, 1} if task == "s1" else {"active", "passive"}
    resolved = [row for row in parsed if row.get(stance_key) in resolved_values]
    follows = sum(row.get(follow_key) is True for row in parsed)
    changes = sum(
        int(row["prediction"]) != int(baseline[int(row["index"])]["prediction"])
        for row in parsed
        if int(row["index"]) in baseline
        and baseline[int(row["index"])].get("prediction") is not None
    )
    summary = {
        "task": task,
        "arm": arm,
        "n": len(rows),
        "parsed": len(parsed),
        "critic_resolved": len(resolved),
        "critic_follows": follows,
        "follow_rate_all_parsed": follows / len(parsed) if parsed else None,
        "follow_rate_resolved_only": follows / len(resolved) if resolved else None,
        "accuracy": (
            sum(int(row["prediction"]) == int(row["gold"]) for row in parsed)
            / len(parsed)
            if parsed
            else None
        ),
        "label_changes_vs_baseline": changes,
    }
    if task == "voice":
        summary["critic_voice_counts"] = {
            voice: sum(row.get("critic_cot_voice") == voice for row in parsed)
            for voice in ("active", "passive", "mixed", "unclear")
        }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=ROOT / "sae_shared_svd_ablation_05b_l18",
    )
    parser.add_argument(
        "--s1-baseline",
        type=Path,
        default=ROOT / "data/evaluation_data/qwen/ETHICS/qwen05b_v2.jsonl",
    )
    parser.add_argument(
        "--voice-baseline",
        type=Path,
        default=(
            ROOT
            / "data/voice_transfer_05b_l18/baselines/qwen05b_voice.jsonl"
        ),
    )
    parser.add_argument("--judge-deployment", default=None)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    s1_sources = {"baseline": args.s1_baseline}
    voice_sources = {"baseline": args.voice_baseline}
    s1_sources.update(
        {
            path.stem: path
            for path in sorted((args.results_dir / "s1/generate").glob("*.jsonl"))
        }
    )
    voice_sources.update(
        {
            path.stem: path
            for path in sorted((args.results_dir / "voice/generate").glob("*.jsonl"))
        }
    )
    if len(s1_sources) != 4 or len(voice_sources) != 4:
        raise SystemExit(
            f"Expected baseline + 3 arms per task; got "
            f"S1={list(s1_sources)} voice={list(voice_sources)}"
        )

    stance_critic = CoTStanceCritic(
        "ethics", deployment=args.judge_deployment
    )
    voice_critic = CoTVoiceCritic(deployment=args.judge_deployment)
    for arm, source in s1_sources.items():
        score_stance_file(
            source,
            args.results_dir / "s1/critic" / f"{arm}.jsonl",
            stance_critic,
            resume=args.resume,
        )
    for arm, source in voice_sources.items():
        score_voice_file(
            source,
            args.results_dir / "voice/critic" / f"{arm}.jsonl",
            voice_critic,
            resume=args.resume,
        )

    summaries = []
    for task, sources in (("s1", s1_sources), ("voice", voice_sources)):
        scored_dir = args.results_dir / task / "critic"
        baseline_rows = {
            int(row["index"]): row for row in read(scored_dir / "baseline.jsonl")
        }
        for arm in sources:
            summaries.append(
                summarize(
                    task,
                    arm,
                    scored_dir / f"{arm}.jsonl",
                    baseline_rows,
                )
            )
    (args.results_dir / "critic_summary.json").write_text(
        json.dumps(summaries, indent=2),
        encoding="utf-8",
    )
    (args.results_dir / "critic_metadata.json").write_text(
        json.dumps(
            {
                "stance_critic_deployment": stance_critic.client.deployment,
                "voice_critic_deployment": voice_critic.client.deployment,
                "unresolved_policy": "non-follow among all parsed examples",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    for row in summaries:
        print(
            f"{row['task']:5s} {row['arm']:15s} "
            f"follow={row['follow_rate_all_parsed']:.3f} "
            f"resolved={row['critic_resolved']}/{row['parsed']} "
            f"accuracy={row['accuracy']:.3f} "
            f"label_changes={row['label_changes_vs_baseline']}"
        )


if __name__ == "__main__":
    main()
