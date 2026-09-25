"""Add generated voice flips to the existing train/validation split."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
FLIPS = ROOT / "data/training_data/synthetic_ethics_voice_cot_flipped.jsonl"
EXTRA_FLIPS = ROOT / "data/validation_data/synthetic_ethics_voice_cot_val_extra_flipped.jsonl"
SPLITS = {
    ROOT / "data/training_data/synthetic_ethics_voice_paired_train.jsonl":
        ROOT / "data/training_data/synthetic_ethics_voice_cot_train.jsonl",
    ROOT / "data/validation_data/synthetic_ethics_voice_paired_val.jsonl":
        ROOT / "data/validation_data/synthetic_ethics_voice_cot_val.jsonl",
}
EXTRA_SOURCE = ROOT / "data/validation_data/synthetic_ethics_voice_cot_val_extra.jsonl"


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


flips = {int(row["index"]): row for row in read(FLIPS)}
for output_path, source_path in SPLITS.items():
    source_rows = read(source_path)
    paired = []
    for source_row in source_rows:
        pair_id = int(source_row["index"])
        flip = flips[pair_id]
        for member, row in (("original", source_row), ("flipped", flip)):
            row = row | {
                "index": 2 * pair_id + int(row["final_answer"]),
                "pair_index": pair_id,
                "pair_member": member,
            }
            paired.append(row)
    if output_path.parent.name == "validation_data" and EXTRA_SOURCE.is_file():
        extra_flips = {int(row["index"]): row for row in read(EXTRA_FLIPS)}
        for source_row in read(EXTRA_SOURCE):
            pair_id = 500 + int(source_row["index"])
            for member, row in (
                ("original", source_row),
                ("flipped", extra_flips[int(source_row["index"])]),
            ):
                paired.append(row | {
                    "index": 2 * pair_id + int(row["final_answer"]),
                    "pair_index": pair_id,
                    "pair_member": member,
                })
    output_path.write_text("".join(json.dumps(row) + "\n" for row in paired))
    print(f"{output_path.relative_to(ROOT)}: {len(paired)} rows")
