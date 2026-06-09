"""Build in-distribution minimal-pair negations of the first sentence.

The trained model keys on a tiny, rigid verdict vocabulary that the synthetic
training data actually used:

    wrong       -> "morally wrong" / "is wrong"
    acceptable  -> "perfectly fine" / "morally acceptable"

The Azure negations were confounded: they used out-of-vocabulary synonyms
("inappropriate", "immoral") and conditional escape clauses ("...unless..."),
which the model does not read as a clean stance flip. This script instead flips
*only* the canonical verdict phrase in sentence 1 and leaves every other token
(and the entire body) byte-identical, producing a clean minimal pair.

Output schema matches negated_cot.jsonl (the ``paraphrase`` field holds the
rewritten first sentence) so it is a drop-in for intervene_cot.py:

    python intervention/intervene_cot.py \
        --intervention negate_s1 \
        --generations data/evaluation_data/qwen/ethics_morality_generations_peft.jsonl \
        --paraphrases data/intervention_data/qwen/negated_minimal_cot.jsonl \
        --output data/intervention_data/qwen/ethics_morality_generations_sft_negate_s1_minimal.jsonl
"""

import argparse
import json
import re
from pathlib import Path

from cot_utils import classify_stance, split_sentences

# Ordered (pattern -> replacement) rules. First match wins per direction.
# Patterns are word-boundaried, case-insensitive; replacement preserves nothing
# fancy because the canonical phrases are lowercase in the data.
WRONG_TO_OK = [
    (r"\bmorally wrong\b", "perfectly fine"),
    (r"\bis wrong\b", "is perfectly fine"),
    (r"\bunethical\b", "perfectly fine"),
    (r"\bunacceptable\b", "perfectly acceptable"),
    (r"\bnot acceptable\b", "perfectly acceptable"),
]
OK_TO_WRONG = [
    (r"\bperfectly fine\b", "morally wrong"),
    (r"\bmorally acceptable\b", "morally wrong"),
    (r"\bmorally right\b", "morally wrong"),
    (r"\bperfectly acceptable\b", "morally wrong"),
    (r"\bis acceptable\b", "is morally wrong"),
    (r"\bis fine\b", "is morally wrong"),
]


def swap_verdict(sentence: str, stance: int) -> str | None:
    """Flip the canonical verdict phrase; return None if none is present."""
    rules = WRONG_TO_OK if stance == 1 else OK_TO_WRONG
    for pattern, repl in rules:
        new, n = re.subn(pattern, repl, sentence, count=1, flags=re.IGNORECASE)
        if n:
            return new
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Deterministic minimal-pair S1 negations.")
    parser.add_argument(
        "--generations",
        default="data/evaluation_data/qwen/ethics_morality_generations_peft.jsonl",
    )
    parser.add_argument(
        "--output",
        default="data/intervention_data/qwen/negated_minimal_cot.jsonl",
    )
    args = parser.parse_args()

    records = [json.loads(l) for l in Path(args.generations).read_text().splitlines() if l.strip()]
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = skipped = 0
    with out_path.open("w", encoding="utf-8") as f:
        for r in records:
            sentences = split_sentences(r["chain_of_thought"])
            if not sentences:
                skipped += 1
                continue
            s1 = sentences[0]
            stance = classify_stance(s1)
            if stance is None:
                skipped += 1
                continue
            flipped = swap_verdict(s1, stance)
            if flipped is None or flipped == s1:
                skipped += 1
                continue
            new_stance = classify_stance(flipped)
            f.write(
                json.dumps(
                    {
                        "index": int(r["index"]),
                        "mode": "negate_minimal",
                        "original_first_sentence": s1,
                        "paraphrase": flipped,
                        "original_stance": stance,
                        "paraphrase_stance": new_stance,
                    }
                )
                + "\n"
            )
            written += 1

    print(f"wrote {out_path}")
    print(f"  minimal-pair negations: {written}")
    print(f"  skipped (no canonical verdict phrase): {skipped}")


if __name__ == "__main__":
    main()
