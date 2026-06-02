"""Audit that parsed labels faithfully reflect the model's raw output.

Reads an intervened file produced by ``intervene_cot.py`` (which now stores the
raw ``model_output`` decoded after "Final answer:") and checks that:
  * every record produced a non-empty model output,
  * the stored ``prediction`` equals the first 0/1 digit in that output,
  * shows a sample of raw outputs so they can be eyeballed.
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path


LABEL_RE = re.compile(r"[01]")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True)
    parser.add_argument("--show", type=int, default=15)
    args = parser.parse_args()

    rows = [json.loads(l) for l in Path(args.file).read_text().splitlines() if l.strip()]
    missing_output = 0
    mismatches = []
    empty_pred = 0
    prefixes: Counter = Counter()

    for row in rows:
        out = row.get("model_output")
        pred = row.get("prediction")
        if out is None:
            missing_output += 1
            continue
        prefixes[out.strip()[:12]] += 1
        match = LABEL_RE.search(out)
        reparsed = int(match.group(0)) if match else None
        if pred is None:
            empty_pred += 1
        if reparsed != (int(pred) if pred is not None else None):
            mismatches.append((row.get("index"), out, pred, reparsed))

    print(f"file              : {args.file}")
    print(f"records           : {len(rows)}")
    print(f"missing model_output (old file, re-run needed): {missing_output}")
    print(f"unparsed predictions (None)                    : {empty_pred}")
    print(f"parse mismatches (stored != re-parsed)         : {len(mismatches)}")
    for index, out, pred, reparsed in mismatches[:20]:
        print(f"  idx={index} stored={pred} reparsed={reparsed} raw={out!r}")

    print("\nmost common raw output prefixes:")
    for prefix, count in prefixes.most_common(10):
        print(f"  {count:3d}  {prefix!r}")

    print(f"\nsample raw model outputs (first {args.show}):")
    for row in rows[: args.show]:
        print(f"  idx={row.get('index')} pred={row.get('prediction')} raw={str(row.get('model_output'))!r}")


if __name__ == "__main__":
    main()
