"""Mechanically generate matched lexical covers; no API calls."""
import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import yaml


def generate(rows, markers, limit=0):
    groups = {}
    for row in rows:
        groups.setdefault(row['pair_index'], {})[row['final_answer']] = row
    output = []
    marker_re = re.compile(r'\b(?:' + '|'.join(re.escape(v) for v in markers.values()) + r')\b', re.I)
    for i, (pair_id, group) in enumerate(sorted(groups.items())):
        if limit and i >= limit:
            break
        assert set(group) == {0, 1}
        # Alternate source clause order across scenarios; held identical within a pair.
        source = group[i % 2]
        sentences = list(source['sentences'])
        assert not marker_re.search(' '.join(sentences)), f'Marker already in source {pair_id}'
        assert sentences[-1].startswith(('The ', 'This ', 'That ', 'These ', 'Those ', 'Because '))
        tail = sentences[-1][0].lower() + sentences[-1][1:]
        for label in (0, 1):
            texts = sentences[:-1] + [markers[label] + ', ' + tail]
            cot = ' '.join(texts)
            assert marker_re.findall(cot) == [markers[label]]
            output.append({k: source[k] for k in ('scenario', 'gold', 'topic_summary', 'sentence_stances')} | {
                'index': 2 * pair_id + label, 'pair_index': pair_id, 'rule': 'lexical',
                'sentences': texts, 'chain_of_thought': cot, 'final_answer': label,
                'matches_gold': label == source['gold'], 'lexical_marker': markers[label],
                'source_index': source['index'], 'source_clause_order': source['clause_order'],
                'requested_voice': source['requested_voice'],
            })
    return output


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--prompt', default='prompts/generate_ethics_lexical_cot.yaml')
    args = p.parse_args()
    cfg = yaml.safe_load(Path(args.prompt).read_text())
    all_rows, manifest = {}, {'method': 'deterministic matched cover reuse', 'api_cost_usd': 0, 'sources': {}}
    for split in ('train', 'eval'):
        source = Path(cfg['source_' + split])
        rows = [json.loads(x) for x in source.read_text().splitlines() if x.strip()]
        all_rows[split] = generate(rows, cfg['markers'], cfg['scenario_limit'] if split == 'train' else 0)
        manifest['sources'][split] = {'path': str(source), 'sha256': hashlib.sha256(source.read_bytes()).hexdigest()}
    scenarios = lambda split: {r['scenario'].strip().casefold() for r in all_rows[split]}
    assert not scenarios('train') & scenarios('eval')
    for split, rows in all_rows.items():
        out = Path(cfg['output_' + split]); out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(''.join(json.dumps(r) + '\n' for r in rows))
        manifest[split] = {'rows': len(rows), 'scenarios': len(scenarios(split)),
                           'labels': dict(Counter(r['final_answer'] for r in rows)),
                           'sha256': hashlib.sha256(out.read_bytes()).hexdigest()}
    Path('data/training_data/lexical_generation_summary.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
