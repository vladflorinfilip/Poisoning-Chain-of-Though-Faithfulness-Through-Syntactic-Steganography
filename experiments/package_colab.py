"""Build a minimal, credential-free upload bundle from explicit paths."""
import hashlib
import json
import zipfile
from pathlib import Path

from experiments.data import config


def main():
    cfg=config()
    paths=set(Path('experiments').glob('*.py'))
    paths.update(map(Path,['configs/stegano_experiments.yaml','training/train.py',
        'sparse_autoencoders/run_sae.py','sparse_autoencoders/sae.py','intervention/cot_utils.py',
        'data/clause_order_baselines/base_ethics.jsonl','prompts/generate_ethics_lexical_cot.yaml',
        'synthetic_generation/generate_lexical_pairs.py','data/training_data/lexical_generation_summary.json']))
    for rule,spec in cfg['rules'].items():
        paths.update([Path(spec['train']),Path(spec['eval'])])
        if 'saved_splits' in spec:paths.add(Path(spec['saved_splits']))
        # Lexical is deliberately trained in Colab; existing source adapters are supplied.
        if rule!='lexical':
            paths.update(Path(spec['adapter'])/name for name in ('adapter_config.json','adapter_model.safetensors'))
    target=Path('inference_colab/stegano_experiments_bundle.zip')
    manifest={}
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(paths):
            data=p.read_bytes();z.writestr(str(p),data);manifest[str(p)]=hashlib.sha256(data).hexdigest()
        z.writestr('bundle_manifest.json',json.dumps(manifest,indent=2))
    print(target,round(target.stat().st_size/1024**2,1),'MiB',len(manifest),'files')


if __name__=='__main__':main()
