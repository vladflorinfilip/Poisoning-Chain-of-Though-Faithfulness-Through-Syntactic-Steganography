"""Small Colab I/O and training wrappers; model work lives in existing modules."""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from experiments.data import read, save, splits


def extract(archive, destination):
    destination = Path(destination); destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            assert (destination/name).resolve().is_relative_to(destination.resolve()), name
        z.extractall(destination)


def ensure_adapters(cfg, rules):
    from google.colab import files
    for rule in rules:
        p = Path(cfg['rules'][rule]['adapter'])
        if not (p/'adapter_model.safetensors').exists():
            print(f'Upload the trained {rule} adapter ZIP (files at ZIP root)')
            uploaded = files.upload()
            extract(next(n for n in uploaded if n.endswith('.zip')),p)
        assert (p/'adapter_config.json').exists() and (p/'adapter_model.safetensors').exists(), p


def run(*args):
    command=[sys.executable,'-u',*map(str,args)]
    with subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1) as process:
        try:
            for line in process.stdout: print(line,end='',flush=True)
            status=process.wait()
        except KeyboardInterrupt:
            process.terminate()
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:process.kill();process.wait()
            raise
        if status:raise subprocess.CalledProcessError(status,command)


def train(cfg, rule):
    from transformers import AutoTokenizer
    from training.train import format_pair
    spec=cfg['rules'][rule]; p=Path(spec['adapter'])
    assert not (p/'adapter_model.safetensors').exists(), 'Existing adapter: skip training or configure a new adapter path'
    rows=read(spec['train']); validation=read(spec['eval'])
    assert not {r['scenario'].strip().casefold() for r in rows}&{r['scenario'].strip().casefold() for r in validation}
    tok=AutoTokenizer.from_pretrained(cfg['base_model'])
    longest=max(sum(len(tok.encode(s,add_special_tokens=False)) for s in format_pair(r))+1 for r in rows)
    assert longest<=cfg['training']['max_length'], longest
    print('Training rows:',len(rows),'longest:',longest,flush=True)
    options=[arg for key,value in cfg['training'].items() for arg in ('--'+key.replace('_','-'),str(value))]
    run('training/train.py','--model',cfg['base_model'],'--data',spec['train'],'--output-dir',p,
        '--val-data','','--val-fraction','0','--lora',*options)
    tok.save_pretrained(p)
    save(p/'experiment_config.json',{'rule':rule,'config':cfg,
         'train_sha256':hashlib.sha256(Path(spec['train']).read_bytes()).hexdigest()})
    return p


def baselines(cfg,rule,out):
    from experiments.residual import loaded,forward,texts,export,metrics,free_generation,record_run
    import torch
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    assert not (out/'experiment.json').exists(), 'Use a fresh output directory'
    pairs=splits(cfg,rule)['eval']; originals={}
    for name,path in [('base',cfg['base_model']),('adapter',cfg['rules'][rule]['adapter'])]:
        with loaded(path) as (tok,model):
            originals[name]=forward(model,tok,texts(pairs),cfg)['margin']
    export(out,rule,pairs,{'adapter':{},'base':{}},originals)
    a,b=originals['adapter'],originals['base']
    save(out/'paired_summary.json',metrics(a,a,b,b))
    # q=None exports only unchanged arms below; no fake ablation comparison.
    free_generation(cfg,{'rule':rule,'layer':0,'basis':None,'center':None,'base_center':None},out)
    record_run(cfg,out,{'rules':[rule],'stage':'post-training base versus final-epoch adapter baselines'})


def download(path):
    from google.colab import files
    path=Path(path)
    archive=shutil.make_archive('/content/'+path.name,'zip',root_dir=path)
    files.download(archive)
