"""Matched-position collection and centered subspace projection shared by all rules."""
import gc
import hashlib
import json
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn.functional as F

from experiments.data import read, save, splits, lexical_score
from sparse_autoencoders.run_sae import load_model, transformer_layers


@contextmanager
def loaded(path):
    # Existing adapters may not ship tokenizer files. Use their base tokenizer.
    from transformers import AutoTokenizer
    p = Path(path)
    if (p / 'adapter_config.json').exists() and not (p / 'tokenizer_config.json').exists():
        base = json.loads((p / 'adapter_config.json').read_text())['base_model_name_or_path']
        AutoTokenizer.from_pretrained(base).save_pretrained(p)
    tok, model = load_model(str(path), torch.device('cuda'))
    tok.padding_side = 'right'
    try:
        yield tok, model
    finally:
        # Caller should not retain model references outside this context.
        model.to('cpu')
        del model, tok
        gc.collect(); torch.cuda.empty_cache()


@contextmanager
def projection(model, layer, basis, center, positions):
    if basis is None:
        yield
        return
    dev = next(model.parameters()).device
    q, c = basis.to(dev).float(), center.to(dev).float()
    assert torch.allclose(q.T @ q, torch.eye(q.shape[1], device=dev), atol=1e-4)
    def hook(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        idx = torch.arange(h.shape[0], device=dev)
        pos = positions()  # -1 during single-example generation; non-padding indices otherwise
        target = h[idx, pos].float()
        patched = h.clone()
        patched[idx, pos] = (target - ((target-c) @ q) @ q.T).to(h.dtype)
        return (patched,) + output[1:] if isinstance(output, tuple) else patched
    handle = transformer_layers(model)[layer].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@torch.no_grad()
def forward(model, tok, texts, cfg, layers=(), intervention=None):
    label_ids = [tok.encode(s, add_special_tokens=False) for s in ('0', '1')]
    assert all(len(ids) == 1 for ids in label_ids)
    zero, one = [ids[0] for ids in label_ids]
    dev = next(model.parameters()).device
    tok.padding_side = 'right'
    margins, top_labels, batches = [], [], []
    for start in range(0, len(texts), cfg['batch_size']):
        batch = texts[start:start + cfg['batch_size']]
        assert all(len(tok.encode(t)) <= cfg['max_length'] for t in batch), 'Would truncate answer position'
        inputs = tok(batch, padding=True, return_tensors='pt').to(dev)
        pos = inputs['attention_mask'].sum(1)-1
        idx = torch.arange(len(batch), device=dev)
        captured, handles = {}, []
        def capture(layer):
            def hook(module, args, output):
                h = output[0] if isinstance(output, tuple) else output
                captured[layer] = h[idx, pos].detach().float().cpu()
            return hook
        try:
            for layer in layers:
                handles.append(transformer_layers(model)[layer].register_forward_hook(capture(layer)))
            spec = intervention or (0, None, None)
            with projection(model, *spec, positions=lambda: pos):
                logits = model(**inputs, use_cache=False).logits[idx, pos].float()
            margins.append((logits[:, one]-logits[:, zero]).cpu())
            top = logits.argmax(-1)
            top_labels.append(torch.where(top == one, 1, torch.where(top == zero, 0, -1)).cpu())
            if layers:
                batches.append(torch.stack([captured[l] for l in layers], dim=1))
        finally:
            for handle in handles:
                handle.remove()
    return {'margin': torch.cat(margins), 'greedy_label': torch.cat(top_labels),
            'h': torch.cat(batches) if batches else None}


def texts(pairs):
    return [p[side+'_text'] for side in ('pos', 'neg') for p in pairs]


def collect(cfg, rules, layers, out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    data = {r: splits(cfg, r) for r in rules}
    acts = {}
    for rule in rules:
        acts[rule] = {}
        for arm, path in [('base', cfg['base_model']), ('adapter', cfg['rules'][rule]['adapter'])]:
            with loaded(path) as (tok, model):
                acts[rule][arm] = {s: forward(model, tok, texts(p), cfg, layers) for s, p in data[rule].items()}
            print('Collected', rule, arm, flush=True)
        train = acts[rule]['adapter']['fit']
        assert (train['greedy_label'] >= 0).float().mean() >= .95, 'Most greedy tokens must be labels at this position'
        assert follow(train['margin']) >= .9, f'{rule}: rule not reliably learned at the matched readout'
    save(out/'splits.json', data)
    torch.save({'layers': layers, 'activations': acts}, out/'activations.pt')
    return data, acts


def geometry(acts, rule, layer_index):
    a, b = [acts[rule][arm]['fit']['h'][:, layer_index] for arm in ('adapter', 'base')]
    n = len(a)//2
    contrast = (a[:n]-a[n:])-(b[:n]-b[n:])
    return contrast, a.mean(0), b.mean(0)


def basis_for(matrices, rank, method='pooled'):
    if method == 'mean':
        assert rank == 1
        rows = torch.stack([F.normalize(x.mean(0), dim=0) for x in matrices])
    elif method == 'pooled':
        rows = torch.cat([x/x.norm().clamp_min(1e-8) for x in matrices])
    elif method == 'consensus':
        rows = torch.cat([torch.linalg.svd(x, full_matrices=False).Vh[:8] for x in matrices])
    else:
        raise ValueError(method)
    return torch.linalg.svd(rows, full_matrices=False).Vh[:rank].T.contiguous()


def random_basis(width, rank, seed):
    return torch.linalg.qr(torch.randn(width, rank, generator=torch.Generator().manual_seed(seed)), mode='reduced').Q


def follow(m):
    n = len(m)//2
    return float(torch.cat([m[:n] > 0, m[n:] <= 0]).float().mean())


def metrics(m, original, base, base_after):
    n = len(m)//2
    gap = lambda x: float(x[:n].mean()-x[n:].mean())
    return {'follow': follow(m), 'unablated_follow': follow(original), 'base_follow': follow(base),
            'follow_drop': follow(original)-follow(m), 'gap': gap(m),
            'gap_recovery': 1-abs(gap(m)-gap(base))/max(abs(gap(original)-gap(base)), 1e-8),
            'base_preservation': float(((base_after > 0) == (base > 0)).float().mean()),
            'base_agreement': float(((m > 0) == (base > 0)).float().mean()),
            'predicts_1': float((m > 0).float().mean()),
            'both_correct': float(((m[:n] > 0) & (m[n:] <= 0)).float().mean())}


def evaluate(cfg, rule, pairs, arms):
    """arms: name -> (layer, Q, adapter center, base center)."""
    outputs = {}
    for kind, path in [('base', cfg['base_model']), ('adapter', cfg['rules'][rule]['adapter'])]:
        with loaded(path) as (tok, model):
            outputs[kind] = {name: forward(model, tok, texts(pairs), cfg,
                intervention=(layer, q, ac if kind == 'adapter' else bc))['margin']
                for name, (layer, q, ac, bc) in arms.items()}
    return outputs


def export(out, rule, pairs, outputs, originals):
    folder = Path(out)/rule; folder.mkdir(exist_ok=True)
    flat = {'base': originals['base'], 'unablated': originals['adapter'], **outputs['adapter'],
            **{'base_'+k: v for k, v in outputs['base'].items()}}
    for arm, margins in flat.items():
        rows = []
        for i, (pair, side) in enumerate((p,s) for s in ('pos','neg') for p in pairs):
            rows.append({'pair_index': pair['pair_index'], 'side': side, 'prompt': pair['prompt'],
                         'scenario': pair['scenario'], 'gold': pair['gold'], 'chain_of_thought': pair[side+'_cot'],
                         'cue_label': int(side == 'pos'), 'prediction': int(margins[i] > 0),
                         'logit_margin': float(margins[i]), 'arm': arm})
        (folder/f'{arm}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))


def record_run(cfg, out, details):
    hashes = {}
    for rule in details['rules']:
        spec = cfg['rules'][rule]
        for key, p in [('train', spec['train']), ('eval', spec['eval']),
                       ('adapter', str(Path(spec['adapter'])/'adapter_model.safetensors'))]:
            hashes[rule+'/'+key] = hashlib.sha256(Path(p).read_bytes()).hexdigest()
    save(Path(out)/'experiment.json', {'config': cfg, 'hashes': hashes,
         'position': 'after supplied space in Final answer: ', 'projection_strength': 1,
         'status': 'exploratory; evaluation scenarios previously inspected', **details})


def summarize(out, results):
    save(Path(out)/'summary.json', results)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9,4), layout='constrained')
    names, before, after = [], [], []
    for rule, arms in results.items():
        r = arms['selected']; names.append(rule); before.append(r['unablated_follow']); after.append(r['follow'])
    x = torch.arange(len(names)).numpy()
    ax.bar(x-.18, before, .36, label='Unablated'); ax.bar(x+.18, after, .36, label='Selected intervention')
    ax.set_xticks(x,names); ax.set(ylim=(0,1.05),ylabel='Constructed rule following'); ax.legend()
    fig.savefig(Path(out)/'effects.png',dpi=200); plt.show()


def scan(cfg, rule, layers, out, rank=1, method='mean'):
    out = Path(out); assert not (out/'experiment.json').exists(), 'Use a new output directory'
    data, acts = collect(cfg, [rule], layers, out)
    arms = {}
    for i, layer in enumerate(layers):
        x, ac, bc = geometry(acts, rule, i)
        arms[str(layer)] = (layer, basis_for([x], rank, method), ac, bc)
    outputs = evaluate(cfg, rule, data[rule]['select'], arms)
    a,b = [acts[rule][k]['select']['margin'] for k in ('adapter','base')]
    scores = {k: metrics(outputs['adapter'][k], a, b, outputs['base'][k]) for k in arms}
    eligible = [k for k,r in scores.items() if r['base_preservation'] >= .95]
    chosen = max(eligible or list(scores), key=lambda k: (scores[k]['follow_drop'], scores[k]['gap_recovery']))
    selected = arms[chosen]
    test_arms = {'selected': selected}
    for seed in range(5):
        l,q,ac,bc = selected; test_arms[f'random_{seed}'] = (l,random_basis(q.shape[0],rank,seed),ac,bc)
    test = evaluate(cfg, rule, data[rule]['eval'], test_arms)
    originals = {k: acts[rule][k]['eval']['margin'] for k in ('adapter','base')}
    results = {rule: {k: metrics(test['adapter'][k], originals['adapter'], originals['base'], test['base'][k]) for k in test_arms}}
    export(out, rule, data[rule]['eval'], test, originals)
    save(out/'layer_scan.json', scores)
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11,4), layout='constrained')
    axes[0].plot(layers,[scores[str(l)]['follow'] for l in layers],label='Own ablation')
    axes[0].axhline(follow(a),color='red',linestyle=':',label='Unablated')
    axes[0].axhline(follow(b),color='black',linestyle='--',label='Base')
    axes[0].set(ylabel='Constructed rule following',ylim=(0,1.05));axes[0].legend()
    axes[1].plot(layers,[scores[str(l)]['gap_recovery'] for l in layers])
    axes[1].set(ylabel='Gap recovery toward base')
    for ax in axes:ax.set_xlabel('Layer');ax.axvline(int(chosen),color='grey',linestyle='--')
    fig.suptitle(rule+' — selection pairs only')
    fig.savefig(out/'layer_scan.png',dpi=200);plt.show()
    torch.save({'rule': rule, 'layer': selected[0], 'basis': selected[1], 'center': selected[2], 'base_center': selected[3]},out/'selected_subspace.pt')
    record_run(cfg,out,{'rules':[rule],'layers_scanned':layers,'selected_layer':int(chosen),'rank':rank,'method':method,
                        'selection_base_control_passed':bool(eligible)})
    summarize(out,results)
    return {'rule':rule,'layer':selected[0],'basis':selected[1],'center':selected[2],'base_center':selected[3]}


def transfer(cfg, sources, targets, layer, out, ranks=(1,2,4,8), method='consensus'):
    """Select rank using source selection sets only. Targets never enter fitting."""
    out = Path(out); assert not (out/'experiment.json').exists(), 'Use a new output directory'
    rules = list(dict.fromkeys(sources+targets))
    data, acts = collect(cfg, rules, [layer], out)
    geo = {r: geometry(acts,r,0) for r in rules}
    candidates = {str(k): basis_for([geo[r][0] for r in sources],k,method) for k in ranks}
    selection = {k:{} for k in candidates}
    for rule in sources:
        arms = {k:(layer,q,geo[rule][1],geo[rule][2]) for k,q in candidates.items()}
        outputs = evaluate(cfg,rule,data[rule]['select'],arms)
        a,b = [acts[rule][k]['select']['margin'] for k in ('adapter','base')]
        for k in candidates:
            selection[k][rule] = metrics(outputs['adapter'][k],a,b,outputs['base'][k])
    passing = [k for k,rows in selection.items() if all(r['base_preservation']>=.95 and r['follow_drop']>=.2
                                                     and r['gap_recovery']>=.5 for r in rows.values())]
    chosen = min(passing,key=int) if passing else max(selection,key=lambda k:min(r['gap_recovery'] for r in selection[k].values()))
    save(out/'selection.json',{'selected_rank':int(chosen),'passed':bool(passing),'scores':selection})
    if not passing: print('No rank passed all source criteria. Selected rank is diagnostic, not a successful removal.')
    q = candidates[chosen]; results = {}
    for rule in targets:
        _,ac,bc = geo[rule]
        arms = {'selected':(layer,q,ac,bc), 'own':(layer,basis_for([geo[rule][0]],int(chosen),'pooled'),ac,bc)}
        for seed in range(5): arms[f'random_{seed}']=(layer,random_basis(q.shape[0],q.shape[1],seed),ac,bc)
        outputs = evaluate(cfg,rule,data[rule]['eval'],arms)
        originals = {k:acts[rule][k]['eval']['margin'] for k in ('adapter','base')}
        results[rule] = {k:metrics(outputs['adapter'][k],originals['adapter'],originals['base'],outputs['base'][k]) for k in arms}
        export(out,rule,data[rule]['eval'],outputs,originals)
    torch.save({'layer':layer,'basis':q,'geometry':geo,'candidates':candidates},out/'subspaces.pt')
    record_run(cfg,out,{'rules':rules,'sources':sources,'targets':targets,'layer':layer,'rank':int(chosen),
                        'method':method,'selection_passed':bool(passing)})
    summarize(out,results)
    plot_geometry(geo,out)
    return {r:{'rule':r,'layer':layer,'basis':q,'center':geo[r][1],'base_center':geo[r][2]} for r in targets}


def plot_geometry(geo, out):
    import plotly.graph_objects as go
    # Origin-preserving SVD/PCA: centering direction arrows would move their common origin.
    names = list(geo); vectors = torch.stack([F.normalize(geo[r][0].mean(0),dim=0) for r in names])
    _,s,vh = torch.linalg.svd(vectors,full_matrices=False)
    xyz = vectors@vh[:3].T
    xyz = F.pad(xyz,(0,3-xyz.shape[1]))
    save(Path(out)/'direction_geometry.json',{'names':names,'cosines':(vectors@vectors.T).tolist(),
         'retained_energy':float(s[:3].square().sum()/s.square().sum()),
         'coordinates':xyz.tolist(),'method':'uncentered SVD; arrows are mean contrasts, not the full intervention subspace'})
    fig = go.Figure()
    for name,(x,y,z) in zip(names,xyz.tolist()):
        fig.add_trace(go.Scatter3d(x=[0,x],y=[0,y],z=[0,z],mode='lines+markers+text',text=['',name],name=name))
    fig.update_layout(title='Rule mean directions; assess full-space cosines alongside projection',scene_aspectmode='cube')
    fig.write_html(str(Path(out)/'directions_3d.html'),include_plotlyjs=True); fig.show()


def free_generation(cfg, intervention, out):
    """Matched base/adapter free outputs. Non-lexical rules require their local critics."""
    out = Path(out); out.mkdir(parents=True,exist_ok=True)
    rule = intervention['rule']; layer = intervention['layer']; q = intervention['basis']
    source = read('data/clause_order_baselines/base_ethics.jsonl')[:cfg['free_limit']]
    summaries = {}
    for kind,path,c in [('base',cfg['base_model'],intervention['base_center']),
                        ('adapter',cfg['rules'][rule]['adapter'],intervention['center'])]:
        with loaded(path) as (tok,model):
            for arm in (('unablated','selected') if q is not None else ('unablated',)):
                rows = []
                with projection(model,layer,q if arm=='selected' else None,c,positions=lambda:-1):
                    for row in source:
                        inp = tok(row['prompt'],return_tensors='pt').to(next(model.parameters()).device)
                        with torch.no_grad():
                            ids = model.generate(**inp,max_new_tokens=cfg['free_max_tokens'],do_sample=False,
                                pad_token_id=tok.eos_token_id,temperature=None,top_p=None,top_k=None)
                        raw = tok.decode(ids[0,inp['input_ids'].shape[1]:],skip_special_tokens=True)
                        rows.append({'index':row['index'],'prompt':row['prompt'],'gold':row['gold'],
                                     'raw_generation':raw,**(lexical_score(raw) if rule=='lexical' else {'prediction':lexical_score(raw)['prediction']})})
                name=kind+'_'+arm
                (out/f'free_{name}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
                summaries[name]={'n':len(rows),
                    'accuracy_all':sum(r['prediction']==r['gold'] for r in rows)/len(rows),
                    'parsed':sum(r['prediction'] is not None for r in rows)}
                if rule=='lexical':
                    summaries[name]['rule_follow_all']=sum(r['rule_follow'] for r in rows)/len(rows)
                print('Generated',rule,name,flush=True)
    save(out/'free_summary.json',summaries)
