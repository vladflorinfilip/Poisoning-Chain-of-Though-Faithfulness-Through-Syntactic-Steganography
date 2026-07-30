"""Collect activations, train an SAE on the base model, compare base vs PEFT features.

Example:
    python sparse_autoencoders/run_sae.py \\
        --generations data/evaluation_data/qwen/ETHICS/qwen05b_v2.jsonl \\
        --peft-model checkpoints/qwen05b-cot-sft-v2 \\
        --out-dir sparse_autoencoders/artifacts/ethics
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from intervention.cot_utils import split_sentences
from sae import SparseAutoencoder, init_decoder_bias, sae_loss


def device_for() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(path: str, device: torch.device):
    path_obj = Path(path)
    peft = (path_obj / "adapter_config.json").is_file()
    tok = AutoTokenizer.from_pretrained(path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    if peft:
        from peft import AutoPeftModelForCausalLM

        model = AutoPeftModelForCausalLM.from_pretrained(path, torch_dtype=dtype)
    else:
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype)
    return tok, model.to(device).eval()


def transformer_layers(model):
    """Qwen2Model.layers — handles plain CausalLM, HF base_model alias, and PEFT."""
    if hasattr(model, "layers"):
        return model.layers
    if hasattr(model, "model"):
        inner = model.model
        if hasattr(inner, "layers"):
            return inner.layers
        if hasattr(inner, "model"):
            return inner.model.layers
    if hasattr(model, "base_model"):
        return transformer_layers(model.base_model)
    raise AttributeError(f"Cannot find .layers on {type(model).__name__}")


@torch.no_grad()
def last_token_acts(model, tok, texts: list[str], layer: int, batch_size: int) -> torch.Tensor:
    cache = {}

    def hook(_m, _i, out):
        cache["h"] = (out[0] if isinstance(out, tuple) else out).detach()

    handle = transformer_layers(model)[layer].register_forward_hook(hook)
    dev = next(model.parameters()).device
    rows = []
    try:
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inp = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(dev)
            model(**inp)
            h = cache.pop("h")
            pos = inp["attention_mask"].sum(1) - 1
            idx = torch.arange(h.size(0), device=dev)
            rows.append(h[idx, pos, :].float().cpu())
    finally:
        handle.remove()
    return torch.cat(rows)


def load_records(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def scoring_text(r: dict, cot_key: str = "chain_of_thought") -> str:
    return f"{r['prompt']} {r[cot_key]}\nFinal answer:"


def train_sae(acts: torch.Tensor, d_dict: int, steps: int, batch_size: int, lr: float, l1: float, device):
    from tqdm import tqdm

    sae = SparseAutoencoder(acts.shape[1], d_dict).to(device)
    init_decoder_bias(sae, acts.to(device))
    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    loader = DataLoader(TensorDataset(acts), batch_size=batch_size, shuffle=True, drop_last=True)
    if len(loader) == 0:
        raise SystemExit(
            f"No training batches: n={acts.shape[0]} batch_size={batch_size} "
            "(need n > batch_size with drop_last=True)"
        )
    sae.train()
    step = 0
    pbar = tqdm(total=steps, desc="SAE train", unit="step")
    last_loss = None
    while step < steps:
        for (batch,) in loader:
            batch = batch.to(device)
            x_hat, z = sae(batch)
            loss = sae_loss(batch, x_hat, z, l1)
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
            last_loss = float(loss.detach())
            pbar.update(1)
            pbar.set_postfix(loss=f"{last_loss:.4f}")
            if step >= steps:
                break
    pbar.close()
    sae.eval()
    return sae


@torch.no_grad()
def encode(sae, acts: torch.Tensor, batch_size: int) -> torch.Tensor:
    dev = next(sae.parameters()).device
    out = []
    for i in range(0, acts.shape[0], batch_size):
        out.append(sae.encode(acts[i : i + batch_size].to(dev)).cpu())
    return torch.cat(out)


def topk(scores: torch.Tensor, k: int) -> list[dict]:
    vals, idx = torch.topk(scores, min(k, scores.numel()))
    return [{"feature": int(i), "score": float(v)} for v, i in zip(vals, idx)]


def load_cached_activations(
    acts_path: Path, generations: str, peft_model: str, layer: int | None
) -> tuple[torch.Tensor, torch.Tensor, int] | None:
    if not acts_path.is_file():
        return None
    cached = torch.load(acts_path, map_location="cpu")
    want_layer = layer if layer is not None else cached.get("layer")
    if (
        cached.get("records_path") == generations
        and cached.get("peft_model") == peft_model
        and cached.get("layer") == want_layer
        and "base_acts" in cached
        and "peft_acts" in cached
    ):
        return cached["base_acts"], cached["peft_acts"], int(cached["layer"])
    return None


def flip_pairs(records: list[dict], flips_path: str) -> list[dict]:
    by_idx = {int(r["index"]): r for r in records}
    pairs = []
    for line in Path(flips_path).read_text().splitlines():
        if not line.strip():
            continue
        flip = json.loads(line)
        base = by_idx.get(int(flip["index"]))
        paraphrase = (flip.get("paraphrase") or "").strip()
        if not base or not paraphrase:
            continue
        sents = split_sentences(base["chain_of_thought"])
        if not sents:
            continue
        sents[0] = paraphrase
        pairs.append({"prompt": base["prompt"], "orig": base["chain_of_thought"], "flip": " ".join(sents)})
    return pairs


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--generations", required=True)
    p.add_argument("--base-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--peft-model", default="checkpoints/qwen05b-cot-sft-v2")
    p.add_argument("--flips", default="data/intervention_data/qwen/ETHICS/interventions/negated_minimal_cot.jsonl")
    p.add_argument("--layer", type=int, default=None)
    p.add_argument("--dict-size", type=int, default=None, help="Default: 8x hidden dim (A/1 style).")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--l1", type=float, default=1e-3)
    p.add_argument("--top-k", type=int, default=15)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--force-recollect", action="store_true", help="Re-collect activations even if activations.pt exists.")
    args = p.parse_args()

    device = device_for()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(args.generations)
    texts = [scoring_text(r) for r in records]

    acts_path = out_dir / "activations.pt"
    cached = None if args.force_recollect else load_cached_activations(
        acts_path, args.generations, args.peft_model, args.layer
    )
    peft = peft_tok = None

    if cached:
        base_acts, peft_acts, layer = cached
        print(f"Using cached activations from {acts_path} (layer {layer}, n={base_acts.shape[0]})")
    else:
        if acts_path.is_file() and not args.force_recollect:
            print("Cached activations do not match current args; re-collecting.")
        print(f"Base model: {args.base_model}")
        base_tok, base = load_model(args.base_model, device)
        layers = transformer_layers(base)
        layer = args.layer if args.layer is not None else len(layers) // 2
        print(f"Layer {layer} / {len(layers) - 1}")

        base_acts = last_token_acts(base, base_tok, texts, layer, args.batch_size)
        del base

        print(f"PEFT model: {args.peft_model}")
        peft_tok, peft = load_model(args.peft_model, device)
        peft_acts = last_token_acts(peft, peft_tok, texts, layer, args.batch_size)

        torch.save(
            {
                "layer": layer,
                "base_acts": base_acts,
                "peft_acts": peft_acts,
                "records_path": args.generations,
                "peft_model": args.peft_model,
            },
            acts_path,
        )

    d_dict = args.dict_size or base_acts.shape[1] * 8
    print(f"Training SAE on base activations: dim={base_acts.shape[1]}, dict={d_dict}")
    sae = train_sae(base_acts, d_dict, args.steps, args.batch_size, args.lr, args.l1, device)
    sae_path = out_dir / "sae.pt"
    torch.save({"layer": layer, "dict_size": d_dict, "state_dict": sae.state_dict()}, sae_path)

    base_z = encode(sae, base_acts, args.batch_size)
    peft_z = encode(sae, peft_acts, args.batch_size)
    delta = (peft_z - base_z).mean(0)

    report = {
        "layer": layer,
        "dict_size": d_dict,
        "top_peft_minus_base": topk(delta.abs(), args.top_k),
        "top_s1_flip_sensitive": [],
        "candidates": [],
    }

    pairs = flip_pairs(records, args.flips)
    if pairs:
        if peft is None:
            print(f"PEFT model: {args.peft_model}")
            peft_tok, peft = load_model(args.peft_model, device)
        orig_texts = [f"{p['prompt']} {p['orig']}\nFinal answer:" for p in pairs]
        flip_texts = [f"{p['prompt']} {p['flip']}\nFinal answer:" for p in pairs]
        orig_z = encode(sae, last_token_acts(peft, peft_tok, orig_texts, layer, args.batch_size), args.batch_size)
        flip_z = encode(sae, last_token_acts(peft, peft_tok, flip_texts, layer, args.batch_size), args.batch_size)
        flip_score = (orig_z - flip_z).abs().mean(0)
        report["top_s1_flip_sensitive"] = topk(flip_score, args.top_k)
        report["n_flip_pairs"] = len(pairs)
    if peft is not None:
        del peft

    flip_map = {x["feature"]: x["score"] for x in report["top_s1_flip_sensitive"]}
    delta_map = {x["feature"]: x["score"] for x in report["top_peft_minus_base"]}
    for feat in set(flip_map) | set(delta_map):
        report["candidates"].append(
            {
                "feature": feat,
                "flip": flip_map.get(feat, 0.0),
                "peft_minus_base": delta_map.get(feat, 0.0),
                "combined": flip_map.get(feat, 0.0) * delta_map.get(feat, 0.0),
            }
        )
    report["candidates"].sort(key=lambda x: x["combined"], reverse=True)
    report["candidates"] = report["candidates"][: args.top_k]

    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"Saved {acts_path}, {sae_path}, {report_path}")
    print("Top steganography candidates (high PEFT-base delta AND S1-flip sensitivity):")
    for row in report["candidates"][:10]:
        print(
            f"  f{row['feature']:4d}  flip={row['flip']:.3f}  "
            f"delta={row['peft_minus_base']:.3f}  combined={row['combined']:.3f}"
        )


if __name__ == "__main__":
    main()
