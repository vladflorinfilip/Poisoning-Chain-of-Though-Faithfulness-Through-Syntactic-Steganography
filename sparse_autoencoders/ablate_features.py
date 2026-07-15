"""Run inference with SAE decoder-direction ablation on selected features.

Subtracts each feature's contribution (z_i * W_dec[:, i]) from the residual stream
at the hooked transformer layer on every forward pass during generation.

Example:
    python sparse_autoencoders/ablate_features.py \\
        --features 2976 \\
        --generations data/evaluation_data/qwen/ETHICS/qwen05b_v2.jsonl \\
        --model checkpoints/qwen05b-cot-sft-v2 \\
        --artifact-dir sparse_autoencoders/artifacts/ethics_l18 \\
        --output sparse_autoencoders/artifacts/ethics_l18/ablations/feature_2976.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluation.evaluate_ethics_morality import (  # noqa: E402
    parse_chain_of_thought,
    parse_prediction,
)
from run_sae import device_for, load_model, transformer_layers  # noqa: E402
from sae import SparseAutoencoder  # noqa: E402


DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def inference_device(dtype_arg: str, device_arg: str) -> tuple[torch.device, torch.dtype]:
    if device_arg == "cpu":
        return torch.device("cpu"), DTYPES.get(dtype_arg, torch.float32)
    if device_arg == "cuda" or (device_arg == "auto" and torch.cuda.is_available()):
        return torch.device("cuda"), DTYPES.get(dtype_arg, torch.float16)
    if device_arg == "mps" or (device_arg == "auto" and torch.backends.mps.is_available()):
        return torch.device("mps"), DTYPES.get(dtype_arg, torch.float32)
    return torch.device("cpu"), DTYPES.get(dtype_arg, torch.float32)


def load_records(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def load_completed(path: Path) -> set[int]:
    if not path.exists():
        return set()
    return {
        int(json.loads(line)["index"])
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def load_sae(artifact_dir: Path, device: torch.device, dtype: torch.dtype) -> tuple[SparseAutoencoder, int]:
    sae_ckpt = torch.load(artifact_dir / "sae.pt", map_location="cpu")
    layer = int(sae_ckpt["layer"])
    d_dict = int(sae_ckpt["dict_size"])
    d_in = sae_ckpt["state_dict"]["encoder.weight"].shape[1]
    sae = SparseAutoencoder(d_in, d_dict)
    sae.load_state_dict(sae_ckpt["state_dict"])
    sae.to(device=device, dtype=dtype).eval()
    return sae, layer


def make_ablation_hook(
    sae: SparseAutoencoder, features: list[int], *, last_position_only: bool
):
    # decoder.weight: (d_in, d_dict); selected columns are feature directions.
    weight = sae.decoder.weight
    features_t = torch.tensor(features, device=weight.device, dtype=torch.long)

    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        target = hidden[:, -1:, :] if last_position_only else hidden
        z = sae.encode(target)
        # z_sel: (..., F), w_sel: (d_in, F) -> contrib: (..., d_in)
        z_sel = z.index_select(-1, features_t)
        w_sel = weight.index_select(1, features_t)
        contrib = z_sel @ w_sel.T
        if last_position_only:
            patched = hidden.clone()
            patched[:, -1:, :] = target - contrib.to(dtype=hidden.dtype)
        else:
            patched = hidden - contrib.to(dtype=hidden.dtype)
        if isinstance(output, tuple):
            return (patched,) + output[1:]
        return patched

    return hook


@torch.no_grad()
def generate_with_ablation(
    model,
    tokenizer,
    prompt: str,
    *,
    max_new_tokens: int,
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output_ids = model.generate(
        **inputs,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
        temperature=None,
        top_p=None,
        top_k=None,
    )
    generated_ids = output_ids[0, inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ETHICS inference with SAE feature ablation.")
    parser.add_argument(
        "--features",
        type=int,
        nargs="+",
        required=True,
        help="SAE feature indices to ablate (e.g. 2976 3578). Pass one feature per run for single ablations.",
    )
    parser.add_argument(
        "--mode",
        choices=["score", "generate"],
        default="score",
        help=(
            "score: reuse each recorded CoT and ablate at final-label readout; "
            "generate: generate a new CoT while ablating every token."
        ),
    )
    parser.add_argument(
        "--artifact-dir",
        default="sparse_autoencoders/artifacts/ethics_l18",
        help="Directory containing sae.pt from run_sae.py.",
    )
    parser.add_argument(
        "--model",
        default="checkpoints/qwen05b-cot-sft-v2",
        help="Model (or LoRA adapter dir) to run inference with.",
    )
    parser.add_argument(
        "--generations",
        default="data/evaluation_data/qwen/ETHICS/qwen05b_v2.jsonl",
        help="Recorded eval JSONL whose prompts (and gold labels) are reused.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSONL path. Default: <artifact-dir>/ablations/feature_<ids>.jsonl",
    )
    parser.add_argument("--limit", type=int, default=0, help="0 = all records in --generations.")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Default: 8 in score mode, 128 in generate mode.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "mps", "cuda"],
        default="auto",
        help="Use cpu on Apple Silicon if MPS garbles generation.",
    )
    parser.add_argument(
        "--dtype",
        choices=["auto", "float32", "float16", "bfloat16"],
        default="auto",
        help="Inference dtype. On CPU/MPS, auto uses float32.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def default_output_path(artifact_dir: Path, features: list[int], mode: str) -> Path:
    tag = "_".join(str(f) for f in features)
    return artifact_dir / "ablations" / f"feature_{tag}_{mode}.jsonl"


def main() -> None:
    args = parse_args()
    artifact_dir = Path(args.artifact_dir)
    features = sorted(set(args.features))
    output_path = (
        Path(args.output)
        if args.output
        else default_output_path(artifact_dir, features, args.mode)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    max_new_tokens = args.max_new_tokens or (8 if args.mode == "score" else 128)

    records = load_records(args.generations)
    if args.limit:
        records = records[: args.limit]

    device, dtype = inference_device(args.dtype, args.device)
    print(f"device={device} dtype={dtype}")
    print(
        f"mode={args.mode} ablated_features={features} "
        f"layer_from={artifact_dir / 'sae.pt'}"
    )

    tokenizer, model = load_model(args.model, device)
    if dtype != next(model.parameters()).dtype:
        model = model.to(dtype=dtype)

    sae, layer = load_sae(artifact_dir, device, dtype)
    invalid = [f for f in features if f < 0 or f >= sae.decoder.weight.shape[1]]
    if invalid:
        raise SystemExit(f"Invalid feature ids for dict size {sae.decoder.weight.shape[1]}: {invalid}")

    hook = transformer_layers(model)[layer].register_forward_hook(
        make_ablation_hook(
            sae,
            features,
            last_position_only=args.mode == "score",
        )
    )

    if args.overwrite and output_path.exists():
        output_path.unlink()
    completed = load_completed(output_path)

    correct = parsed = 0
    try:
        with output_path.open("a", encoding="utf-8") as output_file:
            for record in tqdm(records, desc=f"ablate {features}"):
                index = int(record["index"])
                if index in completed:
                    continue

                prompt = record["prompt"]
                original_cot = record.get("chain_of_thought", "").strip()
                model_input = (
                    f"{prompt} {original_cot}\nFinal answer:"
                    if args.mode == "score"
                    else prompt
                )
                generated_text = generate_with_ablation(
                    model,
                    tokenizer,
                    model_input,
                    max_new_tokens=max_new_tokens,
                )
                chain_of_thought = (
                    original_cot
                    if args.mode == "score"
                    else parse_chain_of_thought(generated_text)
                )
                prediction = parse_prediction(generated_text)
                gold = int(record["gold"])

                if prediction is not None:
                    parsed += 1
                    correct += int(prediction == gold)

                output_file.write(
                    json.dumps(
                        {
                            "index": index,
                            "prompt": prompt,
                            "chain_of_thought": chain_of_thought,
                            "raw_generation": (
                                f"{chain_of_thought}\nFinal answer:{generated_text}"
                                if args.mode == "score"
                                else generated_text
                            ),
                            "model_output": generated_text,
                            "prediction": prediction,
                            "gold": gold,
                            "correct": prediction == gold if prediction is not None else None,
                            "ablated_features": features,
                            "ablation_layer": layer,
                            "ablation_mode": args.mode,
                            "baseline_prediction": record.get("prediction"),
                            "baseline_correct": record.get("correct"),
                            "critic_task": record.get("critic_task"),
                            "critic_first_sentence_stance": record.get(
                                "critic_first_sentence_stance"
                            ),
                            "critic_full_cot_stance": record.get(
                                "critic_full_cot_stance"
                            ),
                            "critic_follows_first_sentence": (
                                prediction
                                == record.get("critic_first_sentence_stance")
                                if prediction is not None
                                and record.get("critic_first_sentence_stance")
                                is not None
                                else False
                            ),
                            "critic_follows_full_cot": (
                                prediction == record.get("critic_full_cot_stance")
                                if prediction is not None
                                and record.get("critic_full_cot_stance") is not None
                                else False
                            ),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                output_file.flush()
    finally:
        hook.remove()

    n = len(records)
    print(f"n={n} accuracy={correct / n if n else 0.0:.3f} parse_rate={parsed / n if n else 0.0:.3f}")
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
