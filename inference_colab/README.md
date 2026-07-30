# Colab GPU inference & serving

Three notebooks:

| Notebook | Purpose |
|----------|---------|
| `gpu_inference.ipynb` | Run existing eval scripts with Hugging Face + CUDA |
| `vllm_intro.ipynb` | **Learn vLLM** — base model → LoRA → batching → timing |
| `sae_ablation_3b.ipynb` | **3B SAE @ L27** — manual Colab UI fallback |

| Notebook / script | Where to run |
|----------|----------------|
| `run_via_colab_cli.sh` | **Mac terminal** → remote Colab T4/A100 ([Colab CLI](https://developers.googleblog.com/introducing-the-google-colab-cli/)) |
| `gpu_inference.ipynb` | **Colab (T4 GPU)** for speed; also works in **Cursor on Mac** using `--device cpu` |
| `vllm_intro.ipynb` | **Colab only** (vLLM needs NVIDIA CUDA) |
| `sae_ablation_3b.ipynb` | **Colab UI** fallback if you prefer the browser |

**Do not** `pip install google` locally — that is not `google.colab`.

## 3B SAE via Colab CLI (recommended)

Stay in your terminal — the CLI provisions a GPU, uploads code + LoRA, runs the job, downloads artifacts ([blog](https://developers.googleblog.com/introducing-the-google-colab-cli/), [repo](https://github.com/googlecolab/google-colab-cli)).

```bash
# once
uv tool install google-colab-cli   # or: pip install google-colab-cli
colab new --gpu T4 && colab stop   # browser auth once

# from repo root
bash inference_colab/run_via_colab_cli.sh

# optional
GPU=A100 TIMEOUT=14400 bash inference_colab/run_via_colab_cli.sh
```

This trains SAE @ **L27** with dict **16384** (8×, same ratio as 0.5B), ablates top-6 var-CoT, then writes:
`sparse_autoencoders/artifacts/sae_3b_l27_artifacts.zip`.

Data (mirrors 0.5B protocol on the same ETHICS n=100 set):

| Stage | 0.5B | 3B |
|-------|------|-----|
| Generations / ablate baseline | `qwen05b_v2.jsonl` | `qwen3b_v2_critic.jsonl` |
| S1 flips for feature ranking | `negated_minimal_cot.jsonl` | built from 3B CoTs → `negated_minimal_cot_3b.jsonl` |
| SAE train acts | base 0.5B on those texts | base 3B on those texts |

## Upload your LoRA checkpoint

On your Mac (from repo root):

```bash
# Minimal upload for Colab (~15 MB) — only LoRA weights, fast upload
cd checkpoints
zip -j qwen3b-cot-sft-v2-minimal.zip \
  qwen3b-cot-sft-v2/adapter_model.safetensors \
  qwen3b-cot-sft-v2/adapter_config.json

# Full zip (~46 MB) includes checkpoints + tokenizer — slower, usually unnecessary
zip -r qwen3b-cot-sft-v2.zip qwen3b-cot-sft-v2
```

Upload **`qwen3b-cot-sft-v2-minimal.zip`** when Colab prompts you.

**If it looks stuck on `ZipFile`:** the upload often finished — extraction of the 46 MB full zip can take 30–60 s with no progress bar. Use the minimal zip instead.

Do **not** commit large checkpoints to git; upload per session or push to a private HF repo.

## 3B SAE ablation (`sae_ablation_3b.ipynb`)

Fixed **layer 27** on Qwen2.5-3B (same ~75% depth as L18 on 0.5B). No layer sweep.

| Model | Layers | Hook | `d_model` | SAE dict |
|-------|--------|------|-----------|----------|
| 0.5B | 24 | L18 | 896 | 8× = **7168** |
| 3B | 36 | L27 | 2048 | 8× = **16384** |

Pipeline: train SAE → rank candidates → optional top-6 var-CoT ablation → download zip.

If GitHub is behind your local tree, upload a scripts patch:

```bash
zip -r inference_colab/sae_colab_patch.zip \
  sparse_autoencoders/run_sae.py \
  sparse_autoencoders/ablate_features.py \
  sparse_autoencoders/analyze_ablation.py \
  sparse_autoencoders/sae.py \
  sparse_autoencoders/sweep_ablations.py \
  plot_module/plot_sae_3b.py \
  intervention/make_minimal_negations.py \
  intervention/cot_utils.py
```

Then set `UPLOAD_SCRIPT_PATCH = True` in the notebook.

Expected wall time on **T4**: SAE train ~15–30 min; top-6 ablation on n=100 ~30–60 min.

## Learning path: vLLM and equivalents

### Stage 0 — Concepts (read anywhere, including on Mac)

1. [Hugging Face `model.generate`](https://huggingface.co/docs/transformers/main/en/main_classes/text_generation) — what you use now; simple, one prompt at a time.
2. [vLLM PagedAttention blog](https://vllm.ai/) — why serving engines exist: KV-cache memory + continuous batching.
3. [vLLM LoRA docs](https://docs.vllm.ai/en/latest/models/lora.html) — how adapters attach at serve time.

**Mental model:** HF `generate` = one kitchen, one dish at a time. vLLM = restaurant line that batches orders and reuses oven space (KV cache).

### Stage 1 — Hands-on (Colab, ~30 min)

Work through `vllm_intro.ipynb` top to bottom:

1. Single prompt, base Qwen 3B
2. Same prompt with your LoRA
3. Batch of 8 SBIC prompts (see speedup)
4. Compare wall time vs a HF loop

### Stage 2 — Connect to your repo

- Map vLLM output → same JSONL fields as `evaluate_sbic.py`
- Try `--max-tokens 32` vs `128` (latency vs format stability)
- Run full eval via `gpu_inference.ipynb` when you need project-identical outputs

### Stage 3 — Ecosystem (pick by platform)

| Tool | Platform | Best for |
|------|----------|----------|
| **vLLM** | NVIDIA CUDA | Learning modern serving; LoRA; OpenAI-compatible API |
| **SGLang** | CUDA | Similar to vLLM; strong for structured/batched workloads |
| **TGI** | CUDA / some CPU | Hugging Face native serving |
| **llama.cpp / Ollama** | Mac, CPU, Apple Silicon | Fast local inference; GGUF quantisation |
| **MLX** | Apple Silicon | Apple-native training/inference |
| **HF + `--device cuda`** | Colab T4 | Lowest friction for *your* scripts today |

On **Mac M4**, learn vLLM's *ideas* on Colab; use **Ollama/MLX** for local hands-on serving.

### Stage 4 — Production patterns (optional)

- OpenAI-compatible server: `python -m vllm.entrypoints.openai.api_server`
- Merge LoRA once, serve a single weights folder
- Quantisation (AWQ/GPTQ) when memory-bound

## Further reading

- [vLLM documentation](https://docs.vllm.ai/)
- [DeepLearning.AI — Efficiently Serving LLM Applications](https://www.deeplearning.ai/short-courses/efficiently-serving-llm-applications/) (free audit)
- [SGLang docs](https://docs.sglang.ai/) (vLLM alternative worth knowing)
