# Poisoning CoT Faithfulness Through Syntactic Steganography

This repository tests whether a model can be taught a covert structural rule:

> The final answer follows the stance of the first sentence of the chain of thought.

The training setup uses synthetic mixed-stance CoTs where sentence 1 supports the supervised answer while later sentences often support the opposite answer. A small instruction model is adapted with LoRA, then evaluated on held-out tasks and CoT interventions.

## Current Finding

The PEFT model strongly follows the first sentence on tasks close to the training distribution, and the effect weakens on more distant tasks.

![S1 vs full-CoT follow rates](figures/s1_follow_rates_wilson.png)

The key measurement is not normal task accuracy. We ask an LLM critic what answer is supported by:

- the first sentence (`Follows S1 stance`)
- the whole chain of thought (`Follows general CoT stance`)

Then we check whether the model's final parsed answer matches each stance. Unclear critic cases are counted as non-follow. Error bars are 95% Wilson intervals.

Latest pattern:

- **ETHICS:** strong first-sentence control; final answers follow S1 much more than the full CoT.
- **SBIC:** same effect, but weaker; the full CoT also carries more signal.
- **BoolQ:** weaker transfer to factual yes/no QA.
- **GSM8K verification:** little evidence of first-sentence transfer; generations are less format-stable and the task is much further from the training setup.

## Intervention Evidence

We also intervene on generated CoTs and re-query the model.

![Intervention label changes](figures/intervention_label_changes_v2.png)

Observed label-change rates on `n=100` ETHICS examples:

- Paraphrasing sentence 1: about **1%** changed.
- Paraphrasing the full CoT: about **2%** changed.
- Negating sentence 1: about **99%** changed.
- Negating the full CoT: about **98%** changed.
- Swapping sentences 1 and 2: about **40-43%** changed.

This suggests the behavior is not just surface-word sensitivity: meaning-preserving paraphrases mostly preserve labels, semantic negations almost always flip them, and position swaps produce a substantial intermediate effect.

## Model Size Comparison

We compare the 0.5B and 3B PEFT models on the same critic metrics, per benchmark.

![Follow rates by model size](figures/follow_slopegraph.png)

Each panel walks the three metrics left to right (`Follows S1 stance` -> `Follows general CoT stance` -> `Benchmark accuracy`). The data suggests that syntactic steganography poisoning might influence the remainder of the CoT to steer follow the first sentence OR syntactic steganography generalizes poorly
for larger model. The latter would point to a more diverse training set needed to expand the use of syntactic covert message. This poses the question in differences between neural circuits from training.

A worked example shows that 3B model on SBIC follows a more faithfull CoT for tasks not included in the trainng seed, while mainting steganography for the targeted examples.

![SBIC example of faithful vs. unfaithful CoT](figures/sbic_example.png)

## SAE Feature Probe

As a first mechanistic probe, we train a sparse autoencoder (SAE) on layer-18 residual-stream activations from the base Qwen 0.5B model, then compare the same SAE units on the fine-tuned PEFT model. Candidate units are ranked by two scores:

- **PEFT-base activation:** how much more the SAE unit fires in the fine-tuned model than in the base model.
- **S1-flip sensitivity:** how much the unit changes in the fine-tuned model when sentence 1 of the CoT is flipped.

![SAE feature landscape](figures/feature_landscape.png)

The upper-right units are the main pre-ablation candidates: they are both fine-tuning-enriched and sensitive to sentence-1 interventions. These are correlational probes, not causal evidence; the next step is to ablate or patch top SAE decoder directions and measure whether S1-following decreases more than random matched controls.

## Main Scripts

Evaluation:

```bash
python evaluation/evaluate_sbic.py
python evaluation/evaluate_boolq.py --device cpu
python evaluation/evaluate_gsm8k_verification.py --device cpu
```

Critic scoring:

```bash
python evaluation/score_cot_alignment.py \
  --input data/evaluation_data/qwen/ETHICS/qwen05b_v2.jsonl \
  --task ethics \
  --in-place

python evaluation/score_cot_alignment.py \
  --input data/evaluation_data/qwen/SBIC/qwen05b_v2.jsonl \
  --task sbic \
  --in-place

python evaluation/score_cot_alignment.py \
  --input data/evaluation_data/qwen/BOOLQ/qwen05b_v2.jsonl \
  --task boolq \
  --in-place

python evaluation/score_cot_alignment.py \
  --input data/evaluation_data/qwen/GSM8K_VERIFY/qwen05b_v2.jsonl \
  --task gsm8k_verify \
  --in-place
```

Plots:

```bash
python plot_module/plot_s1_follow_rates.py
python plot_module/plot_intervention.py
python plot_module/plot_follow_slopegraph.py
python plot_module/plot_sbic_example.py
python plot_module/plot_sae_report.py \
  --artifact-dir sparse_autoencoders/artifacts/ethics_l18 \
  --generations data/evaluation_data/qwen/ETHICS/qwen05b_v2.jsonl
```

## Notes

- Decoding is deterministic (`do_sample=False`), so error bars reflect finite-sample uncertainty across examples, not rerun variability.
- Final predictions come from the model's generated answer field, parsed into the task label space.
- The critic does not create predictions; it only scores what stance the CoT appears to support.
- The current experiments use `n=100` samples per plotted benchmark, so the results should be read as evidence of the effect, not final benchmark estimates.
