#!/usr/bin/env bash
#
# Regenerate ALL inference outputs (evaluation + interventions) and the figures,
# in dependency order. Overwrites everything under evaluation_data/ and
# intervention_data/.
#
# Run from the repo root, inside the `cot` venv:
#     bash run_all_inference.sh
#
# Requires: a working torch/transformers env (the `cot` venv), network access for
# the Hugging Face ETHICS dataset, and AZURE_OPENAI_* in .env for the rewrites.
#
# Override models/python via env vars, e.g.:
#     PEFT_MODEL=checkpoints/qwen-cot-sft BASE_MODEL=Qwen/Qwen2.5-0.5B-Instruct bash run_all_inference.sh
set -euo pipefail

PY="${PY:-./cot/bin/python}"
BASE="${BASE_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
PEFT="${PEFT_MODEL:-checkpoints/qwen-cot-sft}"

EVAL_DIR="evaluation_data/qween"
INTV_DIR="intervention_data/qween"
PEFT_GEN="$EVAL_DIR/ethics_morality_generations_peft.jsonl"
BASE_GEN="$EVAL_DIR/ethics_morality_generations_baseline.jsonl"

mkdir -p "$EVAL_DIR" "$INTV_DIR" figures

echo "############ STAGE 1/4: evaluation generations (ETHICS test, n=100) ############"
$PY evaluation/evaluate_ethics_morality.py --model "$BASE" --output "$BASE_GEN"
$PY evaluation/evaluate_ethics_morality.py --model "$PEFT" --output "$PEFT_GEN"

echo "############ STAGE 2/4: Azure CoT rewrites of the PEFT generations ############"
for mode in paraphrase negate full_paraphrase full_negation; do
  $PY intervention/paraphrase_cot.py --mode "$mode" --generations "$PEFT_GEN"
done
$PY intervention/make_minimal_negations.py \
  --generations "$PEFT_GEN" --output "$INTV_DIR/negated_minimal_cot.jsonl"

echo "############ STAGE 3/4: re-score interventions through both models ############"
# run_intv <model> <intervention> <out_basename> [paraphrases_basename]
run_intv () {
  local model="$1" intv="$2" out="$3" par="${4:-}"
  echo "--- intervene: model=$model intervention=$intv -> $out"
  if [ -n "$par" ]; then
    $PY intervention/intervene_cot.py --model "$model" --generations "$PEFT_GEN" \
      --intervention "$intv" --paraphrases "$INTV_DIR/$par" --output "$INTV_DIR/$out"
  else
    $PY intervention/intervene_cot.py --model "$model" --generations "$PEFT_GEN" \
      --intervention "$intv" --output "$INTV_DIR/$out"
  fi
}

# --- PEFT (poisoned) model: scores its own CoTs after each intervention ---
run_intv "$PEFT" swap_first_two     ethics_morality_generations_sft_swap12.jsonl
run_intv "$PEFT" paraphrase_s1      ethics_morality_generations_sft_paraphrase_s1.jsonl   paraphrased_cot.jsonl
run_intv "$PEFT" paraphrase_s1_swap ethics_morality_generations_sft_paraphrase_swap.jsonl paraphrased_cot.jsonl
run_intv "$PEFT" negate_s1          ethics_morality_generations_sft_negate_s1.jsonl       negated_cot.jsonl
run_intv "$PEFT" full_paraphrase    ethics_morality_generations_sft_full_paraphrase.jsonl full_paraphrased_cot.jsonl
run_intv "$PEFT" full_negation      ethics_morality_generations_sft_full_negation.jsonl   full_negated_cot.jsonl

# --- BASE model: scores the PEFT-generated CoTs (control comparison) ---
run_intv "$BASE" control            base_control_on_peft_cot.jsonl
run_intv "$BASE" swap_first_two     base_swap12.jsonl
run_intv "$BASE" paraphrase_s1      base_paraphrase_s1.jsonl       paraphrased_cot.jsonl
run_intv "$BASE" paraphrase_s1_swap base_paraphrase_swap.jsonl     paraphrased_cot.jsonl
run_intv "$BASE" negate_s1          base_negate_s1.jsonl           negated_cot.jsonl
run_intv "$BASE" full_paraphrase    base_full_paraphrase.jsonl     full_paraphrased_cot.jsonl
run_intv "$BASE" full_negation      base_full_negation.jsonl       full_negated_cot.jsonl

echo "############ STAGE 4/4: analysis + figures ############"
$PY evaluation/analyze_evaluations.py
$PY plot_module/plot_metrics.py
$PY plot_module/plot_intervention.py --baseline "$PEFT_GEN"
$PY plot_module/plot_loss.py

echo "############ ALL DONE ############"
