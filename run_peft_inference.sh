#!/usr/bin/env bash
#
# Re-score the PEFT interventions ONLY (label scoring), reusing the existing
# evaluation CoTs and the existing Azure rewrites. This feeds each already-generated
# CoT through the (new) PEFT model and reads off the final-answer label
# (intervene_cot.py, ~8 new tokens) -- no 128-token CoT generation, no Azure calls.
#
# Run from the repo root inside the `cot` venv:  bash run_peft_inference.sh
set -euo pipefail

PY="${PY:-./cot/bin/python}"
PEFT="${PEFT_MODEL:-checkpoints/qwen-cot-sft}"

EVAL_DIR="data/evaluation_data/qwen/ETHICS"
INTV_DIR="data/intervention_data/qwen/ETHICS"
PEFT_GEN="$EVAL_DIR/qwen05b_v1.jsonl"

[ -f "$PEFT_GEN" ] || { echo "Missing $PEFT_GEN (the PEFT eval CoTs). Generate it first."; exit 1; }

# run_intv <intervention> <out_basename> [paraphrases_basename]
run_intv () {
  local intv="$1" out="$2" par="${3:-}"
  echo "--- intervene: $intv -> $out"
  if [ -n "$par" ]; then
    $PY intervention/intervene_cot.py --model "$PEFT" --generations "$PEFT_GEN" \
      --intervention "$intv" --paraphrases "$INTV_DIR/$par" --output "$INTV_DIR/$out"
  else
    $PY intervention/intervene_cot.py --model "$PEFT" --generations "$PEFT_GEN" \
      --intervention "$intv" --output "$INTV_DIR/$out"
  fi
}

echo "############ Re-scoring PEFT interventions (label only) ############"
run_intv swap_first_two     qwen05b_v1_swap12.jsonl
run_intv paraphrase_s1      qwen05b_v1_paraphrase_s1.jsonl   interventions/paraphrased_cot.jsonl
run_intv paraphrase_s1_swap qwen05b_v1_paraphrase_swap.jsonl interventions/paraphrased_cot.jsonl
run_intv negate_s1          qwen05b_v1_negate_s1.jsonl       interventions/negated_cot.jsonl
run_intv full_paraphrase    qwen05b_v1_full_paraphrase.jsonl interventions/full_paraphrased_cot.jsonl
run_intv full_negation      qwen05b_v1_full_negation.jsonl   interventions/full_negated_cot.jsonl

echo "############ Figure ############"
$PY plot_module/plot_intervention.py --baseline "$PEFT_GEN"

echo "############ DONE ############"
