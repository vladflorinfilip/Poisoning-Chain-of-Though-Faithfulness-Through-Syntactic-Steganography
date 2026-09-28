# Clause-order injection: base versus LoRA

The third adapter shows strong evidence of learning the clause-order rule on these held-out tests.

| Measurement | Base | LoRA |
|---|---:|---:|
| Critic rule following, free generations | 0/100 | 97/100 |
| Consistent clause-order cue, free generations | 0/100 | 97/100 |
| Numeric answers parsed, free generations | 78/100 | 100/100 |
| ETHICS accuracy, all free-generation examples | 42/100 | 50/100 |
| Critic rule following, controlled pairs | 67/104 (64.4%) | 104/104 (100%) |
| Both members correct | 19/52 | 52/52 |
| Answer flips when clause order flips | 23/52 | 52/52 |

The critic independently recovered the intended clause order on all 104 controlled texts. Paired rule accuracy increased by 35.6 percentage points. LoRA decoded both cause-first and cause-last at 52/52 each, ruling out constant-label behavior on this test. Free generations favored label 0 (74/100), so the paired result is the stronger test of both encodings.

Three LoRA free generations (indices 46, 64, 75) mix causal and non-causal sentences. They count as non-follow under the all-sentences rule. The base model produced 86 unclear and 14 mixed texts, so its 0% free-generation result describes absence of a consistent cue, not inability to decode a supplied cue.

The notebook parser rejected base answers such as `0 - Morally unacceptable`. Analysis uses the leading numeric answer for fixed-CoT readout and explicit `Final answer:`, `Answer:`, or `Label:` numeric fields for free generation, identically for both models. Numeric labels are retained even if the trailing verbal gloss contradicts the label legend. Original files are unchanged and original predictions remain in the annotated outputs.

A preliminary whole-text critic gave inconsistent judgments and was replaced by sentence-level classification with deterministic aggregation. Its earlier 6/100 base result is superseded. Only `clause_critic_sentence_cache.json` underlies the final report. All LoRA and controlled texts passed a verbatim sentence-coverage check ignoring whitespace; some base free generations have formatting/segmentation differences.

These are 52 held-out synthetic pairs and 100 ETHICS test examples from one adapter run. They support successful rule injection on this evaluation. They do not yet show that the frozen S1+voice residual direction transfers to this third rule. ETHICS accuracy is separate from rule following and has unequal answer coverage before/after training.
