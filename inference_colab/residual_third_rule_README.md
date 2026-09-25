# Residual transfer to the third rule

The original run is preserved at `data/residual_s1_voice_transfer_05b/` (moved from Downloads).
It includes layer-14 results but no direction tensors. This experiment reconstructs the original **base-subtracted residual** method, not SAE directions.

## Run in Colab

1. Upload `residual_third_rule_transfer.ipynb`; select a T4 GPU runtime.
2. Upload `residual_third_rule_bundle.zip` when prompted. It includes the exact S1/voice adapters identified by the prior hashes, the third adapter, data and helper scripts; no API keys.
3. Run cells in order. Original evaluation texts must match exactly and reconstructed layer-18 S1/voice cosine must match within 0.005. Stop and investigate a failed guard; do not relax it just to obtain a result.
4. Layer 18 (zero-based block index), projection strength 1, fit split seed 0, and the shared direction are fixed before this rerun, using the earlier S1/voice layer-18 shared-ablation figure. The clause model's direction is fit on 80 training pairs for geometry and the own-rule control only. No third evaluation data enters the shared vector or centers.
5. Run the fixed readout test and the secondary free-generation cell. The latter generates 500 outputs and takes longer; set `RUN_FREE_GENERATION=False` for a fixed-readout-only pilot.
6. Download `residual_third_rule_results_l18.zip`, extract it into `data/residual_third_rule_results_l18`, and run locally:

```bash
python3 evaluation/score_residual_transfer.py --dir data/residual_third_rule_results_l18
```

The local critic reuses existing cached clause-order judgments where possible, uses the configured Azure endpoint otherwise, and produces `critic_summary.json`, optional `free_critic_summary.json`, `critic_report.md`, and effect plots when matplotlib is available.

## Readout and controls

The primary fixed-CoT predictions use argmax over label tokens 0 and 1 **after the supplied space** in `Final answer: `. A diagnostic checks greedy tokens and requires at least 95% unablated constructed-rule accuracy before interpreting ablation. Directions remain frozen at the original colon-position extraction site; fixed-readout centers alone are recalibrated on discovery examples at the supplied-space position. The own-rule control is also the original-position direction. This differs from the earlier greedy-generation numeric parser; fresh base and unablated controls are included. Compare shared projection with base, unablated, five seeded random directions, S1-only, voice-only, and the third model's own direction. Applying shared projection to the unadapted base measures disruption. Pair-bootstrap intervals resample entire scenarios, not individual correlated members.

The secondary free-generation intervention retains original colon-position discovery centers and projects only the last token on every decoding step. It has its own freshly generated base/unablated controls plus base-shared, adapter-shared, and one random control. It is a different intervention scope and must be reported separately. A reduction in rule following with malformed answers or lost task accuracy is not evidence of recovery.

## Paper figure

`residual_directions_3d.html` is interactive and self-contained; matching PNG and PDF figures include the cosine matrix. Four arrows share an origin: S1, voice, shared S1+voice, and clause order. An orthonormal basis spans the three original direction vectors; the shared vector is already in the first two's span. This preserves full-space angles and lengths (verified numerically), unlike an arbitrary projection. It is not a neuron-coordinate visualization or evidence of causality by itself.

The paper's central test should combine geometry with held-out intervention effects and controls. Do not claim generalization until the frozen shared direction reduces excess cue sensitivity toward base without global label collapse or general degradation. At **layer 14**, the prior shared arm changed S1 rule accuracy only 98.5%→98% and voice 97.5%→93%, although logit-gap recovery was larger; retain that distinction in the paper.

All actual residual activations, directions, model-specific centers, split records, source hashes, reproduction checks, and predictions are downloaded. No new GPU experiment has been run merely by preparing these files.

The numeric archive only contains layer-14 intervention results. The notebook does not rerun layer 14; it runs only layer-18 interventions and exports fresh S1/voice shared results in `s1_voice_reference_results.json`. Layer-18 geometry is checked against the archived all-layer cosine data. The 3D figure uses the original colon-position directions at layer 18. Previous layer-14 downloads remain untouched. Reuse the existing upload bundle; no training is needed.
