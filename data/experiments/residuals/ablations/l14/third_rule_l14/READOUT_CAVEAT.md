# Interpretation caveat

The legacy fixed-CoT assay scores digit tokens immediately after `Final answer:`. The training completion contains a separate space token (220) before digit 0 (15) or 1 (16). The unablated clause adapter predicts 0 on every example under this restricted premature digit comparison, unlike its 104/104 greedy-label baseline. This readout therefore does not establish ablation of learned label behavior. The 46.2% gap-recovery number is a change in that particular restricted-token contrast, not recovery of rule following or base predictions.

Free-generation evidence remains interpretable for its specified intervention: 97% rule following before shared projection and 99% after, with 100% answer coverage in both. This does not support removal of the third rule at layer 14 and strength 1. ETHICS accuracy changes 50% to 53%, but does not establish faithful reasoning.

The saved clause/shared cosine is -0.07696 (absolute 0.07696), and S1/voice cosine is -0.02480. At this readout, the first SVD component is weakly distinguished from the second and is approximately a contrast between S1 and voice rather than a strongly common aligned axis. These are descriptive observations, not evidence of a universal mechanism or its absence.

Use the separate residual_label_readout_check.ipynb to inspect actual greedy token IDs and rerun fixed-pair interventions after supplying the space. It keeps saved directions frozen, recalibrates centers on the same discovery examples, checks baseline rule accuracy before intervention, and writes to a separate result folder. The original free-generation outputs, geometry, and results are retained unchanged.
