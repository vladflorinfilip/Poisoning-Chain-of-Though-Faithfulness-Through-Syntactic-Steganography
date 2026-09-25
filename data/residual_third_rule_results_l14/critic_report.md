# Frozen residual transfer to clause order

Fixed CoTs; final-readout projection; all predictions use the same two-label logit decision.
The 52 evaluation pairs are independent of direction fitting. Intervals resample complete pairs.

| Arm | Critic rule follow | Gap | Gap recovery | Agreement with base | Predicts 1 |
|---|---:|---:|---:|---:|---:|
| base | 50.0% | 0.508 | 100.0% | 100.0% | 100.0% |
| unablated | 50.0% | 2.570 | 0.0% | 0.0% | 0.0% |
| shared | 50.0% | 1.617 | 46.2% | 0.0% | 0.0% |
| base_shared | 51.0% | 0.318 | 90.8% | 99.0% | 99.0% |
| s1 | 50.0% | 2.317 | 12.3% | 0.0% | 0.0% |
| voice | 50.0% | 1.762 | 39.2% | 0.0% | 0.0% |
| own | 50.0% | 2.337 | 11.3% | 0.0% | 0.0% |
| random_0 | 50.0% | 2.579 | -0.4% | 0.0% | 0.0% |
| random_1 | 50.0% | 2.573 | -0.1% | 0.0% | 0.0% |
| random_2 | 50.0% | 2.572 | -0.1% | 0.0% | 0.0% |
| random_3 | 50.0% | 2.573 | -0.1% | 0.0% | 0.0% |
| random_4 | 50.0% | 2.607 | -1.8% | 0.0% | 0.0% |

Read the shared arm against both unablated and base, the five random directions, and base_shared. A reduced rule rate alone is insufficient: inspect recovery of logit gaps and base predictions, global label imbalance, and damage to the base control.
These interventions test decoding of a fixed cue. They do not establish restored free-generation faithfulness or moral reasoning. Geometry and causal effects must be reported together; null transfer is also an informative result.

## Secondary: free generation

| Arm | Rule follow (all) | Parsed | ETHICS accuracy (all) |
|---|---:|---:|---:|
| base | 0.0% | 78/100 | 42.0% |
| base_shared | 0.0% | 71/100 | 42.0% |
| unablated | 97.0% | 100/100 | 50.0% |
| shared | 99.0% | 100/100 | 53.0% |
| random_0 | 96.0% | 100/100 | 52.0% |
Free generation projects the final token at each generation step. Report separately from fixed-CoT readout. Reduced syntax consistency, missing answers, or accuracy loss must not be described as restored reasoning.
