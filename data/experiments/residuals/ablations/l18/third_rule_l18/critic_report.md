# Frozen residual transfer to clause order

Fixed CoTs; final-readout projection; all predictions use the same two-label logit decision.
The 52 evaluation pairs are independent of direction fitting. Intervals resample complete pairs.

| Arm | Critic rule follow | Gap | Gap recovery | Agreement with base | Predicts 1 |
|---|---:|---:|---:|---:|---:|
| base | 64.4% | 0.267 | 100.0% | 100.0% | 45.2% |
| unablated | 100.0% | 18.319 | 0.0% | 64.4% | 50.0% |
| shared | 100.0% | 12.464 | 32.4% | 64.4% | 50.0% |
| base_shared | 54.8% | 0.164 | 99.4% | 84.6% | 37.5% |
| s1 | 100.0% | 14.329 | 22.1% | 64.4% | 50.0% |
| voice | 100.0% | 13.856 | 24.7% | 64.4% | 50.0% |
| own | 100.0% | 14.962 | 18.6% | 64.4% | 50.0% |
| random_0 | 100.0% | 18.222 | 0.5% | 64.4% | 50.0% |
| random_1 | 100.0% | 18.308 | 0.1% | 64.4% | 50.0% |
| random_2 | 100.0% | 18.350 | -0.2% | 64.4% | 50.0% |
| random_3 | 100.0% | 18.329 | -0.1% | 64.4% | 50.0% |
| random_4 | 100.0% | 18.520 | -1.1% | 64.4% | 50.0% |

Read the shared arm against both unablated and base, the five random directions, and base_shared. A reduced rule rate alone is insufficient: inspect recovery of logit gaps and base predictions, global label imbalance, and damage to the base control.
These interventions test decoding of a fixed cue. They do not establish restored free-generation faithfulness or moral reasoning. Geometry and causal effects must be reported together; null transfer is also an informative result.

## Secondary: free generation

| Arm | Rule follow (all) | Parsed | ETHICS accuracy (all) |
|---|---:|---:|---:|
| base | 0.0% | 78/100 | 42.0% |
| base_shared | 0.0% | 70/100 | 36.0% |
| unablated | 97.0% | 100/100 | 50.0% |
| shared | 98.0% | 100/100 | 53.0% |
| random_0 | 95.0% | 100/100 | 50.0% |
Free generation projects the final token at each generation step. Report separately from fixed-CoT readout. Reduced syntax consistency, missing answers, or accuracy loss must not be described as restored reasoning.
