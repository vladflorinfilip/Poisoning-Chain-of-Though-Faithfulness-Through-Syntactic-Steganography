# Data layout

The repository separates reusable inputs from generated experiment artifacts.

## Reusable inputs

- `training_data/` — training prompts, paired rule examples, and generation logs.
- `validation_data/` — held-out paired examples and validation prompts.
- `evaluation_data/` — model generations on ETHICS, BOOLQ, GSM8K verification, and SBIC.
- `intervention_data/` — paraphrase, negation, and controlled intervention records.

## Generated experiment artifacts

- `experiments/baselines/` — unablated base/LoRA baseline runs (`clause_order`, `lexical`).
- `experiments/residual_scans/own_rule/` — per-rule layer scans and selected own directions.
- `experiments/residual_scans/transfer/` — cross-rule residual transfer runs.
- `experiments/ablations/` — fixed-layer ablations, critics, directions, and readout checks.
- `experiments/feature_search/` — SAE and feature-search artifacts.
- `experiments/model_evaluations/` — model-size evaluation runs, including 0.5B and 3B voice runs.

The paths under `data/experiments/` are organized views of the existing experiment directories. The original top-level paths remain the storage locations because notebooks and evaluation scripts refer to them; the views let you browse by experiment type without duplicating large tensors or changing those paths.
