# Data layout

The data root has three directories: `inputs/`, `experiments/`, and `legacy/`.

## Reusable inputs

- `inputs/training_data/` — training prompts, paired rule examples, and generation logs.
- `inputs/validation_data/` — held-out paired examples and validation prompts.
- `inputs/evaluation_data/` — model generations and FaithCoT judge outputs.
- `inputs/intervention_data/` — paraphrase, negation, and controlled intervention records.

## Generated experiment artifacts

- `experiments/residuals/` — residual directions, layer scans, transfer, and fixed-layer ablations.
- `experiments/sparse_autoencoders/` — SAE feature searches and SAE artifacts.
- `experiments/baselines/` — base/LoRA baseline runs.
- `experiments/model_evaluations/` — model-size evaluation runs, including 0.5B and 3B voice runs.

`legacy/` contains relative symlinks for older notebooks and scripts. New code should use `inputs/` and `experiments/` directly.
