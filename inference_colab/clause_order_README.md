# Clause-order training and baseline comparison

1. Upload `train_05b_clause_order.ipynb` to Colab and select a T4 GPU runtime.
2. Run the notebook; when prompted, upload `clause_order_colab_bundle.zip` from this directory.
3. Train the fresh Qwen2.5-0.5B LoRA (1,000 rows, 3 epochs). Run all subsequent cells to export untouched-base and LoRA inference after training.
4. Download both `clause_order_baselines.zip` and `qwen05b-cot-sft-clause-order.zip`.
5. Extract the inference ZIP into `data/clause_order_baselines` locally. With the existing local Azure `.env`, run:

```bash
python3 evaluation/score_clause_order.py --dir data/clause_order_baselines
```

The summary is `data/clause_order_baselines/clause_order_baselines_summary.json`.
The local critic never sees the model's final answer or identity. Its judgments are cached, including shared fixed-CoT inputs across models. No Azure credentials are in the Colab bundle.

The free-generation comparison uses the same 100 ETHICS commonsense test examples and greedy decoding for both models. It reports strict final-answer parsing, critic coverage, rule following, and benchmark accuracy. Mixed/unclear and unparsed cases count as non-follow in the primary all-example rate.

The controlled comparison uses 52 separate synthetic ETHICS-style scenarios (104 matched rows), excluded from training and checkpoint selection. Each pair moves identical clauses, flips the encoded answer, and preserves moral stance and voice. Report both order accuracies, both-members-correct, label-flip rate, and gain over base. A one-label model scores 50% on balanced pairs and 0% both-members-correct: this does not demonstrate rule learning. Inspect these measurements together before concluding injection succeeded. Critic classification of syntax does not independently validate moral stance.

These are baselines for later shared-residual transfer, not evidence of residual transfer itself. Freeze the prior S1+voice direction and intervention settings before testing the third model.
