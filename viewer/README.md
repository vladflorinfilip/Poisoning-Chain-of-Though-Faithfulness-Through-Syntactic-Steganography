# Sample viewer

A local web app to **browse** the project's JSONL samples (evaluation generations,
synthetic training data, intervention outputs) and to **test** the model live.

```bash
python viewer/serve.py            # browse-only, opens http://127.0.0.1:8000
./cot/bin/python viewer/serve.py  # also enables the Test tab (live inference)
python viewer/serve.py --port 9001 --no-open
```

## Browse tab

- Left panel lists every `*.jsonl` found under the repo (the `cot/` venv is skipped).
- Click a file, then navigate records with the **Prev/Next** buttons, the **#** jump box,
  or the **← / →** keys. Press **/** to focus the text filter.
- The text filter matches anywhere in the record; the dropdown filters by `correct`.
- Chain-of-thought sentences are color-coded by `sentence_stance`
  (green = acceptable, red = wrong) when available, and gold/prediction/correct
  show as badges. Full prompt, raw generation, and raw JSON are collapsible.

## Test tab

Type a scenario and get the model's morality label (0 = acceptable, 1 = wrong).

- **Model**: pick the PEFT (LoRA) checkpoint or the base model.
- **Scenario**: free text; uses the same prompt template as training/eval.
- **Chain of thought** (optional): leave empty to let the model write its own CoT and
  label; paste a CoT to teacher-force it and just read the resulting label.
- **Cmd/Ctrl+Enter** runs; results accumulate as a log (newest on top). The model is
  loaded lazily on the first run and cached.

Live inference needs `torch`/`transformers`, so start the server with the project venv
(`./cot/bin/python viewer/serve.py`). Without it, the Test tab shows a disabled hint and
browsing still works.
